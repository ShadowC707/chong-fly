"""Explicit continuation constraints; no changes to the inference equations."""
import math
import hashlib

import torch

from core.invariants import verify_recurrent_contract


class RecurrentConstraints:
    """Freeze tau/A and project masked recurrent weights after native post_step.

    The bound is captured AFTER initial scaling, or restored verbatim from
    checkpoint metadata. A Frobenius bound is not a settling-time guarantee.
    """
    VERSION = 'frozen-tau-a-frobenius-v1'
    CORE_VERSION = 'frozen-recurrent-core-v1'

    def __init__(self, policy, *, metadata=None, freeze_core=False):
        if type(freeze_core) is not bool:
            raise ValueError('freeze_core must be boolean')
        if metadata is not None:
            freeze_core = metadata.get('version') == self.CORE_VERSION
        self.freeze_core = freeze_core
        self.policy = policy
        self.frozen_parameters = ['tau_raw', 'A'] + (['_w_macro_param'] if freeze_core else [])
        version = self.CORE_VERSION if freeze_core else self.VERSION
        cell = policy.cfc_network.cell
        if cell.mode != 'masked' or not cell.preserve_signs:
            raise ValueError('Expected masked, sign-preserving recurrent weights')
        weight = cell._effective_W().detach()
        if not torch.isfinite(weight).all():
            raise ValueError('Nonfinite recurrent weights')
        bound = float(weight.norm()) if metadata is None else metadata.get('w_max_norm')
        if (isinstance(bound, bool) or not isinstance(bound, (int, float))
                or not math.isfinite(bound) or bound <= 0):
            raise ValueError('Expected a positive finite saved recurrent bound')
        if metadata is not None and (metadata.get('version') != version
                or metadata.get('frozen_parameters') != self.frozen_parameters
                or metadata.get('norm') != 'frobenius'
                or metadata.get('projection_order') != ('native_post_step_then_verify' if freeze_core else 'native_post_step_then_norm')):
            raise ValueError('Incompatible continuation constraint metadata')
        # Loading must never quietly rescale a checkpoint to make it compliant.
        if float(weight.norm()) > bound * (1 + 4*torch.finfo(weight.dtype).eps):
            raise ValueError('Checkpoint already exceeds its recorded recurrent bound')
        verify_recurrent_contract(cell)
        self.w_max_norm = float(bound)
        self.cell = cell
        self.tau_raw = cell.tau_raw.detach().clone()
        self.amplitude = cell.A.detach().clone()
        cell.tau_raw.requires_grad_(False)
        cell.A.requires_grad_(False)
        cell.tau_raw.grad = None
        cell.A.grad = None
        self.steps = self.projections = 0
        self.max_observed_norm = float(weight.norm())
        if freeze_core:
            if not policy.neutral_origin or policy.sensor_layer.encoding != 'proximity-v1':
                raise ValueError('Full-core control requires proximity-v1 and neutral_origin')
            self.reference_W = cell._w_macro_param.detach().clone()
            self.reference_sha256 = hashlib.sha256(self.reference_W.cpu().numpy().tobytes()).hexdigest()
            self.initial_decoder_norms = decoder_norms(policy) if metadata is None else metadata['initial_decoder_norms']
            if metadata is not None and (metadata.get('W_reference_sha256') != self.reference_sha256
                    or metadata.get('decoder_norm_policy') != 'monitor_only'):
                raise ValueError('Frozen core checkpoint differs from recorded reference')
            cell._w_macro_param.requires_grad_(False)
            cell._w_macro_param.grad = None
            if any(p.requires_grad for n,p in policy.named_parameters()
                   if not n.startswith(('sensor_layer.', 'cfc_network.cell.W_in.', 'dn_head.'))):
                raise ValueError('Full-core control permits only interface learning')
            verify_clear_drive(policy)

    def metadata(self):
        result = dict(version=self.CORE_VERSION if self.freeze_core else self.VERSION,
                    frozen_parameters=self.frozen_parameters,
                    w_max_norm=self.w_max_norm, norm='frobenius',
                    projection_order='native_post_step_then_verify' if self.freeze_core else 'native_post_step_then_norm')
        if self.freeze_core:
            result.update(W_reference_sha256=self.reference_sha256,
                          initial_decoder_norms=self.initial_decoder_norms, decoder_norm_policy='monitor_only')
        return result

    @torch.no_grad()
    def __call__(self, policy):
        cell = policy.cfc_network.cell
        if cell is not self.cell:
            raise ValueError('Constraint belongs to another policy')
        if (cell.tau_raw.requires_grad or cell.A.requires_grad
                or not torch.equal(cell.tau_raw, self.tau_raw.to(cell.tau_raw.device))
                or not torch.equal(cell.A, self.amplitude.to(cell.A.device))):
            raise ValueError('Frozen dynamics changed during continuation')
        # Fail if native projection was omitted; do not silently repair ordering.
        verify_recurrent_contract(cell)
        if self.freeze_core:
            if cell._w_macro_param.requires_grad or not torch.equal(
                    cell._w_macro_param, self.reference_W.to(cell._w_macro_param.device)):
                raise ValueError('Frozen recurrent W changed during interface training')
            verify_clear_drive(policy)
            self.steps += 1
            return
        norm = float(cell._effective_W().norm())
        if not math.isfinite(norm):
            raise ValueError('Nonfinite recurrent norm')
        if norm > self.w_max_norm:
            # Leave a tiny rounding margin to keep float32 norms <= the bound.
            scale = (self.w_max_norm / norm) * (1 - 4*torch.finfo(cell._w_macro_param.dtype).eps)
            cell._w_macro_param.mul_(scale)
            self.projections += 1
        actual = float(cell._effective_W().norm())
        if actual > self.w_max_norm:
            raise ValueError('Recurrent projection did not enforce its bound')
        self.max_observed_norm = max(self.max_observed_norm, actual)
        self.steps += 1

    def audit(self):
        result = dict(**self.metadata(), optimizer_steps=self.steps, projections=self.projections,
                    max_post_projection_norm=self.max_observed_norm,
                    final_norm=float(self.cell._effective_W().detach().norm()),
                    tau_unchanged=torch.equal(self.cell.tau_raw.detach(), self.tau_raw.to(self.cell.tau_raw.device)),
                    A_unchanged=torch.equal(self.cell.A.detach(), self.amplitude.to(self.cell.A.device)))
        if self.freeze_core:
            norms = decoder_norms(self.policy)
            result.update(W_unchanged=torch.equal(self.cell._w_macro_param.detach(),
                self.reference_W.to(self.cell._w_macro_param.device)),
                clear_drive_max_abs=verify_clear_drive(self.policy), decoder_norms=norms,
                decoder_frobenius_ratio=norms['frobenius']/self.initial_decoder_norms['frobenius']
                    if self.initial_decoder_norms['frobenius'] else None,
                decoder_yaw_norm_ratio=norms['yaw_l2']/self.initial_decoder_norms['yaw_l2']
                    if self.initial_decoder_norms['yaw_l2'] else None)
        return result


@torch.no_grad()
def decoder_norms(policy):
    weight = policy.dn_head.proj.weight * policy.dn_head.proj.route_mask
    if not torch.isfinite(weight).all():
        raise ValueError('Nonfinite decoder weights')
    return dict(frobenius=float(weight.norm()), yaw_l2=float(weight[3].norm()))


@torch.no_grad()
def verify_clear_drive(policy):
    """Exact zero external drive for the probe's clear, zero-flow observations."""
    cell = policy.cfc_network.cell
    obs = torch.ones(1, policy.sensor_dim, device=cell.A.device, dtype=cell.A.dtype)
    obs[:, :2] = 0
    drive = cell.W_in(policy.sensor_layer(obs)) + cell.b
    if not torch.isfinite(drive).all() or torch.count_nonzero(drive):
        raise ValueError('Recovery probe has nonzero external drive')
    return float(drive.abs().max())
