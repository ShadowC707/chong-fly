"""Explicit engineering input/output routes; no inferred biological connections."""
import hashlib
from collections import deque
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

CHANNELS = ("throttle", "roll", "pitch", "yaw")
ROUTING_VERSION = "connectome-routing-v1"


def _indices(values, size, label):
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{label} mapping indices must be a list")
    if any(isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < size for i in values):
        raise ValueError(f"{label} mapping indices must be integers in [0, {size})")
    return sorted(set(int(i) for i in values))


def sensor_mask(hidden_size, input_size, populations, input_routes=None):
    """Return [target neuron, input channel] mask; unmapped memory stays disconnected.

    Physical 2/66/74-D layouts use declared flow/ToF populations. For other
    experiments input_routes explicitly maps input-column indices to neurons.
    """
    mask = torch.zeros(hidden_size, input_size, dtype=torch.bool)
    if input_routes is not None:
        for channel, targets in input_routes.items():
            _indices([channel], input_size, "input routing")
            mask[_indices(targets, hidden_size, "input routing"), channel] = True
        return mask
    if input_size not in (2, 66, 74):
        raise ValueError("Custom input layouts require explicit input_routes mapping")
    unknown = set(populations) - {"lptc_flow", "lc_looming", "memory"}
    if unknown:
        raise ValueError(f"Unknown sensor population mapping: {sorted(unknown)}")
    groups = [("lptc_flow", 0, 2)]
    if input_size >= 66:
        groups.append(("lc_looming", 2, 66))
    for key, start, end in groups:
        targets = _indices(populations.get(key, []), hidden_size, key)
        if not targets:
            raise ValueError(f"Missing sensor population mapping: {key}")
        mask[targets, start:end] = True
    if "memory" in populations:
        targets = _indices(populations["memory"], hidden_size, "memory")
        if input_size == 74:
            mask[targets, 66:74] = True
    return mask


def motor_mask(hidden_size, mapping, allow_dense_fallback=False):
    """Only exact channel names and the explicit legacy pitch_roll alias are accepted."""
    unknown = set(mapping) - set(CHANNELS) - {"pitch_roll"}
    if unknown:
        raise ValueError(f"Unknown motor channel mapping: {sorted(unknown)}")
    for key, values in mapping.items():
        _indices(values, hidden_size, key)
    mask = torch.zeros(4, hidden_size, dtype=torch.bool)
    for row, key in enumerate(CHANNELS):
        targets = mapping.get(key, mapping.get("pitch_roll", []) if key in ("pitch", "roll") else [])
        targets = _indices(targets, hidden_size, key)
        if not targets:
            if not allow_dense_fallback:
                raise ValueError(f"Missing motor channel mapping: {key}")
            mask[row] = True
        else:
            mask[row, targets] = True
    return mask


def structural_rank(mask):
    """Maximum readout rank permitted by the mask, not the rank of learned weights."""
    neighbors = [row.nonzero().flatten().tolist() for row in mask.cpu()]
    owners = {}
    def assign(row, seen):
        for column in neighbors[row]:
            if column in seen:
                continue
            seen.add(column)
            if column not in owners or assign(owners[column], seen):
                owners[column] = row
                return True
        return False
    return sum(assign(row, set()) for row in range(len(neighbors)))


def sensor_motor_paths(adjacency, populations, readout_mask):
    """Directed paths in a source→target graph; connectivity is not controllability.

    Hops count recurrent edges (a direct sensor/readout overlap has zero hops).
    The result is a snapshot of the supplied graph, not of future learned weights.
    """
    from scipy.sparse import issparse
    if issparse(adjacency):
        adjacency = adjacency.tocsr(copy=True)
        adjacency.sum_duplicates()
        adjacency.eliminate_zeros()
        neighbors = [adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i + 1]].tolist()
                     for i in range(adjacency.shape[0])]
    else:
        adjacency = torch.as_tensor(adjacency, dtype=torch.bool).cpu()
        neighbors = [row.nonzero().flatten().tolist() for row in adjacency] if adjacency.ndim == 2 else []
    readout_mask = readout_mask.bool().cpu()
    if (adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]
            or readout_mask.shape != (len(CHANNELS), adjacency.shape[0])):
        raise ValueError("Path diagnostics require square adjacency and a matching motor mask")
    targets = [row.nonzero().flatten().tolist() for row in readout_mask]
    report = {}
    for name, nodes in populations.items():
        distance = {i: 0 for i in _indices(nodes, len(neighbors), name)}
        queue = deque(distance)
        while queue:
            source = queue.popleft()
            for target in neighbors[source]:
                if target not in distance:
                    distance[target] = distance[source] + 1
                    queue.append(target)
        counts, hops = {}, {}
        for channel, indices in zip(CHANNELS, targets):
            reachable = [distance[i] for i in indices if i in distance]
            counts[channel] = len(reachable)
            hops[channel] = min(reachable) if reachable else None
        reachable_mask = torch.tensor([i in distance for i in range(len(neighbors))])
        report[name] = {"minimum_hops": hops, "reachable_motor_counts": counts,
                        "reachable_readout_rank": structural_rank(readout_mask & reachable_mask)}
    return report


class RoutedLinear(nn.Linear):
    """Masked linear adapter. Routes are rebuilt from config, never loaded as weights."""
    def __init__(self, mask):
        super().__init__(mask.shape[1], mask.shape[0], bias=True)
        self.register_buffer("route_mask", mask.clone().bool(), persistent=False)
        self.register_buffer("bias_mask", mask.any(dim=1), persistent=False)
        digest = hashlib.sha256(mask.to(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        self._routing_contract = {"version": ROUTING_VERSION, "mask_sha256": digest,
                                  "shape": list(mask.shape)}
        self.apply_mask()

    def forward(self, x):
        return F.linear(x, self.weight*self.route_mask, self.bias*self.bias_mask)

    @torch.no_grad()
    def apply_mask(self):
        self.weight.mul_(self.route_mask)
        self.bias.mul_(self.bias_mask)

    def get_extra_state(self):
        return dict(self._routing_contract)

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("Incompatible routing contract in checkpoint")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if state_dict.get(prefix + "_extra_state") != self.get_extra_state():
            error_msgs.append(prefix + "incompatible or missing routing contract")
            return
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
