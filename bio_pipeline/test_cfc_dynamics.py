"""
bio_pipeline/test_cfc_dynamics.py
==================================
Pre-evolutionary dynamical validation of BiologicalCfCCell.

Tests
-----
1. Syntax / import sanity
2. Cell forward pass – shape, dtype, no NaN/Inf
3. Gradient flow through all trainable parameters
4. Topology modes:
   a) fixed   – W_macro has zero gradient
   b) masked  – pruned entries stay zero after optimizer step
   c) free    – all entries trainable
5. from_reduced_model() factory (spectral k=64)
6. build_network_from_meta() factory + sequence forward
7. Sensory / motor index map propagation

Run
---
    python bio_pipeline/test_cfc_dynamics.py
"""

import os, sys
import math
import numpy as np
import torch
import torch.nn as nn

# Make project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bio_pipeline.models import (
    BiologicalCfCCell,
    BiologicalCfCNetwork,
    build_network_from_meta,
)

# ── helpers ──────────────────────────────────────────────────────────────────

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
_results = []

def check(name: str, cond: bool, detail: str = ""):
    sym = PASS if cond else FAIL
    print(f"  {sym}  {name}" + (f"  [{detail}]" if detail else ""))
    _results.append(cond)
    return cond


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ── 1. Import sanity ──────────────────────────────────────────────────────────

section("1. Import sanity")
try:
    from bio_pipeline.models import BiologicalCfCCell, BiologicalCfCNetwork
    check("BiologicalCfCCell importable", True)
except Exception as e:
    check("BiologicalCfCCell importable", False, str(e))

# ── 2. Cell forward pass ──────────────────────────────────────────────────────

section("2. Cell forward pass")

K      = 32
INPUT  = 3
BATCH  = 4

cell = BiologicalCfCCell(hidden_size=K, input_size=INPUT, mode="free", dt=0.004)
W_init = torch.randn(K, K) * 0.05
cell.set_w_macro(W_init)

hx    = torch.zeros(BATCH, K)
u     = torch.randn(BATCH, INPUT)
h_new = cell(u, hx)

check("output shape",   h_new.shape == (BATCH, K),   str(h_new.shape))
check("output dtype",   h_new.dtype == torch.float32, str(h_new.dtype))
check("no NaN",         not h_new.isnan().any().item())
check("no Inf",         not h_new.isinf().any().item())
check("state changes",  not torch.allclose(h_new, hx))   # must update

# ── 3. Gradient flow ──────────────────────────────────────────────────────────
# NOTE: hx must be non-zero so hx @ W.T contributes a gradient to W_macro.

section("3. Gradient flow through all trainable params")

cell_g = BiologicalCfCCell(hidden_size=K, input_size=INPUT, mode="free", dt=0.004)
cell_g.set_w_macro(torch.randn(K, K) * 0.05)

hx = torch.randn(BATCH, K)   # non-zero: ensures grad flows into W_macro
u  = torch.randn(BATCH, INPUT)
h  = cell_g(u, hx)
loss = h.sum()
loss.backward()

for name, p in cell_g.named_parameters():
    has_grad = p.grad is not None and p.grad.abs().sum().item() > 0
    check(f"  grad({name})", has_grad)

# ── 4a. Fixed mode – W_macro frozen ──────────────────────────────────────────

section("4a. mode='fixed' — W_macro has no gradient")

cell_fix = BiologicalCfCCell(hidden_size=K, input_size=INPUT, mode="fixed", dt=0.004)
cell_fix.set_w_macro(torch.randn(K, K) * 0.05)

hx = torch.zeros(BATCH, K)
u  = torch.randn(BATCH, INPUT)
h  = cell_fix(u, hx)
h.sum().backward()

check("W_macro is buffer (no grad)", not hasattr(cell_fix, '_w_macro_param')
      or cell_fix._w_macro_param is None)
# backbone still gets grads
bb_grad = any(p.grad is not None for p in cell_fix.backbone.parameters())
check("backbone grad OK in fixed mode", bb_grad)

# ── 4b. Masked mode – pruned entries stay zero ────────────────────────────────

section("4b. mode='masked' — pruned entries zeroed after step")

K2 = 16
cell_mask = BiologicalCfCCell(hidden_size=K2, input_size=INPUT, mode="masked", dt=0.004)

W_dense = torch.randn(K2, K2) * 0.1
# Pruning: zero out lower-left quadrant
mask = torch.ones(K2, K2, dtype=torch.float32)
mask[K2//2:, :K2//2] = 0.0

cell_mask.set_w_macro(W_dense, mask=mask.bool())

opt = torch.optim.Adam(cell_mask.parameters(), lr=1e-3)

for _ in range(3):
    opt.zero_grad()
    hx = torch.randn(BATCH, K2)   # non-zero hx → grad flows into W_macro
    u  = torch.randn(BATCH, INPUT)
    h  = cell_mask(u, hx)
    h.sum().backward()
    opt.step()
    cell_mask.apply_topology_mask()     # hard-zero pruned entries

W_after = cell_mask._w_macro_param.detach()
pruned_region = W_after[K2//2:, :K2//2]
check("pruned entries remain zero", pruned_region.abs().max().item() < 1e-7,
      f"max={pruned_region.abs().max().item():.2e}")
check("unmasked entries updated", not torch.allclose(
    W_after[: K2//2, :], W_dense[:K2//2, :]))

# ── 4c. Free mode ─────────────────────────────────────────────────────────────

section("4c. mode='free' — full W_macro trainable")

cell_free = BiologicalCfCCell(hidden_size=K, input_size=INPUT, mode="free")
cell_free.set_w_macro(torch.randn(K, K) * 0.05)
opt2 = torch.optim.SGD(cell_free.parameters(), lr=0.01)
hx  = torch.randn(BATCH, K)   # non-zero hx → grad to W_macro
u   = torch.randn(BATCH, INPUT)
h   = cell_free(u, hx)
h.sum().backward()
W_before = cell_free._w_macro_param.detach().clone()
opt2.step()
W_after  = cell_free._w_macro_param.detach()
check("W_macro updated by optimizer", not torch.allclose(W_before, W_after))

# ── 5. from_reduced_model factory ────────────────────────────────────────────

section("5. from_reduced_model() factory")

META_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "reduced_models", "meta_spectral_k64.json"
)

if os.path.exists(META_PATH):
    try:
        from bio_pipeline.graph_reducer import ReducedModel
        rm = ReducedModel.load(META_PATH)
        cell_bio = BiologicalCfCCell.from_reduced_model(
            rm, input_size=2, mode="masked",
            backbone_units=32, backbone_layers=1, dt=0.004
        )
        hx = torch.zeros(2, rm.k)
        u  = torch.randn(2, 2)
        h  = cell_bio(u, hx)
        check("factory shape ok", h.shape == (2, rm.k), str(h.shape))
        check("sensor_indices propagated", len(cell_bio.sensor_indices) > 0)
        check("motor_indices propagated",  len(cell_bio.motor_indices) > 0)
        print(f"       sensor_indices: {cell_bio.sensor_indices}")
        print(f"       motor_indices:  {cell_bio.motor_indices}")
    except Exception as e:
        check("from_reduced_model", False, str(e))
else:
    print(f"  ⚠  Skipped: {META_PATH} not found — run graph_reducer.py first")

# ── 6. BiologicalCfCNetwork sequence forward ──────────────────────────────────

section("6. BiologicalCfCNetwork — sequence forward")

K3   = 32
T    = 20
cell_net = BiologicalCfCCell(hidden_size=K3, input_size=2, mode="free")
cell_net.set_w_macro(torch.randn(K3, K3) * 0.05)

net = BiologicalCfCNetwork(cell_net, output_dim=4, return_sequences=True)
seq_in = torch.randn(BATCH, T, 2)
motor_out, h_last = net(seq_in)

check("output shape (seq)",   motor_out.shape == (BATCH, T, 4),   str(motor_out.shape))
check("h_last shape",         h_last.shape    == (BATCH, K3),     str(h_last.shape))
check("no NaN in motor_out",  not motor_out.isnan().any().item())

# last-step only
net_ls = BiologicalCfCNetwork(cell_net, output_dim=4, return_sequences=False)
mo_ls, _ = net_ls(seq_in)
check("output shape (last)",  mo_ls.shape == (BATCH, 4), str(mo_ls.shape))

# post_step mask enforcement
net.post_step()
check("post_step runs without error", True)

# ── 7. build_network_from_meta factory ───────────────────────────────────────

section("7. build_network_from_meta() factory")

if os.path.exists(META_PATH):
    try:
        full_net = build_network_from_meta(
            meta_path=META_PATH,
            input_size=2,
            output_dim=4,
            mode="masked",
            backbone_units=32,
            backbone_layers=1,
            dt=0.004,
        )
        seq_in2 = torch.randn(2, 10, 2)
        out2, _ = full_net(seq_in2)
        check("factory network output ok", out2.shape == (2, 4), str(out2.shape))
    except Exception as e:
        check("build_network_from_meta", False, str(e))
else:
    print("  ⚠  Skipped: meta file not found")

# ── 8. tau positivity ─────────────────────────────────────────────────────────

section("8. Membrane time constants τ always positive")

cell_tau = BiologicalCfCCell(hidden_size=K, input_size=INPUT, mode="free")
cell_tau.set_w_macro(torch.randn(K, K) * 0.05)
# Perturb tau_raw to extreme negative
with torch.no_grad():
    cell_tau.tau_raw.fill_(-100.0)
check("tau > 0 after extreme init", (cell_tau.tau > 0).all().item())

# ── Summary ───────────────────────────────────────────────────────────────────

total  = len(_results)
passed = sum(_results)
failed = total - passed

print(f"\n{'='*60}")
print(f"  Results: {passed}/{total} passed"
      + (f"  — {failed} FAILED" if failed else "  — all OK ✓"))
print(f"{'='*60}\n")

if failed:
    sys.exit(1)
