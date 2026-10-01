"""
simulation/test_policy.py
==========================
Validation tests for ChongFlyMSPPolicy.

Tests
-----
1. PWMOutputLayer — affine ranges exact
2. SensorInputLayer — pack, shape, learnable scale
3. DNProjectionHead — sparse mask, output shape
4. ChongFlyMSPPolicy.from_meta() — end-to-end single step
5. step_np() — numpy interface, PWM bounds
6. Sequence forward — (batch, T, 4) output
7. Stateful inference — hx propagates across calls
8. post_step() mask enforcement
9. Gradient flow through full policy

Run
---
    python simulation/test_policy.py
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from simulation.policy import (
    ChongFlyMSPPolicy,
    SensorInputLayer,
    DNProjectionHead,
    PWMOutputLayer,
    SENSOR_DIM, N_CONTROLS,
    PWM_MIN, PWM_MAX, PWM_MID, PWM_HALF,
    CH_THROTTLE, CH_ROLL, CH_PITCH, CH_YAW,
)

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
_results = []

def check(name, cond, detail=""):
    sym = PASS if cond else FAIL
    print(f"  {sym}  {name}" + (f"  [{detail}]" if detail else ""))
    _results.append(cond)
    return cond

def section(title):
    print(f"\n{'─'*62}")
    print(f"  {title}")
    print(f"{'─'*62}")


META_PATH = os.path.join(_ROOT, "data", "reduced_models", "meta_spectral_k64.json")
W_PATH    = os.path.join(_ROOT, "data", "reduced_models", "w_spectral_k64.npy")
HAS_META  = os.path.exists(META_PATH) and os.path.exists(W_PATH)
policy    = None

BATCH = 4

# ── 1. PWMOutputLayer — affine ranges ────────────────────────────────────────

section("1. PWMOutputLayer — affine transform ranges")

pwm_layer = PWMOutputLayer()

# Max positive logit → upper bound
big  = torch.full((1, 4),  10.0)
# Max negative logit → lower bound
small= torch.full((1, 4), -10.0)
# Zero logit → neutral
zero = torch.zeros(1, 4)

pwm_big   = pwm_layer(big)[0].numpy()
pwm_small = pwm_layer(small)[0].numpy()
pwm_zero  = pwm_layer(zero)[0].numpy()

check("throttle upper bound ≈ 2000", abs(pwm_big[CH_THROTTLE]   - 2000) < 1.0,
      f"{pwm_big[CH_THROTTLE]:.1f}")
check("throttle lower bound ≈ 1000", abs(pwm_small[CH_THROTTLE] - 1000) < 1.0,
      f"{pwm_small[CH_THROTTLE]:.1f}")
check("throttle neutral ≈ 1500",     abs(pwm_zero[CH_THROTTLE]  - 1500) < 1.0,
      f"{pwm_zero[CH_THROTTLE]:.1f}")

for ch, name in [(CH_ROLL, "roll"), (CH_PITCH, "pitch"), (CH_YAW, "yaw")]:
    check(f"{name} upper bound ≈ 2000", abs(pwm_big[ch]   - 2000) < 1.0,
          f"{pwm_big[ch]:.1f}")
    check(f"{name} lower bound ≈ 1000", abs(pwm_small[ch] - 1000) < 1.0,
          f"{pwm_small[ch]:.1f}")
    check(f"{name} neutral = 1500",     abs(pwm_zero[ch]  - 1500) < 0.1,
          f"{pwm_zero[ch]:.2f}")

# decode_np
decoded = PWMOutputLayer.decode_np(pwm_zero[...])
check("decode_np throttle_pct = 50.0", abs(decoded["throttle_pct"] - 50.0) < 0.1,
      str(decoded["throttle_pct"]))
check("decode_np roll_norm ≈ 0.0",     abs(decoded["roll_norm"]) < 0.01,
      str(decoded["roll_norm"]))

# ── 2. SensorInputLayer ───────────────────────────────────────────────────────

section("2. SensorInputLayer — packing and shape")

sl = SensorInputLayer(SENSOR_DIM, learnable_scale=True)

flow = np.array([0.3, -0.1], dtype=np.float32)
tof  = np.random.rand(64).astype(np.float32)
obs_t = sl.from_numpy(flow, tof)

check("from_numpy shape", obs_t.shape == (1, 66), str(obs_t.shape))
check("flow values packed", abs(float(obs_t[0, 0]) - 0.3) < 1e-5)
check("tof values packed",  abs(float(obs_t[0, 2]) - float(tof[0])) < 1e-5)

# Forward with batch
obs_batch = torch.randn(BATCH, SENSOR_DIM)
out_sl    = sl(obs_batch)
check("forward output shape", out_sl.shape == (BATCH, SENSOR_DIM))

# Learnable scale changes output
with torch.no_grad():
    sl.gain.fill_(2.0)
out_scaled = sl(obs_batch)
check("learnable gain applied", torch.allclose(out_scaled, obs_batch * 2.0 + sl.bias))

# ── 3. DNProjectionHead ───────────────────────────────────────────────────────

section("3. DNProjectionHead — sparse motor cluster readout")

K = 32
motor_map = {
    "throttle":   [0, 1, 2],
    "roll":       [5, 6],
    "pitch":      [10, 11],
    "yaw":        [15, 16, 17],
}
dn_head = DNProjectionHead(K, motor_map)

h = torch.randn(BATCH, K)
logits = dn_head(h)
check("output shape", logits.shape == (BATCH, N_CONTROLS), str(logits.shape))
check("no NaN",        not logits.isnan().any().item())
check("no Inf",        not logits.isinf().any().item())

# Sparse mask coverage
mask = dn_head._proj_mask
check("mask shape", mask.shape == (N_CONTROLS, K), str(mask.shape))
for ch, key in enumerate(("throttle", "roll", "pitch", "yaw")):
    active = mask[ch].sum().item()
    check(f"  {key} mask non-zero", active > 0, f"{int(active)} active cols")

# Without motor map (full fallback)
dn_full = DNProjectionHead(K, motor_index_map={})
logits_f = dn_full(h)
check("fallback output shape", logits_f.shape == (BATCH, N_CONTROLS))

# ── 4. ChongFlyMSPPolicy.from_meta() ─────────────────────────────────────────

section("4. from_meta() factory — end-to-end single step")

if HAS_META:
    try:
        policy = ChongFlyMSPPolicy.from_meta(
            meta_path=META_PATH,
            mode="masked",
            backbone_units=32,
            backbone_layers=1,
            dt=0.004,
        )
        obs = torch.randn(1, SENSOR_DIM)
        pwm, hx = policy(obs)

        check("pwm shape",   pwm.shape == (1, N_CONTROLS), str(pwm.shape))
        check("pwm in [1000,2000]",
              (pwm >= 1000).all().item() and (pwm <= 2000).all().item(),
              f"min={pwm.min():.1f} max={pwm.max():.1f}")
        print(f"       pwm = {pwm[0].detach().numpy().round(1)}")
    except Exception as e:
        check("from_meta()", False, str(e))
else:
    print("  ⚠  Skipped: meta file not found")

# ── 5. step_np() — numpy interface + PWM bounds ───────────────────────────────

section("5. step_np() — numpy flight-loop interface")

if HAS_META and policy is not None:
    policy.reset_state()
    flow_xy = np.array([0.05, -0.02], dtype=np.float32)
    tof_grid = np.random.rand(64).astype(np.float32) * 0.5

    pwm_np = policy.step_np(flow_xy, tof_grid)

    check("step_np dtype",    pwm_np.dtype == np.float32, str(pwm_np.dtype))
    check("step_np shape",    pwm_np.shape == (4,),       str(pwm_np.shape))
    check("all PWM ≥ 1000",   (pwm_np >= 1000).all(),     str(pwm_np.round(1)))
    check("all PWM ≤ 2000",   (pwm_np <= 2000).all(),     str(pwm_np.round(1)))

    decoded_np = PWMOutputLayer.decode_np(pwm_np)
    print(f"       throttle: {decoded_np['throttle_us']:.1f} µs "
          f"({decoded_np['throttle_pct']:.1f}%)")
    print(f"       roll:     {decoded_np['roll_us']:.1f} µs "
          f"(norm={decoded_np['roll_norm']:.3f})")
    print(f"       pitch:    {decoded_np['pitch_us']:.1f} µs")
    print(f"       yaw:      {decoded_np['yaw_us']:.1f} µs")
else:
    print("  ⚠  Skipped")

# ── 6. Sequence forward ───────────────────────────────────────────────────────

section("6. Sequence forward (batch, T, 66) → (batch, T, 4)")

if HAS_META and policy is not None:
    T  = 15
    obs_seq = torch.randn(BATCH, T, SENSOR_DIM)
    pwm_seq, h_last = policy(obs_seq)

    check("seq pwm shape",  pwm_seq.shape == (BATCH, T, N_CONTROLS), str(pwm_seq.shape))
    check("seq h_last shape", h_last.shape == (BATCH, policy.cfc_network.cell.hidden_size))
    check("seq PWM in [1000,2000]",
          (pwm_seq >= 1000).all().item() and (pwm_seq <= 2000).all().item())
    check("seq no NaN", not pwm_seq.isnan().any().item())
else:
    print("  ⚠  Skipped")

# ── 7. Stateful inference — hx propagates ────────────────────────────────────

section("7. Stateful inference — hidden state propagation")

if HAS_META and policy is not None:
    policy.reset_state()
    obs1 = torch.randn(1, SENSOR_DIM)
    obs2 = torch.randn(1, SENSOR_DIM)

    pwm1 = policy.step(obs1)
    hx_after_1 = policy._hx.clone()

    pwm2 = policy.step(obs2)
    hx_after_2 = policy._hx.clone()

    check("hx changes after step 1", not torch.allclose(hx_after_1, torch.zeros_like(hx_after_1)))
    check("hx changes after step 2", not torch.allclose(hx_after_1, hx_after_2))

    # Reset clears state
    policy.reset_state()
    check("hx is None after reset", policy._hx is None)
else:
    print("  ⚠  Skipped")

# ── 8. post_step mask enforcement ────────────────────────────────────────────

section("8. post_step() — W_macro mask enforcement")

if HAS_META and policy is not None:
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    obs = torch.randn(BATCH, SENSOR_DIM)
    pwm_t, _ = policy(obs)
    pwm_t.sum().backward()
    opt.step()
    policy.post_step()
    check("post_step runs without error", True)
    # Verify cell mode is masked
    check("cell mode is masked", policy.cfc_network.cell.mode == "masked")
else:
    print("  ⚠  Skipped")

# ── 9. Gradient flow through full policy ─────────────────────────────────────

section("9. Gradient flow through all policy parameters")

if HAS_META and policy is not None:
    policy.zero_grad()
    obs  = torch.randn(BATCH, SENSOR_DIM)
    hx_0 = torch.randn(BATCH, policy.cfc_network.cell.hidden_size)
    pwm_out, _ = policy(obs, hx=hx_0)
    loss = pwm_out.sum()
    loss.backward()

    no_grad = []
    for name, p in policy.named_parameters():
        # motor_head is intentionally bypassed in ChongFlyMSPPolicy.forward
        if "motor_head" in name:
            continue
        if p.requires_grad and (p.grad is None or p.grad.abs().sum().item() == 0):
            no_grad.append(name)

    check("all active params receive grad", len(no_grad) == 0,
          f"zero-grad: {no_grad}" if no_grad else "")
else:
    print("  ⚠  Skipped")

# ── Summary ───────────────────────────────────────────────────────────────────

total  = len(_results)
passed = sum(_results)
failed = total - passed

print(f"\n{'='*62}")
print(f"  Results: {passed}/{total} passed"
      + (f"  — {failed} FAILED" if failed else "  — all OK ✓"))
print(f"{'='*62}\n")

if failed:
    sys.exit(1)
