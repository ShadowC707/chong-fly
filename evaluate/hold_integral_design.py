"""Round 6: choose the integral gain of the velocity hold on the identified brake model, before runs.

Model (round 2): tilt[n+1] = a*(tilt-trim) + b*u + trim, v += dt*(beta*tilt - gamma*v), flow = v/scale + noise.
Hold: 1 s moving average of flow, P = -kp*flow_mean, I += -ki*flow_mean*dt (frozen while the total is
saturated), u = clip(P + I, +-cap), one tick of delay. The trim of card 26 is inferred from its trace:
steady drift -0.024 m/s with a P-only correction of +2.6 PWM gives trim ~ -0.22 deg (the model then
reproduces the observed drift, see `check_card26`).
Rule: the largest ki on the grid whose response, for trims -0.03 / -0.22 / -0.5 deg and 5 noise seeds,
(a) never exceeds the cap, (b) has no velocity overshoot of the opposite sign beyond the 0.03 m/s
reverse threshold, and (c) gives the smallest 20 s displacement.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DT, NOISE, WINDOW = .02, .0095, 50


def model():
    return json.loads((ROOT/'output/pipeline_v1_round2/brake_identification.json').read_text(encoding='utf-8'))['model']


def simulate(m, kp, ki, trim, seed, v0=-.02, seconds=20., cap=10.):
    a, b, beta, gamma, scale = m['a'], m['b'], m['beta'], m['gamma'], m['flow_scale_mps']
    rng = np.random.default_rng(seed)
    tilt, v = trim, v0
    buf = [v0/scale]*WINDOW
    queue, integral = [0.], 0.
    xs, vs, us, x = [], [], [], 0.
    for _ in range(int(seconds/DT)):
        buf.append(v/scale+rng.normal(0, NOISE)); buf.pop(0)
        mean = float(np.mean(buf))
        p = -kp*mean
        candidate = integral-ki*mean*DT
        u = float(np.clip(p+candidate, -cap, cap))
        if abs(p+candidate) <= cap:  # anti-windup: integrate only while not saturated
            integral = candidate
        queue.append(u)
        uu = queue.pop(0)
        tilt = a*(tilt-trim)+b*uu+trim
        v += DT*(beta*tilt-gamma*v)
        x += v*DT
        xs.append(x); vs.append(v); us.append(uu)
    return dict(displacement_m=x, max_u=float(np.max(np.abs(us))), max_v=float(np.max(vs)), min_v=float(np.min(vs)),
                final_v=float(np.mean(vs[-250:])))


def check_card26(m):
    # P-only steady state with trim -0.22 deg: v = (beta/gamma)*(trim + g*u), u = -kp*v/scale
    g = m['tilt_gain_deg_per_pwm']
    trim = -.22
    k = m['beta']/m['gamma']
    v = k*trim/(1+k*g*400./m['flow_scale_mps'])
    return dict(trim_deg=trim, predicted_drift_mps=v, predicted_correction_pwm=-400.*v/m['flow_scale_mps'],
                observed_drift_mps=-.024, observed_correction_pwm=2.6)


def main(out):
    m = model()
    grid = []
    for ki in (0., 50., 100., 200., 400., 800., 1600.):
        runs = [simulate(m, 400., ki, trim, seed) for trim in (-.03, -.22, -.5) for seed in range(5)]
        grid.append(dict(ki=ki, max_u=max(r['max_u'] for r in runs), worst_overshoot_mps=max(r['max_v'] for r in runs),
                         worst_abs_displacement_m=max(abs(r['displacement_m']) for r in runs),
                         worst_final_v_mps=max(abs(r['final_v']) for r in runs)))
    ok = [g for g in grid if g['max_u'] < 10.-1e-9 and g['worst_overshoot_mps'] <= .03]
    choice = min(ok, key=lambda g: g['worst_abs_displacement_m']) if ok else None
    report = dict(card26=check_card26(m), grid=grid, chosen=choice, rule=__doc__.split('Rule: ')[1].strip())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args().out)
