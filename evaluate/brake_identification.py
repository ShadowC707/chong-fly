"""Round 2 (stage 1, rollback): identify the Pitch -> tilt -> forward-speed dynamics from recorded
pipeline_v1_r1 working-set traces and choose the speed-proportional brake gain k BEFORE any run.

Model (20 ms ticks, body forward axis):
  tilt[n+1]  = a*tilt[n] + b*u[n-d]               u = applied Pitch - 1500 (PWM)
  v[n+1]     = v[n] + dt*(beta*tilt[n] - gamma*v[n])
  flow       = v / flow_scale, filtered with tau 0.06 s exactly as stage 1 does.
Closed loop: the network holds a constant brake u_net < 0; stage 1 passes
  u = -min(|u_net|, k*max(filtered_flow, 0)).
The chosen k is the largest one on the grid for which no start speed in the observed range and no
network brake down to the observed minimum produces any reverse motion faster than 0.03 m/s (the
reverse threshold of the section-2 metric), i.e. no counted rollback at all.
"""
import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np

DT = .02
TAU_FLOW = .06
REVERSE_MPS = .03


def load(paths):
    runs = []
    for path in paths:
        ticks = pickle.loads(Path(path).read_bytes())
        t = np.array([k['time_s'] for k in ticks])
        ok = np.array([k['coordinated'] is not None and k['stages'] is not None for k in ticks])
        yaw = np.radians([k['attitude_deg'][2] for k in ticks])
        vel = np.array([k['velocity_mps'] for k in ticks])
        v = vel[:, 0]*np.cos(yaw)+vel[:, 1]*np.sin(yaw)
        tilt = np.array([k['attitude_deg'][1] for k in ticks])
        u = np.array([(k['coordinated'] or [np.nan]*4)[2]-1500. for k in ticks])
        net = np.array([(k['network'] or [np.nan]*4)[2]-1500. for k in ticks])
        flow = np.array([k['stages'][0].get('filtered_flow', np.nan) if k['stages'] else np.nan for k in ticks])
        runs.append(dict(name=Path(path).stem, t=t, ok=ok, v=v, tilt=tilt, u=u, net=net, flow=flow))
    return runs


def fit(runs, max_delay=6):
    best = None
    for d in range(max_delay+1):
        X, y = [], []
        for r in runs:
            n = len(r['t'])
            idx = np.arange(d, n-1)
            keep = r['ok'][idx] & r['ok'][idx-d] & np.isfinite(r['u'][idx-d])
            idx = idx[keep]
            X.append(np.c_[r['tilt'][idx], r['u'][idx-d]])
            y.append(r['tilt'][idx+1])
        X, y = np.concatenate(X), np.concatenate(y)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        rss = float(((X @ coef-y)**2).mean())
        if best is None or rss < best['rss']:
            best = dict(delay_ticks=d, a=float(coef[0]), b=float(coef[1]), rss=rss,
                        r2=1-rss/float(y.var()))
    X, y = [], []
    for r in runs:
        idx = np.arange(len(r['t'])-1)
        idx = idx[r['ok'][idx]]
        X.append(np.c_[r['tilt'][idx], -r['v'][idx]])
        y.append((r['v'][idx+1]-r['v'][idx])/DT)
    X, y = np.concatenate(X), np.concatenate(y)
    (beta, gamma), *_ = np.linalg.lstsq(X, y, rcond=None)
    rss = float(((X @ [beta, gamma]-y)**2).mean())
    best.update(beta=float(beta), gamma=float(gamma), accel_r2=1-rss/float(y.var()))
    # flow scale: filtered flow is a lagged v/scale; compare the filtered speed with the stage value
    num, den = 0., 0.
    for r in runs:
        vf = np.zeros_like(r['v'])
        for i in range(1, len(vf)):
            vf[i] = vf[i-1]+(-math.expm1(-DT/TAU_FLOW))*(r['v'][i]-vf[i-1])
        m = r['ok'] & np.isfinite(r['flow'])
        num += float((vf[m]*r['flow'][m]).sum())
        den += float((r['flow'][m]**2).sum())
    best['flow_scale_mps'] = num/den
    best['tilt_tau_s'] = -DT/math.log(best['a'])
    best['tilt_gain_deg_per_pwm'] = best['b']/(1-best['a'])
    return best


def stage1_v1(u_net, flow, v_stop=.0105):
    x = min(max(flow/v_stop, 0.), 1.)
    return u_net*x*x*(3-2*x)


def stage1_prop(u_net, flow, k):
    return -min(-u_net, k*max(flow, 0.))


def simulate(model, law, v0, u_net, seconds=4.):
    d = model['delay_ticks']
    alpha = -math.expm1(-DT/TAU_FLOW)
    tilt, v, flow = 0., v0, v0/model['flow_scale_mps']
    # start in steady forward cruise: tilt that balances drag at v0
    tilt = model['gamma']*v0/model['beta']
    queue = [tilt/model['tilt_gain_deg_per_pwm']]*(d+1)
    reverse_m, min_v, stop_m = 0., v0, 0.
    for _ in range(int(seconds/DT)):
        flow += alpha*(v/model['flow_scale_mps']-flow)
        queue.append(law(u_net, flow))
        u = queue.pop(0)
        tilt = model['a']*tilt+model['b']*u
        v = v+DT*(model['beta']*tilt-model['gamma']*v)
        min_v = min(min_v, v)
        if v < -REVERSE_MPS:
            reverse_m += -v*DT
        if v > 0:
            stop_m += v*DT
    return dict(reverse_m=reverse_m, min_v_mps=min_v, stop_m=stop_m)


def main(paths, out):
    runs = load(paths)
    model = fit(runs)
    brakes = np.concatenate([r['net'][r['ok'] & (r['net'] < -10)] for r in runs])  # raw network brake
    speeds = np.concatenate([r['v'][r['ok']] for r in runs])
    v_max = float(np.quantile(speeds, .995))
    u_min = float(brakes.min())
    starts = np.linspace(.05, v_max, 8)
    nets = np.linspace(-20, u_min, 8)
    check_v1 = max((simulate(model, stage1_v1, v0, un) for v0 in starts for un in nets),
                   key=lambda r: r['reverse_m'])
    grid = []
    for k in np.arange(200., 4001., 100.):
        worst = [simulate(model, lambda un, f: stage1_prop(un, f, k), v0, un) for v0 in starts for un in nets]
        grid.append(dict(k=float(k), worst_reverse_m=max(w['reverse_m'] for w in worst),
                         worst_min_v_mps=min(w['min_v_mps'] for w in worst),
                         stop_m_at_vmax_full_brake=simulate(model, lambda un, f: stage1_prop(un, f, k),
                                                            v_max, u_min)['stop_m']))
    safe = [g for g in grid if g['worst_min_v_mps'] >= -REVERSE_MPS]
    choice = max(safe, key=lambda g: g['k']) if safe else None
    report = dict(model=model, samples=int(sum(r['ok'].sum() for r in runs)), runs=[r['name'] for r in runs],
                  observed=dict(speed_p995_mps=v_max, min_network_brake_pwm=u_min),
                  v1_law_worst=check_v1, grid=grid, chosen=choice,
                  rule='largest k with min forward speed >= -0.03 m/s for all starts 0.05..p99.5 speed '
                       'and all constant network brakes -20..observed minimum')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(dict(model=model, observed=report['observed'], v1_law_worst=check_v1, chosen=choice), indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('traces', nargs='+', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.traces, a.out)
