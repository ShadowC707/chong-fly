"""Repeatable live-loop audit. Exact arrays and decisions, no numerical tolerance."""
import argparse
import json
from pathlib import Path
from time import sleep

import numpy as np

from demo.recording import save_recording
from demo.server import Pacer
from demo.sitl import SitlRunner, recompute
from evaluate.sitl_long_flight import digest, trace_metrics, write


SMOKE = [
    dict(seed=20261021, density=6, style='uniform', start=[-.7, 0., 1.], yaw_deg=0),
    dict(seed=20261022, density=18, style='clustered', start=[.4, -.6, 1.], yaw_deg=-45),
    dict(seed=20261023, density=30, style='uniform', start=[-.4, .6, 1.], yaw_deg=75),
]


def compare(a, b):
    with np.load(a/'trace.npz') as old, np.load(b/'trace.npz') as new:
        arrays = {k: k in new and old[k].dtype==new[k].dtype and
                  old[k].shape==new[k].shape and old[k].tobytes()==new[k].tobytes() for k in old.files}
    def commands(path):
        rows = json.loads((path/'command_trace.json').read_text())
        for row in rows:
            # Session identity and its manifest signature describe provenance.
            row.pop('session_id')
            row.pop('manifest_sha256')
            row['selected'].pop('session_id')
        return rows
    return dict(arrays=arrays,
        decisions=(a/'decisions.json').read_bytes()==(b/'decisions.json').read_bytes(),
        commands=commands(a)==commands(b))


def run_case(out, *, schedule, **kwargs):
    out.mkdir()
    r = SitlRunner(archive_path=out/'flight', **kwargs)
    pacer = Pacer(r)
    now = 0.
    try:
        r.event(dict(type='resume', mode='AUTO'))
        for i in range(100_000):
            t = r.time_s
            if r.terminal:
                break
            schedule(r, t, pacer, now)
            # Irregular polling, including periods shorter than a control step.
            now += (.007, .055, .014)[i%3]
            pacer.poll(now)
        if not r.terminal:
            raise RuntimeError('Audit exhausted its poll budget')
        data = r.recording()
        if data['frames']:
            save_recording(data, out/'recording.json.gz')
        commands = [f for f in r.rows if f['plant_input_pwm'] is not None]
        with np.load(out/'flight/trace.npz') as trace:
            state = trace['state']
            clearance = trace['clearance']
            clock_passed = bool(clearance.size and
                trace_metrics(trace['time_s'], state, clearance, trace['acks'])['clock_passed'])
        result = dict(duration_s=r.time_s, terminal_reason=r.pause_reason, collision=r.view.collision,
            error=r.view.error, min_clearance_m=float(clearance.min()) if clearance.size else None,
            real_time_factor=r.snapshot()['real_time_factor'],
            auto_first_s=next((f['time_s'] for f in commands if f['authority']=='AUTO'), None),
            accepted_frames=sum(f['trace']['accepted_pwm'] is not None for f in commands),
            max_height_error_m=float(np.max(np.abs(state[:,2]-1))), clock_passed=clock_passed,
            frames=len(r.rows))
        write(out/'summary.json',result)
        return data, result
    finally:
        r.close()


def run(out, reference, batch):
    out.mkdir(parents=True, exist_ok=False)
    write(out/'protocol.json',dict(reference=str(reference), batch=str(batch),
        baseline_sha256={name:digest(reference/name) for name in ('manifest.json','trace.npz',
            'command_trace.json','decisions.json')}, exact_numeric=True, pause_at_s=[.2, 9., 40.],
        stop_at_s=40., horizon_s=45., smoke_horizon_s=10., smoke=SMOKE,
        takeover=dict(seed=SMOKE[0]['seed'],key_down_s=1.,key_up_s=1.4)))
    seen = set()
    def reference_schedule(r, t, pacer, now):
        tick = r.step_index
        if tick not in (800, 36000, 160000) or tick in seen:
            return
        seen.add(tick)
        r.event(dict(type='pause'))
        count, before = len(r.rows), r.snapshot()['time_s']
        for i in range(5):
            sleep(.02)
            pacer.poll(now+i*.1)
            assert len(r.rows)==count and r.time_s==before
        r.event(dict(type='resume'))
        if tick==160000:
            r.event(dict(type='stop'))

    _, long = run_case(out/'reference', schedule=reference_schedule, scene='long_arena',
        start=[0.,0.,1.], max_duration_s=45., run_id='interactive-reference')
    baseline_vs_batch = compare(reference, batch)
    batch_vs_live = compare(batch, out/'reference/flight')
    write(out/'parity.json',dict(baseline_vs_batch=baseline_vs_batch,batch_vs_live=batch_vs_live,
                                live=long, paused_ticks=sorted(seen)))
    if any(not (all(c['arrays'].values()) and c['decisions'] and c['commands'])
           for c in (baseline_vs_batch, batch_vs_live)) or long['error']:
        raise RuntimeError('Interactive origin trace differs from batch')

    smoke = []
    for spec in SMOKE:
        _, result = run_case(out/f"seed-{spec['seed']}", schedule=lambda *_: None,
            seed=spec['seed'], start=spec['start'], yaw=float(np.radians(spec['yaw_deg'])),
            scene_options=dict(obstacle_density_pct=spec['density'],layout_style=spec['style']),
            max_duration_s=10.)
        smoke.append(dict(spec, **result))
        write(out/'smoke.json', smoke)

    seen.clear()
    def takeover(r, t, *_):
        if t not in (1., 1.4) or t in seen:
            return
        seen.add(t)
        r.event(dict(type='key',code='KeyW',down=t==1.))

    data, manual = run_case(out/'takeover', schedule=takeover, seed=SMOKE[0]['seed'],
        start=SMOKE[0]['start'], scene_options=dict(obstacle_density_pct=6,layout_style='uniform'),
        max_duration_s=2.)
    frames = data['frames']
    override = [f for f in frames if 1. <= f['time_s'] < 1.4]
    manual.update(takeover_passed=bool(override) and all(f['authority']=='MANUAL'
        and f['trace']['accepted_pwm'] is None and f['receiver']['tripped'] for f in override)
        and override[-1]['fc_rc_pwm'][2]!=1500,
        exact_recording_repeat=data['frames']==recompute(data)['frames'])
    write(out/'takeover/summary.json', manual)
    print(json.dumps(dict(reference=long, smoke=smoke, takeover=manual), indent=2), flush=True)
    if (not manual['takeover_passed'] or not manual['exact_recording_repeat']
            or any(r['error'] or not r['clock_passed'] for r in smoke)):
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('out', 'reference', 'batch'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.out, args.reference, args.batch)
