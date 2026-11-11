"""Compare neural/teacher decisions on identical inputs and teacher-driven flights.

Research overlay only: leaves the pinned runtime and coordinator untouched.
The teacher uses only the policy's normalized observation and reset history.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class PolicyComparison:
    def __init__(self, policies, teacher, driver):
        if 'before' not in policies or driver not in {*policies, 'teacher'}:
            raise ValueError('Unknown driver or missing baseline')
        self.policies, self.teacher, self.driver = policies, teacher, driver
        self.records = []
        self.segment = -1
        self.reset_state()

    def __getattr__(self, key):
        # Demo diagnostics retain the baseline's neural state. The manifest
        # explicitly declares that teacher-driven actions are not neural actions.
        return getattr(self.policies['before'], key)

    def reset_state(self):
        for policy in self.policies.values():
            policy.reset_state()
        self.teacher.reset()
        self.segment += 1

    def step_np(self, flow_xy, tof_8x8, memory_ring=None, dt=None):
        dt = self.default_dt if dt is None else dt
        memory = np.ones(8, dtype=np.float32) if memory_ring is None else memory_ring
        x = np.concatenate([np.ravel(flow_xy), np.ravel(tof_8x8), memory]).astype(np.float32)
        outputs = {name: p.step_np(x[:2], x[2:66], x[66:], dt=dt)
                   for name, p in self.policies.items()}
        outputs['teacher'] = self.teacher.step(x, dt=dt)
        self.records.append(dict(input=x.copy(), driver=self.driver, outputs={k:v.copy() for k,v in outputs.items()},
                                 segment=self.segment))
        return outputs[self.driver].copy()


def run(output):
    from demo.runner import DemoRunner
    from demo.recording import plain
    from edge.system import FlightSystem
    from edge.runtime import NumpyPolicy, sha256
    from evaluate.compare_maneuver_repair import summarize
    from generator.generate_reflex_dataset import make_teacher

    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    source = ROOT/'output/neural_flow_balance_v1'
    cases = ('068fe108', '89d81a9f', '9d5431c1', 'b0a9f212')
    metadata = json.loads((ROOT/'data/reflex_training_v5_pilot/train_report.json').read_text())['metadata']
    protocol = dict(version='maneuver-teacher-check-v1', cases=cases, horizon_s=45,
        drivers=['before', 'teacher'], comparison_models=['before','control','active_fast'],
        teacher_version=metadata['teacher_version'], teacher_parameters=metadata['teacher_parameters'],
        coordinator='both/original_v1 unchanged', reset='Same runtime resets for all observers',
        script_sha256=sha256(__file__), model_sha256={a:sha256(source/a/'checkpoint.pt')
            for a in ('before','control','active_fast')},
        promotion=False, handover='Both use baseline until first admitted AUTO; teacher then takes over without state reset.',
        note='Teacher-driven flights are counterfactual, not teacher replay labels. Teacher cruise exceeds neutral handover envelope.')
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    reports = {}
    for case in cases:
        reports[case] = {}
        m = json.loads((source/'before'/case/'manifest.json').read_text())
        for driver in ('before','teacher'):
            dest = output/(case+'_'+driver); dest.mkdir()
            r = DemoRunner(config=source/'before/runtime_config.json', scene=m['scene'], seed=m['seed'],
                start=m['start']['position_m'], yaw=m['start']['yaw_rad'], scene_options=m['scene_options'],
                collision_radius_m=.2, max_duration_s=45, contact_model='sphere-sweep-v1')
            try:
                for key in ('room','boxes','cylinders'):
                    assert r.manifest()['geometry'][key] == m['geometry'][key]
                policies = {a:NumpyPolicy(source/a/'numpy') for a in protocol['comparison_models']}
                comparison = PolicyComparison(policies, make_teacher(metadata['teacher_version'],
                    **metadata['teacher_parameters']), 'before')
                r.system.runtime.policy = comparison
                manifest = r.system.manifest
                manifest['version'] = 'teacher-check-overlay-v1'
                manifest['experiment'] = dict(driver=driver, research_only=True,
                    neural_display='baseline observer, not teacher state', source_sha256=sha256(__file__))
                r.system = FlightSystem(r.system.config, r.system.runtime, manifest)
                r._manifest = plain(r._build_manifest())
                r.resume(); r.event(dict(type='mode',mode='AUTO'))
                for _ in range(11250):
                    if driver == 'teacher' and r.system.bridge.state == 'AUTO':
                        comparison.driver = 'teacher'
                    n = len(comparison.records)
                    now = r.time_s
                    r.advance()
                    if len(comparison.records) > n:
                        rec = comparison.records[-1]
                        rec.update(time_s=now, position_m=r.rows[-1]['position_m'],
                            applied_pwm=r.rows[-1]['plant_input_pwm'],
                            coordinated_pwm=r.rows[-1]['trace'].get('coordinated_pwm'),
                            plant_source=r.rows[-1]['plant_source'])
                    if r.terminal: break
                report = dict(metrics=summarize(r.rows, r.time_s), resets=comparison.segment,
                              decisions=len(comparison.records), events=r.events,
                              teacher_takeover_s=next((v['time_s'] for v in comparison.records if v['driver']=='teacher'),None))
                if report['metrics']['auto_s'] <= 0:
                    raise RuntimeError('No AUTO: not a valid behavior comparison')
                if driver == 'before':
                    with np.load(source/'before'/case/'trace.npz') as trace:
                        controls=r.rows[::5]
                        np.testing.assert_array_equal(trace['pos'],[row['position_m'] for row in controls])
                        np.testing.assert_array_equal(trace['network'],[row['trace'].get('network_pwm') or [np.nan]*4 for row in controls])
                    report['baseline_exact'] = True
                records = comparison.records
                np.savez_compressed(dest/'decisions.npz', X=np.array([v['input'] for v in records]),
                    time=[v['time_s'] for v in records], segment=[v['segment'] for v in records],
                    **{a:np.array([v['outputs'][a] for v in records]) for a in (*policies,'teacher')})
                (dest/'decisions.json').write_text(json.dumps(plain(records), indent=2), encoding='utf-8')
                (dest/'report.json').write_text(json.dumps(plain(report), indent=2), encoding='utf-8')
                reports[case][driver] = report
                print(case, driver, report['metrics'], flush=True)
            finally:
                r.env.close()
    (output/'report.json').write_text(json.dumps(plain(reports), indent=2), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args().out)
