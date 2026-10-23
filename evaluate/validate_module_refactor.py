"""Before/after evidence for the module-boundary refactor; no controller tuning."""
import argparse
import json
from pathlib import Path

import numpy as np

from artifacts.io import file_sha256, write_json
from edge.bundle_validation import replay_bundle
from edge.system import RunConfig
from edge.validate_numpy import _replay_inputs
from evaluate.validate_transport_loop import run as run_physics


def compare(before, config_path, reference, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = RunConfig.from_json(config_path)
    cases = [dict(coordinator_mode=m, coordinator_preset=p, levy_explore=b,
                  stop_hover=b, seed=20261021, sensor_mode='multi_rate_v1')
             for m,p,b in [('baseline','original_v1',False), ('both','original_v1',False),
                           ('both','near_zero_v1',False), ('both','original_v1',True)]]
    inputs = _replay_inputs(2000)
    stop = np.zeros((4,2000),bool)
    stop[-1,-250:] = True
    write_json(output/'protocol.json',dict(before=str(Path(before).resolve()),
        after=config.to_dict(),reference_physics=str(Path(reference).resolve()),
        input_sha256={k:__import__('hashlib').sha256(v.tobytes()).hexdigest() for k,v in inputs.items()},
        replay_cases=cases,steps_per_case=2000,comparison='exact arrays and JSON numerical values'))
    old = replay_bundle(before,inputs,cases,stop_requests=stop,assembly=True)
    new = replay_bundle(config.bundle_dir,inputs,cases,stop_requests=stop,assembly=True)
    equal = {k:old[k].dtype==new[k].dtype and old[k].shape==new[k].shape
             and old[k].tobytes()==new[k].tobytes() for k in ('network_pwm','supervised_pwm','hidden')}
    reasons = old['report']['reasons']==new['report']['reasons']
    weights = file_sha256(Path(before)/'model.npz')==file_sha256(Path(config.bundle_dir)/'model.npz')
    result = run_physics(config_path,output/'physics')
    rows=[]
    for r in result['runs']:
        name=r['case']['name']
        equality={}
        for kind in ('physics','traces','receiver_events','link_events'):
            def read(path):
                records=[json.loads(line) for line in path.read_text().splitlines()]
                if kind=='traces':
                    # The changed executing-code identity must differ. All actual
                    # commands, sensor ages, states and receipt times must match.
                    for row in records: row.pop('manifest_sha256',None)
                return records
            equality[kind]=read(Path(reference)/(name+'_'+kind+'.jsonl'))==read(
                output/'physics'/(name+'_'+kind+'.jsonl'))
        rows.append(dict(case=name,exact_equal=equality,checks_passed=r['passed']))
    passed=all(equal.values()) and reasons and weights and result['passed'] and all(
        all(r['exact_equal'].values()) for r in rows)
    report=dict(passed=passed,weights_unchanged=weights,bitwise_replay=equal,
        reasons_equal=reasons,replay_ticks=8000,physical_cases=rows,
        before_execution={k:v for k,v in old['report'].items() if k!='reasons'},
        after_execution={k:v for k,v in new['report'].items() if k!='reasons'},
        limitation='Same host/NumPy, frozen default config. Nondefault height is tested separately; no hardware claims.')
    write_json(output/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,default=Path('data/edge/k306_tau025_hold_v1'))
    p.add_argument('--config',type=Path,default=Path('configs/mvp_runtime_v3.json'))
    p.add_argument('--reference',type=Path,default=Path('data/diagnostics/transport_loop_v3'))
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args()
    report=compare(a.before,a.config,a.reference,a.out_dir)
    print(json.dumps(dict(passed=report['passed'],report=str(a.out_dir/'report.json'))))
    if not report['passed']: raise SystemExit(1)


if __name__=='__main__': main()
