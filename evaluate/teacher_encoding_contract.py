"""Construct a finite witness of teacher targets aliased by proximity-v1."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def histories():
    pair=[]
    for distance in (.85,1.05):
        x=np.ones((70,74),np.float32);x[:,:2]=0
        x[:20,2:66]=.5/3.5
        x[20:,2:66]=distance/3.5
        pair.append(x)
    return np.stack(pair)


def teacher_targets(x):
    from generator.generate_reflex_dataset import make_teacher
    result=[]
    for episode in x:
        teacher=make_teacher('geometry-reflex-v5',noise_std_pwm=0,pwm_cruise_pitch=1530.)
        result.append([teacher.step(row,dt=.02) for row in episode])
    return np.asarray(result)


def run(out):
    import torch
    from simulation.policy import SensorInputLayer
    from edge.runtime import NumpyPolicy,sha256
    torch.set_num_threads(1)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    x=histories();y=teacher_targets(x)
    encoder=SensorInputLayer(encoding='proximity-v1',learnable_scale=False)
    with torch.no_grad(): encoded=encoder(torch.from_numpy(x)).numpy()
    np.testing.assert_array_equal(encoded[0],encoded[1])
    model_results={};traces={}
    for label,folder in [('before','before'),('candidate','plastic')]:
        path=ROOT/'output/balanced_maneuver_v1'/folder/'numpy'
        p=NumpyPolicy(path);pred=[]
        for ep in x:
            p.reset_state()
            pred.append([p.step_np(row[:2],row[2:66],row[66:],dt=.02) for row in ep])
        pred=np.asarray(pred);np.testing.assert_array_equal(pred[0],pred[1])
        model_results[label]=dict(model_sha256=sha256(path/'model.npz'),outputs_exactly_equal=True,
                                  final_pwm=pred[:,-1].tolist())
        traces[label]=pred
    result=dict(version='teacher-proximity-alias-witness-v1',
        description='Same initial state and 0.4s threat; then 1s at 0.85m or 1.05m, identical zero flow and memory',
        encoding='proximity-v1 clips every range >=0.8m to zero',
        teacher='v5/1530 releases latch only beyond 0.95m for the configured dwell',
        encoded_sequences_exactly_equal=True,teacher_final_pwm=y[:,-1].tolist(),
        final_target_difference_pwm=(y[0,-1]-y[1,-1]).tolist(),models=model_results,
        conclusion='No deterministic policy using only these encoded histories and identical reset can match both teacher sequences exactly',
        limitation='Constructed open-loop witness; not proof this alias caused all observed collisions or that both sequences occur equally often in flight',
        sources={p:sha256(ROOT/p) for p in ['simulation/policy.py','edge/runtime.py',
                  'generator/generate_reflex_dataset.py','generator/continuous_reflex_teacher.py']},
        no_training=True,no_contract_changes=True)
    np.savez_compressed(out/'witness.npz',X=x,encoded=encoded,teacher=y,**traces)
    (out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args().out)
