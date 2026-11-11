"""Open-loop history distinguishability; not a physical flight or impossibility proof."""
import argparse
from pathlib import Path
import numpy as np


def histories():
    """Last 150 frames identical; only earlier threat/settling histories differ."""
    clear=np.ones((200,74),np.float32); clear[:,:2]=[.12,0.]
    threat=clear.copy();threat[:25,:2]=[.01,0.]
    threat[:25,2:66]=np.tile(np.r_[np.full(4,.5/3.5),np.ones(4)],8)
    settled=threat.copy();settled[25:50,:2]=0
    return dict(always_clear=clear,threat_then_clear=threat,threat_then_settle=settled)


def probe(directory):
    from edge.runtime import NumpyPolicy,sha256
    from generator.generate_reflex_dataset import make_teacher
    from artifacts.io import write_json
    d=Path(directory); result={}; traces={}
    for arm in ('before','control','mixed'):
        result[arm]={}
        for name,x in histories().items():
            p=NumpyPolicy(d/arm/'numpy'); teacher=make_teacher('geometry-reflex-v5',noise_std_pwm=0)
            predicted=[];targets=[]
            for row in x:
                predicted.append(p.step_np(row[:2],row[2:66],row[66:],dt=.02))
                targets.append(teacher.step(row,dt=.02))
            result[arm][name]=dict(pitch=float(predicted[-1][2]),teacher_pitch=float(targets[-1][2]),
                final_hidden_norm=float(np.linalg.norm(p.h)))
            traces[arm+'_'+name]=np.asarray(predicted)
            traces['teacher_'+name]=np.asarray(targets)
        pairs=list(result[arm].values())
        result[arm]['pitch_history_span_pwm']=max(p['pitch'] for p in pairs)-min(p['pitch'] for p in pairs)
    np.savez_compressed(d/'history_probe_traces.npz',**traces,**histories())
    write_json(d/'history_alias_probe.json',dict(protocol=dict(dt=.02,steps=200,
        threat_steps=25,settle_steps=25,identical_suffix_steps=150,
        description='Inputs imposed externally; no physical plant, coordinator or teacher actions alter them'),
        models={a:sha256(d/a/'numpy/model.npz') for a in result},
        source_sha256=sha256(__file__),results=result,
        interpretation='Empirical distinguishability of these checkpoints; not proof that all permitted weights must behave identically'))
    flows={}
    for arm in result:
        flows[arm]=[]
        for value in (-.2,-.1,0.,.1,.2):
            p=NumpyPolicy(d/arm/'numpy')
            for _ in range(100):y=p.step_np(np.array([value,0.]),np.ones(64))
            flows[arm].append(dict(flow=value,pitch=float(y[2]),yaw=float(y[3])))
    write_json(d/'clear_flow_probe.json',dict(dt=.02,steps=100,results=flows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    probe(p.parse_args().directory)
