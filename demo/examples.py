"""Small reproducible examples, generated with the same live runner."""
import hashlib
import json
from pathlib import Path
from .runner import DemoRunner
from .recording import save_recording, load_recording, plain

SCENARIOS = {
    '01_manual': dict(scene='clear',seconds=4,events={0:[dict(type='key',code='KeyW',down=True)],
        150:[dict(type='key',code='KeyW',down=False)],
        500:[dict(type='key',code='ArrowLeft',down=True)],
        600:[dict(type='key',code='ArrowLeft',down=False)]}),
    '02_shadow': dict(scene='obstacle_left',seconds=4,events={0:[dict(type='mode',mode='SHADOW'),
        dict(type='key',code='KeyW',down=True)],150:[dict(type='key',code='KeyW',down=False)]}),
    '03_auto_stop_takeover': dict(scene='obstacle_left',seconds=9,events={
        0:[dict(type='mode',mode='SHADOW')],150:[dict(type='mode',mode='AUTO')],
        1500:[dict(type='stop')],2000:[dict(type='mode',mode='MANUAL')]}),
    '04_denied_near_obstacle': dict(scene='obstacle_center',start=(.4,0,1),seconds=1.2,
        events={0:[dict(type='mode',mode='AUTO')]})}


def run_example(name):
    spec=SCENARIOS[name]
    r=DemoRunner(scene=spec['scene'],start=spec.get('start',(-.7,0,1)),run_id=name,
                 collision_radius_m=.1,max_duration_s=60,contact_model='expanded-primitives-v1')
    r.event(dict(type='resume'))
    for step in range(round(spec['seconds']/.004)):
        for event in spec['events'].get(step,[]): r.event(event)
        r.advance()
        if r.terminal: break
    data=r.recording()
    r.env.close()
    return data


def recompute(data):
    """Replay applied input at original ticks, including focus/disconnect pauses."""
    from .recording import validate_recording
    validate_recording(data)
    m=data['manifest']
    if m.get('kind')!='live_recording':
        raise ValueError('Legacy replay has no complete interactive input schedule')
    session=m['system']['run_config']['session_id']
    run_id,generation=session.rsplit('-',1)
    r=DemoRunner(scene=m['scene'],seed=m['seed'],start=m['start']['position_m'],
                 yaw=m['start']['yaw_rad'],run_id=run_id,scene_options=m.get('scene_options'),
                 controller_mode=m.get('controller_mode','original'),
                 recorded_config_sha256=m['config_sha256'],
                 collision_radius_m=m.get('geometry',{}).get('collision_radius_m',.1),
                 max_duration_s=m.get('max_duration_s',60),
                 contact_model=m.get('contact_model','expanded-primitives-v1'))
    if int(generation):
        r.generation=int(generation)-1; r.reset()
    if r.manifest()['config_sha256']!=m['config_sha256']:
        raise ValueError('Config differs from recorded identity')
    schedule=data['input_schedule']
    end=round(data.get('end_time_s',data['frames'][-1]['time_s']+.004)/.004)
    pending=0
    for step in range(end+1):
        while pending<len(schedule) and schedule[pending]['step']==step:
            r.event(schedule[pending]['event']); pending+=1
        if step<end and not r.terminal:
            if r.paused: raise ValueError('Input schedule leaves simulation paused before its end')
            r.advance()
    result=r.recording(); r.env.close()
    return result


def examples(verify=False):
    directory=Path(__file__).parent/'recordings'
    for name in SCENARIOS:
        path=directory/(name+'.json.gz')
        data=run_example(name)
        if verify:
            saved=load_recording(path)
            if data != saved: raise ValueError('Recording differs: '+name)
        else: save_recording(data,path)
        frames=data['frames']
        print(json.dumps(dict(name=name,verified=verify,frames=len(frames),
            end_s=frames[-1]['time_s'],auto_frames=sum(x['plant_source']=='auto' for x in frames),
            collisions=sum(bool(x['collision']) for x in frames),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()),ensure_ascii=False),flush=True)
