"""Small physical handover experiment; no motors, fixed bridge parameters."""
import json
from pathlib import Path
import numpy as np
from edge.runtime import FlightRuntime
from edge.flight_bridge import FlightBridge, SensorFrame
from simulation.benchmark_scenes import NavigationScene
from simulation.altitude_control import AltitudeHold, AltitudeConfig, sample_from_observation, AltitudeSample


def run(output):
    output=Path(output)
    if output.exists(): raise FileExistsError('Use a new report')
    results=[]
    for name,velocity,height in [('hover',0.,1.),('moving',.6,1.),('other_height',0.,1.4),('dropout',0.,1.)]:
        runtime=FlightRuntime('data/edge/k306_tau025_numpy_behaviors_v2',coordinator_mode='both',stop_hover=True)
        bridge=FlightBridge(runtime); env=NavigationScene('clear')
        manual_hold=AltitudeHold(AltitudeConfig(target_height_m=height))
        trace=[]
        try:
            obs=env.reset(initial_pos=[0.,0.,height],seed=20261021)
            env.physics.vel[:]=[velocity,0,0]
            obs=env._sample_observation(0.)
            for i in range(200):
                t=i*.02; sample=sample_from_observation(env.get_isaac_obs())
                manual=manual_hold.apply([1213.,1500.,1500.,1500.],sample,now_s=t)
                supplied=sample
                if name=='dropout' and i==100:
                    supplied=AltitudeSample(sample.range_m,sample.roll_rad,sample.pitch_rad,sample.timestamp_s,False)
                request='MANUAL' if i>=150 else 'AUTO'
                decision=bridge.tick(SensorFrame(i,t,obs[:2],obs[2:],supplied),now_s=t,request=request,manual_pwm=manual)
                applied=manual if decision['lease'] is None else decision['lease'].pwm
                trace.append(dict(time_s=t,state=decision['state'],reason=decision['reason'],
                    ready=decision['ready'],authority=decision['lease'] is not None,
                    speed_mps=float(np.linalg.norm(env.physics.vel[:2])),height_m=float(env.physics.pos[2]),
                    manual_pwm=manual.tolist(),applied_pwm=list(applied)))
                for _ in range(5): obs,*_=env.step(applied)
            auto=[r for r in trace if r['authority']]
            results.append(dict(scenario=name,initial_velocity_mps=velocity,initial_height_m=height,
                first_auto_s=auto[0]['time_s'] if auto else None,
                speed_at_engage_mps=auto[0]['speed_mps'] if auto else None,
                faults=sum(r['state']=='FAULT' for r in trace),trace=trace))
        finally: env.close()
    hover,moving,height,drop=results
    assert hover['first_auto_s'] is not None, 'Hover never became ready'
    assert moving['first_auto_s'] is None or moving['speed_at_engage_mps']<.10
    assert height['first_auto_s'] is None
    assert not drop['trace'][100]['authority'] and drop['trace'][100]['state']=='FAULT'
    assert all(not r['authority'] for result in results for r in result['trace'][150:])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(version='flight-bridge-physical-v1',runs=results,
        limitation='Manual fallback is a simulation-only altitude controller, not an ESP failsafe implementation.'),indent=2)+'\n',encoding='utf-8')
    for r in results: print({k:v for k,v in r.items() if k!='trace'})


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--out',required=True)
    run(parser.parse_args().out)
