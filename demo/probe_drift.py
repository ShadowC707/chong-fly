"""Reproduce W-release drift and AUTO admission without altering the model."""
import json
import numpy as np
from .runner import DemoRunner


def probe():
    r=DemoRunner(); r.resume()
    r.event(dict(type='key',code='KeyW',down=True)); r.advance(250)
    r.event(dict(type='key',code='KeyW',down=False))
    r.event(dict(type='mode',mode='AUTO'))
    samples=[]
    for _ in range(17):
        samples.append(dict(time_s=r.time_s,speed_xy_mps=float(np.linalg.norm(r.env.physics.vel[:2])),
            actual_authority=r.rows[-1]['authority'],request=r.request,admission=r.admission))
        r.advance(250)
    r.event(dict(type='mode',mode='MANUAL'))
    result=dict(protocol='W 1 s, release and request AUTO, sample every simulation second',
        drag_expression='F_drag = -drag_linear * velocity * norm(velocity)',
        drag_coefficient=r.env.params.drag_linear,mass_kg=r.env.physics.total_mass,
        samples=samples,manual_override=dict(receiver_mode=r.receiver.mode,request=r.request,
            receiver_pwm=r.receiver.gate.pwm,speed_xy_mps=float(np.linalg.norm(r.env.physics.vel[:2]))))
    r.env.close(); return result


if __name__=='__main__': print(json.dumps(probe(),indent=2))
