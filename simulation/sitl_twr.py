"""Experimental collective-RPM adapter; not calibrated hardware control."""
import math
import numpy as np
from simulation.sitl_lockstep import infer_hover


def thrust_to_weight(p):
    return 4*p['thrust_coeff']*p['max_rpm']**2/(p['mass']*p['g'])


class ThrustAdapter:
    def __init__(self,reference,target_twr,rows,*,min_pwm=1100.,max_pwm=1400.):
        if not math.isfinite(target_twr) or target_twr<=0:
            raise ValueError('Invalid target TWR')
        if not 1000<min_pwm<max_pwm<=2000: raise ValueError('Invalid reference throttle bounds')
        self.reference=dict(reference);self.dynamics=dict(reference);self.rows=[dict(r) for r in rows]
        self.target_twr=float(target_twr);self.min_pwm=min_pwm;self.max_pwm=max_pwm
        if target_twr!=thrust_to_weight(reference):
            self.dynamics['max_rpm']=math.sqrt(target_twr*reference['mass']*reference['g']/(4*reference['thrust_coeff']))
        self.scale=reference['max_rpm']/self.dynamics['max_rpm']
        self.bounds=[(v-1000)/1000*self.scale for v in (min_pwm,max_pwm)]
        for value in self.bounds: infer_hover(self.rows,value)
        self.hover_motor=1/math.sqrt(target_twr)
        self.hover_rc_pwm=infer_hover(self.rows,self.hover_motor)

    def command(self,reference_pwm):
        pwm=np.asarray(reference_pwm,dtype=float).copy()
        if (pwm.shape!=(4,) or not np.isfinite(pwm).all() or np.any((pwm<1000)|(pwm>2000))
                or not self.min_pwm<=pwm[0]<=self.max_pwm):
            raise ValueError('Reference command outside declared actuator contract')
        pwm[0]=infer_hover(self.rows,(pwm[0]-1000)/1000*self.scale)
        return pwm

    def contract(self):
        return dict(version='collective-rpm-twr-v1',target_twr=self.target_twr,
            reference_max_rpm=self.reference['max_rpm'],plant_max_rpm=self.dynamics['max_rpm'],
            normalized_motor_scale=self.scale,reference_throttle_bounds=[self.min_pwm,self.max_pwm],
            requested_motor_bounds=self.bounds,physical_hover_motor=self.hover_motor,
            physical_hover_rc_pwm=self.hover_rc_pwm,
            rule='u_new=u_ref*n_ref/n_new; measured inverse RC map; no clipping or extrapolation',
            limit='Collective feedforward only; FC mixing, RPM lag and torque response remain physical',
            calibrated_hardware=False)
