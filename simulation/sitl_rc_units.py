"""Experimental units adapter for ONE pinned virtual ACTUAL-rates profile.

Betaflight 79065c96b: rc.c applyActualRates, pid.c pidLevel. Not a generic
hardware adapter. RC smoothing/feedforward/PID transients remain firmware-owned.
"""
import numpy as np


def actual_rate_dps(stick):
    x=np.asarray(stick,dtype=float)
    if not np.isfinite(x).all() or np.any(abs(x)>1):raise ValueError('Invalid normalized stick')
    return 70*x+600*x*abs(x)


def reference_axes_to_actual(pwm):
    a=np.asarray(pwm,dtype=float).copy()
    if a.shape!=(4,) or not np.isfinite(a).all() or np.any((a<1000)|(a>2000)):
        raise ValueError('Invalid reference PWM')
    x=(a[1:]-1500)/500
    # Old roll/pitch targets: +/-35 degrees; yaw: +/-200 degrees/sec.
    # New ANGLE target = 60 * actual_rate / 670.
    desired=x*np.array([35/60*670,35/60*670,200])
    magnitude=abs(desired)
    # Stable positive root of 600*u² + 70*u = magnitude.
    u=2*magnitude/(70+np.sqrt(70**2+4*600*magnitude))
    a[1:]=1500+500*np.sign(desired)*u
    return a


def validate_virtual_profile(path,params):
    settings={}
    for line in path.read_text().splitlines():
        if line.startswith('set '):
            key,value=line[4:].split('=',1);settings[key.strip()]=value.strip()
    expected={'rates_type':'ACTUAL','angle_limit':'60'}
    for axis in ('roll','pitch','yaw'):
        expected.update({axis+'_rc_rate':'7',axis+'_srate':'67',axis+'_expo':'0'})
    if any(settings.get(k)!=v for k,v in expected.items()):
        raise ValueError('Adapter requires the pinned virtual ACTUAL profile')
    if not np.isclose(params['max_angle_rad'],np.radians(35),rtol=0,atol=1e-12) or not np.isclose(
            params['max_yaw_rate_rads'],np.radians(200),rtol=0,atol=1e-12):
        raise ValueError('Adapter requires the original reference axis units')
