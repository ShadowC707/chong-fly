"""Contracts for an isolated ready-simulator feasibility spike, not deployment."""
import math
import struct
import xml.etree.ElementTree as ET
import numpy as np

UPSTREAM_COMMIT='348b5c18a4bb770a48928b9d044decf77ca33302'
BETAFLIGHT_COMMIT='79065c96ba0bb5cdc675e67d7093e05dab8b330e'


def _pwm(values, size):
    a=np.asarray(values,dtype=float)
    if a.shape!=(size,) or not np.isfinite(a).all() or np.any((a<1000)|(a>2000)):
        raise ValueError('Expected finite RC values in [1000,2000]')
    return a


def rc_setpoints(pwm, params):
    """Our T/R/P/Y command -> normalized RPM request, roll/pitch, FLU yaw rate."""
    t,r,p,y=_pwm(pwm,4)
    return ((t-1000)/1000,(r-1500)/500*params['max_angle_rad'],
            (p-1500)/500*params['max_angle_rad'],
            -(y-1500)/500*params['max_yaw_rate_rads'])


def map_motors(motors):
    a=np.asarray(motors)
    if a.shape!=(4,) or not np.isfinite(a).all() or np.any(a<0):
        raise ValueError('Expected four finite nonnegative motor values')
    return a[[0,1,3,2]]


def make_urdf(path,p):
    """Emit independent mesh-free asset preserving the existing force/moment map.

    The native library applies forces at each link inertial origin. The
    custom rotor order is RR, FR, FL, RL; yaw signs match its CF2X solver.
    Collision cylinder is only a smoke-test body, NOT the demo's sphere.
    """
    if p['payload_mass']!=0:
        raise ValueError('Spike only supports explicitly matched zero-payload reference')
    root=ET.Element('robot',name='chong_reference_uncalibrated')
    props=dict(arm=p['arm_length'],kf=p['thrust_coeff'],km=p['torque_coeff'],
        thrust2weight=4*p['thrust_coeff']*p['max_rpm']**2/(p['mass']*p['g']),
        max_speed_kmh=30,gnd_eff_coeff=0,prop_radius=.03,drag_coeff_xy=0,
        drag_coeff_z=0,dw_coeff_1=0,dw_coeff_2=0,dw_coeff_3=0)
    ET.SubElement(root,'properties',**{k:str(v) for k,v in props.items()})
    link=ET.SubElement(root,'link',name='base_link')
    inertial=ET.SubElement(link,'inertial')
    ET.SubElement(inertial,'origin',rpy='0 0 0',xyz='0 0 0')
    ET.SubElement(inertial,'mass',value=str(p['mass']))
    ET.SubElement(inertial,'inertia',ixx=str(p['ixx']),iyy=str(p['iyy']),izz=str(p['izz']),ixy='0',ixz='0',iyz='0')
    for name in ('visual','collision'):
        item=ET.SubElement(link,name);ET.SubElement(item,'origin',xyz='0 0 0',rpy='0 0 0')
        geo=ET.SubElement(item,'geometry');ET.SubElement(geo,'cylinder',radius='.2',length='.04')
    d=p['arm_length']/math.sqrt(2)
    for i,xyz in enumerate(((-d,-d,0),(d,-d,0),(d,d,0),(-d,d,0),(0,0,0))):
        name=f'prop{i}_link' if i<4 else 'center_of_mass_link'
        link=ET.SubElement(root,'link',name=name)
        inertial=ET.SubElement(link,'inertial')
        ET.SubElement(inertial,'origin',xyz=' '.join(map(str,xyz)),rpy='0 0 0')
        ET.SubElement(inertial,'mass',value='0')
        ET.SubElement(inertial,'inertia',ixx='0',iyy='0',izz='0',ixy='0',ixz='0',iyz='0')
        joint=ET.SubElement(root,'joint',name=f'joint{i}',type='fixed')
        ET.SubElement(joint,'parent',link='base_link');ET.SubElement(joint,'child',link=name)
    ET.indent(root)
    ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)


def sitl_rc_packet(pwm, *, timestamp_s, aux):
    """Encode only; no sockets, arming defaults or real FC output.

    Linux/x86-64 little-endian SITL ABI d + 16 unsigned shorts. The four
    RC slots are A/E/T/R. Remaining AUX channels MUST be explicit.
    This is NOT the onboard CFHW/CFMI protocol.
    """
    if not math.isfinite(timestamp_s) or timestamp_s<0:
        raise ValueError('Invalid simulation timestamp')
    t,r,p,y=_pwm(pwm,4)
    a=_pwm(aux,12)
    return struct.pack('<d16H',timestamp_s,*[round(x) for x in (r,p,t,y,*a)])
