import math
import struct
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from simulation.ready_sim_contract import rc_setpoints, map_motors, make_urdf, sitl_rc_packet


def params():
    return dict(mass=.13,payload_mass=0.,arm_length=.075,g=9.81,
        ixx=3.5e-4,iyy=3.5e-4,izz=6.7e-4,thrust_coeff=8.9286e-9,
        torque_coeff=1.6e-10,max_rpm=28000.,max_angle_rad=math.radians(35),
        max_yaw_rate_rads=math.radians(200))


def test_rc_contract_is_angles_and_right_positive_yaw_not_motor_pwm():
    thrust,roll,pitch,yaw=rc_setpoints([1213.,1550.,1450.,1640.],params())
    assert thrust==pytest.approx(.213)
    assert roll==pytest.approx(math.radians(3.5))
    assert pitch==pytest.approx(-math.radians(3.5))
    assert yaw==pytest.approx(-math.radians(56))
    for command in ([1,2,3], [1213,1500,float('nan'),1500], [1213,1500,1500,2500]):
        with pytest.raises(ValueError):rc_setpoints(command,params())


def test_generated_motor_positions_and_permutation_preserve_all_three_moments(tmp_path):
    p=params(); path=tmp_path/'craft.urdf';make_urdf(path,p)
    tree=ET.parse(path).getroot()
    positions=np.array([[float(x) for x in tree.find(f"link[@name='prop{i}_link']/inertial/origin").get('xyz').split()] for i in range(4)])
    for motor in range(4):
        legacy=np.zeros(4);legacy[motor]=1
        mapped=map_motors(legacy)
        assert np.array_equal(mapped,legacy[[0,1,3,2]])
        roll=np.dot(positions[:,1],mapped)
        pitch=-np.dot(positions[:,0],mapped)
        yaw=np.dot([-1,1,-1,1],mapped)
        d=p['arm_length']/math.sqrt(2)
        assert roll==pytest.approx(d*np.dot([-1,-1,1,1],legacy))
        assert pitch==pytest.approx(d*np.dot([1,-1,1,-1],legacy))
        assert yaw==np.dot([-1,1,1,-1],legacy)


def test_urdf_carries_explicit_mass_inertia_and_no_vendor_mesh(tmp_path):
    path=tmp_path/'craft.urdf';p=params();make_urdf(path,p)
    t=ET.parse(path).getroot()
    assert float(t.find('link/inertial/mass').get('value'))==p['mass']
    assert float(t.find('link/inertial/inertia').get('izz'))==p['izz']
    assert not t.findall('.//mesh')
    with pytest.raises(ValueError):make_urdf(tmp_path/'bad.urdf',p|{'payload_mass':.1})


def test_sitl_rc_packet_reorders_channels_without_assuming_arm_mapping():
    packet=sitl_rc_packet([1213,1510,1520,1530],timestamp_s=2.,aux=[1100]*12)
    assert len(packet)==40
    values=struct.unpack('<d16H',packet)
    assert values[:5]==(2.,1510,1520,1213,1530)
    assert values[5:]==(1100,)*12
    with pytest.raises(ValueError):sitl_rc_packet([1213,1500,1500,1500],timestamp_s=0,aux=[1000])
    with pytest.raises(ValueError):sitl_rc_packet([1213,1500,1500,1500],timestamp_s=-1,aux=[1000]*12)


def test_fixed_rc_protocol_has_five_tick_hold_and_explicit_brake():
    from evaluate.ready_sim_spike import command
    for tick in range(249, 752):
        np.testing.assert_array_equal(command('pitch_brake', tick, 1213),
                                      command('pitch_brake', (tick // 5)*5, 1213))
    assert command('pitch_brake', 250, 1213)[2] == 1550
    assert command('pitch_brake', 500, 1213)[2] == 1450
    assert command('pitch_brake', 750, 1213)[2] == 1500
    assert command('yaw_right', 250, 1213)[3] == 1640
    assert command('yaw_right', 500, 1213)[3] == 1500
