import math
import struct

import numpy as np
import pytest

from simulation.sitl_contract import fdm_packet, motor_packet, msp_request, parse_msp


def quat(roll, pitch, yaw):
    cr, cp, cy = np.cos(np.array([roll, pitch, yaw]) / 2)
    sr, sp, sy = np.sin(np.array([roll, pitch, yaw]) / 2)
    return np.array([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                     cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy])


def test_fdm_level_is_full_packet_with_gravity_not_zero_accelerometer():
    data = fdm_packet(2., [0,0,1], [0,0,0], [1,0,0,0], [0,0,0], [0,0,9.81])
    values = struct.unpack('<18d', data)
    assert len(data) == 144
    assert values[4:7] == (0.,0.,-9.81)
    assert values[7:11] == (1.,0.,0.,0.)
    assert values[14:17] == (0.,0.,-1.)
    assert 101000 < values[17] < 101325


def test_pinned_sitl_quaternion_hacks_and_gyro_produce_consistent_control_signs():
    # Reproduce the audited 79065c96b imu.c SITL Euler path, including two
    # sign changes in rMat[1][0] and rMat[2][0]. This is NOT generic MAVLink.
    angles = np.radians([10.,20.,-30.])
    values = struct.unpack('<18d', fdm_packet(1., [1,2,3], [4,5,6],
        quat(*angles), [.1,.2,.3], [1,2,9]))
    w,x,y,z = values[7:11]
    r20 = -2*(x*z-w*y)
    r21, r22 = 2*(y*z+w*x), 1-2*(x*x+y*y)
    r10, r00 = -2*(x*y+w*z), 1-2*(y*y+z*z)
    reported = [math.atan2(r21,r22), math.asin(-r20), -math.atan2(r10,r00)]
    np.testing.assert_allclose(reported, np.radians([10,20,30]), atol=1e-12)
    np.testing.assert_allclose(np.array(values[1:4])*[1,-1,-1], [.1,.2,.3])


def test_normalized_motor_packet_undoes_firmware_gazebo_permutation():
    np.testing.assert_allclose(motor_packet(struct.pack('<4f', .2,.3,.4,.1)), [.1,.2,.3,.4])
    with pytest.raises(ValueError): motor_packet(b'wrong')
    with pytest.raises(ValueError): motor_packet(struct.pack('<4f', 0,0,0,float('nan')))


def test_msp_checksum_and_partial_tcp_frame():
    request = msp_request(108)
    assert request == b'$M<\x00ll'
    response = bytearray(b'$M>\x06\x6c' + struct.pack('<hhh', 100,200,30))
    checksum = 0
    for b in response[3:]: checksum ^= b
    response.append(checksum)
    buffer = bytearray(b'noise' + response[:-1])
    assert parse_msp(buffer) is None
    buffer.append(checksum)
    assert parse_msp(buffer) == (108, struct.pack('<hhh',100,200,30))
    assert not buffer
    response[-1] ^= 1
    with pytest.raises(ValueError): parse_msp(response)


def test_fdm_rejects_nonphysical_or_nonfinite_inputs():
    for q in ([0,0,0,0], [2,0,0,0], [float('nan'),0,0,0]):
        with pytest.raises(ValueError): fdm_packet(0, [0,0,1], [0,0,0], q, [0,0,0], [0,0,9.81])
    with pytest.raises(ValueError): fdm_packet(-1, [0,0,1], [0,0,0], [1,0,0,0], [0,0,0], [0,0,9.81])
