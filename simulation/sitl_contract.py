"""Local experimental ABI for Betaflight 79065c96b on Linux x86-64.

Not an ESP32 protocol. Assumes the audited default USE_IMU_CALC-disabled
quaternion path, board alignment zero and virtual GPS disabled. Firmware's
SITL-only matrix sign corrections require this version-specific adapter.
"""
import math
import struct
import numpy as np


def _vector(value, length):
    value = np.asarray(value, dtype=float)
    if value.shape != (length,) or not np.isfinite(value).all():
        raise ValueError(f'Expected {length} finite numbers')
    return value


def fdm_packet(timestamp_s, position_flu, velocity_flu, quaternion_wxyz,
               omega_body_flu, specific_force_body_flu):
    """State -> firmware ABI; specific force excludes gravity, in m/s².

    For this pinned firmware MSP attitude reads (roll, pitch, -yaw) and gyro
    (omega_x, omega_y, omega_z). Its RC yaw command is internally negated.
    Physical motor spin signs must match yaw_motors_reversed=ON in our asset.
    This validates control-axis consistency;
    it does NOT model accelerometer/gyro noise or exercise the AHRS.
    """
    if not math.isfinite(timestamp_s) or timestamp_s < 0:
        raise ValueError('Invalid timestamp')
    pos, vel = _vector(position_flu, 3), _vector(velocity_flu, 3)
    q = _vector(quaternion_wxyz, 4)
    if not np.isclose(np.linalg.norm(q), 1., atol=1e-5, rtol=0):
        raise ValueError('Expected normalized quaternion')
    omega, force = _vector(omega_body_flu, 3), _vector(specific_force_body_flu, 3)
    pressure_base = 1 - pos[2] / 44330.
    if pressure_base <= 0:
        raise ValueError('Outside pressure-model altitude range')
    pressure = 101325. * pressure_base ** 5.255
    return struct.pack('<18d', timestamp_s, *(omega * [1,-1,-1]), *(-force),
                       *(q * [1,1,-1,-1]), *(vel * [1,-1,-1]),
                       *(pos * [1,-1,-1]), pressure)


def motor_packet(data):
    """Undo sitl.c Gazebo remap; return native QUADX RR,FR,RL,FL outputs.

    Values are (motor PWM - idlePulse)/1000, NOT thrust or RPM. The caller
    must declare the actuator mapping; do not inherit BetaAviary's sqrt law.
    """
    if len(data) != 16:
        raise ValueError('Expected 16-byte motor packet')
    values = _vector(struct.unpack('<4f', data), 4)
    if np.any(values < -1e-6) or np.any(values > 1.000001):
        raise ValueError('Motor output outside normal non-3D range')
    return values[[3,0,1,2]]


def msp_request(command, payload=b''):
    if not 0 <= command <= 255 or len(payload) > 255:
        raise ValueError('Expected MSP v1 command/payload')
    body = bytes([len(payload), command]) + payload
    checksum = 0
    for value in body:
        checksum ^= value
    return b'$M<' + body + bytes([checksum])


def parse_msp(buffer):
    """Consume one checksummed response from a bytearray; preserve partial TCP."""
    while len(buffer) >= 3 and buffer[:3] not in (b'$M>', b'$M!'):
        del buffer[0]
    if len(buffer) < 6:
        return None
    length, command = buffer[3:5]
    if len(buffer) < length + 6:
        return None
    frame = bytes(buffer[:length+6])
    del buffer[:length+6]
    checksum = 0
    for value in frame[3:-1]:
        checksum ^= value
    if frame[-1] != checksum:
        raise ValueError('MSP checksum mismatch')
    if frame[2:3] == b'!':
        raise ValueError(f'Firmware rejected MSP command {command}')
    return command, frame[5:-1]
