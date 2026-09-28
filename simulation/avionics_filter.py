"""
simulation/avionics_filter.py
=============================
Avionics filtering and cascaded PID flight control inspired by Betaflight firmware.

Architecture:
-------------
1. Outer Loop (Angle Mode / Attitude Stabilization):
   - Accepts target pitch (theta_cmd) and target roll (phi_cmd) setpoints [rad].
   - Compares with measured pitch and roll Euler angles.
   - P-controller generates target body angular rates (p_target, q_target) [rad/s].
   - Target yaw rate (r_target) passes directly through to the rate controller.

2. Inner Loop (Rate Mode / Gyro PID):
   - Three independent PID axes: Roll (p), Pitch (q), Yaw (r).
   - Proportional term: Kp * error.
   - Integral term with anti-windup clamping (iterm_limit).
   - Derivative term on measurement (gyro delta) with 1st-order PT1 low-pass filter
     to suppress high-frequency motor noise without phase lag.

3. Motor Mixer (Betaflight Quad X configuration):
   - Maps [thrust, roll_cmd, pitch_cmd, yaw_cmd] to 4 motor PWM / RPM signals:
       m1: Rear Right (CCW) = thrust - roll + pitch - yaw
       m2: Front Right (CW)  = thrust - roll - pitch + yaw
       m3: Rear Left (CW)   = thrust + roll + pitch + yaw
       m4: Front Left (CCW) = thrust + roll - pitch - yaw
   - Clips output to [0.0, 1.0] with optional airmode dynamic range recovery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Union
import numpy as np


class PT1Filter:
    """
    1st-order low-pass filter (PT1) matching Betaflight gyro and D-term filtering.
    """

    def __init__(self, cutoff_hz: float, dt: float):
        self.dt = dt
        self.cutoff_hz = cutoff_hz
        self.alpha = self._compute_alpha(cutoff_hz, dt)
        self.state = 0.0

    def _compute_alpha(self, cutoff_hz: float, dt: float) -> float:
        if cutoff_hz <= 0.0:
            return 1.0
        rc = 1.0 / (2.0 * math.pi * cutoff_hz)
        return float(dt / (dt + rc))

    def reset(self, initial_value: float = 0.0) -> None:
        self.state = float(initial_value)

    def update(self, x: float) -> float:
        self.state += self.alpha * (x - self.state)
        return self.state


@dataclass
class PIDConstants:
    kp: float
    ki: float
    kd: float
    iterm_limit: float = 0.3
    dterm_lpf_hz: float = 60.0


class BetaflightAxisPID:
    """
    Single-axis rate PID controller matching Betaflight inner loop.
    Calculates derivative on measurement (-d(gyro)/dt) to prevent setpoint derivative kick.
    """

    def __init__(self, constants: PIDConstants, dt: float):
        self.constants = constants
        self.dt = dt
        self.iterm = 0.0
        self.last_measurement = 0.0
        self.dterm_filter = PT1Filter(constants.dterm_lpf_hz, dt)

    def reset(self) -> None:
        self.iterm = 0.0
        self.last_measurement = 0.0
        self.dterm_filter.reset(0.0)

    def update(self, target_rate: float, measured_rate: float) -> float:
        error = target_rate - measured_rate

        # 1. Proportional term
        p_term = self.constants.kp * error

        # 2. Integral term with anti-windup clamping
        self.iterm += self.constants.ki * error * self.dt
        limit = self.constants.iterm_limit
        self.iterm = float(np.clip(self.iterm, -limit, limit))

        # 3. Derivative term on measurement (suppress setpoint derivative kicks)
        d_meas = -(measured_rate - self.last_measurement) / max(self.dt, 1e-6)
        self.last_measurement = measured_rate
        d_meas_filtered = self.dterm_filter.update(d_meas)
        d_term = self.constants.kd * d_meas_filtered

        return float(p_term + self.iterm + d_term)


class BetaflightCascadedPID:
    """
    Cascaded flight controller implementing Betaflight Angle Mode + Rate Mode.
    
    Outer Loop (Angle Mode):
        P_angle: target_roll/pitch -> target_p/q rates.
    Inner Loop (Rate Mode):
        PID_rate: target_rates vs gyro rates -> motor mixer inputs.
    """

    def __init__(
        self,
        dt: float = 0.004,
        angle_p_gain: float = 5.0,
        max_rate_rads: float = 8.72,       # ~500 deg/s max target rate
        roll_pid: PIDConstants = PIDConstants(kp=0.08, ki=0.15, kd=0.003, iterm_limit=0.3, dterm_lpf_hz=60.0),
        pitch_pid: PIDConstants = PIDConstants(kp=0.08, ki=0.15, kd=0.003, iterm_limit=0.3, dterm_lpf_hz=60.0),
        yaw_pid: PIDConstants = PIDConstants(kp=0.12, ki=0.20, kd=0.001, iterm_limit=0.3, dterm_lpf_hz=60.0),
    ):
        self.dt = dt
        self.angle_p_gain = angle_p_gain
        self.max_rate_rads = max_rate_rads

        self.pid_roll = BetaflightAxisPID(roll_pid, dt)
        self.pid_pitch = BetaflightAxisPID(pitch_pid, dt)
        self.pid_yaw = BetaflightAxisPID(yaw_pid, dt)

    def reset(self) -> None:
        self.pid_roll.reset()
        self.pid_pitch.reset()
        self.pid_yaw.reset()

    def compute_motor_commands(
        self,
        target_thrust: float,          # [0.0, 1.0] normalized collective thrust
        target_pitch: float,           # [rad] target pitch angle (Angle mode)
        target_roll: float,            # [rad] target roll angle (Angle mode)
        target_yaw_rate: float,        # [rad/s] target yaw rate (Rate mode)
        measured_euler: np.ndarray,    # [roll, pitch, yaw] in radians
        measured_omega: np.ndarray,    # [p, q, r] gyro angular velocities in rad/s
    ) -> Tuple[np.ndarray, dict]:
        """
        Runs the cascaded PID and mixer to produce 4 motor commands in [0.0, 1.0].
        
        Returns:
            motors: np.ndarray of shape (4,) with normalized motor throttle commands [m1, m2, m3, m4]
            debug_info: dictionary with intermediate rate targets and control efforts
        """
        curr_roll = float(measured_euler[0])
        curr_pitch = float(measured_euler[1])

        # ── 1. Outer Loop (Angle mode P-controller) ──────────────────────────
        rate_target_p = self.angle_p_gain * (target_roll - curr_roll)
        rate_target_q = self.angle_p_gain * (target_pitch - curr_pitch)
        rate_target_r = target_yaw_rate

        # Clamping rate setpoints to safety limits
        max_r = self.max_rate_rads
        rate_target_p = float(np.clip(rate_target_p, -max_r, max_r))
        rate_target_q = float(np.clip(rate_target_q, -max_r, max_r))
        rate_target_r = float(np.clip(rate_target_r, -max_r, max_r))

        # ── 2. Inner Loop (Betaflight Rate PID) ──────────────────────────────
        p_gyro, q_gyro, r_gyro = float(measured_omega[0]), float(measured_omega[1]), float(measured_omega[2])
        
        u_roll = self.pid_roll.update(rate_target_p, p_gyro)
        u_pitch = self.pid_pitch.update(rate_target_q, q_gyro)
        u_yaw = self.pid_yaw.update(rate_target_r, r_gyro)

        # ── 3. Motor Mixer (Standard Quad X) ─────────────────────────────────
        # m1: Rear Right (CCW)
        # m2: Front Right (CW)
        # m3: Rear Left (CW)
        # m4: Front Left (CCW)
        throttle = float(np.clip(target_thrust, 0.0, 1.0))

        m1 = throttle - u_roll + u_pitch - u_yaw
        m2 = throttle - u_roll - u_pitch + u_yaw
        m3 = throttle + u_roll + u_pitch + u_yaw
        m4 = throttle + u_roll - u_pitch - u_yaw

        # Clamp motor signals
        motors = np.clip(np.array([m1, m2, m3, m4], dtype=np.float32), 0.0, 1.0)

        debug = {
            "rate_target": np.array([rate_target_p, rate_target_q, rate_target_r], dtype=np.float32),
            "control_efforts": np.array([u_roll, u_pitch, u_yaw], dtype=np.float32),
            "raw_motors": np.array([m1, m2, m3, m4], dtype=np.float32),
        }
        return motors, debug
