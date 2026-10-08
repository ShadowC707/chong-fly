"""Collect clean reflex targets in geometric scenes; never splice terminated episodes."""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
from pathlib import Path
from collections import Counter
from copy import deepcopy
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from configs.flight_config import (
    SENSOR_DIM, PWM_MIN, PWM_MAX, PWM_HOVER, PWM_MID, PWM_LEVEL_ROLL,
    PWM_CRUISE_PITCH, PWM_NEUTRAL_YAW, DEFAULT_DT, CONTROL_DT, COORDINATE_VERSION,
    TOF_RAYCASTER_MAX_RANGE_M, APF_DISTANCE_THRESHOLD_M, APF_NOISE_STD_PWM,
    APF_K_REPULSIVE, APF_MAX_PITCH_BRAKE_PWM, APF_REPULSION_SCALE, APF_YAW_GAIN,
    APF_MIN_DIST_CLAMP,
)
from generator.reflex_contract import (
    DATASET_VERSION, TEACHER_VERSION, LEGACY_TEACHER_VERSION, BRAKING_TEACHER_VERSION,
    DEFAULT_DATASET_PATH, SCENARIOS, BEHAVIORS, REQUIRED_PATHS,
)
from simulation.drone_env import DroneSimulationEnv, RoomBoundaries, BoxObstacle
from simulation.memory import EgocentricMemoryWrapper
from simulation.altitude_control import AltitudeHold, AltitudeTelemetryError, sample_from_observation
from simulation.control_contract import control_contract


class LegacyExpertReflexPolicy:
    """Flow damping, smooth braking, deterministic latched escape.

    RC yaw is right-positive; image columns run left to right. A symmetric
    threat selects right, with no hidden random label. Sustained clearance
    releases the turn. This heuristic is not a verified flight controller.
    """
    def __init__(self, distance_threshold_m=APF_DISTANCE_THRESHOLD_M,
                 max_sensor_range_m=TOF_RAYCASTER_MAX_RANGE_M,
                 noise_std_pwm=APF_NOISE_STD_PWM, seed=None,
                 k_repulsive=APF_K_REPULSIVE, max_pitch_brake_pwm=APF_MAX_PITCH_BRAKE_PWM,
                 repulsion_scale=APF_REPULSION_SCALE, yaw_gain=APF_YAW_GAIN,
                 min_dist_clamp=APF_MIN_DIST_CLAMP, pwm_hover=PWM_HOVER,
                 pwm_level_roll=PWM_LEVEL_ROLL, pwm_cruise_pitch=PWM_CRUISE_PITCH,
                 pwm_neutral_yaw=PWM_NEUTRAL_YAW, pwm_min=PWM_MIN, pwm_max=PWM_MAX,
                 flow_gain_pwm=400., release_margin_m=.15, clear_hold_s=.12):
        settings = dict(distance_threshold_m=distance_threshold_m, max_sensor_range_m=max_sensor_range_m,
            noise_std_pwm=noise_std_pwm, k_repulsive=k_repulsive, max_pitch_brake_pwm=max_pitch_brake_pwm,
            repulsion_scale=repulsion_scale, yaw_gain=yaw_gain, min_dist_clamp=min_dist_clamp,
            pwm_hover=pwm_hover, pwm_level_roll=pwm_level_roll, pwm_cruise_pitch=pwm_cruise_pitch,
            pwm_neutral_yaw=pwm_neutral_yaw, pwm_min=pwm_min, pwm_max=pwm_max,
            flow_gain_pwm=flow_gain_pwm, release_margin_m=release_margin_m, clear_hold_s=clear_hold_s)
        for key, value in settings.items():
            if not np.isfinite(value): raise ValueError(f'{key} must be finite')
            setattr(self, key, float(value))
        self.parameters = {key: float(value) for key, value in settings.items()}
        if (not 0 < distance_threshold_m < max_sensor_range_m or min_dist_clamp <= 0
                or repulsion_scale <= 0 or clear_hold_s <= 0 or release_margin_m < 0
                or noise_std_pwm < 0 or flow_gain_pwm < 0 or pwm_min >= pwm_max
                or k_repulsive < 0 or yaw_gain < 0 or max_pitch_brake_pwm < 0):
            raise ValueError('Invalid teacher range, gain or timing')
        self.d0 = distance_threshold_m / max_sensor_range_m
        self.rng = np.random.default_rng(seed)
        self.nx = np.tile(np.linspace(-1., 1., 8), (8, 1)).ravel()
        self.reset()

    def reset(self):
        self.latched_turn = None
        self._clear_elapsed = 0.

    def step(self, obs_74, dt=CONTROL_DT):
        obs = np.asarray(obs_74, dtype=np.float32)
        if (obs.shape != (SENSOR_DIM,) or not np.isfinite(obs).all()
                or np.any(abs(obs[:2]) > 1) or np.any(obs[2:] < 0) or np.any(obs[2:] > 1)):
            raise ValueError('Expected finite normalized observation [flow(2), ToF(64), memory(8)]')
        if not np.isfinite(dt) or dt <= 0: raise ValueError('dt must be positive and finite')
        tof = obs[2:66]
        d = np.maximum(tof, self.min_dist_clamp)
        active = d < self.d0
        force = np.zeros(64)
        force[active] = self.k_repulsive * (1/d[active] - 1/self.d0) / d[active]**2
        mean_repulsion = float(force[active].mean()) if active.any() else 0.
        force_x = float((force * -self.nx)[active].mean()) if active.any() else 0.
        braking = self.max_pitch_brake_pwm * mean_repulsion / (self.repulsion_scale + mean_repulsion)
        if self.latched_turn is None and active.any():
            self.latched_turn = 1 if force_x >= -1e-6 else -1
        clear = float(tof.min()) * self.max_sensor_range_m >= self.distance_threshold_m + self.release_margin_m
        self._clear_elapsed = self._clear_elapsed + dt if clear else 0.
        if self._clear_elapsed + 1e-9 >= self.clear_hold_s:
            self.latched_turn = None
        yaw = self.pwm_neutral_yaw
        if self.latched_turn is not None:
            yaw += self.latched_turn * np.clip(abs(force_x) * self.yaw_gain, 120., 350.)
        roll, pitch, yaw = self._navigation_action(obs, braking, yaw, dt, bool(active.any()))
        pwm = np.array([self.pwm_hover, roll, pitch, yaw], dtype=np.float32)
        if self.noise_std_pwm:
            pwm += self.rng.normal(0, self.noise_std_pwm, 4).astype(np.float32)
        return np.clip(pwm, self.pwm_min, self.pwm_max)

    def _navigation_action(self, obs, braking, yaw, dt, active):
        return (self.pwm_level_roll + self.flow_gain_pwm * obs[1],
                self.pwm_cruise_pitch - braking - self.flow_gain_pwm * obs[0], yaw)


class BrakingExpertReflexPolicy(LegacyExpertReflexPolicy):
    """Retain braking after loss of sight until planar flow settles.

    Flow is normalized optical motion, not an absolute speed measurement.
    This simulated teacher is a research heuristic, not a flight safety layer.
    """
    def __init__(self, *, brake_release_flow=.03, brake_settle_s=.20,
                 brake_flow_scale=.10, cruise_recovery_s=.20, **kwargs):
        settings = dict(brake_release_flow=brake_release_flow, brake_settle_s=brake_settle_s,
                        brake_flow_scale=brake_flow_scale, cruise_recovery_s=cruise_recovery_s)
        if (any(not np.isfinite(v) or v <= 0 for v in settings.values())
                or not brake_release_flow < brake_flow_scale <= 1):
            raise ValueError('Invalid braking flow threshold or timing')
        super().__init__(**kwargs)
        for key, value in settings.items():
            setattr(self, key, float(value))
        self.parameters.update({k:float(v) for k,v in settings.items()})

    def reset(self):
        super().reset()
        self._brake_peak = 0.
        self._escape_sign = None
        self._settled_for = 0.
        self._cruise_recovery = 1.

    def _navigation_action(self, obs, braking, yaw, dt, active):
        roll, pitch, yaw = super()._navigation_action(obs, braking, yaw, dt, active)
        if active:
            self._escape_sign = self.latched_turn
            self._brake_peak = max(self._brake_peak, braking)
            self._settled_for = 0.
            self._cruise_recovery = 0.
            return roll, pitch, yaw
        if self._escape_sign is not None:
            self.latched_turn = self._escape_sign
            self._settled_for = (self._settled_for + dt
                if self._clear_elapsed > 0 and np.linalg.norm(obs[:2]) <= self.brake_release_flow else 0.)
            if (self._settled_for + 1e-9 < self.brake_settle_s
                    or self._clear_elapsed + 1e-9 < self.clear_hold_s):
                held_brake = self._brake_peak * np.clip(obs[0]/self.brake_flow_scale, 0., 1.)
                return (roll, PWM_MID - self.flow_gain_pwm*obs[0] - held_brake,
                        self.pwm_neutral_yaw + self._escape_sign*120.)
            self.latched_turn = self._escape_sign = None
            self._brake_peak = 0.
            yaw = self.pwm_neutral_yaw
        self._cruise_recovery = min(1., self._cruise_recovery + dt/self.cruise_recovery_s)
        pitch = PWM_MID + self._cruise_recovery*(self.pwm_cruise_pitch-PWM_MID) - self.flow_gain_pwm*obs[0]
        return roll, pitch, yaw


class ExpertReflexPolicy(BrakingExpertReflexPolicy):
    """Brake before rotating the body away from its translation direction."""
    def __init__(self, *, turn_full_flow=.05, turn_stop_flow=.12, **kwargs):
        if (not np.isfinite(turn_full_flow) or not np.isfinite(turn_stop_flow)
                or not 0 <= turn_full_flow < turn_stop_flow <= 1):
            raise ValueError('Invalid turn flow thresholds')
        super().__init__(**kwargs)
        self.turn_full_flow, self.turn_stop_flow = float(turn_full_flow), float(turn_stop_flow)
        self.parameters.update(turn_full_flow=self.turn_full_flow, turn_stop_flow=self.turn_stop_flow)

    def _navigation_action(self, obs, braking, yaw, dt, active):
        roll, pitch, yaw = super()._navigation_action(obs, braking, yaw, dt, active)
        if active and self.max_pitch_brake_pwm > 0:
            # Remove the cruise contribution gradually with the APF strength,
            # rather than retaining forward thrust throughout emergency braking.
            pitch -= (self.pwm_cruise_pitch-PWM_MID)*braking/self.max_pitch_brake_pwm
        if self._escape_sign is not None:
            flow = float(np.linalg.norm(obs[:2]))
            turn_fraction = float(np.clip((self.turn_stop_flow-flow)
                                         /(self.turn_stop_flow-self.turn_full_flow), 0., 1.))
            yaw = self.pwm_neutral_yaw + (yaw-self.pwm_neutral_yaw)*turn_fraction
        return roll, pitch, yaw


def make_teacher(version, **parameters):
    """Never silently replay historical labels with a different controller."""
    if version == LEGACY_TEACHER_VERSION:
        return LegacyExpertReflexPolicy(**parameters)
    if version == BRAKING_TEACHER_VERSION:
        return BrakingExpertReflexPolicy(**parameters)
    if version == TEACHER_VERSION:
        return ExpertReflexPolicy(**parameters)
    raise ValueError(f'Unknown teacher version: {version}')


def _make_scenario(name, seed, profile='standard-v1'):
    """The same real boxes participate in raycasting and collision detection."""
    rng = np.random.default_rng(seed)
    env = DroneSimulationEnv(dt=DEFAULT_DT, engine='standalone', headless=True,
                             room=RoomBoundaries(-8, 8, -8, 8, 0, 4))
    env.boxes, env.cylinders = [], []
    varied = profile == 'varied-v1'
    scene_parameters = {'profile':profile}
    if name.startswith('obstacle_'):
        distance = float(rng.uniform(.35, 1.25) if varied else rng.uniform(.45, .70))
        width = float(rng.uniform(.25, .9)) if varied else .5
        y_min, y_max = {'obstacle_left': (.02, width), 'obstacle_right': (-width, -.02),
                        'obstacle_center': (-width*.7, width*.7)}[name]
        env.boxes = [BoxObstacle(distance, distance + .15, y_min, y_max, .2, 2.)]
        scene_parameters.update(obstacle_distance_m=distance, obstacle_width_m=width)
    height = float(rng.uniform(.85, 1.5)) if varied else 1.
    env.reset(initial_pos=np.array([0., 0., height]), seed=seed)
    env.physics.quat[:] = [1., 0., 0., 0.]
    env.physics.vel[:] = [rng.uniform(.1, 1.5) if varied else rng.uniform(.15, .35), 0, 0]
    if name in {'drift_left', 'drift_right'}:
        env.physics.vel[1] = (rng.uniform(.25, .8) if varied else rng.uniform(.35, .65)) * (1 if name == 'drift_left' else -1)
    if name in {'drift_forward', 'drift_backward'}:
        env.physics.vel[0] = (rng.uniform(.8, 2.2) if varied else rng.uniform(1.4, 1.8)) * (1 if name == 'drift_forward' else -1)
    scene_parameters.update(initial_height_m=height, initial_speed_mps=float(env.physics.vel[0]),
                            initial_lateral_speed_mps=float(env.physics.vel[1]))
    env.scene_parameters = scene_parameters
    obs = env._sample_observation(elapsed_dt=0.)
    return env, obs


def _behavior(obs, action):
    if action[3] > PWM_MID + 50: return 1
    if action[3] < PWM_MID - 50: return 2
    if obs[1] > .03: return 3
    if obs[1] < -.03: return 4
    if action[2] < PWM_MID: return 5
    return 0


def generate_reflex_dataset(num_episodes=60, seq_len=100, dt=CONTROL_DT,
                            distance_threshold_m=APF_DISTANCE_THRESHOLD_M,
                            noise_std_pwm=APF_NOISE_STD_PWM, seed=42, output_path=None,
                            pwm_hover=PWM_HOVER, pwm_level_roll=PWM_LEVEL_ROLL,
                            pwm_cruise_pitch=PWM_CRUISE_PITCH, pwm_neutral_yaw=PWM_NEUTRAL_YAW,
                            pwm_min=PWM_MIN, pwm_max=PWM_MAX, scene_profile='standard-v1'):
    """One continuous trajectory per row, explicit valid-prefix padding on termination.

    dt is the policy period, a multiple of the 4 ms physics tick. Optional
    Navigation noise perturbs execution only; Y stores clean expert labels.
    Y[..., 0] is an unused interface placeholder, never a learning target.
    Applied throttle comes exclusively from the range/attitude controller.
    Standard collection aborts on a telemetry fault. Varied collection records
    the fault and retains only the nonempty valid prefix of that episode;
    a fault before its first valid frame still aborts collection.
    Equal scenario allocation is supplemented with per-frame behavior counts.
    """
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in (num_episodes, seq_len)):
        raise ValueError('num_episodes and seq_len must be positive integers')
    if scene_profile not in {'standard-v1', 'varied-v1'}:
        raise ValueError('Unknown scene_profile')
    if not np.isfinite(dt) or dt <= 0: raise ValueError('dt must be positive and finite')
    ratio = dt/DEFAULT_DT
    if ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-8, rel_tol=0):
        raise ValueError('Control dt must be an integer multiple of physics dt')
    if not np.isfinite(noise_std_pwm) or noise_std_pwm < 0:
        raise ValueError('Execution noise must be finite and nonnegative')
    # Scene selection cannot depend on trajectory length or an early crash.
    episode_seeds = np.random.default_rng(seed).integers(0, 2**31-1, size=num_episodes)
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    X = np.zeros((num_episodes, seq_len, SENSOR_DIM), dtype=np.float32)
    Y = np.full((num_episodes, seq_len, 4), PWM_MID, dtype=np.float32)
    applied_actions = np.full_like(Y, PWM_MID)
    valid = np.zeros((num_episodes, seq_len), dtype=bool)
    behavior = np.full((num_episodes, seq_len), -1, dtype=np.int64)
    scenarios, records = [], []
    for ep in range(num_episodes):
        name = SCENARIOS[ep % len(SCENARIOS)]
        scenarios.append(name)
        ep_seed = int(episode_seeds[ep])
        env, obs = (_make_scenario(name, ep_seed) if scene_profile == 'standard-v1'
                    else _make_scenario(name, ep_seed, profile=scene_profile))
        expert = ExpertReflexPolicy(distance_threshold_m=distance_threshold_m,
            max_sensor_range_m=env.tof.max_range, noise_std_pwm=0, seed=ep_seed,
            pwm_hover=pwm_hover, pwm_level_roll=pwm_level_roll, pwm_cruise_pitch=pwm_cruise_pitch,
            pwm_neutral_yaw=pwm_neutral_yaw, pwm_min=pwm_min, pwm_max=pwm_max)
        memory = EgocentricMemoryWrapper()
        altitude = AltitudeHold()
        ticks, done, info = 0, False, {}
        try:
            for t in range(seq_len):
                yaw = float(env.physics.quaternion_to_euler(env.physics.quat)[2])
                ring = memory.update(obs[2:66], current_yaw_rad=-yaw)
                x = np.concatenate([obs, ring]).astype(np.float32)
                label = expert.step(x, dt=dt)
                action = label.copy()
                action[1:] = np.clip(action[1:] + rng.normal(0, noise_std_pwm, 3), pwm_min, pwm_max)
                sample = sample_from_observation(env.get_isaac_obs())
                try:
                    action = altitude.apply(action, sample, now_s=ticks*DEFAULT_DT)
                except AltitudeTelemetryError as exc:
                    # New challenge collection preserves the last valid prefix
                    # and explicitly records the fault. Never fabricate an
                    # action/label for the invalid sample or reset mid-sequence.
                    if scene_profile == 'standard-v1' or t == 0:
                        raise
                    done, info = True, {'fatal_failure':True, 'failure_reason':str(exc)}
                    break
                X[ep, t], Y[ep, t], valid[ep, t] = x, label, True
                applied_actions[ep, t] = action
                behavior[ep, t] = _behavior(x, label)
                for _ in range(round(ratio)):
                    obs, _, done, info = env.step(action, action_type='pwm')
                    ticks += 1
                    if done: break
                if done: break
            records.append({'seed': ep_seed, 'scenario': name, 'valid_steps': int(valid[ep].sum()),
                            'physics_ticks': ticks, 'duration_s': ticks*DEFAULT_DT,
                            'terminated': bool(done), 'crashed': bool(info.get('crashed', False)),
                            'collision_kind': info.get('collision_kind')})
            if scene_profile != 'standard-v1':
                records[-1]['scene_parameters'] = env.scene_parameters
                records[-1]['failure_reason'] = info.get('failure_reason')
        finally:
            env.close()
    counts = Counter(BEHAVIORS[int(b)] for b in behavior[valid])
    dataset = {'X': torch.from_numpy(X), 'Y': torch.from_numpy(Y),
               'applied_actions': torch.from_numpy(applied_actions),
               'valid': torch.from_numpy(valid), 'behavior': torch.from_numpy(behavior),
               'metadata': {'dataset_version': DATASET_VERSION, 'teacher_version': TEACHER_VERSION,
                   'coordinate_version': COORDINATE_VERSION, 'observation_source': 'geometric_raycast',
                   'dt': dt, 'physics_dt': DEFAULT_DT, 'tof_max_range_m': TOF_RAYCASTER_MAX_RANGE_M,
                   'sensor_dim': SENSOR_DIM, 'action_dim': 4, 'seed': seed,
                   'scene_profile':scene_profile,
                   'episode_seed_schedule_version':'independent-scene-seeds-v1',
                   'control_contract': control_contract(),
                   'num_episodes': num_episodes, 'seq_len': seq_len,
                   'distance_threshold_m': distance_threshold_m, 'execution_noise_std_pwm': noise_std_pwm,
                   'teacher_parameters': expert.parameters,
                   'label_noise_std_pwm': 0., 'episode_scenarios': scenarios,
                   'scenario_counts': dict(Counter(scenarios)), 'behavior_names': list(BEHAVIORS),
                   'behavior_counts': dict(counts), 'episodes': records,
                   'required_sensor_motor_paths': deepcopy(REQUIRED_PATHS)}}
    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torch.save(dataset, output_path)
        report = {'metadata': dataset['metadata'], 'valid_frames': int(valid.sum()),
                  'crashed_episodes': sum(row['crashed'] for row in records),
                  'action_min': Y[valid].min(axis=0).tolist(), 'action_max': Y[valid].max(axis=0).tolist()}
        report['applied_action_min'] = applied_actions[valid].min(axis=0).tolist()
        report['applied_action_max'] = applied_actions[valid].max(axis=0).tolist()
        path = Path(output_path)
        path.with_name(path.stem + '_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=60)
    parser.add_argument('--seq_len', type=int, default=100)
    parser.add_argument('--output', default=DEFAULT_DATASET_PATH)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scene-profile', choices=['standard-v1', 'varied-v1'], default='standard-v1')
    args = parser.parse_args()
    path = os.path.join(_ROOT, args.output)
    data = generate_reflex_dataset(args.episodes, args.seq_len, seed=args.seed, output_path=path,
                                   scene_profile=args.scene_profile)
    print(f'Saved {path}: {tuple(data["X"].shape)}, valid frames={int(data["valid"].sum())}')
    print(f'Behavior counts: {data["metadata"]["behavior_counts"]}')
