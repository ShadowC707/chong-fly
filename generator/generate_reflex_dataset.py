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
    DATASET_VERSION, TEACHER_VERSION, DEFAULT_DATASET_PATH, SCENARIOS, BEHAVIORS, REQUIRED_PATHS,
)
from simulation.drone_env import DroneSimulationEnv, RoomBoundaries, BoxObstacle
from simulation.memory import EgocentricMemoryWrapper
from simulation.altitude_control import AltitudeHold, sample_from_observation
from simulation.control_contract import control_contract


class ExpertReflexPolicy:
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
        pwm = np.array([self.pwm_hover,
                        self.pwm_level_roll + self.flow_gain_pwm * obs[1],
                        self.pwm_cruise_pitch - braking - self.flow_gain_pwm * obs[0], yaw], dtype=np.float32)
        if self.noise_std_pwm:
            pwm += self.rng.normal(0, self.noise_std_pwm, 4).astype(np.float32)
        return np.clip(pwm, self.pwm_min, self.pwm_max)


def _make_scenario(name, seed):
    """The same real boxes participate in raycasting and collision detection."""
    rng = np.random.default_rng(seed)
    env = DroneSimulationEnv(dt=DEFAULT_DT, engine='standalone', headless=True,
                             room=RoomBoundaries(-8, 8, -8, 8, 0, 4))
    env.boxes, env.cylinders = [], []
    if name.startswith('obstacle_'):
        distance = float(rng.uniform(.45, .70))
        y_min, y_max = {'obstacle_left': (.02, .5), 'obstacle_right': (-.5, -.02),
                        'obstacle_center': (-.35, .35)}[name]
        env.boxes = [BoxObstacle(distance, distance + .15, y_min, y_max, .2, 2.)]
    env.reset(initial_pos=np.array([0., 0., 1.]), seed=seed)
    env.physics.quat[:] = [1., 0., 0., 0.]
    env.physics.vel[:] = [rng.uniform(.15, .35), 0, 0]
    if name in {'drift_left', 'drift_right'}:
        env.physics.vel[1] = rng.uniform(.35, .65) * (1 if name == 'drift_left' else -1)
    if name in {'drift_forward', 'drift_backward'}:
        env.physics.vel[0] = rng.uniform(1.4, 1.8) * (1 if name == 'drift_forward' else -1)
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
                            pwm_min=PWM_MIN, pwm_max=PWM_MAX):
    """One continuous trajectory per row, explicit valid-prefix padding on termination.

    dt is the policy period, a multiple of the 4 ms physics tick. Optional
    Navigation noise perturbs execution only; Y stores clean expert labels.
    Y[..., 0] is an unused interface placeholder, never a learning target.
    Applied throttle comes exclusively from the range/attitude controller.
    A telemetry fault aborts collection and does not publish a partial dataset.
    Equal scenario allocation is supplemented with per-frame behavior counts.
    """
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in (num_episodes, seq_len)):
        raise ValueError('num_episodes and seq_len must be positive integers')
    if not np.isfinite(dt) or dt <= 0: raise ValueError('dt must be positive and finite')
    ratio = dt/DEFAULT_DT
    if ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-8, rel_tol=0):
        raise ValueError('Control dt must be an integer multiple of physics dt')
    if not np.isfinite(noise_std_pwm) or noise_std_pwm < 0:
        raise ValueError('Execution noise must be finite and nonnegative')
    rng = np.random.default_rng(seed)
    X = np.zeros((num_episodes, seq_len, SENSOR_DIM), dtype=np.float32)
    Y = np.full((num_episodes, seq_len, 4), PWM_MID, dtype=np.float32)
    applied_actions = np.full_like(Y, PWM_MID)
    valid = np.zeros((num_episodes, seq_len), dtype=bool)
    behavior = np.full((num_episodes, seq_len), -1, dtype=np.int64)
    scenarios, records = [], []
    for ep in range(num_episodes):
        name = SCENARIOS[ep % len(SCENARIOS)]
        scenarios.append(name)
        ep_seed = int(rng.integers(0, 2**31 - 1))
        env, obs = _make_scenario(name, ep_seed)
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
                action = altitude.apply(action, sample, now_s=ticks*DEFAULT_DT)
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
    args = parser.parse_args()
    path = os.path.join(_ROOT, args.output)
    data = generate_reflex_dataset(args.episodes, args.seq_len, seed=args.seed, output_path=path)
    print(f'Saved {path}: {tuple(data["X"].shape)}, valid frames={int(data["valid"].sum())}')
    print(f'Behavior counts: {data["metadata"]["behavior_counts"]}')
