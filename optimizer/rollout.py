"""Deterministic, fixed-horizon standalone benchmark; no synthetic exploration cues."""
import inspect
import math

import numpy as np
import torch

from configs.flight_config import (
    CONTROL_DT, DEFAULT_DT, FLOW_DIM, TOF_DIM, PWM_MIN, PWM_MAX,
    PWM_SATURATION_LOW, PWM_SATURATION_HIGH, TOF_RAYCASTER_MAX_RANGE_M,
)
from simulation.memory import EgocentricMemoryWrapper
from simulation.control_contract import control_contract
from simulation.metrics import (
    VoxelTracker, PhysicsTelemetryTracker, calculate_jitter_pr, calculate_saccades_yaw,
    measure_yaw_bursts,
)

# v9 records corrected finite-cylinder caps and parallel box rays. Do not
# merge these flight results into studies evaluated with the old geometry.
BENCHMARK_VERSION = "flight-benchmark-v9"
RAW_BENCHMARK_VERSION = "flight-benchmark-v9-raw"


def _observation(result):
    if isinstance(result, tuple) and len(result) == 2:
        if isinstance(result[1], dict):
            return _observation(result[0])
        flow, tof = (np.asarray(x, dtype=np.float32).ravel() for x in result)
    else:
        flat = np.asarray(result, dtype=np.float32)
        if flat.shape != (FLOW_DIM + TOF_DIM,):
            raise ValueError("Expected a 66-element observation")
        flow, tof = flat[:FLOW_DIM], flat[FLOW_DIM:]
    if flow.shape != (FLOW_DIM,) or tof.shape != (TOF_DIM,):
        raise ValueError("Expected flow[2] and ToF[64]")
    if not (np.isfinite(flow).all() and np.isfinite(tof).all()):
        raise ValueError("Non-finite sensor observation")
    if np.any(tof < 0) or np.any(tof > 1):
        raise ValueError("ToF must be normalized to [0, 1]")
    return flow.copy(), tof.copy()


def _state(env):
    physics = getattr(env, "physics", env)
    pos = np.asarray(physics.pos, dtype=float)
    vel = np.asarray(physics.vel, dtype=float)
    if hasattr(physics, "quaternion_to_euler"):
        yaw = float(physics.quaternion_to_euler(physics.quat)[2])
    else:
        yaw = float(env.att[2])
    if pos.shape != (3,) or vel.shape != (3,) or not all(np.isfinite(x).all() for x in (pos, vel, yaw)):
        raise ValueError("Non-finite or malformed physical state")
    body_vel = np.array([vel[0]*math.cos(yaw) + vel[1]*math.sin(yaw),
                         -vel[0]*math.sin(yaw) + vel[1]*math.cos(yaw), vel[2]])
    return pos, vel, yaw, body_vel


def _policy_action(policy, flow, tof, memory, dt):
    if hasattr(policy, "step_np"):
        # Adapt optional arguments without swallowing TypeError from inside a policy.
        signature = inspect.signature(policy.step_np)
        kwargs = {"memory_ring": memory, "dt": dt}
        if not any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
            kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
        action = policy.step_np(flow, tof, **kwargs)
    elif callable(policy):
        action = policy(np.concatenate([flow, tof, memory]))
    else:
        raise TypeError("Policy must expose step_np or be callable")
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    return np.asarray(action, dtype=float)


@torch.no_grad()
def simulate_policy_rollout(policy, env=None, eval_steps=100, dt=None, seed=42, altitude_hold=None,
                            record_trace=False, coordinator=None, stop_at_s=None):
    """Evaluate ``eval_steps`` control periods, holding each RC action between calls.

    Physics time and controller time are distinct. The latter must be an integer
    multiple of env.dt. A crash stops at the actual physics tick. Feasibility
    requires a complete horizon, valid commands/state, and no reported crash.
    It is a benchmark gate, not a real-aircraft safety certificate.
    Optional altitude_hold owns throttle using the sensor snapshot. Its results
    carry the v7 navigation benchmark version only with the standard contract.
    Omission preserves the historical raw-policy v6 evaluation explicitly.
    """
    control_dt = CONTROL_DT if dt is None else float(dt)
    if not np.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("control dt must be positive and finite")
    if isinstance(eval_steps, bool) or not isinstance(eval_steps, (int, np.integer)) or eval_steps <= 0:
        raise ValueError("eval_steps must be a positive integer")
    if type(record_trace) is not bool:
        raise ValueError('record_trace must be boolean')
    if stop_at_s is not None and (not math.isfinite(stop_at_s) or stop_at_s < 0
            or coordinator is None or not coordinator.stop_hover):
        raise ValueError('Scheduled STOP requires a nonnegative time and stop_hover coordinator')
    trace = {'control': [], 'physics': []} if record_trace else None
    owned = env is None
    if owned:
        from simulation.drone_env import DroneSimulationEnv
        env = DroneSimulationEnv(dt=DEFAULT_DT, engine="standalone", headless=True)
    was_training = getattr(policy, "training", None)
    try:
        physics_dt = float(env.dt)
        if not np.isfinite(physics_dt) or physics_dt <= 0:
            raise ValueError("physics dt must be positive and finite")
        ratio = control_dt / physics_dt
        substeps = round(ratio)
        if substeps < 1 or not math.isclose(ratio, substeps, rel_tol=0, abs_tol=1e-8):
            raise ValueError("control dt must be an integer multiple of physics dt")
        if hasattr(policy, "eval"):
            policy.eval()
        if hasattr(policy, "reset_state"):
            policy.reset_state()
        if altitude_hold is not None:
            altitude_hold.reset()
        if coordinator is not None:
            coordinator.reset()
        intervention_steps = 0
        intervention_reasons = {}
        reset_result = env.reset(seed=seed)
        voxels, telemetry = VoxelTracker(), PhysicsTelemetryTracker()
        memory = EgocentricMemoryWrapper()
        pwms, speeds, clearances = [], [], []
        min_clearance_event = None
        yaw_rates = []
        altitude_errors = []
        ticks = saturation_count = 0
        total_energy = 0.0
        crashed = fatal = terminated = False
        failure_reason = None
        tof_range = float(getattr(getattr(env, "tof", None), "max_range", TOF_RAYCASTER_MAX_RANGE_M))
        info = {}
        try:
            flow, tof = _observation(reset_result)
            pos, vel, yaw, body_vel = _state(env)
            voxels.update(pos)
            if trace is not None:
                trace['physics'].append(dict(time_s=0., position_m=pos.tolist(),
                    yaw_rad=float(yaw), body_velocity_mps=body_vel.tolist()))
        except ValueError as exc:
            fatal, failure_reason = True, str(exc)

        for _ in range(eval_steps):
            if fatal or terminated:
                break
            # Memory's existing API uses right-positive yaw; physics uses +Z/left-positive.
            ring = memory.update(tof, current_yaw_rad=-yaw)
            pwm = _policy_action(policy, flow, tof, ring, control_dt)
            if pwm.shape != (4,) or not np.isfinite(pwm).all() or np.any(pwm < PWM_MIN) or np.any(pwm > PWM_MAX):
                fatal, failure_reason = True, "Invalid RC command (expected four finite PWM values in [1000, 2000])"
                break
            network_pwm = pwm.copy()
            decision = dict(intervened=False, reasons=[])
            if coordinator is not None:
                pwm = coordinator.apply(pwm, flow_xy=flow, tof_m=tof*tof_range, dt=control_dt,
                    stop_requested=bool(stop_at_s is not None and ticks*physics_dt + 1e-9 >= stop_at_s))
                decision = coordinator.last_decision.copy()
                intervention_steps += int(decision['intervened'])
                for reason in decision['reasons']:
                    intervention_reasons[reason] = intervention_reasons.get(reason, 0) + 1
            if altitude_hold is not None:
                from simulation.altitude_control import sample_from_observation, AltitudeTelemetryError
                try:
                    if not hasattr(env, 'get_isaac_obs'):
                        raise AltitudeTelemetryError('Environment has no range-attitude snapshot')
                    sample = sample_from_observation(env.get_isaac_obs())
                    pwm = altitude_hold.apply(pwm, sample, now_s=ticks*physics_dt)
                    altitude_errors.append(altitude_hold.last_command.height_m-altitude_hold.config.target_height_m)
                except AltitudeTelemetryError as exc:
                    fatal, failure_reason = True, f'Altitude control: {exc}'
                    break
            pwms.append(pwm.copy())
            omega = getattr(getattr(env, 'physics', env), 'omega', None)
            yaw_rates.append(float(omega[2]) if omega is not None else float('nan'))
            if trace is not None:
                trace['control'].append(dict(time_s=ticks*physics_dt, rc_pwm=pwm.tolist(),
                    network_pwm=network_pwm.tolist(), applied_pwm=pwm.tolist(), coordinator=decision,
                    flow_xy=flow.tolist(), min_tof_m=float(np.min(tof)*tof_range),
                    yaw_rate_rad_s=float(omega[2]) if omega is not None and np.isfinite(omega[2]) else None))
            saturation_count += int(np.any(pwm < PWM_SATURATION_LOW) or np.any(pwm > PWM_SATURATION_HIGH))
            for _ in range(substeps):
                next_obs, _, done, info = env.step(pwm)
                ticks += 1
                crashed = crashed or bool(info.get("crashed", False))
                terminated = bool(done or crashed)
                try:
                    flow, tof = _observation(next_obs)
                    pos, vel, yaw, body_vel = _state(env)
                    power = float(info["power_w"])
                    clearance = float(info.get("clearance_m", np.min(tof)*tof_range))
                    if not np.isfinite(power) or power < 0 or not np.isfinite(clearance):
                        raise ValueError("Invalid power/clearance telemetry")
                    if info.get("invalid_state", False):
                        raise ValueError("Invalid physics state")
                    if trace is not None:
                        trace['physics'].append(dict(time_s=ticks*physics_dt, position_m=pos.tolist(),
                            yaw_rad=float(yaw), body_velocity_mps=body_vel.tolist(), clearance_m=clearance))
                    if min_clearance_event is None or clearance < min_clearance_event['clearance_m']:
                        min_clearance_event = dict(time_s=ticks*physics_dt, position_m=pos.tolist(),
                            clearance_m=clearance,
                            source='body_envelope' if 'clearance_m' in info else 'minimum_tof_fallback')
                    total_energy += power * physics_dt
                    clearances.append(clearance)
                    speeds.append(float(body_vel[0]))
                    voxels.update(pos)
                    telemetry.update_step([clearance], body_vel)
                    if crashed:
                        mass = getattr(getattr(env, "physics", env), "total_mass", None)
                        telemetry.register_crash(vel, mass=mass)
                except ValueError as exc:
                    fatal, failure_reason = True, str(exc)
                if terminated or fatal:
                    break

        tensors = []
        if hasattr(policy, "parameters"):
            tensors.extend(policy.parameters())
        if hasattr(policy, "buffers"):
            tensors.extend(policy.buffers())
        # Dense tensor footprint, including zeros/buffers/heads, not peak inference RAM.
        storage_bytes = sum(t.numel()*t.element_size() for t in tensors)
        elapsed = ticks * physics_dt
        complete = ticks == eval_steps * substeps
        summary = telemetry.compute_summary()
        jitter = float(calculate_jitter_pr(pwms))
        saccades = calculate_saccades_yaw(pwms)
        yaw_bursts = (measure_yaw_bursts(yaw_rates, dt=control_dt)
                      if yaw_rates and np.isfinite(yaw_rates).all() else
                      {'version':'physical-yaw-bursts-v1', 'count':None, 'events':[], 'available':False})
        feasible = bool(complete and not terminated and not crashed and not fatal and not info.get("user_closed", False))
        metrics = {
            "benchmark_version": (RAW_BENCHMARK_VERSION if altitude_hold is None else
                                  BENCHMARK_VERSION if control_contract(altitude_hold) == control_contract()
                                  else BENCHMARK_VERSION + '+custom-altitude'),
            "fatal_failure": fatal, "failure_reason": failure_reason,
            "sensor_geometry_version": getattr(getattr(env, 'tof', None), 'geometry_version', None),
            "crashed": crashed, "feasible": feasible,
            "horizon_completed": complete, "terminated": terminated,
            "physics_steps": ticks, "control_steps": len(pwms),
            "physics_dt": physics_dt, "control_dt": control_dt,
            "survival_time_s": elapsed, "survival_ratio": ticks/(eval_steps*substeps),
            "model_storage_bytes": int(storage_bytes),
            "total_energy_j": total_energy, "mean_power_w": total_energy/elapsed if elapsed else 0.0,
            "mean_fwd_speed": float(np.mean(speeds)) if speeds else 0.0,
            "saturation_ratio": saturation_count/max(1, len(pwms)),
            "jitter_pr_l2": jitter, "saccades_yaw_count": yaw_bursts['count'],
            "yaw_bursts":yaw_bursts, "legacy_yaw_jump_count":saccades,
            "roughness_score": float(jitter + saccades),
            "coverage_count": voxels.get_coverage_count(), "coverage_volume": voxels.get_coverage_volume(),
            "mean_clearance": summary["mean_clearance"],
            "min_clearance_m": min(clearances) if clearances else 0.0,
            "min_clearance_event": min_clearance_event,
            "impact_energy_j": summary["impact_energy_j"],
            "forward_ratio_median": summary["forward_ratio_median"],
            "is_crab_flight": summary["is_crab_flight"],
            "memory_sectors": memory.get_memory().tolist(),
            "collision_kind": info.get('collision_kind'),
        }
        if altitude_hold is not None:
            metrics['control_contract'] = control_contract(altitude_hold)
            metrics['altitude_control'] = altitude_hold.contract()
            metrics['altitude_tracking'] = {
                'sensor_samples': len(altitude_errors),
                'mean_abs_error_m': float(np.mean(np.abs(altitude_errors))) if altitude_errors else None,
                'max_abs_error_m': float(np.max(np.abs(altitude_errors))) if altitude_errors else None,
            }
        if coordinator is not None:
            metrics['benchmark_version'] += '+' + coordinator.contract()['version']
            metrics['coordinator'] = dict(contract=coordinator.contract(),
                intervention_steps=intervention_steps, reason_counts=intervention_reasons, stop_at_s=stop_at_s)
        if trace is not None:
            metrics['trace'] = trace
        return (float(storage_bytes), jitter, float(voxels.get_coverage_count())), metrics
    finally:
        if was_training is not None:
            policy.train(was_training)
        if owned:
            env.close()
