"""
simulation/drone_env.py
=======================
High-Fidelity Isaac Drone Simulation Environment with Betaflight Flight Controller
and Multi-Modal Sensor Emulation (NVIDIA Isaac Gym & 6-DOF Vectorized Engine).

Quadcopter Specifications:
--------------------------
1. Physical Frame & Mass:
   - Base Mass: 130 grams (0.130 kg dry weight)
   - Dimensions: 15 cm x 15 cm (0.15 m x 0.15 m frame span)
   - Payload Capacity: 1.2 kg - 1.5 kg (maximum thrust up to 28 N, TWR >= 1.7 at 1.63 kg AUW)
   - Quad-X Motor Geometry: Moment arm d = 0.053 m (150 mm diagonal wheelbase)
   - Brushless Motor Dynamics: Max 28,000 RPM, thrust constant k_f = 8.9286e-9 N/RPM^2

2. Sensor Suite:
   - Downward Laser Rangefinder:
       Single-beam ToF distance sensor pointing straight down along body -Z.
       Measures true AGL (Altitude Above Ground) with raycasting against floor & obstacles.
   - 8x8 Depth Matrix (VL53L5CX):
       64 directional rays with 45° Field of View (horizontal & vertical).
       Normalized depth output [0, 1] for obstacle detection and avoidance.
   - Optical Displacement & Flow Sensor (PMW3901):
       Tracks surface flow rates (FlowX, FlowY) and incremental/cumulative displacement [dx, dy].
       Scaled dynamically by the downward laser altitude reading.

3. Avionics & Flight Interface:
   - Cascaded Betaflight PID (Angle Mode outer loop + Rate PID inner loop)
   - Accepts high-level setpoints [Thrust, Roll, Pitch, YawRate] or RC PWM [1000..2000 µs]
   - Fully modular Isaac interface (IsaacObservation, IsaacAction, BaseIsaacModel)
   - Real-time 3D visualization in NVIDIA Isaac Gym with laser beam rendering & HUD
"""

from __future__ import annotations

import os
import sys

# Ensure Python environment binaries (ninja, etc.) are in PATH for Isaac Gym
venv_bin = os.path.join(sys.prefix, "bin")
if venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")

# IMPORTANT: NVIDIA Isaac Gym must be imported BEFORE torch to prevent CUDA symbol clashes
try:
    from isaacgym import gymapi, gymtorch
    HAS_ISAACGYM = True
except ImportError:
    HAS_ISAACGYM = False

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Import avionics PID and optical flow sensor
from simulation.avionics_filter import BetaflightCascadedPID, PIDConstants
from simulation.pmw3901_emulator import PMW3901FlowSensor
from simulation.drone_interface import (
    DownwardLaserSensor,
    LaserHitResult,
    IsaacObservation,
    IsaacAction,
    BaseIsaacModel,
    ConnectomeModelAdapter,
    AutonomousLaserNavigatorModel,
)
from simulation.isaac_hud import IsaacGymHUD
from configs.flight_config import (
    DEFAULT_DT,
    GRAVITY,
    DRONE_MASS_ISAAC_KG,
    TARGET_ALTITUDE_M,
    ROOM_X_MIN,
    ROOM_X_MAX,
    ROOM_Y_MIN,
    ROOM_Y_MAX,
    ROOM_Z_MIN,
    ROOM_Z_MAX,
    TOF_ROWS,
    TOF_COLS,
    TOF_RAYCASTER_MAX_RANGE_M,
    PWM_MIN,
    PWM_MID,
    PWM_MAX,
    PWM_HALF,
    PWM_HOVER,
    PWM_LEVEL_ROLL,
    PWM_NEUTRAL_YAW,
    FLOW_DIM,
    TOF_DIM,
    SENSOR_DIM_BASE,
    LASER_MIN_RANGE_M,
    LASER_MAX_RANGE_M,
    LASER_NOISE_STD,
    FLOW_MIN_ALTITUDE_M,
    FLOW_MAX_ALTITUDE_M,
    GROUND_CRASH_ALT_M,
    TUMBLE_ANGLE_THRESHOLD_RAD,
    SENSORS,
    PHYSICS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-Configured Biological Connectome Models Registry
# ─────────────────────────────────────────────────────────────────────────────

AVAILABLE_MODELS = [
    {
        "id": 1,
        "key": "1",
        "name": "Spectral k=64",
        "tag": "Balanced Connectome (Full CX)",
        "meta": "data/reduced_models/meta_spectral_k64.json",
        "desc": "Default 64-cluster connectome with Central Complex",
    },
    {
        "id": 2,
        "key": "2",
        "name": "Spectral k=16",
        "tag": "Micro-MCU / Low Latency",
        "meta": "data/reduced_models/meta_spectral_k16.json",
        "desc": "Ultra-compact 16 clusters for low-power MCUs",
    },
    {
        "id": 3,
        "key": "3",
        "name": "Spectral k=32",
        "tag": "Fast 32-cluster",
        "meta": "data/reduced_models/meta_spectral_k32.json",
        "desc": "32-cluster connectome optimized for speed",
    },
    {
        "id": 4,
        "key": "4",
        "name": "Spectral k=128",
        "tag": "High Capacity",
        "meta": "data/reduced_models/meta_spectral_k128.json",
        "desc": "128 clusters with rich multi-sensory representation",
    },
    {
        "id": 5,
        "key": "5",
        "name": "Spectral k=256",
        "tag": "Deep Connectome",
        "meta": "data/reduced_models/meta_spectral_k256.json",
        "desc": "High-resolution 256-cluster Drosophila connectome",
    },
    {
        "id": 6,
        "key": "6",
        "name": "Spectral k=64 No-CX",
        "tag": "Reflex Only (CX Ablated)",
        "meta": "data/reduced_models/meta_spectral_k64_nocx.json",
        "desc": "Central Complex severed; reflex-only flight behavior",
    },
    {
        "id": 7,
        "key": "7",
        "name": "Centrality k=64",
        "tag": "Hub Prior Clustering",
        "meta": "data/reduced_models/meta_centrality_k64.json",
        "desc": "Clustering preserving high betweenness-centrality hubs",
    },
    {
        "id": 8,
        "key": "8",
        "name": "Magnitude Pruned p80",
        "tag": "80% Sparse Graph",
        "meta": "data/reduced_models/meta_magnitude_p80_k12942.json",
        "desc": "80% synaptic weight pruning on raw connectome",
    },
    {
        "id": 9,
        "key": "9",
        "name": "Autonomous Laser+8x8",
        "tag": "All-Sensors Multi-Modal (Laser+ToF+Flow)",
        "meta": "autonomous_laser_nav",
        "desc": "Autonomous controller using Downward Laser + 8x8 Depth + Optical Displacement",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# NVIDIA Isaac Gym PhysX Simulation Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class IsaacGymDroneSim:
    """
    NVIDIA Isaac Gym PhysX simulation backend for micro-quadcopters.
    
    Provides:
      - PhysX GPU/CPU physics engine setup via gymapi.acquire_gym()
      - Interactive 3D visualizer (gym.create_viewer) with camera tracking
      - URDF loading of assets/chong_micro_quad.urdf
      - Batched PyTorch tensor wrapping via gymtorch.wrap_actor_root_state_tensor
      - Real-time keyboard event processing (WASD, Space/C, 1-8 model switching, R reset)
      - Dynamic camera tracking modes (Chase Cam, Arena Cam, Free Cam)
      - Multi-environment parallel scaling (num_envs = 1..4096)
    """

    def __init__(
        self,
        num_envs: int = 1,
        dt: float = DEFAULT_DT,
        headless: bool = False,
        compute_device_id: int = 0,
        graphics_device_id: int = 0,
        asset_root: str = "assets",
        asset_file: str = "chong_micro_quad.urdf",
    ):
        self.num_envs = num_envs
        self.dt = dt
        self.headless = headless
        self.compute_device_id = compute_device_id
        self.graphics_device_id = graphics_device_id
        self.asset_root = os.path.join(_ROOT, asset_root)
        self.asset_file = asset_file
        self.render_every = 5  # Decoupled 50 FPS graphics at 250 Hz physics

        # Keyboard & Camera state
        self.key_states: Dict[str, bool] = {}
        self.camera_modes = ["chase", "arena", "free"]
        self.camera_mode_idx = 0

        if not HAS_ISAACGYM:
            raise RuntimeError(
                "NVIDIA Isaac Gym is not installed in the current Python environment.\n"
                "To enable native Isaac Gym with 3D PhysX acceleration:\n"
                "  1. Download IsaacGym_Preview_4_Package.tar.gz from developer.nvidia.com\n"
                "  2. Install via: pip install -e isaacgym/python (requires Python 3.7/3.8 and NVIDIA GPU)\n"
                "Chong-Fly includes a high-performance standalone 6-DOF engine that runs on CPU without Isaac Gym."
            )

        self._init_isaacgym()

    def _init_isaacgym(self):
        self.gym = gymapi.acquire_gym()

        # Sim params
        self.sim_params = gymapi.SimParams()
        self.sim_params.dt = self.dt
        self.sim_params.substeps = 2
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -GRAVITY)

        self.sim_params.physx.solver_type = 1  # TGS
        self.sim_params.physx.num_position_iterations = 4
        self.sim_params.physx.num_velocity_iterations = 1
        self.sim_params.physx.num_threads = 4
        self.sim_params.physx.use_gpu = (self.compute_device_id >= 0)
        self.sim_params.use_gpu_pipeline = self.sim_params.physx.use_gpu

        self.sim = self.gym.create_sim(
            self.compute_device_id,
            self.graphics_device_id,
            gymapi.SIM_PHYSX,
            self.sim_params,
        )

        # Add ground plane with grid
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0.0
        self.gym.add_ground(self.sim, plane_params)

        # Load drone asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.angular_damping = 0.002
        asset_options.linear_damping = 0.15
        self.drone_asset = self.gym.load_asset(self.sim, self.asset_root, self.asset_file, asset_options)

        # Procedural obstacle assets matching ToF raycaster arena
        obs_opts = gymapi.AssetOptions()
        obs_opts.fix_base_link = True
        obs_opts.disable_gravity = True
        box_asset = self.gym.create_box(self.sim, 0.8, 0.4, 2.5, obs_opts)
        cap1_asset = self.gym.create_capsule(self.sim, 0.3, 3.0, obs_opts)
        cap2_asset = self.gym.create_capsule(self.sim, 0.25, 3.0, obs_opts)

        # Create envs
        spacing = 4.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.envs = []
        self.actors = []
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, max(1, int(math.sqrt(self.num_envs))))
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            actor = self.gym.create_actor(env_ptr, self.drone_asset, pose, f"chong_drone_{i}", i, 0)
            self.envs.append(env_ptr)
            self.actors.append(actor)

            # Arena obstacles (visual & physical in PhysX)
            self.gym.create_actor(env_ptr, box_asset, gymapi.Transform(gymapi.Vec3(0.0, 2.4, 1.25)), f"box_{i}", i, 0)
            self.gym.create_actor(env_ptr, cap1_asset, gymapi.Transform(gymapi.Vec3(1.5, 0.0, 1.5)), f"pillar1_{i}", i, 0)
            self.gym.create_actor(env_ptr, cap2_asset, gymapi.Transform(gymapi.Vec3(-1.5, 1.0, 1.5)), f"pillar2_{i}", i, 0)

        self.gym.prepare_sim(self.sim)

        # Wrap PyTorch state tensors
        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self.root_tensor)

        # 3D Visualizer & Input Handling
        if not self.headless:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 1280
            camera_props.height = 720
            self.viewer = self.gym.create_viewer(self.sim, camera_props)
            # Position camera with clear initial perspective of drone
            cam_pos = gymapi.Vec3(1.6, 1.6, 1.5)
            cam_target = gymapi.Vec3(0.0, 0.0, 1.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            self.last_cam_pos = np.array([1.6, 1.6, 1.5], dtype=np.float32)
            self.last_cam_target = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            self.hud = IsaacGymHUD()
            self._setup_keyboard_events()
        else:
            self.viewer = None
            self.last_cam_pos = np.array([1.6, 1.6, 1.5], dtype=np.float32)
            self.last_cam_target = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            self.hud = IsaacGymHUD()

    def _setup_keyboard_events(self):
        """Subscribes to viewer keyboard events for interactive control and hotkeys."""
        if self.viewer is None:
            return
        g = self.gym
        v = self.viewer
        # Manual flight controls (WASD)
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_W, "pitch_fwd")
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_S, "pitch_back")
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_A, "roll_left")
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_D, "roll_right")
        # Throttle (Space / Shift / C / Arrow Up/Down)
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_SPACE, "throttle_up")
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_UP, "throttle_up")
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_LEFT_SHIFT, "throttle_down")
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_DOWN, "throttle_down")
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_C, "throttle_down")
        # Yaw (Q / E / Arrow Left/Right)
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_Q, "yaw_left")
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_E, "yaw_right")
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_LEFT, "yaw_left")
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_RIGHT, "yaw_right")
        # Mode & Utility
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_M, "toggle_mode")
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_R, "reset")
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_P, "pause")
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_V, "cycle_camera")
        g.subscribe_viewer_keyboard_event(v, gymapi.KEY_ESCAPE, "quit")

        # In-Viewer Model Selector Menu Toggle & Cycling
        if hasattr(gymapi, "KEY_TAB"):
            g.subscribe_viewer_keyboard_event(v, gymapi.KEY_TAB, "toggle_menu")
        if hasattr(gymapi, "KEY_N"):
            g.subscribe_viewer_keyboard_event(v, gymapi.KEY_N, "next_model")
        if hasattr(gymapi, "KEY_B"):
            g.subscribe_viewer_keyboard_event(v, gymapi.KEY_B, "prev_model")

        # Hover Target Altitude Adjustment (+/- 0.25 m)
        if hasattr(gymapi, "KEY_U"):
            g.subscribe_viewer_keyboard_event(v, gymapi.KEY_U, "alt_up")
        if hasattr(gymapi, "KEY_J"):
            g.subscribe_viewer_keyboard_event(v, gymapi.KEY_J, "alt_down")

        # Dynamic Payload Adjustment hotkeys ('[' / '-' to decrease, ']' / '=' to increase payload by 0.1 kg)
        if hasattr(gymapi, "KEY_LEFT_BRACKET"):
            g.subscribe_viewer_keyboard_event(v, gymapi.KEY_LEFT_BRACKET, "payload_down")
        if hasattr(gymapi, "KEY_MINUS"):
            g.subscribe_viewer_keyboard_event(v, gymapi.KEY_MINUS, "payload_down")
        if hasattr(gymapi, "KEY_RIGHT_BRACKET"):
            g.subscribe_viewer_keyboard_event(v, gymapi.KEY_RIGHT_BRACKET, "payload_up")
        if hasattr(gymapi, "KEY_EQUAL"):
            g.subscribe_viewer_keyboard_event(v, gymapi.KEY_EQUAL, "payload_up")

        # Number keys 1-9 for live model switching (1-8 biological connectomes, 9 autonomous laser nav)
        for i in range(1, 10):
            key_attr = f"KEY_{i}"
            if hasattr(gymapi, key_attr):
                g.subscribe_viewer_keyboard_event(v, getattr(gymapi, key_attr), f"model_{i}")

    def render_hud(
        self,
        drone_pos: np.ndarray,
        drone_rot: np.ndarray,
        drone_vel: np.ndarray,
        laser_hit_point: Optional[np.ndarray],
        depth_8x8: np.ndarray,
        optical_flow: np.ndarray,
        active_model_idx: int,
        models_list: List[Dict[str, Any]],
        control_mode: str,
        payload_kg: float,
        total_mass_kg: float,
        hover_throttle: float,
        laser_alt_m: float,
        target_alt_m: float,
        motors: np.ndarray,
        power_w: float,
        is_paused: bool = False,
        is_crashed: bool = False,
    ):
        """Draws all 3D sensor rays and camera-fixed HUD dashboard lines in Isaac Gym."""
        if self.viewer is None or not self.envs:
            return

        self.gym.clear_lines(self.viewer)
        verts, cols, num_lines = self.hud.generate_frame_lines(
            cam_pos=self.last_cam_pos,
            cam_target=self.last_cam_target,
            drone_pos=drone_pos,
            drone_rot=drone_rot,
            drone_vel=drone_vel,
            laser_hit_point=laser_hit_point,
            depth_8x8=depth_8x8,
            optical_flow=optical_flow,
            active_model_idx=active_model_idx,
            models_list=models_list,
            control_mode=control_mode,
            payload_kg=payload_kg,
            total_mass_kg=total_mass_kg,
            hover_throttle=hover_throttle,
            laser_alt_m=laser_alt_m,
            target_alt_m=target_alt_m,
            motors=motors,
            power_w=power_w,
            is_paused=is_paused,
            is_crashed=is_crashed,
        )
        if num_lines > 0:
            self.gym.add_lines(self.viewer, self.envs[0], num_lines, verts, cols)

    def draw_laser_beam(self, origin: np.ndarray, hit_point: np.ndarray):
        """Draws visual downward laser beam in Isaac Gym 3D visualizer."""
        if self.viewer is None or not self.envs:
            return
        self.gym.clear_lines(self.viewer)
        verts = np.array([
            float(origin[0]), float(origin[1]), float(origin[2]),
            float(hit_point[0]), float(hit_point[1]), float(hit_point[2]),
        ], dtype=np.float32)
        colors = np.array([1.0, 0.15, 0.15], dtype=np.float32) # Red laser ray
        self.gym.add_lines(self.viewer, self.envs[0], 1, verts, colors)

    def poll_input(self) -> Tuple[Dict[str, bool], List[str]]:
        """
        Polls viewer keyboard events.
        Returns:
            key_states: dict of active keys currently held down
            triggers: list of action names that were just pressed down this step
        """
        if self.viewer is None:
            return {}, []

        triggers = []
        events = self.gym.query_viewer_action_events(self.viewer)
        for evt in events:
            is_down = (evt.value > 0.5)
            self.key_states[evt.action] = is_down
            if is_down:
                triggers.append(evt.action)

        return self.key_states, triggers

    def update_camera(self, pos: np.ndarray, euler: np.ndarray):
        """Updates camera position according to current camera mode."""
        if self.viewer is None:
            return
        mode = self.camera_modes[self.camera_mode_idx]
        if mode == "chase":
            yaw = float(euler[2])
            cam_pos = gymapi.Vec3(
                float(pos[0]) - 0.85 * math.cos(yaw),
                float(pos[1]) - 0.85 * math.sin(yaw),
                float(pos[2]) + 0.35,
            )
            cam_target = gymapi.Vec3(
                float(pos[0]) + 0.25 * math.cos(yaw),
                float(pos[1]) + 0.25 * math.sin(yaw),
                float(pos[2]) + 0.05,
            )
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            self.last_cam_pos = np.array([cam_pos.x, cam_pos.y, cam_pos.z], dtype=np.float32)
            self.last_cam_target = np.array([cam_target.x, cam_target.y, cam_target.z], dtype=np.float32)
        elif mode == "arena":
            cam_pos = gymapi.Vec3(
                float(pos[0]) + 1.4,
                float(pos[1]) + 1.4,
                float(pos[2]) + 0.9,
            )
            cam_target = gymapi.Vec3(float(pos[0]), float(pos[1]), float(pos[2]))
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            self.last_cam_pos = np.array([cam_pos.x, cam_pos.y, cam_pos.z], dtype=np.float32)
            self.last_cam_target = np.array([cam_target.x, cam_target.y, cam_target.z], dtype=np.float32)
        elif mode == "free":
            try:
                ct = self.gym.get_viewer_camera_transform(self.viewer, None)
                self.last_cam_pos = np.array([ct.p.x, ct.p.y, ct.p.z], dtype=np.float32)
                qx, qy, qz, qw = ct.r.x, ct.r.y, ct.r.z, ct.r.w
                fwd_x = 2.0 * (qx * qz + qy * qw)
                fwd_y = 2.0 * (qy * qz - qx * qw)
                fwd_z = 1.0 - 2.0 * (qx * qx + qy * qy)
                self.last_cam_target = self.last_cam_pos + np.array([fwd_x, fwd_y, fwd_z], dtype=np.float32)
            except Exception:
                pass

    def sync_state(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        vel: Optional[np.ndarray] = None,
        omega: Optional[np.ndarray] = None,
    ):
        """
        Synchronizes 6-DOF physics state into Isaac Gym PhysX actor root state tensor.
        quat format in QuadcopterDynamics: [qw, qx, qy, qz]
        quat format in Isaac Gym / PhysX:   [qx, qy, qz, qw]
        """
        if self.root_states is not None:
            self.root_states[0, 0] = float(pos[0])
            self.root_states[0, 1] = float(pos[1])
            self.root_states[0, 2] = float(pos[2])
            # Quat conversion: [qw, qx, qy, qz] -> [qx, qy, qz, qw]
            self.root_states[0, 3] = float(quat[1])
            self.root_states[0, 4] = float(quat[2])
            self.root_states[0, 5] = float(quat[3])
            self.root_states[0, 6] = float(quat[0])
            if vel is not None:
                self.root_states[0, 7] = float(vel[0])
                self.root_states[0, 8] = float(vel[1])
                self.root_states[0, 9] = float(vel[2])
            if omega is not None:
                self.root_states[0, 10] = float(omega[0])
                self.root_states[0, 11] = float(omega[1])
                self.root_states[0, 12] = float(omega[2])
            self.gym.set_actor_root_state_tensor(self.sim, self.root_tensor)

    def step(
        self,
        render: bool = True,
        pos: Optional[np.ndarray] = None,
        euler: Optional[np.ndarray] = None,
    ) -> bool:
        """Advances PhysX simulation by one dt and renders if viewer is active."""
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        if self.viewer is not None and render:
            if self.gym.query_viewer_has_closed(self.viewer):
                return False
            if pos is not None and euler is not None:
                self.update_camera(pos, euler)
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)
        return True

    def render(self) -> bool:
        """Renders current frame to viewer."""
        if self.viewer is not None:
            if self.gym.query_viewer_has_closed(self.viewer):
                return False
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)
        return True

    def close(self):
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
            self.viewer = None
        if hasattr(self, 'sim') and self.sim is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None



# ─────────────────────────────────────────────────────────────────────────────
# 3D Geometric Obstacles for ToF Raycasting
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RoomBoundaries:
    """3D bounding box of the flight arena."""
    x_min: float = ROOM_X_MIN
    x_max: float = ROOM_X_MAX
    y_min: float = ROOM_Y_MIN
    y_max: float = ROOM_Y_MAX
    z_min: float = ROOM_Z_MIN
    z_max: float = ROOM_Z_MAX


@dataclass
class CylinderObstacle:
    """Vertical cylindrical pillar obstacle."""
    center_x: float
    center_y: float
    radius: float
    height: float = 3.0


@dataclass
class BoxObstacle:
    """Axis-aligned box obstacle."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


# ─────────────────────────────────────────────────────────────────────────────
# 8x8 ToF Raycaster (VL53L5CX Emulation)
# ─────────────────────────────────────────────────────────────────────────────

class ToFRaycaster:
    """
    Simulates an 8x8 multi-zone Time-of-Flight sensor (e.g. STMicroelectronics VL53L5CX).
    
    Generates 64 directional rays covering a 45-degree field-of-view, transformed
    by the drone's 3D orientation, and computes intersection distances against
    room boundaries and obstacles.
    """

    def __init__(
        self,
        rows: int = TOF_ROWS,
        cols: int = TOF_COLS,
        fov_h_deg: float = SENSORS.tof_fov_h_deg,
        fov_v_deg: float = SENSORS.tof_fov_v_deg,
        max_range: float = TOF_RAYCASTER_MAX_RANGE_M,
        mount_pitch_deg: float = 0.0,    # forward-facing (0 deg) or tilted down
    ):
        self.rows = rows
        self.cols = cols
        self.num_rays = rows * cols
        self.max_range = max_range
        self.mount_pitch_rad = math.radians(mount_pitch_deg)

        # Precompute unit direction vectors in sensor frame (forward = +X, left = +Y, up = +Z)
        half_fov_h = math.radians(fov_h_deg / 2.0)
        half_fov_v = math.radians(fov_v_deg / 2.0)

        # Azimuth angles (columns) and elevation angles (rows)
        azimuths = np.linspace(-half_fov_h, half_fov_h, cols)
        elevations = np.linspace(half_fov_v, -half_fov_v, rows) # top to bottom

        self.body_ray_dirs = np.zeros((self.num_rays, 3), dtype=np.float32)
        idx = 0
        for elev in elevations:
            for az in azimuths:
                # Sensor convention: +X forward, +Y left, +Z up
                dx = math.cos(elev) * math.cos(az)
                dy = math.cos(elev) * math.sin(az)
                dz = math.sin(elev)
                vec = np.array([dx, dy, dz], dtype=np.float32)
                vec /= np.linalg.norm(vec)

                # Apply mount pitch if any
                if self.mount_pitch_rad != 0.0:
                    cp = math.cos(self.mount_pitch_rad)
                    sp = math.sin(self.mount_pitch_rad)
                    R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
                    vec = R_pitch @ vec

                self.body_ray_dirs[idx] = vec
                idx += 1

    def cast_rays(
        self,
        drone_pos: np.ndarray,
        drone_rot: np.ndarray,
        room: RoomBoundaries,
        cylinders: Optional[List[CylinderObstacle]] = None,
        boxes: Optional[List[BoxObstacle]] = None,
    ) -> np.ndarray:
        """
        Casts 64 rays from drone position into 3D environment and computes hit distances.
        
        Args:
            drone_pos: (3,) position [x, y, z] in world coordinates
            drone_rot: (3, 3) body-to-world rotation matrix R
            room: RoomBoundaries defining arena walls
            cylinders: optional list of vertical cylinder obstacles
            boxes: optional list of 3D box obstacles

        Returns:
            distances: (64,) array of normalized distances in [0.0, 1.0] (1.0 = clear path / max range)
        """
        px, py, pz = float(drone_pos[0]), float(drone_pos[1]), float(drone_pos[2])

        # Rotate ray directions into world coordinates: D_world = D_body @ R.T
        world_dirs = self.body_ray_dirs @ drone_rot.T  # shape (64, 3)
        distances = np.full(self.num_rays, self.max_range, dtype=np.float32)

        # ── 1. Intersect with 6 Planes of the Room ───────────────────────────
        # For each plane, t = (plane_coord - ray_origin) / ray_dir
        # Only valid if t > 0
        eps = 1e-6

        # X walls
        dx = world_dirs[:, 0]
        # x_min
        mask = dx < -eps
        t_xmin = (room.x_min - px) / np.where(mask, dx, -1.0)
        distances = np.where(mask & (t_xmin > 0), np.minimum(distances, t_xmin), distances)
        # x_max
        mask = dx > eps
        t_xmax = (room.x_max - px) / np.where(mask, dx, 1.0)
        distances = np.where(mask & (t_xmax > 0), np.minimum(distances, t_xmax), distances)

        # Y walls
        dy = world_dirs[:, 1]
        # y_min
        mask = dy < -eps
        t_ymin = (room.y_min - py) / np.where(mask, dy, -1.0)
        distances = np.where(mask & (t_ymin > 0), np.minimum(distances, t_ymin), distances)
        # y_max
        mask = dy > eps
        t_ymax = (room.y_max - py) / np.where(mask, dy, 1.0)
        distances = np.where(mask & (t_ymax > 0), np.minimum(distances, t_ymax), distances)

        # Z floor and ceiling
        dz = world_dirs[:, 2]
        # Floor (z_min)
        mask = dz < -eps
        t_floor = (room.z_min - pz) / np.where(mask, dz, -1.0)
        distances = np.where(mask & (t_floor > 0), np.minimum(distances, t_floor), distances)
        # Ceiling (z_max)
        mask = dz > eps
        t_ceil = (room.z_max - pz) / np.where(mask, dz, 1.0)
        distances = np.where(mask & (t_ceil > 0), np.minimum(distances, t_ceil), distances)

        # ── 2. Intersect with Cylindrical Obstacles ──────────────────────────
        if cylinders:
            for cyl in cylinders:
                cx, cy, r = cyl.center_x, cyl.center_y, cyl.radius
                ox = px - cx
                oy = py - cy

                # Ray in 2D horizontal plane: (ox + t*dx)^2 + (oy + t*dy)^2 = r^2
                # A*t^2 + 2B*t + C = 0
                A = dx * dx + dy * dy
                B = ox * dx + oy * dy
                C = ox * ox + oy * oy - r * r

                disc = B * B - A * C
                valid_mask = (disc >= 0) & (A > eps)
                sqrt_disc = np.sqrt(np.maximum(0.0, disc))

                t1 = (-B - sqrt_disc) / np.where(valid_mask, A, 1.0)
                t2 = (-B + sqrt_disc) / np.where(valid_mask, A, 1.0)

                # Check if t1 > 0 and within cylinder height
                hit_z1 = pz + t1 * dz
                hit1_valid = valid_mask & (t1 > 0) & (hit_z1 >= room.z_min) & (hit_z1 <= cyl.height)
                distances = np.where(hit1_valid, np.minimum(distances, t1), distances)

                hit_z2 = pz + t2 * dz
                hit2_valid = valid_mask & (t2 > 0) & (hit_z2 >= room.z_min) & (hit_z2 <= cyl.height)
                distances = np.where(hit2_valid, np.minimum(distances, t2), distances)

        # ── 3. Intersect with Box Obstacles (Slab Method) ────────────────────
        if boxes:
            for box in boxes:
                # Slab intersection for all 64 rays
                inv_d = 1.0 / np.where(np.abs(world_dirs) > eps, world_dirs, np.sign(world_dirs) * eps)
                t_b1 = (box.x_min - px) * inv_d[:, 0]
                t_b2 = (box.x_max - px) * inv_d[:, 0]
                t_near_x = np.minimum(t_b1, t_b2)
                t_far_x = np.maximum(t_b1, t_b2)

                t_b3 = (box.y_min - py) * inv_d[:, 1]
                t_b4 = (box.y_max - py) * inv_d[:, 1]
                t_near_y = np.minimum(t_b3, t_b4)
                t_far_y = np.maximum(t_b3, t_b4)

                t_b5 = (box.z_min - pz) * inv_d[:, 2]
                t_b6 = (box.z_max - pz) * inv_d[:, 2]
                t_near_z = np.minimum(t_b5, t_b6)
                t_far_z = np.maximum(t_b5, t_b6)

                t_enter = np.maximum(np.maximum(t_near_x, t_near_y), t_near_z)
                t_exit = np.minimum(np.minimum(t_far_x, t_far_y), t_far_z)

                box_hit = (t_enter <= t_exit) & (t_exit > 0) & (t_enter > 0)
                distances = np.where(box_hit, np.minimum(distances, t_enter), distances)

        # Clamp and normalize to [0.0, 1.0]
        distances = np.clip(distances, 0.0, self.max_range)
        norm_distances = distances / self.max_range
        return norm_distances.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 6-DOF Micro-Quadcopter Dynamics (130g, 15x15cm, 1.2 - 1.5kg Payload)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DroneDynamicsParams:
    mass: float = DRONE_MASS_ISAAC_KG       # 130g base micro-quadcopter (0.130 kg dry weight)
    payload_mass: float = 0.0               # Attached cargo payload mass in [0.0, 1.5] kg
    arm_length: float = 0.075               # 75 mm motor radius (150 mm diagonal wheelbase, 15x15cm frame)
    g: float = GRAVITY                      # gravity [m/s^2]
    # Base moments of inertia [kg*m^2] for 130g 15x15cm quadcopter
    ixx: float = 3.5e-4
    iyy: float = 3.5e-4
    izz: float = 6.7e-4
    # Motor parameters (calibrated to lift 1.2 - 1.5kg payload at 28,000 RPM max, 28 N peak thrust)
    thrust_coeff: float = 8.9286e-9        # Thrust = k_f * rpm^2 (4 motors deliver up to 28 N peak thrust)
    torque_coeff: float = 1.6e-10          # Torque = k_m * rpm^2 (k_m / k_f ~ 0.018)
    max_rpm: float = 28000.0                # Max brushless motor RPM
    motor_tau: float = 0.025                # Motor time constant [s] (25 ms)
    # Aerodynamic drag coefficients
    drag_linear: float = 0.15               # Linear drag coefficient [N/(m/s)]
    drag_angular: float = 0.002             # Angular damping coefficient [N*m/(rad/s)]
    # Maximum flight angle limits for Angle mode
    max_angle_rad: float = math.radians(35.0)   # +/- 35 deg maximum pitch/roll
    max_yaw_rate_rads: float = math.radians(200.0) # +/- 200 deg/s max yaw rate


class QuadcopterDynamics:
    """
    Rigid-body 6-DOF equations of motion with quaternion kinematics,
    first-order brushless motor lag, and dynamic cargo payload physics.
    """

    def __init__(self, params: DroneDynamicsParams, dt: float = DEFAULT_DT):
        self.params = params
        self.dt = dt

        # State vectors:
        self.pos = np.zeros(3, dtype=np.float32)       # [x, y, z] in world frame (m)
        self.vel = np.zeros(3, dtype=np.float32)       # [vx, vy, vz] in world frame (m/s)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) # [qw, qx, qy, qz]
        self.omega = np.zeros(3, dtype=np.float32)     # [p, q, r] in body frame (rad/s)
        self.motor_rpms = np.zeros(4, dtype=np.float32) # current motor RPMs

        # Initialize inertia tensor and hover throttle based on total mass (base + payload)
        self._update_inertias()

    @property
    def total_mass(self) -> float:
        """Returns total All-Up Weight (AUW) in kg: base mass (0.130 kg) + payload."""
        return self.params.mass + self.params.payload_mass

    def set_payload(self, payload_mass: float) -> None:
        """Dynamically adjusts attached cargo payload mass [0.0, 1.5] kg."""
        self.params.payload_mass = float(np.clip(payload_mass, 0.0, 1.5))
        self._update_inertias()

    def _update_inertias(self) -> None:
        """Updates inertia tensor and hover throttle when payload mass changes."""
        p = self.params
        mp = p.payload_mass

        # Payload modeled as concentrated cargo package (80x80x50 mm)
        delta_ixx = mp * (0.08**2 + 0.05**2) / 12.0
        delta_iyy = mp * (0.08**2 + 0.05**2) / 12.0
        delta_izz = mp * (0.08**2 + 0.08**2) / 12.0

        ixx = p.ixx + delta_ixx
        iyy = p.iyy + delta_iyy
        izz = p.izz + delta_izz

        self.J = np.diag([ixx, iyy, izz]).astype(np.float32)
        self.J_inv = np.diag([1.0 / ixx, 1.0 / iyy, 1.0 / izz]).astype(np.float32)

        # Recalculate hover RPM & hover throttle for total weight
        total_weight = self.total_mass * p.g
        self.hover_rpm = math.sqrt(total_weight / (4.0 * p.thrust_coeff))
        self.hover_throttle = self.hover_rpm / p.max_rpm

    def reset(self, initial_pos: np.ndarray, seed: Optional[int] = None) -> None:
        rng = np.random.default_rng(seed)
        self.pos = initial_pos.astype(np.float32)
        self.vel = rng.uniform(-0.02, 0.02, size=3).astype(np.float32)
        # Small attitude perturbation
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        roll = float(rng.uniform(-0.02, 0.02))
        pitch = float(rng.uniform(-0.02, 0.02))
        yaw = float(rng.uniform(-0.05, 0.05))
        self.quat = self.euler_to_quaternion(roll, pitch, yaw)
        self.omega = np.zeros(3, dtype=np.float32)
        self.motor_rpms = np.full(4, self.hover_rpm, dtype=np.float32)

    @staticmethod
    def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        q = np.array([qw, qx, qy, qz], dtype=np.float32)
        return q / np.linalg.norm(q)

    @staticmethod
    def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw], dtype=np.float32)

    @staticmethod
    def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        R = np.array([
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw),       2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw),       1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw),       2.0 * (qy * qz + qx * qw),       1.0 - 2.0 * (qx * qx + qy * qy)],
        ], dtype=np.float32)
        return R

    def step(self, motor_commands: np.ndarray) -> None:
        """
        Advances 6-DOF dynamics by dt given 4 normalized motor throttle commands [0.0, 1.0].
        """
        dt = self.dt
        p = self.params
        total_m = self.total_mass

        # 1. First-order motor dynamics (lag)
        target_rpms = np.clip(motor_commands, 0.0, 1.0) * p.max_rpm
        alpha_m = dt / (dt + p.motor_tau)
        self.motor_rpms += alpha_m * (target_rpms - self.motor_rpms)

        # 2. Individual motor thrust and drag torque
        thrusts = p.thrust_coeff * (self.motor_rpms ** 2)
        torques = p.torque_coeff * (self.motor_rpms ** 2)

        # 3. Quad-X geometry forces and moments in body frame
        # Motor layout:
        # m1: Rear Right (CCW)  [-d, +d]
        # m2: Front Right (CW)  [+d, +d]
        # m3: Rear Left (CW)    [-d, -d]
        # m4: Front Left (CCW)  [+d, -d]
        d = p.arm_length / math.sqrt(2.0)  # ~0.053m moment arm

        # Total upward thrust along body +Z
        total_thrust_body = np.sum(thrusts)
        F_body = np.array([0.0, 0.0, total_thrust_body], dtype=np.float32)

        # Moments:
        # Roll: right motors push down, left push up -> (m3 + m4 - m1 - m2)
        tau_roll = d * (thrusts[2] + thrusts[3] - thrusts[0] - thrusts[1])
        # Pitch: front motors push down, rear push up -> (m1 + m3 - m2 - m4)
        tau_pitch = d * (thrusts[0] + thrusts[2] - thrusts[1] - thrusts[3])
        # Yaw: reaction torque (CW vs CCW) -> (m2 + m3 - m1 - m4)
        tau_yaw = torques[1] + torques[2] - torques[0] - torques[3]
        tau_body = np.array([tau_roll, tau_pitch, tau_yaw], dtype=np.float32)

        # 4. Newton-Euler rigid body translational dynamics
        R = self.quaternion_to_rotation_matrix(self.quat)
        F_world = R @ F_body
        F_gravity = np.array([0.0, 0.0, -total_m * p.g], dtype=np.float32)
        F_drag = -p.drag_linear * self.vel * np.linalg.norm(self.vel)

        acc = (F_world + F_gravity + F_drag) / total_m

        # Symplectic Euler integration for translation
        self.vel += acc * dt
        self.pos += self.vel * dt

        # Floor collision handling (inelastic with restitution)
        if self.pos[2] < 0.02:
            self.pos[2] = 0.02
            if self.vel[2] < 0.0:
                self.vel[2] = -0.1 * self.vel[2] # slight bounce/damping
            self.vel[0] *= 0.8
            self.vel[1] *= 0.8

        # 5. Rotational dynamics
        # J * omega_dot + omega x (J * omega) = tau_body - drag_angular * omega
        omega = self.omega
        J_omega = self.J @ omega
        gyroscopic_moment = np.cross(omega, J_omega)
        tau_damping = -p.drag_angular * omega
        omega_dot = self.J_inv @ (tau_body - gyroscopic_moment + tau_damping)

        self.omega += omega_dot * dt

        # 6. Quaternion kinematics integration: q_dot = 0.5 * q x [0, omega]
        qw, qx, qy, qz = self.quat[0], self.quat[1], self.quat[2], self.quat[3]
        p_w, q_w, r_w = self.omega[0], self.omega[1], self.omega[2]

        dqw = 0.5 * (-qx * p_w - qy * q_w - qz * r_w)
        dqx = 0.5 * ( qw * p_w + qy * r_w - qz * q_w)
        dqy = 0.5 * ( qw * q_w - qx * r_w + qz * p_w)
        dqz = 0.5 * ( qw * r_w + qx * q_w - qy * p_w)

        self.quat[0] += dqw * dt
        self.quat[1] += dqx * dt
        self.quat[2] += dqy * dt
        self.quat[3] += dqz * dt
        # Normalize quaternion
        q_norm = np.linalg.norm(self.quat)
        if q_norm > 1e-6:
            self.quat /= q_norm
        else:
            self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Complete Simulation Environment (simulation/drone_env.py)
# ─────────────────────────────────────────────────────────────────────────────

class DroneSimulationEnv:
    """
    Simulation Environment for Chong-Fly micro-drone with Betaflight cascaded PID
    and biological sensory emulation (ToF 8x8 + Optical Flow).
    
    Compatible with both Isaac Gym PhysX simulation and standalone vectorized execution.
    """

    def __init__(
        self,
        dt: float = DEFAULT_DT,                 # 250 Hz control loop
        target_altitude: float = TARGET_ALTITUDE_M, # nominal hover altitude (m)
        room: Optional[RoomBoundaries] = None,
        dynamics_params: Optional[DroneDynamicsParams] = None,
        pid_constants: Optional[BetaflightCascadedPID] = None,
        engine: str = "auto",                   # "auto", "isaacgym", or "standalone"
        headless: bool = False,                 # Isaac Gym 3D visualizer viewer
    ):
        self.dt = dt
        self.target_altitude = target_altitude
        self.room = room or RoomBoundaries()
        self.params = dynamics_params or DroneDynamicsParams()
        self.headless = headless

        # Choose simulation engine
        if engine == "auto":
            self.engine = "isaacgym" if HAS_ISAACGYM else "standalone"
        elif engine == "isaacgym":
            if not HAS_ISAACGYM:
                print("⚠️  [Isaac Gym Notice]: Native `isaacgym` package is not installed in the current environment.")
                print("   Running in Chong-Fly standalone vectorized 6-DOF physics engine (CPU/CI mode).\n")
                self.engine = "standalone"
            else:
                self.engine = "isaacgym"
        else:
            self.engine = "standalone"

        # Initialize native Isaac Gym PhysX simulation if active
        self.isaac_sim: Optional[IsaacGymDroneSim] = None
        if self.engine == "isaacgym" and HAS_ISAACGYM:
            self.isaac_sim = IsaacGymDroneSim(num_envs=1, dt=dt, headless=headless)

        # Cascaded Betaflight PID
        self.pid = pid_constants or BetaflightCascadedPID(dt=dt)

        # Sensors: Downward Laser + 8x8 Depth Matrix + PMW3901 Optical Flow/Displacement
        self.downward_laser = DownwardLaserSensor(
            min_range=LASER_MIN_RANGE_M,
            max_range=LASER_MAX_RANGE_M,
            noise_std=LASER_NOISE_STD,
        )
        self.tof = ToFRaycaster(
            rows=TOF_ROWS,
            cols=TOF_COLS,
            fov_h_deg=SENSORS.tof_fov_h_deg,
            fov_v_deg=SENSORS.tof_fov_v_deg,
            max_range=TOF_RAYCASTER_MAX_RANGE_M,
        )
        self.flow_sensor = PMW3901FlowSensor(
            min_altitude=FLOW_MIN_ALTITUDE_M,
            max_altitude=FLOW_MAX_ALTITUDE_M,
            derotate_with_gyro=True,
        )

        # Optical displacement trackers
        self.displacement_step = np.zeros(2, dtype=np.float32)
        self.displacement_total = np.zeros(2, dtype=np.float32)
        self.last_laser_result: Optional[LaserHitResult] = None

        # Physics core
        self.physics = QuadcopterDynamics(self.params, dt=dt)

        # Obstacles list
        self.cylinders: List[CylinderObstacle] = [
            CylinderObstacle(center_x=1.5, center_y=0.0, radius=0.3, height=3.0),
            CylinderObstacle(center_x=-1.5, center_y=1.0, radius=0.25, height=3.0),
        ]
        self.boxes: List[BoxObstacle] = [
            BoxObstacle(x_min=-0.4, x_max=0.4, y_min=2.2, y_max=2.6, z_min=0.0, z_max=2.5),
        ]

        # Tracking metrics
        self.step_count = 0
        self.total_energy_j = 0.0
        self.last_action = np.zeros(4, dtype=np.float32)

        # UI & HUD state tracking
        self.active_model_idx = 0
        self.control_mode = "auto"
        self.is_paused = False
        self.is_crashed = False
        self.last_motors = np.zeros(4, dtype=np.float32)
        self.last_power_w = 0.0

    @property
    def is_isaacgym_active(self) -> bool:
        return self.engine == "isaacgym" and HAS_ISAACGYM

    def set_payload(self, payload_kg: float) -> None:
        """Sets attached payload mass in [0.0, 1.5] kg with instant physics adaptation."""
        self.physics.set_payload(payload_kg)

    def render_hud(
        self,
        active_model_idx: Optional[int] = None,
        models_list: Optional[List[Dict[str, Any]]] = None,
        control_mode: Optional[str] = None,
        target_alt_m: Optional[float] = None,
        is_paused: Optional[bool] = None,
        is_crashed: Optional[bool] = None,
    ):
        """Renders 3D sensor rays and camera-projected HUD dashboard in Isaac Gym."""
        if self.isaac_sim is None or self.isaac_sim.viewer is None:
            return

        rot_mat = self.physics.quaternion_to_rotation_matrix(self.physics.quat)
        m_idx = self.active_model_idx if active_model_idx is None else active_model_idx
        m_list = AVAILABLE_MODELS if models_list is None else models_list
        c_mode = self.control_mode if control_mode is None else control_mode
        t_alt = self.target_altitude if target_alt_m is None else target_alt_m
        paused = self.is_paused if is_paused is None else is_paused
        crashed = self.is_crashed if is_crashed is None else is_crashed

        tof_64 = self.tof.cast_rays(
            drone_pos=self.physics.pos,
            drone_rot=rot_mat,
            room=self.room,
            cylinders=self.cylinders,
            boxes=self.boxes,
        )
        depth_8x8 = tof_64.reshape(self.tof.rows, self.tof.cols)
        laser_alt = float(self.last_laser_result.distance) if self.last_laser_result else float(self.physics.pos[2])
        laser_hit = self.last_laser_result.hit_point if self.last_laser_result else None
        flow_xy = self.flow_sensor.compute_flow(
            v_world=self.physics.vel,
            rot_matrix=rot_mat,
            altitude_above_surface=max(LASER_MIN_RANGE_M, laser_alt),
            omega_body=self.physics.omega,
        )

        self.isaac_sim.render_hud(
            drone_pos=self.physics.pos,
            drone_rot=rot_mat,
            drone_vel=self.physics.vel,
            laser_hit_point=laser_hit,
            depth_8x8=depth_8x8,
            optical_flow=flow_xy,
            active_model_idx=m_idx,
            models_list=m_list,
            control_mode=c_mode,
            payload_kg=float(self.physics.params.payload_mass),
            total_mass_kg=float(self.physics.total_mass),
            hover_throttle=float(self.physics.hover_throttle),
            laser_alt_m=laser_alt,
            target_alt_m=t_alt,
            motors=self.last_motors,
            power_w=self.last_power_w,
            is_paused=paused,
            is_crashed=crashed,
        )

    def reset(
        self,
        initial_pos: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Resets environment to hover state and returns 66-dimensional observation vector.
        """
        self.step_count = 0
        self.total_energy_j = 0.0
        self.displacement_step = np.zeros(2, dtype=np.float32)
        self.displacement_total = np.zeros(2, dtype=np.float32)
        self.last_laser_result = None
        self.zero_flow_time = 0.0
        self.pid.reset()

        if initial_pos is None:
            rng = np.random.default_rng(seed)
            pos = np.array([0.0, 0.0, self.target_altitude + float(rng.uniform(-0.05, 0.05))], dtype=np.float32)
        else:
            pos = np.array(initial_pos, dtype=np.float32)

        self.physics.reset(pos, seed=seed)
        self.last_action = np.array([self.physics.hover_throttle, 0.0, 0.0, 0.0], dtype=np.float32)

        # Synchronize initial state to Isaac Gym PhysX viewer
        if self.isaac_sim is not None:
            self.isaac_sim.sync_state(self.physics.pos, self.physics.quat, self.physics.vel, self.physics.omega)
            self.isaac_sim.render()

        return self._get_observation()

    def step(
        self,
        action: Union[np.ndarray, Tuple[float, float, float, float], IsaacAction],
        action_type: str = "auto", # "auto", "setpoints", or "pwm"
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes one flight step (dt = 0.004 s, 250 Hz).
        
        The neural network does NOT command motor RPMs directly.
        Instead, it sends flight setpoints (Angle Mode: Roll/Pitch, Rate Mode: Yaw Rate, Collective Thrust).
        
        Args:
            action: 4-element array or IsaacAction object:
                - If PWM: [throttle_pwm, roll_pwm, pitch_pwm, yaw_pwm] in [1000, 2000] µs
                - If setpoints: [target_thrust (0..1), target_roll (rad), target_pitch (rad), target_yaw_rate (rad/s)]
            action_type: "auto" detects based on values (> 500 means PWM); or explicitly "pwm" / "setpoints".

        Returns:
            obs: 66-D observation array [FlowX, FlowY, ToF_00 ... ToF_63]
            reward: scalar reward
            done: boolean termination flag
            info: auxiliary metrics (altitude, laser distance, euler angles, motor commands, energy)
        """
        if isinstance(action, IsaacAction):
            if action.action_type == "pwm":
                action = np.array([action.throttle_pwm, action.roll_pwm, action.pitch_pwm, action.yaw_pwm], dtype=np.float32)
                action_type = "pwm"
            else:
                action = np.array([action.thrust, action.roll, action.pitch, action.yaw_rate], dtype=np.float32)
                action_type = "setpoints"
        else:
            action = np.asarray(action, dtype=np.float32)

        # ── 1. Parse Neural Network Action ───────────────────────────────────
        is_pwm = (action_type == "pwm") or (action_type == "auto" and np.any(action > 500.0))
        if is_pwm:
            # Map [1000, 2000] µs RC PWM to flight setpoints
            throttle_pwm = float(action[0])
            roll_pwm = float(action[1])
            pitch_pwm = float(action[2])
            yaw_pwm = float(action[3])

            target_thrust = (throttle_pwm - PWM_MIN) / (PWM_MAX - PWM_MIN)
            target_roll = ((roll_pwm - PWM_MID) / PWM_HALF) * self.params.max_angle_rad
            target_pitch = ((pitch_pwm - PWM_MID) / PWM_HALF) * self.params.max_angle_rad
            target_yaw_rate = ((yaw_pwm - PWM_MID) / PWM_HALF) * self.params.max_yaw_rate_rads
        else:
            # Direct setpoint format: [thrust, roll_cmd, pitch_cmd, yaw_rate_cmd]
            target_thrust = float(action[0])
            target_roll = float(action[1])
            target_pitch = float(action[2])
            target_yaw_rate = float(action[3])

        # Clamp setpoints to physical envelope
        target_thrust = float(np.clip(target_thrust, 0.0, 1.0))
        target_roll = float(np.clip(target_roll, -self.params.max_angle_rad, self.params.max_angle_rad))
        target_pitch = float(np.clip(target_pitch, -self.params.max_angle_rad, self.params.max_angle_rad))
        target_yaw_rate = float(np.clip(target_yaw_rate, -self.params.max_yaw_rate_rads, self.params.max_yaw_rate_rads))

        # ── 2. Run Betaflight Cascaded PID Controller ────────────────────────
        euler = self.physics.quaternion_to_euler(self.physics.quat)
        omega = self.physics.omega

        motors, pid_debug = self.pid.compute_motor_commands(
            target_thrust=target_thrust,
            target_pitch=target_pitch,
            target_roll=target_roll,
            target_yaw_rate=target_yaw_rate,
            measured_euler=euler,
            measured_omega=omega,
        )

        # ── 3. Step 6-DOF Physics Dynamics ──────────────────────────────────
        self.physics.step(motors)
        self.step_count += 1
        self.last_motors = motors.copy()

        # Calculate electrical / mechanical energy expenditure (P = sum(T * omega))
        power_w = float(np.sum(motors ** 2) * 25.0)  # ~25W hover power for micro-drone
        self.last_power_w = power_w
        self.total_energy_j += power_w * self.dt

        # ── 4. Collect Sensor Observations ──────────────────────────────────
        obs = self._get_observation()

        # Synchronize and step Isaac Gym graphics / PhysX
        user_closed = False
        if self.isaac_sim is not None:
            self.isaac_sim.sync_state(self.physics.pos, self.physics.quat, self.physics.vel, self.physics.omega)
            should_render = (self.step_count % self.isaac_sim.render_every == 0)
            if should_render and self.isaac_sim.viewer is not None:
                self.isaac_sim.update_camera(self.physics.pos, euler)
                self.render_hud()
            sim_ok = self.isaac_sim.step(
                render=should_render,
                pos=None,
                euler=None,
            )
            if not sim_ok:
                user_closed = True

        # ── 5. Termination & Reward Evaluation ──────────────────────────────
        pos = self.physics.pos
        crashed = (
            pos[2] <= 0.03 or pos[2] >= (self.room.z_max - 0.05) or
            abs(pos[0]) >= (self.room.x_max - 0.05) or
            abs(pos[1]) >= (self.room.y_max - 0.05) or
            abs(euler[0]) > math.radians(65.0) or
            abs(euler[1]) > math.radians(65.0)
        )
        done = bool(crashed or user_closed)

        # Reward: penalize altitude error, attitude tilt, motor jitter, energy
        alt_error = abs(float(pos[2]) - self.target_altitude)
        tilt_error = float(euler[0]**2 + euler[1]**2)
        pwm_jitter = float(np.mean(np.abs(action - self.last_action))) if is_pwm else 0.0
        self.last_action = action.copy()

        reward = 1.0 - (1.5 * alt_error + 0.8 * tilt_error + 0.001 * power_w)
        if crashed:
            reward -= 10.0

        info = {
            "position": pos.copy(),
            "velocity": self.physics.vel.copy(),
            "euler_rad": euler.copy(),
            "omega_rads": omega.copy(),
            "motor_commands": motors.copy(),
            "motor_rpms": self.physics.motor_rpms.copy(),
            "energy_j": self.total_energy_j,
            "power_w": power_w,
            "altitude_error": alt_error,
            "tilt_error": tilt_error,
            "pid_debug": pid_debug,
            "laser_alt": float(self.last_laser_result.distance) if self.last_laser_result else float(pos[2]),
            "laser_hit_surf": self.last_laser_result.surface_type if self.last_laser_result else "floor",
            "displacement_m": self.displacement_total.copy(),
            "payload_mass_kg": float(self.physics.params.payload_mass),
            "total_mass_kg": float(self.physics.total_mass),
            "hover_throttle": float(self.physics.hover_throttle),
            "crashed": crashed,
            "user_closed": user_closed,
        }
        return obs, reward, done, info

    def close(self):
        """Releases Isaac Gym simulation and viewer resources."""
        if self.isaac_sim is not None:
            self.isaac_sim.close()
            self.isaac_sim = None

    def _get_observation(self) -> np.ndarray:
        """
        Synthesizes multi-sensor observation readings:
        1. Downward Laser Rangefinder: Measures true AGL along body -Z.
        2. Optical Flow (PMW3901): Surface velocity scaled dynamically by Downward Laser distance.
        3. Optical Displacement: Body-frame translation [dx, dy] per step and cumulative.
        4. ToF 8x8 Depth Matrix (VL53L5CX): 64 rays covering 45° FOV.
        
        Returns:
            66-D array [FlowX, FlowY, ToF_00 ... ToF_63] for connectome compatibility.
        """
        rot_mat = self.physics.quaternion_to_rotation_matrix(self.physics.quat)
        pos = self.physics.pos
        vel = self.physics.vel
        omega = self.physics.omega

        # 1. Downward Laser Sensor measurement
        self.last_laser_result = self.downward_laser.measure(
            drone_pos=pos,
            rot_matrix=rot_mat,
            floor_z=self.room.z_min,
            boxes=self.boxes,
            cylinders=self.cylinders,
        )
        laser_alt = self.last_laser_result.distance

        # 2. Optical Flow (PMW3901) - scaled by actual Downward Laser reading!
        altitude_surface = max(SENSORS.laser_min_range_m, float(laser_alt))
        flow_xy = self.flow_sensor.compute_flow(
            v_world=vel,
            rot_matrix=rot_mat,
            altitude_above_surface=altitude_surface,
            omega_body=omega,
        )

        # Generate Levy Noise (Cauchy distribution) if flow is 0 for > 2 seconds
        if np.allclose(flow_xy, 0.0, atol=1e-5):
            self.zero_flow_time += getattr(self, 'dt', DEFAULT_DT)
        else:
            self.zero_flow_time = 0.0

        if self.zero_flow_time > 2.0:
            levy_noise = np.random.standard_cauchy(size=2) * 0.1
            flow_xy += levy_noise.astype(np.float32)


        # 3. Optical Displacement Sensor (body translational displacement)
        v_body = rot_mat.T @ vel
        self.displacement_step = (v_body[:2] * self.dt).astype(np.float32)
        self.displacement_total += self.displacement_step

        # 4. ToF 8x8 Raycasting (VL53L5CX, 45° FOV)
        tof_64 = self.tof.cast_rays(
            drone_pos=pos,
            drone_rot=rot_mat,
            room=self.room,
            cylinders=self.cylinders,
            boxes=self.boxes,
        )

        # 5. Concatenate to (66,)
        obs = np.concatenate([flow_xy, tof_64], axis=0).astype(np.float32)
        return obs

    def get_isaac_obs(self) -> IsaacObservation:
        """
        Returns full structured IsaacObservation object containing all sensors,
        IMU, displacement, attitude, and payload state.
        """
        if self.last_laser_result is None:
            self._get_observation()

        rot_mat = self.physics.quaternion_to_rotation_matrix(self.physics.quat)
        euler = self.physics.quaternion_to_euler(self.physics.quat)
        tof_64 = self.tof.cast_rays(
            drone_pos=self.physics.pos,
            drone_rot=rot_mat,
            room=self.room,
            cylinders=self.cylinders,
            boxes=self.boxes,
        )
        depth_8x8 = tof_64.reshape(self.tof.rows, self.tof.cols)
        flow_xy = self.flow_sensor.compute_flow(
            v_world=self.physics.vel,
            rot_matrix=rot_mat,
            altitude_above_surface=max(LASER_MIN_RANGE_M, self.last_laser_result.distance),
            omega_body=self.physics.omega,
        )

        # Body-frame acceleration estimate (including gravity reaction)
        R = rot_mat
        a_body = R.T @ np.array([0.0, 0.0, self.params.g], dtype=np.float32)

        return IsaacObservation(
            laser_distance=float(self.last_laser_result.distance),
            laser_distance_norm=float(self.last_laser_result.distance_norm),
            laser_hit_point=self.last_laser_result.hit_point.copy(),
            laser_valid=bool(self.last_laser_result.is_valid),
            depth_8x8=depth_8x8.copy(),
            depth_flat=tof_64.copy(),
            optical_flow=flow_xy.copy(),
            displacement_step=self.displacement_step.copy(),
            displacement_total=self.displacement_total.copy(),
            imu_gyro=self.physics.omega.copy(),
            imu_accel=a_body,
            attitude_euler=euler.copy(),
            position=self.physics.pos.copy(),
            velocity=self.physics.vel.copy(),
            payload_mass=float(self.physics.params.payload_mass),
            total_mass=float(self.physics.total_mass),
            hover_throttle=float(self.physics.hover_throttle),
            sim_time=float(self.step_count * self.dt),
            step_count=int(self.step_count),
        )

    def get_chong_fly_obs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convenience helper returning structured (flow_xy, tof_8x8)
        directly consumable by `ChongFlyMSPPolicy.step_np(flow_xy, tof_8x8)`.
        """
        obs = self._get_observation()
        flow_xy = obs[0:FLOW_DIM]
        tof_8x8 = obs[FLOW_DIM:(FLOW_DIM + TOF_DIM)]
        return flow_xy, tof_8x8


# ─────────────────────────────────────────────────────────────────────────────
# Interactive CLI Flight Runner & Real-Time Telemetry HUD
# ─────────────────────────────────────────────────────────────────────────────

def _make_bar(value: float, width: int = 12) -> str:
    """Renders a simple ASCII gauge bar for values in [0.0, 1.0]."""
    filled = int(round(np.clip(value, 0.0, 1.0) * width))
    return "█" * filled + "░" * (width - filled)


def run_flight_simulation(
    meta_path: str = "data/reduced_models/meta_spectral_k64.json",
    duration_s: float = 0.0,            # 0.0 or <= 0 means continuous / infinite flight
    target_altitude: float = 1.0,
    payload_kg: float = 0.0,            # Payload mass in kg [0.0, 1.5]
    realtime: bool = True,
    hud_interval_s: float = 0.1,
    solver_type: str = "CfC",
    pruning_sparsity: Optional[float] = None,
    ablate_cx: Optional[bool] = None,
    engine: str = "auto",
    headless: bool = False,
    control_mode: str = "auto",         # "auto" (neural policy) or "manual" (pilot WASD)
) -> Dict[str, Any]:
    """
    Executes an interactive closed-loop flight simulation with live controls,
    on-the-fly model switching, continuous flight, and real-time HUD telemetry.
    """
    import time
    from simulation.policy import ChongFlyMSPPolicy

    # Initialize Dynamics & Environment with requested payload
    dyn_params = DroneDynamicsParams(payload_mass=payload_kg)
    env = DroneSimulationEnv(
        dt=DEFAULT_DT,
        target_altitude=target_altitude,
        dynamics_params=dyn_params,
        engine=engine,
        headless=headless,
    )
    obs = env.reset(seed=42)

    engine_str = "NVIDIA Isaac Gym (PhysX)" if env.is_isaacgym_active else "Standalone 6-DOF Vectorized (CPU)"

    # Default to real-time playback if visual 3D viewer is active
    if not headless and not realtime:
        realtime = True

    is_infinite = (duration_s is None or duration_s <= 0.0)
    dur_str = "Continuous (Infinite / Press Esc to exit)" if is_infinite else f"{duration_s:.1f} s ({int(duration_s / DEFAULT_DT)} steps @ 250 Hz)"

    # Locate initial model in registry
    active_model_idx = 0
    if str(meta_path).isdigit():
        val = int(meta_path) - 1
        if 0 <= val < len(AVAILABLE_MODELS):
            active_model_idx = val
    else:
        for i, m in enumerate(AVAILABLE_MODELS):
            if m["meta"] == meta_path or m["name"].lower() == str(meta_path).lower() or str(meta_path).lower() in m["name"].lower():
                active_model_idx = i
                break
    active_model_name = AVAILABLE_MODELS[active_model_idx]["name"]

    print("\n" + "=" * 80)
    print("   🚀 CHONG-FLY 6-DOF BETAFLIGHT ISAAC FLIGHT SIMULATION")
    print("=" * 80)
    print(f"  • Physics Engine:       {engine_str}")
    print(f"  • Base Frame & Mass:    130 g (0.130 kg) | 15 cm x 15 cm Frame")
    print(f"  • Payload & AUW:        {payload_kg:.2f} kg (Total AUW: {env.physics.total_mass*1000:.0f} g / Max 1.63 kg)")
    print(f"  • Hover Throttle:       {env.physics.hover_throttle * 100:.1f}% ({env.physics.hover_rpm:.0f} RPM)")
    print(f"  • Sensors Active:       Downward Laser, 8x8 Depth Matrix (45° FOV), PMW3901 Flow")
    print(f"  • Active Model / Brain: [{active_model_idx + 1}] {active_model_name}")
    print(f"  • Flight Mode:          {'🎮 MANUAL PILOT (WASD)' if control_mode == 'manual' else '🤖 AUTONOMOUS (Neural Connectome)'}")
    print(f"  • Flight Duration:      {dur_str}")
    print(f"  • Target Altitude:      {target_altitude:.2f} m")
    print(f"  • Speed Mode:           {'Real-Time Playback (1:1)' if realtime else 'Maximum Speed (Fast)'}")
    print("=" * 80)
    print("  🎮 LIVE 3D VIEWER CONTROLS (IN-VIEWER GUI ACTIVE):")
    print("    • [TAB] Toggle In-Viewer Interactive Model Selector Menu Overlay")
    print("    • [1 - 9] Direct Model Selection   • [N / B] Next / Previous Model")
    print("    • [ [ / ] ] or [- / =] Adjust Cargo Payload Mass (-0.1 kg / +0.1 kg)")
    print("    • [U / J] Adjust Target Hover Altitude (+0.25 m / -0.25 m)")
    print("    • [M] Toggle Mode (Autonomous ↔ Manual Pilot WASD)")
    print("    • [W / S] Pitch Forward / Backward • [A / D] Roll Left / Right")
    print("    • [Space / C] Throttle Up / Down   • [Q / E] Yaw Turn Left / Right")
    print("    • [V] Cycle Camera View (Chase Cam ↔ Arena Cam ↔ Free Mouse Look)")
    print("    • [R] Respawn / Reset Drone        • [P] Pause / Resume Simulation")
    print("    • [Esc] Exit Flight")
    print("=" * 80 + "\n")

    # 1. Initialize Policy
    autonomous_model = AutonomousLaserNavigatorModel(target_altitude=target_altitude)
    policy = None
    if active_model_idx != 8:
        policy = ChongFlyMSPPolicy.from_meta(
            meta_path=meta_path,
            mode="fixed",
            dt=DEFAULT_DT,
            solver_type=solver_type,
            pruning_sparsity=pruning_sparsity,
            ablate_cx=ablate_cx,
        )
        policy.reset_state()

    # Model Switch Helper
    def switch_to_model(m_idx: int):
        nonlocal active_model_idx, active_model_name, policy, status_msg
        if not (0 <= m_idx < len(AVAILABLE_MODELS)):
            return
        try:
            active_model_idx = m_idx
            active_model_name = AVAILABLE_MODELS[m_idx]["name"]
            new_meta = AVAILABLE_MODELS[m_idx]["meta"]
            if m_idx == 8:
                autonomous_model.reset()
                status_msg = f"🧠 Hot-Swapped Brain -> [9] {active_model_name}"
            else:
                policy = ChongFlyMSPPolicy.from_meta(
                    meta_path=new_meta,
                    mode="fixed",
                    dt=DEFAULT_DT,
                    solver_type=solver_type,
                )
                policy.reset_state()
                status_msg = f"🧠 Hot-Swapped Brain -> [{m_idx + 1}] {active_model_name}"
            env.active_model_idx = active_model_idx
            if env.isaac_sim and hasattr(env.isaac_sim, "hud"):
                env.isaac_sim.hud.set_notification(status_msg)
        except Exception as ex:
            status_msg = f"⚠️ Model switch error: {ex}"

    total_steps = sys.maxsize if is_infinite else int(duration_s / env.dt)
    steps_survived = 0
    alt_errors = []
    tilt_errors = []
    pwms = []

    last_hud_time = -1.0
    start_wall_time = time.time()
    info = {}
    is_paused = False
    is_crashed = False
    status_msg = "Flight started."

    # 2. Interactive Flight Loop (250 Hz)
    try:
        step = 0
        while step < total_steps:
            t_sim = step * env.dt

            # ── Keyboard & Hotkey Event Polling ──────────────────────────────
            key_states = {}
            triggers = []
            if env.isaac_sim is not None:
                key_states, triggers = env.isaac_sim.poll_input()

            # Process Triggers (single keypress events)
            for trig in triggers:
                if trig == "quit":
                    print("\n🛑 Flight stopped by user [Esc].")
                    return {}
                elif trig == "pause":
                    is_paused = not is_paused
                    env.is_paused = is_paused
                    status_msg = "⏸️  PAUSED (Press [P] to Resume)" if is_paused else "▶️  RESUMED"
                    if env.isaac_sim and hasattr(env.isaac_sim, "hud"):
                        env.isaac_sim.hud.set_notification(status_msg)
                elif trig == "toggle_menu":
                    if env.isaac_sim and hasattr(env.isaac_sim, "hud"):
                        env.isaac_sim.hud.toggle_menu()
                        status_msg = "HUD Menu: " + ("OPEN" if env.isaac_sim.hud.show_menu else "MINIMIZED")
                elif trig == "next_model":
                    switch_to_model((active_model_idx + 1) % len(AVAILABLE_MODELS))
                elif trig == "prev_model":
                    switch_to_model((active_model_idx - 1) % len(AVAILABLE_MODELS))
                elif trig == "alt_up":
                    target_altitude = min(5.0, target_altitude + 0.25)
                    env.target_altitude = target_altitude
                    autonomous_model.target_altitude = target_altitude
                    status_msg = f"Target Alt -> {target_altitude:.2f} m [U/J]"
                    if env.isaac_sim and hasattr(env.isaac_sim, "hud"):
                        env.isaac_sim.hud.set_notification(status_msg)
                elif trig == "alt_down":
                    target_altitude = max(0.25, target_altitude - 0.25)
                    env.target_altitude = target_altitude
                    autonomous_model.target_altitude = target_altitude
                    status_msg = f"Target Alt -> {target_altitude:.2f} m [U/J]"
                    if env.isaac_sim and hasattr(env.isaac_sim, "hud"):
                        env.isaac_sim.hud.set_notification(status_msg)
                elif trig == "payload_up":
                    new_p = min(1.5, env.physics.params.payload_mass + 0.1)
                    env.set_payload(new_p)
                    status_msg = f"📦 Payload Increased -> {new_p:.2f} kg (AUW: {env.physics.total_mass*1000:.0f}g, Hover Th: {env.physics.hover_throttle*100:.1f}%)"
                    if env.isaac_sim and hasattr(env.isaac_sim, "hud"):
                        env.isaac_sim.hud.set_notification(status_msg)
                elif trig == "payload_down":
                    new_p = max(0.0, env.physics.params.payload_mass - 0.1)
                    env.set_payload(new_p)
                    status_msg = f"📦 Payload Decreased -> {new_p:.2f} kg (AUW: {env.physics.total_mass*1000:.0f}g, Hover Th: {env.physics.hover_throttle*100:.1f}%)"
                    if env.isaac_sim and hasattr(env.isaac_sim, "hud"):
                        env.isaac_sim.hud.set_notification(status_msg)
                elif trig == "toggle_mode":
                    control_mode = "manual" if control_mode == "auto" else "auto"
                    env.control_mode = control_mode
                    status_msg = f"Switched Mode -> {'🎮 MANUAL PILOT' if control_mode == 'manual' else '🤖 AUTONOMOUS'}"
                    if env.isaac_sim and hasattr(env.isaac_sim, "hud"):
                        env.isaac_sim.hud.set_notification(status_msg)
                elif trig == "cycle_camera":
                    if env.isaac_sim:
                        env.isaac_sim.camera_mode_idx = (env.isaac_sim.camera_mode_idx + 1) % len(env.isaac_sim.camera_modes)
                        cam_name = env.isaac_sim.camera_modes[env.isaac_sim.camera_mode_idx].upper()
                        status_msg = f"Camera Mode -> [{cam_name}]"
                        if hasattr(env.isaac_sim, "hud"):
                            env.isaac_sim.hud.set_notification(status_msg)
                elif trig == "reset":
                    # Instant Respawn / Reset
                    env.physics.reset(np.array([0.0, 0.0, target_altitude]))
                    env.pid.reset()
                    if policy:
                        policy.reset_state()
                    autonomous_model.reset()
                    if env.isaac_sim:
                        env.isaac_sim.sync_state(env.physics.pos, env.physics.quat, env.physics.vel, env.physics.omega)
                    is_crashed = False
                    env.is_crashed = False
                    status_msg = "🔄 Drone Respawned at hover altitude!"
                    if env.isaac_sim and hasattr(env.isaac_sim, "hud"):
                        env.isaac_sim.hud.set_notification(status_msg)
                elif trig.startswith("model_"):
                    try:
                        m_idx = int(trig.split("_")[1]) - 1
                        switch_to_model(m_idx)
                    except Exception as ex:
                        status_msg = f"⚠️ Model switch error: {ex}"

            # Keep environment UI states synchronized
            env.active_model_idx = active_model_idx
            env.control_mode = control_mode
            env.is_paused = is_paused
            env.is_crashed = is_crashed

            # If Paused: render HUD overlay and sleep
            if is_paused:
                if env.isaac_sim:
                    env.render_hud()
                    env.isaac_sim.render()
                time.sleep(0.02)
                continue

            # If Crashed: wait for user to press [R] to respawn (or exit in headless mode)
            if is_crashed:
                if headless:
                    break
                if env.isaac_sim:
                    env.render_hud()
                    env.isaac_sim.render()
                time.sleep(0.02)
                continue

            # ── Sensory Readout ──────────────────────────────────────────────
            flow_xy, tof_8x8 = env.get_chong_fly_obs()

            # ── Action Computation ───────────────────────────────────────────
            if control_mode == "auto":
                if active_model_idx == 8:
                    # Autonomous Multi-Modal Laser + 8x8 Depth + Flow Navigator
                    isaac_obs = env.get_isaac_obs()
                    action_cmd = autonomous_model.step(isaac_obs)
                else:
                    # Biological Neural Connectome Policy Step
                    pwm = policy.step_np(flow_xy, tof_8x8)
                    action_cmd = pwm
            else:
                # Manual Flight Control via Keyboard
                hover_th_pwm = PWM_MIN + env.physics.hover_throttle * (PWM_MAX - PWM_MIN)
                th = hover_th_pwm
                roll = PWM_LEVEL_ROLL
                pitch = PWM_MID
                yaw = PWM_NEUTRAL_YAW

                if key_states.get("pitch_fwd"):
                    pitch += 140.0
                if key_states.get("pitch_back"):
                    pitch -= 140.0
                if key_states.get("roll_left"):
                    roll -= 140.0
                if key_states.get("roll_right"):
                    roll += 140.0
                if key_states.get("throttle_up"):
                    th += 200.0
                if key_states.get("throttle_down"):
                    th -= 200.0
                if key_states.get("yaw_left"):
                    yaw -= 160.0
                if key_states.get("yaw_right"):
                    yaw += 160.0

                action_cmd = np.array([th, roll, pitch, yaw], dtype=np.float32)

            # ── Environment & Betaflight Step ────────────────────────────────
            obs, reward, done, info = env.step(action_cmd)
            steps_survived += 1

            pos = info["position"]
            vel = info["velocity"]
            euler = info["euler_rad"]
            omega = info["omega_rads"]
            motors = info["motor_commands"]
            power_w = info["power_w"]
            laser_alt = info.get("laser_alt", float(pos[2]))
            laser_surf = info.get("laser_hit_surf", "floor")
            disp = info.get("displacement_m", np.zeros(2))
            payload = info.get("payload_mass_kg", 0.0)
            total_m = info.get("total_mass_kg", 0.130)
            hover_th = info.get("hover_throttle", 0.213)

            alt_errors.append(abs(float(pos[2]) - target_altitude))
            tilt_errors.append(math.sqrt(float(euler[0]**2 + euler[1]**2)))

            # Handle Collision / Crash (allow respawn with 'R')
            if info.get("crashed", False):
                is_crashed = True
                status_msg = "💥 COLLISION! Press [R] in 3D viewer to Respawn."

            # Handle Window Close
            if info.get("user_closed", False):
                print(f"\n🚪 Isaac Gym 3D Viewer closed by user at T = {t_sim:.3f} s.")
                break

            # ── Real-Time Telemetry HUD Display ──────────────────────────────
            if (t_sim - last_hud_time) >= hud_interval_s or step == 0 or is_crashed:
                last_hud_time = t_sim
                roll_deg = math.degrees(euler[0])
                pitch_deg = math.degrees(euler[1])
                yaw_deg = math.degrees(euler[2])

                tof_mat = tof_8x8.reshape(8, 8)
                center_tof = float(np.mean(tof_mat[3:5, 3:5]))

                cam_name = env.isaac_sim.camera_modes[env.isaac_sim.camera_mode_idx].upper() if env.isaac_sim else "N/A"
                mode_label = "MANUAL PILOT" if control_mode == "manual" else f"AUTO ({active_model_name})"
                total_str = f"/{total_steps}" if not is_infinite else " (Inf)"

                print(f"\r┌─[ T = {t_sim:5.3f}s | Step {step:4d}{total_str} | Mode: {mode_label} | Cam: {cam_name} ]" + "─" * 10 + "┐")
                print(f"│ POS:  X={pos[0]:+6.2f}m  Y={pos[1]:+6.2f}m  Z={pos[2]:5.2f}m (Target: {target_altitude:4.2f}m)  │ VEL: Vx={vel[0]:+5.2f} Vy={vel[1]:+5.2f} Vz={vel[2]:+5.2f} m/s │")
                print(f"│ ATT:  Roll={roll_deg:+5.1f}° Pitch={pitch_deg:+5.1f}° Yaw={yaw_deg:+5.1f}°     │ GYRO: p={omega[0]:+5.2f}  q={omega[1]:+5.2f}  r={omega[2]:+5.2f} rad/s │")
                print(f"│ MOTORS (Betaflight Quad-X):                                              │")
                print(f"│   M4 (FL): [{_make_bar(motors[3])}] {motors[3]*100:4.1f}%     M2 (FR): [{_make_bar(motors[1])}] {motors[1]*100:4.1f}%       │")
                print(f"│   M3 (RL): [{_make_bar(motors[2])}] {motors[2]*100:4.1f}%     M1 (RR): [{_make_bar(motors[0])}] {motors[0]*100:4.1f}%       │")
                print(f"│ SENSORS: Down Laser: {laser_alt:4.2f}m ({laser_surf:<8}) | ToF 8x8 Center: {center_tof*3.5:4.2f}m (45° FOV) │")
                print(f"│ OPTICAL: Flow=[X:{flow_xy[0]:+5.2f}, Y:{flow_xy[1]:+5.2f}] | Disp=[X:{disp[0]:+5.2f}m, Y:{disp[1]:+5.2f}m]         │")
                print(f"│ PAYLOAD: {payload:4.2f} kg (AUW: {total_m:4.2f}kg / Max 1.63kg) | Hover Throttle: {hover_th*100:4.1f}%     │")
                print(f"│ STATUS:  {status_msg:<63} │")
                print(f"└" + "─" * 74 + "┘")

            if realtime:
                # Sleep remainder of dt to maintain real-time pacing
                elapsed = time.time() - start_wall_time
                sleep_time = (step + 1) * env.dt - elapsed
                if sleep_time > 0.0005:
                    time.sleep(sleep_time)

            step += 1

    except KeyboardInterrupt:
        print("\n🛑 Simulation paused / interrupted by user.")

    wall_duration = time.time() - start_wall_time
    fps = steps_survived / max(1e-6, wall_duration)

    # 3. Final Performance Summary
    mean_alt_err = float(np.mean(alt_errors)) if alt_errors else 0.0
    mean_tilt_err = math.degrees(float(np.mean(tilt_errors))) if tilt_errors else 0.0

    if info.get("user_closed", False):
        outcome_str = "🚪 VIEWER CLOSED"
    elif is_crashed:
        outcome_str = "❌ CRASHED"
    else:
        outcome_str = "✅ FLIGHT COMPLETED (STABLE)"

    dur_disp = f"{steps_survived * env.dt:.3f} s / {duration_s:.3f} s" if not is_infinite else f"{steps_survived * env.dt:.3f} s (Continuous)"

    print("\n" + "=" * 80)
    print("   📊 FLIGHT TELEMETRY SUMMARY & EVALUATION REPORT")
    print("=" * 80)
    print(f"  • Flight Outcome:         {outcome_str}")
    print(f"  • Survival Duration:      {dur_disp}")
    print(f"  • Simulation Speed:       {fps:.1f} steps/s ({fps * env.dt:.1f}x real-time)")
    print(f"  • Mean Altitude Error:    {mean_alt_err:.4f} m")
    print(f"  • Mean Attitude Tilt:     {mean_tilt_err:.2f}°")
    print(f"  • Total Energy Consumed:  {env.total_energy_j:.2f} Joules")
    print(f"  • Average Power Demand:   {env.total_energy_j / max(1e-6, steps_survived * env.dt):.2f} Watts")
    print("=" * 80 + "\n")

    # 4. Keep 3D viewer open if active so user can inspect scene
    if env.isaac_sim is not None and env.isaac_sim.viewer is not None and not info.get("user_closed", False):
        print("💡 [Isaac Gym 3D Viewer]: Flight simulation complete! The window will stay open for inspection.")
        print("   (Close the viewer window with 'X' or press Ctrl+C in terminal to exit)\n")
        try:
            while not env.isaac_sim.gym.query_viewer_has_closed(env.isaac_sim.viewer):
                env.isaac_sim.render()
                time.sleep(0.02)
        except KeyboardInterrupt:
            pass

    env.close()

    return {
        "survived": not is_crashed and not info.get("user_closed", False),
        "steps_survived": steps_survived,
        "flight_time_s": steps_survived * env.dt,
        "energy_j": env.total_energy_j,
        "mean_alt_error": mean_alt_err,
        "mean_tilt_error_deg": mean_tilt_err,
    }


def interactive_menu() -> Tuple[str, float, str, float]:
    """
    Displays an interactive CLI launcher menu to select model, flight mode, duration, and payload.
    """
    print("\n" + "=" * 80)
    print("   🚁 CHONG-FLY INTERACTIVE FLIGHT SIMULATION LAUNCHER")
    print("=" * 80)
    print("  Quadcopter: 130g Base Mass | 15x15 cm Frame | 1.2-1.5 kg Payload Capacity")
    print("  Sensors: Downward Laser + 8x8 ToF Matrix (45° FOV) + Optical Flow/Disp")
    print("-" * 80)
    print("  Select Flight Brain / Policy:")
    for m in AVAILABLE_MODELS:
        print(f"    [{m['id']}] {m['name']:<25} • {m['tag']}")
    print("    [M] Manual Piloting Mode (Fly with WASD / Space / C)")
    print("-" * 80)

    try:
        choice = input("  Select Model [1-9] or [M] for Manual (default: 1): ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = "1"

    mode = "auto"
    selected_meta = AVAILABLE_MODELS[0]["meta"]
    if choice.upper() == "M":
        mode = "manual"
    elif choice.isdigit() and 1 <= int(choice) <= len(AVAILABLE_MODELS):
        selected_meta = AVAILABLE_MODELS[int(choice) - 1]["meta"]

    print("\n  Select Cargo Payload Mass [0.0 - 1.5 kg]:")
    print("    [0] 0.00 kg (Unladen, 130g base AUW, extreme agility)")
    print("    [1] 0.50 kg (Medium cargo, 630g AUW)")
    print("    [2] 1.20 kg (Heavy payload, 1.33 kg AUW)")
    print("    [3] 1.50 kg (Maximum rated payload capacity, 1.63 kg AUW)")
    try:
        p_choice = input("  Select Payload [0-3] or enter kg (default: 0): ").strip()
    except (EOFError, KeyboardInterrupt):
        p_choice = "0"

    p_map = {"0": 0.0, "1": 0.5, "2": 1.2, "3": 1.5}
    try:
        payload_kg = p_map.get(p_choice, float(p_choice))
    except ValueError:
        payload_kg = 0.0
    payload_kg = float(np.clip(payload_kg, 0.0, 1.5))

    print("\n  Select Flight Duration:")
    print("    [0] Continuous Flight (Infinite / Fly as long as you want, [R] to respawn)")
    print("    [1] 5.0 seconds")
    print("    [2] 15.0 seconds")
    print("    [3] 30.0 seconds")
    try:
        dur_choice = input("  Select Duration [0-3] (default: 0 - Continuous): ").strip()
    except (EOFError, KeyboardInterrupt):
        dur_choice = "0"

    dur_map = {"0": 0.0, "1": 5.0, "2": 15.0, "3": 30.0}
    duration_s = dur_map.get(dur_choice, 0.0)

    print("=" * 80 + "\n")
    return selected_meta, duration_s, mode, payload_kg


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Chong-Fly 6-DOF Drone Betaflight Isaac Simulation")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to reduced model meta JSON or model name (default: meta_spectral_k64.json)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Flight duration in seconds (default: 0 = continuous flight in GUI mode)")
    parser.add_argument("--alt", type=float, default=1.0,
                        help="Target hover altitude in meters (default: 1.0m)")
    parser.add_argument("--payload", type=float, default=0.0,
                        help="Attached cargo payload in kg [0.0 - 1.5 kg] (default: 0.0 kg)")
    parser.add_argument("--realtime", action="store_true",
                        help="Run at 1x real-time playback speed (default: auto for viewer)")
    parser.add_argument("--fast", action="store_true",
                        help="Run at maximum speed without real-time delay")
    parser.add_argument("--solver", type=str, choices=["CfC", "Euler_dt_0.02"], default="CfC",
                        help="Neural ODE solver: CfC (closed-form, 250Hz) or Euler_dt_0.02 (default: CfC)")
    parser.add_argument("--sparsity", type=float, default=None,
                        help="Dynamic weight magnitude pruning sparsity [0.50, 0.95]")
    parser.add_argument("--ablate-cx", action="store_true", default=None,
                        help="Ablate Central Complex (reflex-only mode)")
    parser.add_argument("--engine", type=str, choices=["auto", "isaacgym", "standalone"], default="auto",
                        help="Simulation backend: 'isaacgym' (NVIDIA PhysX GPU/CPU) or 'standalone' (vectorized 6-DOF CPU)")
    parser.add_argument("--headless", action="store_true", default=False,
                        help="Run Isaac Gym in headless mode (no 3D viewer window)")
    parser.add_argument("--menu", action="store_true", default=False,
                        help="Launch interactive terminal menu to select model, payload, and flight mode")
    parser.add_argument("--manual", action="store_true", default=False,
                        help="Start directly in manual pilot mode (WASD + Space/C)")
    parser.add_argument("--infinite", action="store_true", default=False,
                        help="Run continuous flight without time limit")

    args = parser.parse_args()

    mode = "manual" if args.manual else "auto"
    selected_meta = args.model or "data/reduced_models/meta_spectral_k64.json"
    payload_kg = float(np.clip(args.payload, 0.0, 1.5))

    # Default duration: if headless and not specified, 2.0s; if GUI and not specified, continuous (0.0s)
    if args.infinite:
        duration_s = 0.0
    elif args.duration is not None:
        duration_s = args.duration
    else:
        duration_s = 2.0 if args.headless else 0.0

    if args.menu:
        selected_meta, duration_s, mode, payload_kg = interactive_menu()

    realtime = False if args.fast else (True if args.realtime else not args.headless)

    run_flight_simulation(
        meta_path=selected_meta,
        duration_s=duration_s,
        target_altitude=args.alt,
        payload_kg=payload_kg,
        realtime=realtime,
        solver_type=args.solver,
        pruning_sparsity=args.sparsity,
        ablate_cx=args.ablate_cx,
        engine=args.engine,
        headless=args.headless,
        control_mode=mode,
    )


if __name__ == "__main__":
    main()


class OpticalFlowOUWrapper:
    """
    Observation Wrapper that injects Ornstein-Uhlenbeck (OU) noise into 
    the Optical Flow observations (FlowX, FlowY) when the drone is far 
    from obstacles. This induces a smooth, deterministic Levy-like 
    exploration flight pattern without breaking ONNX determinism.
    """
    def __init__(
        self, 
        env: Any,
        theta: float = 0.15,
        mu: float = 0.0,
        sigma: float = 0.3,
        dt: float = DEFAULT_DT,
        clearance_threshold: float = 1.0,
        seed: int = 42
    ):
        self.env = env
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.clearance_threshold = clearance_threshold
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(2, dtype=np.float32)
        
    def _ou_step(self) -> np.ndarray:
        # OU process: dx = theta * (mu - x) * dt + sigma * sqrt(dt) * noise
        noise = self.rng.normal(size=2).astype(np.float32)
        dx = self.theta * (self.mu - self.state) * self.dt + self.sigma * np.sqrt(self.dt) * noise
        self.state += dx
        return self.state.copy()

    def reset(self, **kwargs) -> Any:
        self.state = np.zeros(2, dtype=np.float32)
        result = self.env.reset(**kwargs)
        # Handle cases where reset returns just obs, or (obs, info)
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            obs, info = result
            return self._inject_noise(obs), info
        else:
            return self._inject_noise(result)

    def step(self, action: Any, **kwargs) -> Tuple[Any, float, bool, Dict[str, Any]]:
        # Fast evaluator env step returns: (obs_flow, obs_tof), cost, done, info
        # Standard gym env step returns: obs, reward, done, info
        result = self.env.step(action, **kwargs)
        
        if len(result) == 4:
            obs, reward_or_cost, done, info = result
            return self._inject_noise(obs), reward_or_cost, done, info
        elif len(result) == 5: # terminated, truncated
            obs, reward, term, trunc, info = result
            return self._inject_noise(obs), reward, term, trunc, info
        return result

    def _inject_noise(self, obs: Any) -> Any:
        if isinstance(obs, tuple) and len(obs) == 2:
            obs_flow, obs_tof = obs
            tof = np.asarray(obs_tof).ravel()
            min_dist = np.min(tof) if tof.size > 0 else 1.0
            if min_dist > self.clearance_threshold:
                ou_noise = self._ou_step()
                obs_flow = obs_flow + ou_noise
            return (obs_flow, obs_tof)
        else:
            obs_np = np.asarray(obs, dtype=np.float32).copy()
            tof = obs_np[FLOW_DIM:(FLOW_DIM + TOF_DIM)] if obs_np.size >= SENSOR_DIM_BASE else np.ones(TOF_DIM, dtype=np.float32)
            min_dist = np.min(tof) if tof.size > 0 else 1.0
            if min_dist > self.clearance_threshold:
                ou_noise = self._ou_step()
                obs_np[0:FLOW_DIM] += ou_noise
            return obs_np

    def __getattr__(self, name):
        return getattr(self.env, name)
