"""
simulation/drone_env.py
=======================
High-Fidelity Drone Simulation Environment with Betaflight Flight Controller
and Sensor Emulation (Isaac Gym & High-Throughput Vectorized Engine).

Key Features:
-------------
1. Betaflight Cascaded PID Controller:
   - Outer loop (Angle Mode): Maps Target Pitch and Target Roll angles to target body rates.
   - Inner loop (Rate PID): Gyro rate error PID with PT1 low-pass D-term filter and anti-windup.
   - Standard Quad-X motor mixer: Maps collective thrust + [Roll, Pitch, Yaw] efforts to 4 motors.
   - First-order motor dynamics: Simulates brushless motor lag (spin-up / spin-down time constant).

2. Neural Network Flight Interface:
   - The neural network does NOT control individual motor RPMs directly.
   - The policy commands setpoints:
       ch[0] Throttle / Collective Thrust [0, 1] (or 1000..2000 µs PWM)
       ch[1] Target Roll angle [-max_angle, +max_angle] [rad]
       ch[2] Target Pitch angle [-max_angle, +max_angle] [rad]
       ch[3] Target Yaw rate [-max_yaw_rate, +max_yaw_rate] [rad/s]
   - Seamlessly accepts either raw physical setpoints or ChongFlyMSPPolicy PWM output.

3. Sensor Emulation:
   - 8x8 ToF Distance Matrix (VL53L5CX):
       Raycasting against 3D room boundaries, cylindrical pillars, and dynamic looming obstacles.
       Normalized depth output [0, 1] matching biological LC looming detectors.
   - Optical Flow Vector (PMW3901):
       Calculates translational surface velocity relative to ground altitude (FlowX, FlowY).
       Includes gyro derotation and altitude scaling matching LPTC visual channels.

4. Isaac Gym Compatibility:
   - Supports native Isaac Gym PhysX simulation pipeline when `isaacgym` is installed.
   - Provides a standalone vectorized 6-DOF physics engine (PyTorch / NumPy) for CPU/CI/testing.
"""

from __future__ import annotations

import os
import sys

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
        dt: float = 0.004,
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
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

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
            self._setup_keyboard_events()
        else:
            self.viewer = None

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

        # Number keys 1-8 for live model switching
        for i in range(1, 9):
            key_attr = f"KEY_{i}"
            if hasattr(gymapi, key_attr):
                g.subscribe_viewer_keyboard_event(v, getattr(gymapi, key_attr), f"model_{i}")

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
        elif mode == "arena":
            cam_pos = gymapi.Vec3(
                float(pos[0]) + 1.4,
                float(pos[1]) + 1.4,
                float(pos[2]) + 0.9,
            )
            cam_target = gymapi.Vec3(float(pos[0]), float(pos[1]), float(pos[2]))
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        # "free": do not call viewer_camera_look_at to allow free mouse orbit/pan/zoom

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
    x_min: float = -3.0
    x_max: float = 3.0
    y_min: float = -3.0
    y_max: float = 3.0
    z_min: float = 0.0
    z_max: float = 3.0


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
        rows: int = 8,
        cols: int = 8,
        fov_h_deg: float = 45.0,
        fov_v_deg: float = 45.0,
        max_range: float = 3.5,
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
# 6-DOF Micro-Quadcopter Dynamics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DroneDynamicsParams:
    mass: float = 0.033                     # 33g Crazyflie-class micro-drone
    arm_length: float = 0.046               # 46 mm center-to-motor distance
    g: float = 9.81                         # gravity [m/s^2]
    # Moments of inertia [kg*m^2]
    ixx: float = 1.66e-5
    iyy: float = 1.66e-5
    izz: float = 2.93e-5
    # Motor parameters (calibrated for 50% stick hover at 11,000 RPM, PWM 1500us)
    thrust_coeff: float = 6.6886e-10       # Thrust = k_f * rpm^2 (total hover thrust = m*g at 50% throttle)
    torque_coeff: float = 1.2e-11          # Torque = k_m * rpm^2 (k_m / k_f ~ 0.018)
    max_rpm: float = 22000.0                # Max motor RPM
    motor_tau: float = 0.020                # First-order motor time constant [s]
    # Aerodynamic drag coefficients
    drag_linear: float = 0.15               # Linear drag coefficient [N/(m/s)]
    drag_angular: float = 0.002             # Angular damping coefficient [N*m/(rad/s)]
    # Maximum flight angle limits for Angle mode
    max_angle_rad: float = math.radians(35.0)   # +/- 35 deg maximum pitch/roll
    max_yaw_rate_rads: float = math.radians(200.0) # +/- 200 deg/s max yaw rate


class QuadcopterDynamics:
    """
    Rigid-body 6-DOF equations of motion with quaternion kinematics
    and first-order motor lag.
    """

    def __init__(self, params: DroneDynamicsParams, dt: float = 0.004):
        self.params = params
        self.dt = dt

        # State vectors:
        self.pos = np.zeros(3, dtype=np.float32)       # [x, y, z] in world frame (m)
        self.vel = np.zeros(3, dtype=np.float32)       # [vx, vy, vz] in world frame (m/s)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) # [qw, qx, qy, qz]
        self.omega = np.zeros(3, dtype=np.float32)     # [p, q, r] in body frame (rad/s)
        self.motor_rpms = np.zeros(4, dtype=np.float32) # current motor RPMs

        # Inertia tensor and inverse
        self.J = np.diag([params.ixx, params.iyy, params.izz]).astype(np.float32)
        self.J_inv = np.diag([1.0 / params.ixx, 1.0 / params.iyy, 1.0 / params.izz]).astype(np.float32)

        # Hover RPM calculation: mg = 4 * k_f * rpm_hover^2
        total_weight = params.mass * params.g
        self.hover_rpm = math.sqrt(total_weight / (4.0 * params.thrust_coeff))
        self.hover_throttle = self.hover_rpm / params.max_rpm

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
        d = p.arm_length / math.sqrt(2.0)

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
        F_gravity = np.array([0.0, 0.0, -p.mass * p.g], dtype=np.float32)
        F_drag = -p.drag_linear * self.vel * np.linalg.norm(self.vel)

        acc = (F_world + F_gravity + F_drag) / p.mass

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
        dt: float = 0.004,                      # 250 Hz control loop
        target_altitude: float = 1.0,           # nominal hover altitude (m)
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

        # Sensors
        self.tof = ToFRaycaster(rows=8, cols=8, fov_h_deg=45.0, fov_v_deg=45.0, max_range=3.5)
        self.flow_sensor = PMW3901FlowSensor(min_altitude=0.08, max_altitude=3.5, derotate_with_gyro=True)


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

    @property
    def is_isaacgym_active(self) -> bool:
        return self.engine == "isaacgym" and HAS_ISAACGYM

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
        action: Union[np.ndarray, Tuple[float, float, float, float]],
        action_type: str = "auto", # "auto", "setpoints", or "pwm"
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes one flight step (dt = 0.004 s, 250 Hz).
        
        The neural network does NOT command motor RPMs directly.
        Instead, it sends flight setpoints (Angle Mode: Roll/Pitch, Rate Mode: Yaw Rate, Collective Thrust).
        
        Args:
            action: 4-element array:
                - If PWM: [throttle_pwm, roll_pwm, pitch_pwm, yaw_pwm] in [1000, 2000] µs
                - If setpoints: [target_thrust (0..1), target_roll (rad), target_pitch (rad), target_yaw_rate (rad/s)]
            action_type: "auto" detects based on values (> 500 means PWM); or explicitly "pwm" / "setpoints".

        Returns:
            obs: 66-D observation array [FlowX, FlowY, ToF_00 ... ToF_63]
            reward: scalar reward
            done: boolean termination flag
            info: auxiliary metrics (altitude, euler angles, motor commands, energy)
        """
        action = np.asarray(action, dtype=np.float32)

        # ── 1. Parse Neural Network Action ───────────────────────────────────
        is_pwm = (action_type == "pwm") or (action_type == "auto" and np.any(action > 500.0))
        if is_pwm:
            # Map [1000, 2000] µs RC PWM to flight setpoints
            throttle_pwm = float(action[0])
            roll_pwm = float(action[1])
            pitch_pwm = float(action[2])
            yaw_pwm = float(action[3])

            target_thrust = (throttle_pwm - 1000.0) / 1000.0
            target_roll = ((roll_pwm - 1500.0) / 500.0) * self.params.max_angle_rad
            target_pitch = ((pitch_pwm - 1500.0) / 500.0) * self.params.max_angle_rad
            target_yaw_rate = ((yaw_pwm - 1500.0) / 500.0) * self.params.max_yaw_rate_rads
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

        # Synchronize and step Isaac Gym graphics / PhysX
        user_closed = False
        if self.isaac_sim is not None:
            self.isaac_sim.sync_state(self.physics.pos, self.physics.quat, self.physics.vel, self.physics.omega)
            # Decoupled 50 FPS graphics rendering (every 5 steps at 250 Hz physics)
            sim_ok = self.isaac_sim.step(
                render=(self.step_count % self.isaac_sim.render_every == 0),
                pos=self.physics.pos,
                euler=euler,
            )
            if not sim_ok:
                user_closed = True

        # Calculate electrical / mechanical energy expenditure (P = sum(T * omega))
        power_w = float(np.sum(motors ** 2) * 25.0)  # ~25W hover power for micro-drone
        self.total_energy_j += power_w * self.dt

        # ── 4. Collect Sensor Observations ──────────────────────────────────
        obs = self._get_observation()

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
        Synthesizes 66-dimensional observation vector:
            [0:2]   FlowX, FlowY (optical flow from PMW3901)
            [2:66]  ToF_00 ... ToF_63 (8x8 depth matrix from VL53L5CX)
        """
        rot_mat = self.physics.quaternion_to_rotation_matrix(self.physics.quat)
        pos = self.physics.pos
        vel = self.physics.vel
        omega = self.physics.omega

        # 1. Optical Flow (PMW3901)
        altitude_surface = max(0.08, float(pos[2]))
        flow_xy = self.flow_sensor.compute_flow(
            v_world=vel,
            rot_matrix=rot_mat,
            altitude_above_surface=altitude_surface,
            omega_body=omega,
        )

        # 2. ToF 8x8 Raycasting (VL53L5CX)
        tof_64 = self.tof.cast_rays(
            drone_pos=pos,
            drone_rot=rot_mat,
            room=self.room,
            cylinders=self.cylinders,
            boxes=self.boxes,
        )

        # 3. Concatenate to (66,)
        obs = np.concatenate([flow_xy, tof_64], axis=0).astype(np.float32)
        return obs

    def get_chong_fly_obs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convenience helper returning structured (flow_xy, tof_8x8)
        directly consumable by `ChongFlyMSPPolicy.step_np(flow_xy, tof_8x8)`.
        """
        obs = self._get_observation()
        flow_xy = obs[0:2]
        tof_8x8 = obs[2:66]
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

    # Initialize Environment
    env = DroneSimulationEnv(
        dt=0.004,
        target_altitude=target_altitude,
        engine=engine,
        headless=headless,
    )
    obs = env.reset(seed=42)

    engine_str = "NVIDIA Isaac Gym (PhysX)" if env.is_isaacgym_active else "Standalone 6-DOF Vectorized (CPU)"

    # Default to real-time playback if visual 3D viewer is active
    if not headless and not realtime:
        realtime = True

    is_infinite = (duration_s is None or duration_s <= 0.0)
    dur_str = "Continuous (Infinite / Press Esc to exit)" if is_infinite else f"{duration_s:.1f} s ({int(duration_s / 0.004)} steps @ 250 Hz)"

    # Locate initial model in registry
    active_model_idx = next((i for i, m in enumerate(AVAILABLE_MODELS) if m["meta"] == meta_path), 0)
    active_model_name = AVAILABLE_MODELS[active_model_idx]["name"]

    print("\n" + "=" * 80)
    print("   🚀 CHONG-FLY 6-DOF BETAFLIGHT SITL FLIGHT SIMULATION")
    print("=" * 80)
    print(f"  • Physics Engine:   {engine_str}")
    print(f"  • Initial Brain:    [{active_model_idx + 1}] {active_model_name}")
    print(f"  • Model Meta:       {meta_path}")
    print(f"  • Solver Type:      {solver_type}")
    print(f"  • Flight Mode:      {'🎮 MANUAL PILOT (WASD)' if control_mode == 'manual' else '🤖 AUTONOMOUS (Neural Connectome)'}")
    print(f"  • Flight Duration:  {dur_str}")
    print(f"  • Target Altitude:  {target_altitude:.2f} m")
    print(f"  • Speed Mode:       {'Real-Time Playback (1:1)' if realtime else 'Maximum Speed (Fast)'}")
    print("=" * 80)
    print("  🎮 LIVE 3D VIEWER CONTROLS:")
    print("    • [M] Toggle Mode (Autonomous ↔ Manual Pilot)")
    print("    • [W / S] Pitch Forward / Backward       • [A / D] Roll Left / Right")
    print("    • [Space / C] Throttle Up / Down          • [Q / E] Yaw Turn Left / Right")
    print("    • [1 - 8] Hot-Swap Connectome Brain on the Fly (k=16..256, No-CX, etc.)")
    print("    • [V] Cycle Camera View (Chase Cam ↔ Arena Cam ↔ Free Mouse Look)")
    print("    • [R] Respawn / Reset Drone after Collision")
    print("    • [P] Pause / Resume Simulation          • [Esc] Exit Flight")
    print("=" * 80 + "\n")

    # 1. Initialize Policy
    policy = ChongFlyMSPPolicy.from_meta(
        meta_path=meta_path,
        mode="fixed",
        dt=0.004,
        solver_type=solver_type,
        pruning_sparsity=pruning_sparsity,
        ablate_cx=ablate_cx,
    )
    policy.reset_state()

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
                    status_msg = "⏸️  PAUSED (Press [P] to Resume)" if is_paused else "▶️  RESUMED"
                elif trig == "toggle_mode":
                    control_mode = "manual" if control_mode == "auto" else "auto"
                    status_msg = f"Switched Mode -> {'🎮 MANUAL PILOT' if control_mode == 'manual' else '🤖 AUTONOMOUS'}"
                elif trig == "cycle_camera":
                    if env.isaac_sim:
                        env.isaac_sim.camera_mode_idx = (env.isaac_sim.camera_mode_idx + 1) % len(env.isaac_sim.camera_modes)
                        status_msg = f"Camera Mode -> [{env.isaac_sim.camera_modes[env.isaac_sim.camera_mode_idx].upper()}]"
                elif trig == "reset":
                    # Instant Respawn / Reset
                    env.physics.reset(np.array([0.0, 0.0, target_altitude]))
                    env.pid.reset()
                    policy.reset_state()
                    if env.isaac_sim:
                        env.isaac_sim.sync_state(env.physics.pos, env.physics.quat, env.physics.vel, env.physics.omega)
                    is_crashed = False
                    status_msg = "🔄 Drone Respawned at hover altitude!"
                elif trig.startswith("model_"):
                    try:
                        m_idx = int(trig.split("_")[1]) - 1
                        if 0 <= m_idx < len(AVAILABLE_MODELS):
                            active_model_idx = m_idx
                            active_model_name = AVAILABLE_MODELS[m_idx]["name"]
                            new_meta = AVAILABLE_MODELS[m_idx]["meta"]
                            policy = ChongFlyMSPPolicy.from_meta(
                                meta_path=new_meta,
                                mode="fixed",
                                dt=0.004,
                                solver_type=solver_type,
                            )
                            policy.reset_state()
                            status_msg = f"🧠 Hot-Swapped Brain -> [{m_idx + 1}] {active_model_name}"
                    except Exception as ex:
                        status_msg = f"⚠️ Model switch error: {ex}"

            # If Paused: render frame and sleep
            if is_paused:
                if env.isaac_sim:
                    env.isaac_sim.render()
                time.sleep(0.02)
                continue

            # If Crashed: wait for user to press [R] to respawn
            if is_crashed:
                if env.isaac_sim:
                    env.isaac_sim.render()
                time.sleep(0.02)
                continue

            # ── Sensory Readout ──────────────────────────────────────────────
            flow_xy, tof_8x8 = env.get_chong_fly_obs()

            # ── Action Computation ───────────────────────────────────────────
            if control_mode == "auto":
                # Biological Neural Connectome Policy Step
                pwm = policy.step_np(flow_xy, tof_8x8)
            else:
                # Manual Flight Control via Keyboard
                th = 1500.0  # Hover neutral
                roll = 1500.0
                pitch = 1500.0
                yaw = 1500.0

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

                pwm = np.array([th, roll, pitch, yaw], dtype=np.float32)

            pwms.append(pwm)

            # ── Environment & Betaflight Step ────────────────────────────────
            obs, reward, done, info = env.step(pwm)
            steps_survived += 1

            pos = info["position"]
            vel = info["velocity"]
            euler = info["euler_rad"]
            omega = info["omega_rads"]
            motors = info["motor_commands"]
            power_w = info["power_w"]

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
                print(f"│ SENSORS: Optical Flow=[X:{flow_xy[0]:+5.2f}, Y:{flow_xy[1]:+5.2f}] | ToF Center Dist: {center_tof*3.5:4.2f}m  │")
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
    pwm_arr = np.array(pwms)
    pwm_jitter = float(np.mean(np.abs(np.diff(pwm_arr, axis=0)))) if len(pwm_arr) > 1 else 0.0

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
    print(f"  • Mean PWM Jitter:        {pwm_jitter:.2f} µs/step")
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
        "pwm_jitter": pwm_jitter,
    }


def interactive_menu() -> Tuple[str, float, str]:
    """
    Displays an interactive CLI launcher menu to select model, flight mode, and duration.
    """
    print("\n" + "=" * 80)
    print("   🚁 CHONG-FLY INTERACTIVE FLIGHT SIMULATION LAUNCHER")
    print("=" * 80)
    print("  Select Biological Connectome Neural Model:")
    for m in AVAILABLE_MODELS:
        print(f"    [{m['id']}] {m['name']:<25} • {m['tag']}")
    print("    [M] Manual Piloting Mode (Fly with WASD / Space / C)")
    print("-" * 80)

    try:
        choice = input("  Select Model [1-8] or [M] for Manual (default: 1): ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = "1"

    mode = "auto"
    selected_meta = AVAILABLE_MODELS[0]["meta"]
    if choice.upper() == "M":
        mode = "manual"
    elif choice.isdigit() and 1 <= int(choice) <= len(AVAILABLE_MODELS):
        selected_meta = AVAILABLE_MODELS[int(choice) - 1]["meta"]

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
    return selected_meta, duration_s, mode


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Chong-Fly 6-DOF Drone Betaflight SITL Simulation")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to reduced model meta JSON (default: meta_spectral_k64.json)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Flight duration in seconds (default: 0 = continuous flight in GUI mode)")
    parser.add_argument("--alt", type=float, default=1.0,
                        help="Target hover altitude in meters (default: 1.0m)")
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
                        help="Launch interactive terminal menu to select model and flight mode")
    parser.add_argument("--manual", action="store_true", default=False,
                        help="Start directly in manual pilot mode (WASD + Space/C)")
    parser.add_argument("--infinite", action="store_true", default=False,
                        help="Run continuous flight without time limit")

    args = parser.parse_args()

    mode = "manual" if args.manual else "auto"
    selected_meta = args.model or "data/reduced_models/meta_spectral_k64.json"

    # Default duration: if headless and not specified, 2.0s; if GUI and not specified, continuous (0.0s)
    if args.infinite:
        duration_s = 0.0
    elif args.duration is not None:
        duration_s = args.duration
    else:
        duration_s = 2.0 if args.headless else 0.0

    if args.menu:
        selected_meta, duration_s, mode = interactive_menu()

    realtime = False if args.fast else (True if args.realtime else not args.headless)

    run_flight_simulation(
        meta_path=selected_meta,
        duration_s=duration_s,
        target_altitude=args.alt,
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

