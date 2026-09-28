"""
simulation/isaac_hud.py
========================
In-Viewer Graphical HUD, On-Screen Interactive Menu & 3D Sensor Visualizer
for NVIDIA Isaac Gym (PhysX Engine).

Renders directly on top of the 3D viewport using Isaac Gym's native `add_lines` GPU pipeline:
1. 3D In-World Sensor Visualizations:
   - Downward Laser Rangefinder: Glowing red laser ray from drone to ground/obstacle contact point
     plus a surface target reticle (+) at the hit coordinate.
   - 8x8 ToF Matrix (45° FOV): 3D wireframe viewing frustum projected forward from the sensor.
     Dynamically changes color from soft cyan (clear) to bright amber/red when an obstacle looms.
   - Optical Flow & Velocity Vector: Directional motion arrow showing surface drift.

2. Camera-Locked On-Screen HUD Dashboard:
   - Active Flight Brain badge: e.g. `[9] AUTO LASER+8X8` or `[1] SPECTRAL K64`
   - Control Mode badge: `[AUTONOMOUS]` (Cyan) or `[MANUAL WASD]` (Green)
   - Real-time Cargo Payload visual gauge: 0.00 kg ... 1.50 kg (AUW up to 1.63 kg)
   - Downward Laser altitude tape & Target Altitude indicator
   - 4 Motor throttle gauges (Quad-X FL, FR, RL, RR)
   - Optical Flow & Cumulative Displacement readouts

3. Interactive In-Viewer Model Selector Menu (Toggle with [TAB]):
   - Full on-screen list of all 9 flight models with active cursor
   - Complete on-screen keybindings legend for on-the-fly control without touching the console!
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. High-Performance Vector Stroke Font Definition
# ─────────────────────────────────────────────────────────────────────────────

# Normalized glyph segments defined on [0, 1] x [0, 1] (X horizontal, Y vertical up)
_GLYPHS: Dict[str, List[Tuple[float, float, float, float]]] = {
    'A': [(0, 0, 0.5, 1), (0.5, 1, 1, 0), (0.2, 0.4, 0.8, 0.4)],
    'B': [(0, 0, 0, 1), (0, 1, 0.7, 1), (0.7, 1, 0.85, 0.75), (0.85, 0.75, 0.7, 0.5),
          (0.7, 0.5, 0, 0.5), (0.7, 0.5, 0.85, 0.25), (0.85, 0.25, 0.7, 0), (0.7, 0, 0, 0)],
    'C': [(1, 1, 0.1, 1), (0.1, 1, 0, 0.9), (0, 0.9, 0, 0.1), (0, 0.1, 0.1, 0), (0.1, 0, 1, 0)],
    'D': [(0, 0, 0, 1), (0, 1, 0.6, 1), (0.6, 1, 0.9, 0.7), (0.9, 0.7, 0.9, 0.3), (0.9, 0.3, 0.6, 0), (0.6, 0, 0, 0)],
    'E': [(1, 1, 0, 1), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0.5, 0.7, 0.5)],
    'F': [(1, 1, 0, 1), (0, 1, 0, 0), (0, 0.5, 0.7, 0.5)],
    'G': [(1, 1, 0.1, 1), (0.1, 1, 0, 0.9), (0, 0.9, 0, 0.1), (0, 0.1, 0.1, 0), (0.1, 0, 1, 0), (1, 0, 1, 0.5), (1, 0.5, 0.5, 0.5)],
    'H': [(0, 0, 0, 1), (1, 0, 1, 1), (0, 0.5, 1, 0.5)],
    'I': [(0.2, 1, 0.8, 1), (0.5, 1, 0.5, 0), (0.2, 0, 0.8, 0)],
    'J': [(0.8, 1, 0.8, 0.2), (0.8, 0.2, 0.6, 0), (0.6, 0, 0.2, 0), (0.2, 0, 0, 0.3)],
    'K': [(0, 0, 0, 1), (1, 1, 0, 0.5), (0, 0.5, 1, 0)],
    'L': [(0, 1, 0, 0), (0, 0, 0.9, 0)],
    'M': [(0, 0, 0, 1), (0, 1, 0.5, 0.4), (0.5, 0.4, 1, 1), (1, 1, 1, 0)],
    'N': [(0, 0, 0, 1), (0, 1, 1, 0), (1, 0, 1, 1)],
    'O': [(0.1, 0, 0.9, 0), (0.9, 0, 1, 0.1), (1, 0.1, 1, 0.9), (1, 0.9, 0.9, 1),
          (0.9, 1, 0.1, 1), (0.1, 1, 0, 0.9), (0, 0.9, 0, 0.1), (0, 0.1, 0.1, 0)],
    'P': [(0, 0, 0, 1), (0, 1, 0.7, 1), (0.7, 1, 0.9, 0.75), (0.9, 0.75, 0.7, 0.5), (0.7, 0.5, 0, 0.5)],
    'Q': [(0.1, 0, 0.9, 0), (0.9, 0, 1, 0.1), (1, 0.1, 1, 0.9), (1, 0.9, 0.9, 1),
          (0.9, 1, 0.1, 1), (0.1, 1, 0, 0.9), (0, 0.9, 0, 0.1), (0, 0.1, 0.1, 0), (0.6, 0.3, 1.0, -0.1)],
    'R': [(0, 0, 0, 1), (0, 1, 0.7, 1), (0.7, 1, 0.9, 0.75), (0.9, 0.75, 0.7, 0.5), (0.7, 0.5, 0, 0.5), (0.4, 0.5, 0.9, 0)],
    'S': [(1, 0.9, 0.8, 1), (0.8, 1, 0.2, 1), (0.2, 1, 0, 0.8), (0, 0.8, 0, 0.6), (0, 0.6, 1, 0.4),
          (1, 0.4, 1, 0.2), (1, 0.2, 0.8, 0), (0.8, 0, 0.2, 0), (0.2, 0, 0, 0.1)],
    'T': [(0, 1, 1, 1), (0.5, 1, 0.5, 0)],
    'U': [(0, 1, 0, 0.2), (0, 0.2, 0.2, 0), (0.2, 0, 0.8, 0), (0.8, 0, 1, 0.2), (1, 0.2, 1, 1)],
    'V': [(0, 1, 0.5, 0), (0.5, 0, 1, 1)],
    'W': [(0, 1, 0.2, 0), (0.2, 0, 0.5, 0.6), (0.5, 0.6, 0.8, 0), (0.8, 0, 1, 1)],
    'X': [(0, 0, 1, 1), (0, 1, 1, 0)],
    'Y': [(0, 1, 0.5, 0.5), (1, 1, 0.5, 0.5), (0.5, 0.5, 0.5, 0)],
    'Z': [(0, 1, 1, 1), (1, 1, 0, 0), (0, 0, 1, 0)],
    '0': [(0, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 1), (0, 1, 0, 0), (0, 0, 1, 1)],
    '1': [(0.2, 0.8, 0.5, 1), (0.5, 1, 0.5, 0), (0.2, 0, 0.8, 0)],
    '2': [(0, 0.8, 0.2, 1), (0.2, 1, 0.8, 1), (0.8, 1, 1, 0.8), (1, 0.8, 0, 0), (0, 0, 1, 0)],
    '3': [(0, 1, 1, 1), (1, 1, 0.4, 0.55), (0.4, 0.55, 0.8, 0.55), (0.8, 0.55, 1, 0.35), (1, 0.35, 1, 0.1), (1, 0.1, 0.8, 0), (0.8, 0, 0, 0)],
    '4': [(0.8, 0, 0.8, 1), (0.8, 1, 0, 0.35), (0, 0.35, 1, 0.35)],
    '5': [(1, 1, 0, 1), (0, 1, 0, 0.55), (0, 0.55, 0.8, 0.55), (0.8, 0.55, 1, 0.35), (1, 0.35, 1, 0.1), (1, 0.1, 0.8, 0), (0.8, 0, 0, 0)],
    '6': [(1, 0.85, 0.8, 1), (0.8, 1, 0.2, 1), (0.2, 1, 0, 0.7), (0, 0.7, 0, 0.2), (0, 0.2, 0.2, 0), (0.2, 0, 0.8, 0), (0.8, 0, 1, 0.2), (1, 0.2, 1, 0.5), (1, 0.5, 0, 0.5)],
    '7': [(0, 1, 1, 1), (1, 1, 0.3, 0)],
    '8': [(0, 0.5, 0, 1), (0, 1, 1, 1), (1, 1, 1, 0.5), (1, 0.5, 0, 0.5), (0, 0.5, 0, 0), (0, 0, 1, 0), (1, 0, 1, 0.5)],
    '9': [(1, 0.5, 0, 0.5), (0, 0.5, 0, 0.8), (0, 0.8, 0.2, 1), (0.2, 1, 0.8, 1), (0.8, 1, 1, 0.8), (1, 0.8, 1, 0.2), (1, 0.2, 0.8, 0), (0.8, 0, 0, 0)],
    ' ': [],
    '-': [(0.1, 0.5, 0.9, 0.5)],
    '+': [(0.1, 0.5, 0.9, 0.5), (0.5, 0.1, 0.5, 0.9)],
    '[': [(0.7, 1, 0.3, 1), (0.3, 1, 0.3, 0), (0.3, 0, 0.7, 0)],
    ']': [(0.3, 1, 0.7, 1), (0.7, 1, 0.7, 0), (0.7, 0, 0.3, 0)],
    '(': [(0.7, 1, 0.3, 0.7), (0.3, 0.7, 0.3, 0.3), (0.3, 0.3, 0.7, 0)],
    ')': [(0.3, 1, 0.7, 0.7), (0.7, 0.7, 0.7, 0.3), (0.7, 0.3, 0.3, 0)],
    ':': [(0.45, 0.25, 0.55, 0.25), (0.45, 0.75, 0.55, 0.75)],
    '.': [(0.4, 0.05, 0.6, 0.05)],
    '/': [(0.1, 0, 0.9, 1)],
    '%': [(0.1, 0, 0.9, 1), (0.2, 0.8, 0.3, 0.8), (0.7, 0.2, 0.8, 0.2)],
    '<': [(0.8, 1, 0.2, 0.5), (0.2, 0.5, 0.8, 0)],
    '>': [(0.2, 1, 0.8, 0.5), (0.8, 0.5, 0.2, 0)],
    '=': [(0.1, 0.65, 0.9, 0.65), (0.1, 0.35, 0.9, 0.35)],
    '_': [(0.0, 0.0, 1.0, 0.0)],
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Camera Projection & HUD Geometry Generator
# ─────────────────────────────────────────────────────────────────────────────

class IsaacGymHUD:
    """
    Renders on-screen telemetry, interactive selector menu, and 3D sensor rays
    directly inside the NVIDIA Isaac Gym viewport window.
    """

    def __init__(self):
        # Menu state
        self.show_menu = True               # Toggled on/off with TAB
        self.last_switch_notification = ""
        self.notification_timer = 0.0

    def toggle_menu(self):
        self.show_menu = not self.show_menu

    def set_notification(self, text: str, duration_s: float = 2.0):
        self.last_switch_notification = text
        self.notification_timer = duration_s

    def update_timer(self, dt: float):
        if self.notification_timer > 0.0:
            self.notification_timer -= dt

    def generate_frame_lines(
        self,
        cam_pos: np.ndarray,
        cam_target: np.ndarray,
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
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Builds all 3D sensor lines and camera-fixed HUD lines for the current frame.
        
        Returns:
            vertices: (num_lines * 2, 3) float32 array of line start/end points in world space
            colors:   (num_lines, 3) float32 array of RGB colors
            num_lines: total integer line count
        """
        verts: List[float] = []
        cols: List[float] = []

        def add_line(p1, p2, c):
            verts.extend([p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]])
            cols.extend([c[0], c[1], c[2]])

        # ── 1. 3D In-World Sensor Visualizations ─────────────────────────────

        # A. Downward Laser Beam (Bright Red Ray with Surface Crosshair)
        if laser_hit_point is not None:
            # Laser beam ray
            add_line(drone_pos, laser_hit_point, (1.0, 0.15, 0.15))
            # Surface contact reticle (+)
            hx, hy, hz = float(laser_hit_point[0]), float(laser_hit_point[1]), float(laser_hit_point[2])
            r_sz = 0.08
            add_line((hx - r_sz, hy, hz + 0.005), (hx + r_sz, hy, hz + 0.005), (1.0, 0.2, 0.2))
            add_line((hx, hy - r_sz, hz + 0.005), (hx, hy + r_sz, hz + 0.005), (1.0, 0.2, 0.2))

        # B. Forward 8x8 Depth Matrix (45° Field of View Wireframe Frustum)
        # Check frontal obstacle clearance
        center_clearance = float(np.mean(depth_8x8[2:6, 2:6]))
        frustum_len = 1.2
        half_angle = math.radians(22.5) # 45 deg total FOV
        w_half = frustum_len * math.tan(half_angle)
        h_half = frustum_len * math.tan(half_angle)

        # 4 corners of forward frustum in body frame (body +X is forward)
        p_nose = drone_pos + drone_rot @ np.array([0.045, 0.0, 0.005], dtype=np.float32)
        c_tl = drone_pos + drone_rot @ np.array([frustum_len, +w_half, +h_half], dtype=np.float32)
        c_tr = drone_pos + drone_rot @ np.array([frustum_len, -w_half, +h_half], dtype=np.float32)
        c_br = drone_pos + drone_rot @ np.array([frustum_len, -w_half, -h_half], dtype=np.float32)
        c_bl = drone_pos + drone_rot @ np.array([frustum_len, +w_half, -h_half], dtype=np.float32)

        # Frustum color: cyan if clear, turns bright red if looming obstacle < 0.65
        f_color = (1.0, 0.2, 0.1) if center_clearance < 0.65 else (0.15, 0.75, 0.95)
        # 4 edges from sensor nose
        add_line(p_nose, c_tl, f_color)
        add_line(p_nose, c_tr, f_color)
        add_line(p_nose, c_br, f_color)
        add_line(p_nose, c_bl, f_color)
        # Far rectangle
        add_line(c_tl, c_tr, f_color)
        add_line(c_tr, c_br, f_color)
        add_line(c_br, c_bl, f_color)
        add_line(c_bl, c_tl, f_color)

        # C. Optical Flow / Translational Velocity Vector (Green arrow)
        v_speed = float(np.linalg.norm(drone_vel[:2]))
        if v_speed > 0.05:
            v_dir = drone_pos + np.array([drone_vel[0], drone_vel[1], 0.0], dtype=np.float32) * 0.8
            add_line(drone_pos, v_dir, (0.2, 0.9, 0.3))

        # ── 2. Camera-Fixed Heads-Up Display (HUD) System ────────────────────

        # Establish camera coordinate axes in world frame
        fwd = cam_target - cam_pos
        fwd_len = float(np.linalg.norm(fwd))
        if fwd_len < 1e-4:
            fwd = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            fwd /= fwd_len

        up_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(fwd, up_world)
        r_len = float(np.linalg.norm(right))
        if r_len < 1e-4:
            right = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            right /= r_len

        up = np.cross(right, fwd)
        up /= np.linalg.norm(up)

        # Project 2D normalized HUD coords [-1..1] onto 3D plane at distance d = 0.40m
        d_hud = 0.40
        hud_center = cam_pos + d_hud * fwd

        def hud_to_world(hx: float, hy: float) -> Tuple[float, float, float]:
            """Maps 2D HUD coordinate (hx, hy) to 3D world vector."""
            pt = hud_center + (hx * 0.22) * right + (hy * 0.14) * up
            return (float(pt[0]), float(pt[1]), float(pt[2]))

        def draw_hud_line(x1: float, y1: float, x2: float, y2: float, c: Tuple[float, float, float]):
            w1 = hud_to_world(x1, y1)
            w2 = hud_to_world(x2, y2)
            add_line(w1, w2, c)

        def draw_text(text: str, x0: float, y0: float, char_w: float, char_h: float, c: Tuple[float, float, float], spacing: float = 0.30):
            cx = x0
            for ch in text.upper():
                strokes = _GLYPHS.get(ch, [])
                for (s1, t1, s2, t2) in strokes:
                    draw_hud_line(cx + s1 * char_w, y0 + t1 * char_h, cx + s2 * char_w, y0 + t2 * char_h, c)
                cx += char_w * (1.0 + spacing)

        def draw_hud_box(x: float, y: float, w: float, h: float, c: Tuple[float, float, float]):
            draw_hud_line(x, y, x + w, y, c)
            draw_hud_line(x + w, y, x + w, y + h, c)
            draw_hud_line(x + w, y + h, x, y + h, c)
            draw_hud_line(x, y + h, x, y, c)

        def draw_hud_bar(x: float, y: float, w: float, h: float, val: float, c_fill, c_border):
            draw_hud_box(x, y, w, h, c_border)
            val = float(np.clip(val, 0.0, 1.0))
            if val > 0.02:
                fill_w = val * (w - 0.004)
                # Fill bar with horizontal strokes
                draw_hud_line(x + 0.002, y + h * 0.5, x + 0.002 + fill_w, y + h * 0.5, c_fill)
                draw_hud_line(x + 0.002, y + h * 0.25, x + 0.002 + fill_w, y + h * 0.25, c_fill)
                draw_hud_line(x + 0.002, y + h * 0.75, x + 0.002 + fill_w, y + h * 0.75, c_fill)

        # ── A. Top Status Header ─────────────────────────────────────────────
        c_cyan = (0.2, 0.8, 1.0)
        c_white = (0.9, 0.9, 0.9)
        c_green = (0.2, 0.9, 0.4)
        c_amber = (1.0, 0.75, 0.15)
        c_red = (1.0, 0.25, 0.2)
        c_dim = (0.45, 0.5, 0.55)

        # Top Bar frame
        draw_hud_box(-0.95, 0.75, 1.90, 0.20, c_cyan)

        # Title & Drone Platform specs
        draw_text("CHONG-FLY 130G 15X15CM", -0.92, 0.88, 0.015, 0.028, c_cyan)

        # Mode Badge: [AUTONOMOUS] or [MANUAL]
        if control_mode == "manual":
            draw_text("MODE: MANUAL PILOT (WASD)", -0.15, 0.88, 0.014, 0.026, c_green)
        else:
            draw_text("MODE: AUTONOMOUS", -0.15, 0.88, 0.014, 0.026, c_cyan)

        # Active Model Name
        active_name = models_list[active_model_idx]["name"] if 0 <= active_model_idx < len(models_list) else "MODEL"
        draw_text(f"BRAIN: [{active_model_idx + 1}] {active_name}", -0.92, 0.78, 0.014, 0.026, c_amber)

        # Flight Status (Paused / Collision / Live)
        if is_paused:
            draw_text("STATUS: PAUSED [P]", 0.35, 0.78, 0.014, 0.026, c_amber)
        elif is_crashed:
            draw_text("STATUS: COLLISION! [R] RESPAWN", 0.15, 0.78, 0.014, 0.026, c_red)
        else:
            draw_text("STATUS: STABLE", 0.45, 0.78, 0.014, 0.026, c_green)

        # ── B. Live Sensors & Telemetry Gauges (Left Side Panel) ─────────────
        draw_hud_box(-0.95, 0.05, 0.80, 0.66, c_cyan)
        draw_text("SENSORS & PAYLOAD", -0.92, 0.65, 0.013, 0.024, c_cyan)

        # 1. Downward Laser Altimeter Readout
        laser_c = c_green if abs(laser_alt_m - target_alt_m) < 0.15 else c_amber
        draw_text(f"DOWN LASER: {laser_alt_m:4.2f} M", -0.92, 0.58, 0.012, 0.022, laser_c)
        draw_text(f"TARGET ALT: {target_alt_m:4.2f} M [U/J]", -0.92, 0.52, 0.012, 0.022, c_white)

        # 2. Cargo Payload Gauge (0.0 - 1.5 kg)
        p_ratio = payload_kg / 1.50
        draw_text(f"PAYLOAD: {payload_kg:4.2f} KG [ [ / ] ]", -0.92, 0.44, 0.012, 0.022, c_amber)
        draw_hud_bar(-0.92, 0.38, 0.55, 0.035, p_ratio, c_amber, c_dim)
        draw_text(f"AUW: {total_mass_kg:4.2f} KG | HOVER: {hover_throttle*100:4.1f}%", -0.92, 0.32, 0.010, 0.019, c_white)

        # 3. Optical Flow & Displacement
        draw_text(f"OPTICAL FLOW: [{optical_flow[0]:+4.2f}, {optical_flow[1]:+4.2f}]", -0.92, 0.24, 0.010, 0.019, c_cyan)
        draw_text(f"TOF 8X8 CENTER: {center_clearance * 3.5:4.2f} M", -0.92, 0.18, 0.010, 0.019, f_color)
        draw_text(f"POWER DEMAND: {power_w:4.1f} WATTS", -0.92, 0.12, 0.010, 0.019, c_white)

        # ── C. Motor Mixer Gauges (Right Top Panel) ──────────────────────────
        draw_hud_box(0.40, 0.35, 0.55, 0.36, c_cyan)
        draw_text("BETAFLIGHT MOTORS (QUAD-X)", 0.42, 0.65, 0.010, 0.019, c_cyan)
        # FL, FR, RL, RR vertical bars
        m_labels = ["FL", "FR", "RL", "RR"]
        m_vals = [motors[3], motors[1], motors[2], motors[0]]
        for idx, (lbl, val) in enumerate(zip(m_labels, m_vals)):
            bx = 0.44 + idx * 0.12
            draw_text(lbl, bx, 0.58, 0.010, 0.018, c_white)
            draw_hud_bar(bx, 0.43, 0.08, 0.12, val, c_green, c_dim)
            draw_text(f"{val*100:3.0f}%", bx - 0.01, 0.38, 0.009, 0.016, c_white)

        # ── D. Interactive In-Viewer Model Selector Menu (Full Display) ───────
        if self.show_menu:
            # Center-Right interactive selection window
            menu_x = -0.10
            menu_y = -0.75
            menu_w = 1.05
            menu_h = 1.05
            draw_hud_box(menu_x, menu_y, menu_w, menu_h, c_cyan)
            draw_text("SELECT MODEL ON THE FLY [KEYS 1-9 / N / B]", menu_x + 0.03, menu_y + menu_h - 0.06, 0.012, 0.024, c_amber)

            # Draw list of models with active selector cursor
            y_cursor = menu_y + menu_h - 0.13
            for i, m in enumerate(models_list):
                is_active = (i == active_model_idx)
                prefix = ">" if is_active else " "
                m_color = c_green if is_active else c_white
                m_text = f"{prefix} [{i + 1}] {m['name']:<18} ({m['tag']})"
                draw_text(m_text, menu_x + 0.03, y_cursor, 0.010, 0.020, m_color)
                y_cursor -= 0.055

            # Bottom Keybinding instructions
            draw_hud_line(menu_x, y_cursor + 0.02, menu_x + menu_w, y_cursor + 0.02, c_dim)
            draw_text("KEYBOARD CONTROLS (IN VIEWER WINDOW):", menu_x + 0.03, y_cursor - 0.03, 0.009, 0.018, c_cyan)
            draw_text("[1-9] SWITCH BRAIN | [N/B] NEXT/PREV BRAIN", menu_x + 0.03, y_cursor - 0.08, 0.009, 0.017, c_white)
            draw_text("[ [ / ] ] ADJUST PAYLOAD (-/+ 0.1 KG)", menu_x + 0.03, y_cursor - 0.13, 0.009, 0.017, c_white)
            draw_text("[U / J] HOVER ALTITUDE (-/+ 0.25 M)", menu_x + 0.03, y_cursor - 0.18, 0.009, 0.017, c_white)
            draw_text("[M] MODE (AUTO/PILOT) | [V] CAM | [R] RESPAWN", menu_x + 0.03, y_cursor - 0.23, 0.009, 0.017, c_white)
            draw_text("[TAB] HIDE/SHOW THIS MENU OVERLAY", menu_x + 0.03, y_cursor - 0.28, 0.009, 0.017, c_amber)
        else:
            # Compact bottom tip when menu is collapsed
            draw_text("[TAB] PRESS TAB FOR IN-VIEWER MODEL SELECTOR MENU", -0.65, -0.90, 0.012, 0.022, c_amber)

        num_lines = len(cols) // 3
        verts_np = np.array(verts, dtype=np.float32)
        cols_np = np.array(cols, dtype=np.float32)

        return verts_np, cols_np, num_lines
