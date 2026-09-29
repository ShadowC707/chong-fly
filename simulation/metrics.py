import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch


def calculate_jitter_pr(pwm_history: np.ndarray) -> float:
    """
    Рахує дисперсію похідної (L2 норму різниці) ТІЛЬКИ для Pitch та Roll.
    Штрафує модель за розгойдування горизонту.
    """
    if len(pwm_history) < 2:
        return 0.0

    pwm_array = np.asarray(pwm_history)
    # Рахуємо різницю ШІМ між сусідніми кадрами
    dpwm = pwm_array[1:] - pwm_array[:-1]

    # Беремо квадрати змін тільки для Roll (idx 1) та Pitch (idx 2)
    # і повертаємо середнє значення
    jitter_pr = float(np.mean(dpwm[:, 1:3] ** 2))
    return jitter_pr


def calculate_saccades_yaw(pwm_history: np.ndarray, threshold: float = 150.0) -> int:
    """
    Рахує кількість різких ривків (сакад) по осі Yaw (L1 розрідженість).
    Заохочує модель робити різкі маневри ухилення.
    """
    if len(pwm_history) < 2:
        return 0

    pwm_array = np.asarray(pwm_history)
    # Зміна ШІМ по Yaw (idx 3) за модулем
    dyaw = np.abs(pwm_array[1:, 3] - pwm_array[:-1, 3])

    # Рахуємо, скільки разів зміна перевищила поріг
    saccade_count = int(np.sum(dyaw > threshold))
    return saccade_count


# ... (Клас VoxelTracker залишаєш як був, він згодиться пізніше) ...

class VoxelTracker:
    """
    Robust 3D Voxel Tracker for spatial coverage entropy evaluation.
    Discretizes continuous R^3 positions into Z^3 grid cells.
    """

    def __init__(self, voxel_size: float = 0.5):
        if voxel_size <= 0.0 or not np.isfinite(voxel_size):
            raise ValueError(f"voxel_size must be a positive finite number (> 0), got {voxel_size}")
        self.voxel_size = float(voxel_size)
        self.visited_voxels: set[tuple[int, int, int]] = set()

    def reset(self) -> None:
        """Clears visited voxels history."""
        self.visited_voxels.clear()

    def update(self, position: Any) -> None:
        """
        Track 3D voxel for position [x, y, z].
        Safe against NaN/Inf, non-finite values, and supports list, tuple, np.ndarray, torch.Tensor.
        """
        if hasattr(position, "detach"):
            position = position.detach().cpu().numpy()

        try:
            x = float(position[0])
            y = float(position[1])
            z = float(position[2])
        except (IndexError, TypeError, ValueError):
            return

        # Захист від OverflowError та вибуху координат при NaN / Inf
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            return

        voxel = (
            int(np.floor(x / self.voxel_size)),
            int(np.floor(y / self.voxel_size)),
            int(np.floor(z / self.voxel_size)),
        )
        self.visited_voxels.add(voxel)

    def get_coverage_volume(self) -> float:
        return float(len(self.visited_voxels) * (self.voxel_size ** 3))

    def get_coverage_count(self) -> int:
        return len(self.visited_voxels)

def calculate_area_coverage(trajectory, voxel_size=0.5):
    """
    Трекінг 3D-вокселів для покриття площі.
    trajectory: список або масив позицій [x, y, z]
    """
    tracker = VoxelTracker(voxel_size)
    for pos in trajectory:
        tracker.update(pos)
    return tracker.get_coverage_count()


# ─────────────────────────────────────────────────────────────────────────────
# 2. PhysicsTelemetry (Фізичні фільтри / Kill-Switches)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_forward_ratio(
    vx: float,
    vy: float,
    min_speed: float = 0.05,
    eps: float = 1e-6,
) -> Optional[float]:
    """
    Обчислює коефіцієнт цілеспрямованості через нормалізований косинус:
    r_t = V_x / (math.hypot(V_x, V_y) + eps).

    Діапазон [-1.0, 1.0]:
      1.0  = ідеально вперед (0 градусів)
      0.5  = відхилення 60 градусів (поріг краба)
      0.0  = строго боком (90 градусів, краб)
     -1.0  = строго назад (180 градусів)

    Повертає None, якщо горизонтальна швидкість менша за min_speed (фільтр ховеру).
    """
    try:
        vx_f = float(vx)
        vy_f = float(vy)
    except (TypeError, ValueError):
        return None

    if not (np.isfinite(vx_f) and np.isfinite(vy_f)):
        return None

    h_speed = math.hypot(vx_f, vy_f)
    if h_speed < min_speed:
        return None

    ratio = vx_f / (h_speed + eps)
    return float(np.clip(ratio, -1.0, 1.0))


def calculate_impact_energy(
    mass: float,
    velocity: Any,
    is_crash: bool = True,
) -> float:
    """
    Обчислює кінетичну енергію удару: E_k = 0.5 * m * |V|^2.
    Спрацьовує виключно за умови крашу (is_crash == True).
    Якщо дрон успішно вижив епізод без крашу, повертає 0.0 Дж.
    """
    if not is_crash:
        return 0.0

    if hasattr(velocity, "detach"):
        velocity = velocity.detach().cpu().numpy()

    try:
        vx = float(velocity[0])
        vy = float(velocity[1])
        vz = float(velocity[2])
    except (IndexError, TypeError, ValueError):
        return 0.0

    if not (np.isfinite(vx) and np.isfinite(vy) and np.isfinite(vz)):
        return 0.0

    try:
        m = float(mass)
        if not (np.isfinite(m) and m > 0.0):
            m = 0.130
    except (TypeError, ValueError):
        m = 0.130

    v_sq = vx * vx + vy * vy + vz * vz
    return float(0.5 * m * v_sq)


def calculate_mean_clearance(clearance_history: Sequence[float]) -> float:
    """
    Обчислює середнє арифметичне мінімумів ToF / лазера за час польоту.
    Захищене від пустих історій та некоректних числових значень.
    """
    if not clearance_history:
        return 0.0
    arr = np.asarray(clearance_history, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return 0.0
    return float(np.mean(valid))


class PhysicsTelemetryTracker:
    """
    Збирач фізичної телеметрії та фільтр-вибраковщик (Kill-Switches):
    - Mean Clearance: середнє арифметичне мінімальних дистанцій ToF.
    - Impact Kinetic Energy: E_k = 0.5 * m * |V|^2 при краші.
    - Forward Velocity Ratio: медіана нормалізованого косинуса Vx / (hypot(Vx, Vy) + 1e-6).
    """

    def __init__(
        self,
        min_speed_hover: float = 0.05,
        threshold_crab: float = 0.5,
        default_mass: float = 0.130,
    ):
        self.min_speed_hover = float(min_speed_hover)
        self.threshold_crab = float(threshold_crab)
        self.default_mass = float(default_mass)

        self.clearance_history: List[float] = []
        self.forward_ratios: List[float] = []
        self.impact_energy_j: float = 0.0
        self.is_crash: bool = False

    def reset(self) -> None:
        """Скидає історію телеметрії для нового епізоду."""
        self.clearance_history.clear()
        self.forward_ratios.clear()
        self.impact_energy_j = 0.0
        self.is_crash = False

    def update_step(self, obs_tof: Any, vel: Any) -> None:
        """
        Фіксує телеметрію на поточному кроці:
        - Мінімальне значення сенсорів ToF (кліренс).
        - Нормалізований коефіцієнт швидкості руху вперед (з фільтрацією ховеру).
        """
        # 1. Кліренс ToF
        if obs_tof is not None:
            if hasattr(obs_tof, "detach"):
                obs_tof = obs_tof.detach().cpu().numpy()
            try:
                arr = np.asarray(obs_tof, dtype=np.float64)
                if arr.size > 0:
                    valid_tof = arr[np.isfinite(arr)]
                    if valid_tof.size > 0:
                        min_c = float(np.min(valid_tof))
                        self.clearance_history.append(min_c)
            except (TypeError, ValueError):
                pass

        # 2. Forward Velocity Ratio
        if vel is not None:
            if hasattr(vel, "detach"):
                vel = vel.detach().cpu().numpy()
            try:
                vx = float(vel[0])
                vy = float(vel[1])
                r = calculate_forward_ratio(vx, vy, min_speed=self.min_speed_hover)
                if r is not None:
                    self.forward_ratios.append(r)
            except (IndexError, TypeError, ValueError):
                pass

    def register_crash(self, final_vel: Any, mass: Optional[float] = None) -> float:
        """
        Реєструє краш та обчислює кінетичну енергію удару.
        """
        self.is_crash = True
        m = self.default_mass if mass is None else mass
        self.impact_energy_j = calculate_impact_energy(mass=m, velocity=final_vel, is_crash=True)
        return self.impact_energy_j

    def compute_summary(self) -> Dict[str, Any]:
        """
        Повертає підсумок телеметрії з перевіркою kill-switches.
        """
        mean_c = calculate_mean_clearance(self.clearance_history)
        if len(self.forward_ratios) > 0:
            median_r = float(np.median(self.forward_ratios))
        else:
            # Якщо дрон весь час висів у ховері, рух прямо вважається нейтрально-ідеальним
            median_r = 1.0

        is_crab = bool(median_r < self.threshold_crab)

        return {
            "mean_clearance": mean_c,
            "forward_ratio_median": median_r,
            "impact_energy_j": float(self.impact_energy_j),
            "is_crab_flight": is_crab,
            "is_crash": self.is_crash,
            "steps_tracked": len(self.clearance_history),
            "moving_steps_tracked": len(self.forward_ratios),
        }
