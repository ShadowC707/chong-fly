import numpy as np
import torch

# simulation/metrics.py
import numpy as np


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
    def __init__(self, voxel_size=0.5):
        self.voxel_size = voxel_size
        self.visited_voxels = set()
        
    def update(self, position):
        """Трекінг 3D-вокселів: position [x, y, z]"""
        voxel = (
            int(np.floor(position[0] / self.voxel_size)),
            int(np.floor(position[1] / self.voxel_size)),
            int(np.floor(position[2] / self.voxel_size))
        )
        self.visited_voxels.add(voxel)
        
    def get_coverage_volume(self):
        return len(self.visited_voxels) * (self.voxel_size ** 3)
    
    def get_coverage_count(self):
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
