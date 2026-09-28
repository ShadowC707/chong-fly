import numpy as np

def calculate_jitter_xy(pwm_history):
    """
    Розрахунок дисперсії ШІМ (PWM variance) тільки по осях Roll/Pitch.
    pwm_history: список або масив форми (N, 4) [Throttle, Roll, Pitch, Yaw]
                 або списки словників/об'єктів, але припускаємо, що це numpy array
    """
    if len(pwm_history) == 0:
        return 0.0
    pwm_array = np.array(pwm_history)
    
    # Check if shape is valid for [Throttle, Roll, Pitch, Yaw]
    if len(pwm_array.shape) == 2 and pwm_array.shape[1] >= 3:
        # Roll = index 1, Pitch = index 2
        roll_pitch_pwm = pwm_array[:, 1:3]
        variance = np.var(roll_pitch_pwm, axis=0)
        return float(np.sum(variance))
    return 0.0

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