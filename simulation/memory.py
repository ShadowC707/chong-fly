import gym
import numpy as np
import math

class EgocentricMemoryWrapper(gym.ObservationWrapper):
    def __init__(self, env, array_key='sensor_8', yaw_key='yaw'):
        super().__init__(env)
        self.array_key = array_key
        self.yaw_key = yaw_key

    def observation(self, obs):
        """
        Циклічний зсув масиву з 8 елементів залежно від кута Yaw.
        Кожен елемент відповідає сектору в 45 градусів (pi/4).
        """
        # If observation is a dictionary and contains our keys
        if isinstance(obs, dict) and self.array_key in obs and self.yaw_key in obs:
            yaw = obs[self.yaw_key]
            arr = obs[self.array_key]
            
            # Нормалізація yaw до [0, 2*pi]
            yaw_norm = yaw % (2 * math.pi)
            
            # Обчислюємо кількість кроків для зсуву (1 крок = 45 градусів = pi/4)
            shift_steps = int(round(yaw_norm / (math.pi / 4))) % 8
            
            # Циклічний зсув масиву
            obs[self.array_key] = np.roll(arr, shift=shift_steps, axis=-1)
            
        return obs