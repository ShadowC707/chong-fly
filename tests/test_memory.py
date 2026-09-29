"""
tests/test_memory.py
====================
TDD verification for EgocentricMemoryWrapper (Spatial Memory Ring Buffer).
"""

import math
import os
import sys
import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from simulation.memory import (
    EgocentricMemoryWrapper,
    SECTOR_FRONT,
    SECTOR_FRONT_RIGHT,
    SECTOR_RIGHT,
    SECTOR_BACK_RIGHT,
    SECTOR_BACK,
    SECTOR_BACK_LEFT,
    SECTOR_LEFT,
    SECTOR_FRONT_LEFT,
)


def test_memory_initialization():
    """
    Тест ініціалізації: 8 секторів, усі за замовчуванням дорівнюють 1.0 (безпечна дистанція).
    """
    mem = EgocentricMemoryWrapper()
    state = mem.get_memory()
    assert state.shape == (8,)
    assert np.allclose(state, 1.0)


def test_memory_wall_shift_90_deg_right():
    """
    TDD вимога 1:
    Стіна (0.3 м) подається у Front sector (сектор 0).
    Потім симулюється поворот дрона на 90 градусів праворуч (+Yaw = pi / 2).
    Assert: значення 0.3 коректно зміщується в Left sector (сектор 6).
    """
    mem = EgocentricMemoryWrapper(decay_rate=0.0)

    # 1. Подаємо стіну 0.3 м прямо перед дроном (Sector 0)
    state_0 = mem.update(tof_center_dist=0.3, delta_yaw_rad=0.0)
    assert np.isclose(state_0[SECTOR_FRONT], 0.3), f"Expected 0.3 in Front, got {state_0[SECTOR_FRONT]}"
    assert np.isclose(state_0[SECTOR_LEFT], 1.0)

    # 2. Дрон повертає праворуч на 90 градусів (pi / 2).
    # Попереду перешкоди вже немає (1.0 м)
    state_rot = mem.update(tof_center_dist=1.0, delta_yaw_rad=math.pi / 2.0)

    # Стіна, що була спереду, тепер повинна опинитися ЛІВОРУЧ (сектор 6)
    assert np.isclose(state_rot[SECTOR_LEFT], 0.3), (
        f"Expected 0.3 to shift to Left sector (idx {SECTOR_LEFT}), got {state_rot[SECTOR_LEFT]}"
    )
    # Попереду знову чистий простір
    assert np.isclose(state_rot[SECTOR_FRONT], 1.0)


def test_memory_wall_shift_45_deg_right():
    """
    Тест на поворот на 45 градусів праворуч (+pi / 4):
    Стіна переміщується з Front (0) у Front-Left (7).
    """
    mem = EgocentricMemoryWrapper(decay_rate=0.0)
    mem.update(0.4, delta_yaw_rad=0.0)

    state = mem.update(1.0, delta_yaw_rad=math.pi / 4.0)
    assert np.isclose(state[SECTOR_FRONT_LEFT], 0.4)
    assert np.isclose(state[SECTOR_FRONT], 1.0)


def test_memory_wall_shift_90_deg_left():
    """
    Тест на поворот ліворуч (-pi / 2):
    Якщо дрон повертає ліворуч, стіна спереду зміщується ПРАВОРУЧ (сектор 2).
    """
    mem = EgocentricMemoryWrapper(decay_rate=0.0)
    mem.update(0.5, delta_yaw_rad=0.0)

    state = mem.update(1.0, delta_yaw_rad=-math.pi / 2.0)
    assert np.isclose(state[SECTOR_RIGHT], 0.5)
    assert np.isclose(state[SECTOR_FRONT], 1.0)


def test_memory_decay_mechanism():
    """
    TDD вимога 2:
    Перевірка механізму затухання (decay):
    Значення поступово повертаються до 1.0 (безпека) за кілька кроків.
    """
    decay_step = 0.02
    mem = EgocentricMemoryWrapper(decay_rate=decay_step)

    # 1. Поміщаємо перешкоду 0.4 м спереду і повертаємо на 90 градусів праворуч (у сектор 6)
    mem.update(0.4, delta_yaw_rad=0.0)
    state = mem.update(1.0, delta_yaw_rad=math.pi / 2.0)
    assert np.isclose(state[SECTOR_LEFT], 0.4)

    # 2. Робимо 5 кроків прямо без обертання: перешкода зліва повинна збільшуватись на 0.02 щокроку
    current_val = state[SECTOR_LEFT]
    for step in range(1, 6):
        state = mem.update(1.0, delta_yaw_rad=0.0)
        expected_val = min(1.0, 0.4 + step * decay_step)
        assert np.isclose(state[SECTOR_LEFT], expected_val), (
            f"Step {step}: expected {expected_val}, got {state[SECTOR_LEFT]}"
        )

    # 3. Через 50 кроків значення повністю досягає 1.0 (обмежено кліпом)
    for _ in range(50):
        state = mem.update(1.0, delta_yaw_rad=0.0)
    assert np.isclose(state[SECTOR_LEFT], 1.0)


def test_memory_accumulated_yaw_sub_threshold():
    """
    Тест накопичення курсу:
    Повороти по 15 градусів (pi / 12) не повинні викликати зсув,
    доки накопичений кут не досягне порогу 45 градусів (на 3-му кроці).
    """
    mem = EgocentricMemoryWrapper(decay_rate=0.0)
    mem.update(0.25, delta_yaw_rad=0.0)

    step_angle = math.radians(15.0)

    # Крок 1 (+15 deg): стіна все ще частково перед дроном, сумарний кут 15 < 45 -> зсуву немає
    s1 = mem.update(0.25, delta_yaw_rad=step_angle)
    assert np.isclose(s1[SECTOR_FRONT], 0.25)
    assert np.isclose(s1[SECTOR_FRONT_LEFT], 1.0)

    # Крок 2 (+15 deg): сумарний кут 30 < 45 -> зсуву немає
    s2 = mem.update(0.25, delta_yaw_rad=step_angle)
    assert np.isclose(s2[SECTOR_FRONT], 0.25)
    assert np.isclose(s2[SECTOR_FRONT_LEFT], 1.0)

    # Крок 3 (+15 deg): сумарний кут 45 >= 45 -> дрон відвернувся (попереду 1.0 м),
    # стіна зсунулась у Front-Left (сектор 7)!
    s3 = mem.update(1.0, delta_yaw_rad=step_angle)
    assert np.isclose(s3[SECTOR_FRONT], 1.0)
    assert np.isclose(s3[SECTOR_FRONT_LEFT], 0.25)



def test_memory_center_tof_extraction():
    """
    Тест вилучення центральної зони 4x4 із матриці ToF 8x8.
    """
    mem = EgocentricMemoryWrapper()

    # Створюємо 8x8 сітку, де фон 1.0, а в центрі перешкода 0.15 м
    grid = np.ones((8, 8), dtype=np.float32)
    grid[2:6, 2:6] = 0.15

    state = mem.update(grid.ravel(), delta_yaw_rad=0.0)
    assert np.isclose(state[SECTOR_FRONT], 0.15)


def test_memory_reset():
    """
    Тест скидання стану пам'яті.
    """
    mem = EgocentricMemoryWrapper()
    mem.update(0.2, delta_yaw_rad=math.pi / 2.0)
    assert not np.allclose(mem.get_memory(), 1.0)

    mem.reset()
    assert np.allclose(mem.get_memory(), 1.0)
    assert mem.accumulated_yaw == 0.0
    assert mem.last_yaw is None


def test_policy_sensor_input_layer_with_memory():
    """
    Тест інтеграції з policy.py:
    SensorInputLayer підтримує 74-D вектор (2 flow + 64 ToF + 8 memory),
    а також підтримує 100% зворотну сумісність із застарілими 66-D входами.
    """
    import torch
    from simulation.policy import SensorInputLayer, SENSOR_DIM, SENSOR_DIM_BASE

    layer = SensorInputLayer(sensor_dim=SENSOR_DIM, learnable_scale=True)
    assert layer.sensor_dim == 74

    flow = np.array([0.2, -0.3], dtype=np.float32)
    tof = np.full(64, 0.8, dtype=np.float32)
    memory = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)

    # 1. from_numpy з 8-D пам'яттю -> (1, 74)
    obs_74 = layer.from_numpy(flow, tof, memory_ring=memory)
    assert obs_74.shape == (1, 74)
    assert np.isclose(float(obs_74[0, 66]), 0.1)
    assert np.isclose(float(obs_74[0, 73]), 0.8)

    # 2. forward з 74-D тензором
    out_74 = layer(obs_74)
    assert out_74.shape == (1, 74)

    # 3. Зворотна сумісність: передача 66-D тензора автоматично доповнюється до 74-D одиницями (1.0 = safe)
    obs_66 = torch.zeros(1, 66)
    out_from_66 = layer(obs_66)
    assert out_from_66.shape == (1, 74)
    assert torch.allclose(out_from_66[0, 66:], torch.ones(8))

