# tests/test_metrics.py
import os
import sys
import numpy as np
import pytest

# --- МАГІЯ ДЛЯ ІМПОРТІВ ---
# Знаходимо поточну папку (tests) і піднімаємось на рівень вище (корінь проєкту)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# --------------------------

from simulation.metrics import (
    calculate_jitter_pr,
    calculate_saccades_yaw,
    VoxelTracker,
    calculate_area_coverage,
    calculate_forward_ratio,
    calculate_impact_energy,
    calculate_mean_clearance,
    PhysicsTelemetryTracker,
)
from optimizer.evaluate import compute_composite_cost


def test_jitter_pr():
    # 1. Ідеальний політ (PWM = 1500 для всіх осей на 10 кадрів)
    # Формат: [Throttle, Roll, Pitch, Yaw]
    smooth_flight = np.full((10, 4), 1500.0)

    jitter = calculate_jitter_pr(smooth_flight)
    assert jitter == 0.0, f"Expected 0 jitter, got {jitter}"

    # 2. Дрон розгойдується по Roll (idx 1)
    spiky_flight = np.full((10, 4), 1500.0)
    spiky_flight[1::2, 1] = 1700.0  # Кожен другий кадр Roll стрибає до 1700

    bad_jitter = calculate_jitter_pr(spiky_flight)
    assert bad_jitter > 1000.0, "Jitter should be heavily penalized"


def test_saccades_yaw():
    # 1. Плавний політ (без сакад)
    smooth = np.full((10, 4), 1500.0)
    assert calculate_saccades_yaw(smooth, threshold=150.0) == 0

    # 2. Два різкі ривки по Yaw (idx 3)
    flight_with_saccades = np.full((10, 4), 1500.0)

    # З 2-го кадру і до 5-го дрон різко дає вправо (1500 -> 1900) - це Сакада №1
    flight_with_saccades[2:6, 3] = 1900.0

    # З 6-го кадру і до кінця дрон різко дає вліво (1900 -> 1100) - це Сакада №2
    flight_with_saccades[6:, 3] = 1100.0

    saccades = calculate_saccades_yaw(flight_with_saccades, threshold=150.0)
    assert saccades == 2, f"Expected 2 saccades, got {saccades}"


def test_voxel_tracker_hover():
    """
    Тест на Левітацію: 100 однакових координат повинні давати рівно 1 воксель.
    """
    tracker = VoxelTracker(voxel_size=0.5)
    for _ in range(100):
        tracker.update([0.0, 0.0, 1.0])

    assert tracker.get_coverage_count() == 1
    assert np.isclose(tracker.get_coverage_volume(), 0.5 ** 3)


def test_voxel_tracker_linear_flight():
    """
    Тест на Політ по прямій: рівномірне переміщення по X від 0.0 до 5.0 м.
    При voxel_size = 0.5 к-ть вокселів має бути floor(5.0 / 0.5) + 1 = 11.
    """
    tracker = VoxelTracker(voxel_size=0.5)
    xs = np.linspace(0.0, 5.0, 101)
    for x in xs:
        tracker.update(np.array([x, 0.0, 1.0]))

    expected_count = int(np.floor(5.0 / 0.5)) + 1
    assert tracker.get_coverage_count() == expected_count


def test_voxel_tracker_robustness():
    """
    Тест на надійність (break-free):
    - Відхилення некоректного voxel_size (<= 0)
    - Безпечний пропуск NaN / Inf без OverflowError
    - Підтримка різних контейнерів (list, tuple, np.ndarray, torch.Tensor)
    """
    # 1. Захист від ділення на 0 / від'ємного розміру вокселя
    with pytest.raises(ValueError):
        VoxelTracker(voxel_size=0.0)
    with pytest.raises(ValueError):
        VoxelTracker(voxel_size=-0.5)

    tracker = VoxelTracker(voxel_size=0.5)

    # 2. Передача NaN та Inf не повинна викликати збій чи OverflowError
    tracker.update([float("nan"), 0.0, 1.0])
    tracker.update([float("inf"), 1.0, 2.0])
    tracker.update([-float("inf"), 0.0, 0.0])
    assert tracker.get_coverage_count() == 0  # Невалідні координати відфільтровано

    # 3. Сумісність з різними типами вводу
    import torch

    p_list = [1.2, 0.4, 0.8]
    p_tuple = (1.2, 0.4, 0.8)
    p_np = np.array([1.2, 0.4, 0.8], dtype=np.float32)
    p_torch = torch.tensor([1.2, 0.4, 0.8], dtype=torch.float32)

    tracker.update(p_list)
    tracker.update(p_tuple)
    tracker.update(p_np)
    tracker.update(p_torch)

    # Усі 4 формати вказують на одну й ту ж точку, тому це строго 1 воксель
    assert tracker.get_coverage_count() == 1


def test_voxel_tracker_reset():
    """
    Тест на скидання стану трекера.
    """
    tracker = VoxelTracker(voxel_size=0.5)
    tracker.update([1.0, 2.0, 3.0])
    assert tracker.get_coverage_count() == 1

    tracker.reset()
    assert tracker.get_coverage_count() == 0
    assert tracker.get_coverage_volume() == 0.0


def test_exploration_bonus_gating_suicide_vs_hover():
    """
    Тест на інваріант: бонус за дослідження простору НІКОЛИ не повинен перекривати
    штраф за аварію (Варіант 2: survival_ratio ** 3 + cap).
    
    Модель А ("Камікадзе"): 50 унікальних вокселів, але краш на 25% часу польоту.
    Модель Б ("Обережний ховер"): всього 1 воксель, але 100% виживання.
    
    Очікуваний результат: cost(Камікадзе) >> cost(Ховер).
    """
    # Модель А: краш на survival_ratio = 0.25, покрила 50 вокселів
    cost_kamikaze, bonus_kamikaze = compute_composite_cost(
        survival_ratio=0.25,
        mean_power_w=10.0,
        jitter_pr=0.0,
        saccades_yaw=0,
        coverage_count=50,
    )

    # Модель Б: ідеальне виживання (survival_ratio = 1.0), покрила лише 1 воксель
    cost_hover, bonus_hover = compute_composite_cost(
        survival_ratio=1.0,
        mean_power_w=10.0,
        jitter_pr=0.0,
        saccades_yaw=0,
        coverage_count=1,
    )

    # Бонус камікадзе знецінено за рахунок survival_ratio ** 3
    # 0.25 ** 3 = 0.015625 -> бонус стає мізерним
    assert bonus_kamikaze < 1.0
    # Камікадзе отримує велетенський штраф за краш і суттєво програє обережному польоту
    assert cost_kamikaze > cost_hover + 500.0


# ─────────────────────────────────────────────────────────────────────────────
# TDD-перевірки: 2. Логіка PhysicsTelemetry (Фізичні фільтри / Kill-Switches)
# ─────────────────────────────────────────────────────────────────────────────

def test_impact_kinetic_energy():
    """
    Тест на Кінетичну енергію:
    - Маса: 0.130 кг.
    - Вектор швидкості при краші: [3.0, 4.0, 0.0] (|V| = 5.0 м/с).
    - Очікуваний результат: строго 1.625 Джоуля (E_k = 0.5 * 0.130 * 25).
    - Якщо is_crash == False, повертає строго 0.0 (не було крашу).
    """
    m = 0.130
    v = [3.0, 4.0, 0.0]

    # 1. При краші
    e_k = calculate_impact_energy(mass=m, velocity=v, is_crash=True)
    assert np.isclose(e_k, 1.625), f"Expected 1.625 J, got {e_k}"

    # 2. Якщо успішно долетів (is_crash == False) -> 0.0 Дж
    e_k_safe = calculate_impact_energy(mass=m, velocity=v, is_crash=False)
    assert e_k_safe == 0.0

    # 3. Сумісність з torch.Tensor
    import torch
    v_torch = torch.tensor([3.0, 4.0, 0.0], dtype=torch.float32)
    e_k_torch = calculate_impact_energy(mass=m, velocity=v_torch, is_crash=True)
    assert np.isclose(e_k_torch, 1.625)

    # 4. Безпечна обробка некоректних даних (NaN)
    e_k_nan = calculate_impact_energy(mass=m, velocity=[float("nan"), 0.0, 0.0], is_crash=True)
    assert e_k_nan == 0.0


def test_mean_clearance():
    """
    Тест на Середній кліренс (Mean Clearance):
    - 3 кроки вимірювань ToF із мінімумами [0.5, 0.3, 0.4].
    - Очікуваний результат: mean = (0.5 + 0.3 + 0.4) / 3 = 0.4.
    """
    clearances = [0.5, 0.3, 0.4]
    mean_c = calculate_mean_clearance(clearances)
    assert np.isclose(mean_c, 0.4), f"Expected 0.4, got {mean_c}"

    # Порожня історія повинна повертати 0.0 без ділення на 0
    assert calculate_mean_clearance([]) == 0.0

    # Фільтрація NaN
    assert np.isclose(calculate_mean_clearance([0.5, float("nan"), 0.3]), 0.4)


def test_forward_ratio_normalized_cosine():
    """
    Тест на Коефіцієнт цілеспрямованості (нормалізований косинус):
    r_t = V_x / (math.hypot(V_x, V_y) + 1e-6)
    - [1.0, 0.0] -> 1.0 (ідеально вперед)
    - [0.0, 1.0] -> 0.0 (строго боком, краб)
    - [-1.0, 0.0] -> -1.0 (строго назад)
    - Кут 60 градусів (Vx = 0.5, Vy = sqrt(3)/2 ~ 0.866) -> ~0.5
    - Фільтр ховеру: швидкість < 0.05 м/с повертає None (ігнорується)
    """
    # 1. Ідеально вперед
    r_fwd = calculate_forward_ratio(1.0, 0.0)
    assert np.isclose(r_fwd, 1.0, atol=1e-4)

    # 2. Строго боком (краб)
    r_side = calculate_forward_ratio(0.0, 1.0)
    assert np.isclose(r_side, 0.0, atol=1e-4)

    # 3. Строго назад
    r_back = calculate_forward_ratio(-1.0, 0.0)
    assert np.isclose(r_back, -1.0, atol=1e-4)

    # 4. Кут 60 градусів (cos(60 deg) = 0.5)
    r_60 = calculate_forward_ratio(0.5, np.sqrt(3.0) / 2.0)
    assert np.isclose(r_60, 0.5, atol=1e-3)

    # 5. Фільтр ховеру (дрон практично стоїть на місці: 0.01 м/с < 0.05 м/с)
    assert calculate_forward_ratio(0.01, 0.01, min_speed=0.05) is None


def test_physics_telemetry_ideal_vs_crab_vs_saccade():
    """
    Тест на Політ крабом (Forward Ratio Median):
    1. Траєкторія 1 (Ідеальна): 50 кроків вперед (Vx=1.0, Vy=0.0) -> проходження валідації.
    2. Траєкторія 2 (Краб): 50 кроків боком (Vx=0.0, Vy=1.0) -> вибраковка (is_crab_flight == True).
    3. Траєкторія 3 (Сакади): 45 кроків вперед, 5 кроків ухилу вбік -> медіана ~ 1.0 -> НЕ вибраковується!
    """
    tracker_ideal = PhysicsTelemetryTracker(threshold_crab=0.5)
    tracker_crab = PhysicsTelemetryTracker(threshold_crab=0.5)
    tracker_saccade = PhysicsTelemetryTracker(threshold_crab=0.5)

    # 1. Ідеальний політ
    for _ in range(50):
        tracker_ideal.update_step(obs_tof=np.ones((8, 8)), vel=[1.0, 0.0, 0.0])
    res_ideal = tracker_ideal.compute_summary()
    assert res_ideal["is_crab_flight"] is False
    assert np.isclose(res_ideal["forward_ratio_median"], 1.0, atol=1e-3)

    # 2. Краб
    for _ in range(50):
        tracker_crab.update_step(obs_tof=np.ones((8, 8)), vel=[0.0, 1.0, 0.0])
    res_crab = tracker_crab.compute_summary()
    assert res_crab["is_crab_flight"] is True
    assert np.isclose(res_crab["forward_ratio_median"], 0.0, atol=1e-3)

    # 3. Сакади при обльоті перешкод (робастність медіани)
    for _ in range(45):
        tracker_saccade.update_step(obs_tof=np.ones((8, 8)), vel=[1.0, 0.0, 0.0])
    for _ in range(5):
        tracker_saccade.update_step(obs_tof=np.ones((8, 8)), vel=[0.0, 1.0, 0.0])
    res_saccade = tracker_saccade.compute_summary()
    assert res_saccade["is_crab_flight"] is False
    assert np.isclose(res_saccade["forward_ratio_median"], 1.0, atol=1e-3)


def test_physics_telemetry_tracker_lifecycle():
    """
    Тест життєвого циклу трекера телеметрії (update -> crash -> summary -> reset).
    """
    tracker = PhysicsTelemetryTracker(min_speed_hover=0.05, threshold_crab=0.5, default_mass=0.130)

    # Крок 1-3: політ з кліренсами 0.6, 0.4, 0.2
    tracker.update_step(obs_tof=np.array([0.6, 0.9]), vel=[1.0, 0.2, 0.0])
    tracker.update_step(obs_tof=np.array([0.4, 0.7]), vel=[1.0, 0.1, 0.0])
    tracker.update_step(obs_tof=np.array([0.2, 0.5]), vel=[1.0, 0.0, 0.0])

    # Реєстрація крашу зі швидкістю [2.0, 0.0, 0.0] (|V|^2 = 4)
    # E_k = 0.5 * 0.130 * 4 = 0.26 Дж
    e_k = tracker.register_crash(final_vel=[2.0, 0.0, 0.0], mass=0.130)
    assert np.isclose(e_k, 0.26)

    summary = tracker.compute_summary()
    assert np.isclose(summary["mean_clearance"], 0.4)
    assert np.isclose(summary["impact_energy_j"], 0.26)
    assert summary["is_crash"] is True
    assert summary["is_crab_flight"] is False
    assert summary["steps_tracked"] == 3

    # Перевірка reset
    tracker.reset()
    clean_summary = tracker.compute_summary()
    assert clean_summary["mean_clearance"] == 0.0
    assert clean_summary["impact_energy_j"] == 0.0
    assert clean_summary["is_crash"] is False
    assert clean_summary["steps_tracked"] == 0


def test_simulate_policy_rollout_with_telemetry():
    """
    Інтеграційний TDD-тест: симуляція польоту через DroneSimulationEnv
    з повним збором PhysicsTelemetry та VoxelTracker.
    1. Політика з рухом вперед (ForwardPolicy) успішно проходить валідацію.
    2. Політика з польотом боком (CrabPolicy) вибраковується кіл-світчем.
    """
    from optimizer.evaluate import simulate_policy_rollout, DroneSimulationEnv

    class ForwardPolicy:
        def step_np(self, obs_flow, obs_tof, dt=None):
            # Тангаж вперед (Pitch=1550), утримання висоти
            return np.array([1500.0, 1500.0, 1550.0, 1500.0], dtype=np.float32)

    class CrabPolicy:
        def step_np(self, obs_flow, obs_tof, dt=None):
            # Крен боком (Roll=1700), політ крабом
            return np.array([1500.0, 1700.0, 1500.0, 1500.0], dtype=np.float32)

    # 1. Тест успішного польоту вперед
    env_fwd = DroneSimulationEnv(dt=0.004)
    cost_fwd, metrics_fwd = simulate_policy_rollout(ForwardPolicy(), env=env_fwd, eval_steps=100)

    assert cost_fwd >= 0.0
    assert "mean_clearance" in metrics_fwd
    assert "impact_energy_j" in metrics_fwd
    assert "forward_ratio_median" in metrics_fwd
    assert "coverage_count" in metrics_fwd
    assert "coverage_bonus" in metrics_fwd
    assert "memory_sectors" in metrics_fwd
    assert len(metrics_fwd["memory_sectors"]) == 8
    assert metrics_fwd["fatal_failure"] is False
    assert metrics_fwd["is_crab_flight"] is False
    assert metrics_fwd["forward_ratio_median"] > 0.50

    # 2. Тест вибраковки політу крабом
    env_crab = DroneSimulationEnv(dt=0.004)
    cost_crab, metrics_crab = simulate_policy_rollout(CrabPolicy(), env=env_crab, eval_steps=100)

    assert cost_crab == 9999.0
    assert metrics_crab["fatal_failure"] is True
    assert "Crab Flight" in metrics_crab["failure_reason"]
    assert metrics_crab["is_crab_flight"] is True
    assert "memory_sectors" in metrics_crab

    # 3. Тест передачі 8-D кільцевого буфера в політку
    class MemoryAwarePolicy:
        def __init__(self):
            self.received_memory_len = None

        def step_np(self, obs_flow, obs_tof, memory_ring=None, dt=None):
            if memory_ring is not None:
                self.received_memory_len = len(memory_ring)
            return np.array([1500.0, 1500.0, 1550.0, 1500.0], dtype=np.float32)

    pol_mem = MemoryAwarePolicy()
    env_mem = DroneSimulationEnv(dt=0.004)
    simulate_policy_rollout(pol_mem, env=env_mem, eval_steps=5)
    assert pol_mem.received_memory_len == 8