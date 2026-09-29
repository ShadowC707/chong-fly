"""
tests/test_behavior_cloning.py
==============================
TDD Verification Suite for Behavioral Cloning Pipeline:
1. ExpertReflexPolicy (straight flight, active braking, state latching, stochasticity).
2. generate_reflex_dataset (sequence shapes, PWM bounds, non-degeneracy).
3. pretrain_policy (loss decrease, bagging diversity, biological mask enforcement).
"""

import math
import os
import sys
import numpy as np
import pytest
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from generator.generate_reflex_dataset import ExpertReflexPolicy, generate_reflex_dataset
from optimizer.pretrain import pretrain_policy
from configs.flight_config import (
    PWM_MIN,
    PWM_MID,
    PWM_MAX,
    PWM_HOVER,
    PWM_LEVEL_ROLL,
    PWM_CRUISE_PITCH,
    PWM_NEUTRAL_YAW,
    DEFAULT_DT,
)


def test_expert_policy_straight_flight():
    """
    Тест на чистий політ: коли відстань спереду >= 0.8 м,
    експерт видає Pitch ≈ 1600 (рух вперед) та Yaw ≈ 1500 (стабілізація курсу).
    """
    expert = ExpertReflexPolicy(distance_threshold_m=0.8, noise_std_pwm=0.0)

    # 74-D спостереження: спереду чисто (1.0)
    obs = np.ones(74, dtype=np.float32)
    obs[0:2] = 0.0  # нульовий потік
    obs[2:66] = 1.0  # ToF чистий
    obs[66] = 1.0   # Sector 0 чистий

    pwm = expert.step(obs)

    # [Throttle, Roll, Pitch, Yaw]
    assert np.isclose(pwm[0], PWM_HOVER)
    assert np.isclose(pwm[1], PWM_LEVEL_ROLL)
    assert np.isclose(pwm[2], PWM_CRUISE_PITCH), f"Expected forward pitch {PWM_CRUISE_PITCH}, got {pwm[2]}"
    assert np.isclose(pwm[3], PWM_NEUTRAL_YAW), f"Expected neutral yaw {PWM_NEUTRAL_YAW}, got {pwm[3]}"


def test_expert_policy_apf_smoothness():
    """
    Тест на неперервність (smoothness) Штучних Потенційних Піль (APF).
    При плавному наближенні до стіни Pitch має змінюватися диференційовано (плавно),
    а не стрибком.
    """
    expert = ExpertReflexPolicy(distance_threshold_m=0.8, noise_std_pwm=0.0)

    # Імітуємо плавне наближення від 1.0 (чисто) до 0.1 (краш)
    distances = np.linspace(1.0, 0.1, 50)
    pitches = []

    for d in distances:
        obs = np.ones(74, dtype=np.float32)
        obs[2:66] = d  # Фронтальна стіна
        pwm = expert.step(obs)
        pitches.append(pwm[2])

    pitches = np.array(pitches)
    diffs = np.abs(np.diff(pitches))

    # Максимальний стрибок не повинен бути "сходинкою" (у старій версії був стрибок 300)
    max_jump = np.max(diffs)
    assert max_jump < 50.0, f"Expected smooth pitch changes, but found jump of {max_jump} PWM"
    
    # Pitch має зменшуватися (гальмувати сильніше) в міру наближення стіни
    # Допускаємо невеликі похибки обчислень (< 1e-5), але тренд має бути вниз
    assert np.all(np.diff(pitches) <= 1e-5), "Pitch should monotonically decrease (brake harder) as obstacle gets closer"


def test_expert_policy_apf_directional_repulsion():
    """
    Тест на просторове відштовхування (Directional Repulsion).
    Якщо стіна знаходиться більше зліва — експерт має плавно повертати вправо (Yaw > 1500).
    Якщо справа — вліво (Yaw < 1500).
    """
    expert = ExpertReflexPolicy(distance_threshold_m=0.8, noise_std_pwm=0.0)

    # 1. Перешкода зліва (ToF сітка 8x8, ліва половина: стовпці 0..3)
    obs_left = np.ones(74, dtype=np.float32)
    tof_left = np.ones((8, 8))
    tof_left[:, 0:4] = 0.2  # Близько
    obs_left[2:66] = tof_left.flatten()
    
    pwm_left = expert.step(obs_left)
    assert pwm_left[3] > 1550.0, f"Obstacle on left, expected Yaw > 1550 (turn right), got {pwm_left[3]}"

    # 2. Перешкода справа (стовпці 4..7)
    obs_right = np.ones(74, dtype=np.float32)
    tof_right = np.ones((8, 8))
    tof_right[:, 4:8] = 0.2
    obs_right[2:66] = tof_right.flatten()

    pwm_right = expert.step(obs_right)
    assert pwm_right[3] < 1450.0, f"Obstacle on right, expected Yaw < 1450 (turn left), got {pwm_right[3]}"


def test_expert_policy_apf_symmetric_braking():
    """
    Тест на симетричне гальмування (APF Local Minimum).
    Якщо перешкода ідеально симетрична по центру (ToF = 0.15),
    вектори відштовхування зліва і справа компенсують один одного.
    Yaw має залишатися біля 1500, а Pitch жорстко гальмувати.
    """
    expert = ExpertReflexPolicy(distance_threshold_m=0.8, noise_std_pwm=0.0)
    
    obs_wall = np.ones(74, dtype=np.float32)
    obs_wall[2:66] = 0.15  # Плоска стіна
    
    pwm = expert.step(obs_wall)
    
    # Pitch гальмує
    assert pwm[2] < 1400.0, f"Expected strong braking (Pitch < 1400), got {pwm[2]}"
    # Yaw залишається нейтральним (+- невеликі похибки обчислень)
    assert np.isclose(pwm[3], 1500.0, atol=1.0), f"Expected straight braking (Yaw ~ 1500), got {pwm[3]}"


def test_generate_dataset_structure_and_bounds():
    """
    Тест генерації датасету:
    - Розмірності тензорів X [N, T, 74] та Y [N, T, 4].
    - Діапазон ШІМ у [1000, 2000].
    - Відсутність NaN / Inf.
    """
    data = generate_reflex_dataset(
        num_episodes=4,
        seq_len=25,
        dt=0.004,
        noise_std_pwm=10.0,
        seed=42,
    )

    X = data["X"]
    Y = data["Y"]

    assert isinstance(X, torch.Tensor)
    assert isinstance(Y, torch.Tensor)
    assert X.shape == (4, 25, 74)
    assert Y.shape == (4, 25, 4)

    # Немає NaN або Inf
    assert not torch.isnan(X).any()
    assert not torch.isnan(Y).any()
    assert not torch.isinf(X).any()
    assert not torch.isinf(Y).any()

    # Межі ШІМ
    assert (Y >= 1000.0).all()
    assert (Y <= 2000.0).all()


def test_pretrain_policy_loss_decreases():
    """
    Тест швидкого попереднього навчання (Pre-training):
    - Створює модель із рекурентною динамікою.
    - Тренує 3 епохи.
    - Перевіряє зниження помилки MSE (loss_final < loss_initial).
    """
    class TrainableDummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(74, 4)
            # Ініціалізація біля нейтрального ШІМ (1500)
            nn.init.constant_(self.linear.bias, 1500.0)
            self.post_step_called = False

        def forward(self, x):
            # x shape: [B, T, 74] -> [B, T, 4]
            return self.linear(x)

        def post_step(self):
            self.post_step_called = True

    model = TrainableDummyModel()

    # Створюємо синтетичний датасет
    dataset = generate_reflex_dataset(num_episodes=6, seq_len=30, seed=777)

    trained_model = pretrain_policy(
        policy=model,
        dataset_path=dataset,
        epochs=3,
        subset_ratio=1.0,
        lr=0.02,
        seed=42,
    )

    info = getattr(trained_model, "_pretrain_info", None)
    assert info is not None
    assert len(info["loss_history"]) == 3
    # Перевіряємо, що loss зменшився
    assert info["loss_history"][-1] < info["loss_history"][0]
    # Перевіряємо виклик post_step для збереження біологічної розрідженості
    assert model.post_step_called is True


def test_pretrain_policy_bagging_subsets():
    """
    Тест беггінгу: випадковий відбір 70% даних.
    Різні seed відбирають різні підвибірки.
    """
    data = generate_reflex_dataset(num_episodes=10, seq_len=10, seed=1)

    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))
        def forward(self, x):
            return torch.full((x.shape[0], x.shape[1], 4), 1500.0) + self.p

    m1 = Dummy()
    m2 = Dummy()

    pretrain_policy(m1, dataset_path=data, epochs=1, subset_ratio=0.7, seed=10)
    pretrain_policy(m2, dataset_path=data, epochs=1, subset_ratio=0.7, seed=99)

    info1 = getattr(m1, "_pretrain_info")
    info2 = getattr(m2, "_pretrain_info")

    assert info1["subset_size"] == 7
    assert info2["subset_size"] == 7
    # Різні seed вибирають різні набори епізодів
    assert info1["subset_indices"] != info2["subset_indices"]


def test_optuna_objective_end_to_end():
    """
    End-to-End інтеграційний тест:
    Optuna trial запускає objective() з швидким навчанням (1 епоха)
    та симуляцією польоту (20 кроків).
    Перевіряється, що trial успішно завершується, повертає числову оцінку,
    і записує користувацькі атрибути телеметрії.
    """
    import optuna
    from optimizer.evaluate import objective

    # Компактний датасет для швидкого тесту
    dataset = generate_reflex_dataset(num_episodes=4, seq_len=15, seed=123)

    study = optuna.create_study(directions=["minimize", "minimize", "maximize"])

    study.optimize(
        lambda trial: objective(
            trial=trial,
            dataset_path=dataset,
            pretrain=True,
            pretrain_epochs=1,
            subset_ratio=0.7,
            eval_steps=20,
            seed=42,
        ),
        n_trials=1,
    )

    assert len(study.trials) == 1
    trial = study.trials[0]
    assert trial.state == optuna.trial.TrialState.COMPLETE
    assert isinstance(trial.values, (list, tuple))
    assert len(trial.values) == 3
    assert not any(math.isnan(v) for v in trial.values)
    assert "mean_clearance" in trial.user_attrs
    assert "is_crab_flight" in trial.user_attrs
    assert "fatal_failure" in trial.user_attrs


