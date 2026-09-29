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
    assert np.isclose(pwm[0], 1500.0)
    assert np.isclose(pwm[1], 1500.0)
    assert np.isclose(pwm[2], 1600.0), f"Expected forward pitch 1600, got {pwm[2]}"
    assert np.isclose(pwm[3], 1500.0), f"Expected neutral yaw 1500, got {pwm[3]}"


def test_expert_policy_braking_and_turning():
    """
    Тест на реакцію на стіну: коли відстань < 0.8 м (0.5 / 3.0 ~ 0.167),
    експерт гальмує (Pitch = 1300) та різко повертає (Yaw = 1900 або 1100).
    """
    expert = ExpertReflexPolicy(distance_threshold_m=0.8, noise_std_pwm=0.0, seed=42)

    # Спереду перешкода 0.5 м -> 0.5 / 3.0 ~ 0.167
    obs = np.ones(74, dtype=np.float32)
    obs[2:66] = 0.167
    obs[66] = 0.167

    pwm = expert.step(obs)

    assert np.isclose(pwm[2], 1300.0), f"Expected braking pitch 1300, got {pwm[2]}"
    assert np.isclose(pwm[3], 1900.0) or np.isclose(pwm[3], 1100.0), (
        f"Expected sharp turn Yaw in {{1100, 1900}}, got {pwm[3]}"
    )


def test_expert_policy_state_latching():
    """
    Тест на уникнення деренчання (State Latching / Hysteresis):
    Під час одного маневру ухилення експерт фіксує напрямок повороту і
    НЕ перемикає його між ліво/право щокроку, доки перешкода не зникне.
    """
    expert = ExpertReflexPolicy(distance_threshold_m=0.8, noise_std_pwm=0.0, seed=123)

    obs_obstacle = np.ones(74, dtype=np.float32)
    obs_obstacle[66] = 0.15  # Стіна

    # Крок 1: виявлено стіну, обирається початковий поворот
    pwm_step1 = expert.step(obs_obstacle)
    latched_yaw = pwm_step1[3]

    # Кроки 2-10: стіна все ще спереду, напрямок повороту МАЄ бути незмінним
    for step in range(2, 11):
        pwm = expert.step(obs_obstacle)
        assert np.isclose(pwm[3], latched_yaw), (
            f"Step {step}: Yaw switched unexpectedly from {latched_yaw} to {pwm[3]} during latched evasion!"
        )

    # Крок 11: перешкоду облетіли, спереду чисто (1.0 м)
    obs_clear = np.ones(74, dtype=np.float32)
    obs_clear[66] = 1.0
    pwm_clear = expert.step(obs_clear)
    assert np.isclose(pwm_clear[3], 1500.0)
    assert expert.latched_turn is None


def test_expert_policy_stochastic_distribution():
    """
    Тест на стохастичність (50/50 ліворуч/праворуч):
    Зі 100 незалежних зустрічей зі стіною приблизно половина поворотів
    здійснюється праворуч (1900), а половина — ліворуч (1100).
    """
    right_turns = 0
    total_encounters = 100

    for i in range(total_encounters):
        expert = ExpertReflexPolicy(distance_threshold_m=0.8, noise_std_pwm=0.0, seed=i)
        obs_obstacle = np.ones(74, dtype=np.float32)
        obs_obstacle[66] = 0.15

        pwm = expert.step(obs_obstacle)
        if np.isclose(pwm[3], 1900.0):
            right_turns += 1

    ratio = right_turns / total_encounters
    assert 0.35 <= ratio <= 0.65, f"Expected 50/50 balance, got {ratio * 100:.1f}% right turns"


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

    study = optuna.create_study(direction="minimize")

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
    assert isinstance(trial.value, float)
    assert not math.isnan(trial.value)
    assert "mean_clearance" in trial.user_attrs
    assert "is_crab_flight" in trial.user_attrs
    assert "fatal_failure" in trial.user_attrs


