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

from simulation.metrics import calculate_jitter_pr, calculate_saccades_yaw


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