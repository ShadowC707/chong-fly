import numpy as np
import pytest


def test_scan_request_is_rejected_if_braking_did_not_settle_motion():
    from scripts.scan_probe import ProbeConfig, ScanProbePolicy
    policy = ScanProbePolicy(ProbeConfig())
    for _ in range(251):  # 2 s move + 3 s brake + first scan command
        action = policy.step_np(np.array([.2, 0]), np.ones(64), dt=.02)
    assert policy.blocked_reason and policy.phase == 'blocked'
    assert action[3] == 1500 and action[2] < 1500


def test_settled_scan_retains_flow_damping_and_resumes_afterwards():
    from scripts.scan_probe import ProbeConfig, ScanProbePolicy
    policy = ScanProbePolicy(ProbeConfig())
    for _ in range(250):
        policy.step_np(np.zeros(2), np.ones(64), dt=.02)
    action = policy.step_np(np.array([.01, -.02]), np.ones(64), dt=.02)
    assert policy.phase == 'scan' and action[3] > 1500
    assert action[2] < 1500 and action[1] < 1500
    for _ in range(100):
        action = policy.step_np(np.zeros(2), np.ones(64), dt=.02)
    assert policy.phase == 'resume' and action[2] > 1500 and action[3] == 1500


def test_metrics_measure_max_excursion_not_only_final_displacement_and_unwrap_yaw():
    from scripts.scan_probe import phase_metrics
    def row(t, phase, x, yaw):
        return dict(time_s=t, phase=phase, x_m=x, y_m=0., z_m=1.,
                    yaw_rad=np.deg2rad(yaw), horizontal_speed_mps=.1, flow_norm=0.)
    rows = [row(0,'scan',0,179), row(1,'scan',.3,-180), row(2,'resume',0,-179),
            row(3,'resume',1,-170)]
    metrics = phase_metrics(rows)
    assert metrics['scan']['max_drift_m'] == pytest.approx(.3)
    assert metrics['scan']['yaw_change_deg'] == pytest.approx(-2.)  # report uses right-positive yaw
    assert metrics['scan']['duration_s'] == 2


def test_absent_scan_metrics_are_missing_not_zero_success():
    from scripts.scan_probe import phase_metrics
    assert phase_metrics([]) == {}


@pytest.mark.parametrize('settings', [dict(scan_yaw_offset_pwm=float('nan')),
    dict(scan_yaw_offset_pwm=400), dict(flow_gain_pwm=-1)])
def test_invalid_probe_settings_are_rejected(settings):
    from scripts.scan_probe import ProbeConfig
    with pytest.raises(ValueError):
        ProbeConfig(**settings)


def test_real_probe_logs_applied_throttle_and_all_phases():
    from scripts.scan_probe import ProbeConfig, run_probe
    report, rows = run_probe(ProbeConfig(), seed=42)
    assert report['rollout']['horizon_completed'] and not report['rollout']['crashed']
    assert set(report['phases']) == {'move','brake','scan','resume'}
    assert rows[-1]['time_s'] == pytest.approx(10.)
    assert any(abs(r['throttle_pwm']-1213.) > 1 for r in rows)
    assert report['phases']['scan']['duration_s'] == pytest.approx(2.)
