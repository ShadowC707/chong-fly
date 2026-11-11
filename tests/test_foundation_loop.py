"""Two mock ground devices, real loopback sockets, pinned model, C++ app core."""
from edge.foundation_mock import run_foundation_mock
from tests.test_esp32_foundation import foundation_host


def test_two_ground_devices_exchange_canonical_frames_and_only_leased_commands(foundation_host):
    report=run_foundation_mock(foundation_host)
    assert report['hardware_tested'] is False and report['fc_attempts']==0
    assert report['accepted_count']==10
    assert report['wire_sizes']==dict(sensor=733,command=69,receipt=130,challenge=40,manual=67)
    assert report['final_auto_expired'] and report['final_selected_source']=='none'
    assert report['model_bundle']=='data/edge/k306_tau025_integrated_v1'
    assert all(r['accepted_pwm']==r['issued_pwm'] for r in report['accepted'])


def test_notebook_takeover_revokes_pi_and_disconnect_has_no_fabricated_fallback(foundation_host):
    report=run_foundation_mock(foundation_host,takeover=True)
    assert report['takeover_source']=='manual'
    assert report['auto_revoked_by_takeover']
    assert report['final_selected_source']=='none' and report['manual_fault_after_disconnect']
    assert report['fc_attempts']==0
