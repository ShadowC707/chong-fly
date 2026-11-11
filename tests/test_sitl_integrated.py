import numpy as np
import pytest

from tests.test_edge_runtime import exported
from tests.test_system_assembly import config
from edge.system import build_system
from edge.altitude_control import AltitudeSample
from evaluate.sitl_integrated import IntegratedControl


def make_loop(exported, profile='multi_rate'):
    _, bundle = exported
    system = build_system(config(bundle, sensor_mode='multi_rate_v1',
                                 handover_mode='freeze_sync_v1'))
    return IntegratedControl(system, profile)


def tick(loop, i, **kw):
    return loop.tick(i, (0., 0.), np.ones(64),
                     AltitudeSample(.982, 0., 0., i*.02, True), **kw)


@pytest.mark.parametrize('profile', ['sync', 'multi_rate', 'tof_only', 'altitude_only'])
def test_actual_admission_and_five_command_stages(exported, profile):
    loop = make_loop(exported, profile)
    for i in range(25):
        r = tick(loop, i)
        assert r['trace'].issued_pwm is None
        assert r['selected']['source'] == 'fallback'
    r = tick(loop, 25)
    assert r['trace'].authority == 'AUTO'
    assert r['trace'].issued_pwm == r['manual_pwm']
    assert r['trace'].accepted_pwm == r['trace'].issued_pwm
    assert r['selected']['applied_pwm'] == r['trace'].accepted_pwm
    assert r['trace'].receiver_status == 'accepted'
    assert loop.system.runtime.altitude._integral == 0.


def test_held_range_keeps_capture_attitude(exported):
    loop = make_loop(exported)
    tick(loop, 0)
    r = loop.tick(1, (0., 0.), np.ones(64), AltitudeSample(.99, .1, .12, .02, True))
    assert r['frame'].altitude.value == (.982, 0., 0.)
    assert r['frame'].attitude.value == (.1, .12)
    channels = r['trace'].to_dict()['sensor_channels']
    assert channels['altitude']['state'] == 'held'
    assert channels['altitude']['age_s'] == .02
    assert channels['attitude']['state'] == 'new'


def test_watchdog_without_new_packet_is_latched(exported):
    loop = make_loop(exported)
    for i in range(31): tick(loop, i)
    r = loop.output(.641)
    assert r['accepted_pwm'] is None and r['tripped']
    assert r['reason'] == 'watchdog_expired'
    assert r['applied_pwm'][1:] == (1500., 1500., 1500.)
    r = tick(loop, 33)
    assert r['selected']['tripped']
    assert r['trace'].issued_pwm is None
    with pytest.raises(ValueError, match='MANUAL'):
        loop.receiver.set_mode('AUTO', now_s=.66)
        loop.receiver.reset(session_id='new', now_s=.66)


def test_invalid_measurement_is_not_silently_promoted(exported):
    loop = make_loop(exported)
    with pytest.raises(ValueError):
        loop.tick(0, (0., 0.), np.ones(64), AltitudeSample(.982, 0., 0., 0., False))
    assert loop.receiver.gate.pwm is None


def test_profile_typo_fails_closed(exported):
    with pytest.raises(ValueError, match='profile'):
        make_loop(exported, 'typo')


@pytest.mark.parametrize('profile,counts', [('tof_only', (300, 1000)),
                                          ('altitude_only', (1000, 600))])
def test_single_channel_hold_has_expected_capture_counts_and_angles(profile, counts):
    from simulation.sensor_schedule import SampleSchedule
    schedule = SampleSchedule(profile)
    seen = {k: set() for k in ('tof', 'altitude', 'flow', 'attitude')}
    for i in range(1000):
        t = i*.02
        frame = schedule.frame(i, (0., 0.), np.full(64, i/1000),
            AltitudeSample(.982, t/100, -t/100, t, True))
        for k in seen:
            seen[k].add(getattr(frame, k).sample_id)
        m = frame.altitude
        assert m.value[1:] == (m.timestamp_s/100, -m.timestamp_s/100)
        assert frame.attitude.timestamp_s == t
        if profile == 'tof_only': assert m.timestamp_s == t
        else: assert frame.tof.timestamp_s == t
    assert (len(seen['tof']), len(seen['altitude'])) == counts
    assert len(seen['flow']) == len(seen['attitude']) == 1000
