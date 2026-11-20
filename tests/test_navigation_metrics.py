from pathlib import Path

import numpy as np
import pytest

from simulation.navigation_metrics import (generated_share, network_share, share_metrics,
                                           trace_network_share)

TRACE = (Path(__file__).resolve().parents[1]
         / 'output/standalone_navigation_live_regression_v1/phased_side_guard/trace.npz')


def commands(n=10):
    return np.tile([1213., 1500., 1530., 1500.], (n, 1))


def test_identical_commands_give_full_share():
    assert network_share(commands(), commands()) == 1.


def test_throttle_is_not_part_of_the_share():
    coordinated = commands()
    coordinated[:, 0] += 300
    assert network_share(commands(), coordinated) == 1.


@pytest.mark.parametrize('channel', [1, 2, 3])
def test_every_navigation_channel_counts_on_its_own(channel):
    coordinated = commands()
    coordinated[:3, channel] += 25
    assert network_share(commands(), coordinated) == pytest.approx(.7)


def test_tolerance_is_ten_pwm_inclusive():
    coordinated = commands(4)
    coordinated[0, 2] += 10.
    coordinated[1, 2] -= 10.
    coordinated[2, 2] += 10.5
    coordinated[3, 3] -= 11.
    assert network_share(commands(4), coordinated) == .5


def test_ticks_are_counted_only_when_all_three_channels_agree():
    coordinated = commands(4)
    coordinated[0, 1] += 50
    coordinated[1, 2] += 50
    coordinated[2, 3] += 50
    assert network_share(commands(4), coordinated) == .25


def test_mask_restricts_the_analysed_ticks():
    coordinated = commands(10)
    coordinated[:5, 2] += 100
    mask = np.arange(10) >= 5
    assert network_share(commands(), coordinated, mask) == 1.
    assert network_share(commands(), coordinated) == .5
    assert network_share(commands(), coordinated, np.zeros(10, bool)) is None


def test_nan_is_rejected_inside_the_analysis_and_ignored_outside_it():
    network = commands()
    network[:2] = np.nan
    with pytest.raises(ValueError):
        network_share(network, commands())
    assert network_share(network, commands(), np.arange(10) >= 2) == 1.


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        network_share(commands(3), commands(4))
    with pytest.raises(ValueError):
        network_share(commands(3)[:, :3], commands(3)[:, :3])


def record(kind, active):
    return dict(stage='s', kind=kind, active=active, reason='', before=None, after=None)


def test_generated_share_counts_only_active_generating_stages():
    log = [[record('limits', True), record('generates', False)],
           [record('limits', True), record('generates', True)],
           [record('generates', True), record('generates', True)],
           [record('limits', False)]]
    assert generated_share(log) == .5
    assert generated_share(log, [False, True, True, False]) == 1.
    assert generated_share(log, [False] * 4) is None


def test_share_metrics_separates_stop_ticks_and_uses_auto_only():
    coordinated = commands(10)
    coordinated[2, 2] += 100          # not AUTO: ignored
    coordinated[8, 2] += 100          # STOP tick: ignored
    coordinated[5, 2] += 100          # analysed tick that departs from the network
    auto = np.arange(10) >= 3
    stop = np.arange(10) >= 8
    out = share_metrics(commands(), coordinated, auto, stop)
    assert out['network_share'] == pytest.approx(1-1/5)
    assert out['share_analyzed_ticks'] == 5 and out['share_stop_ticks'] == 2
    assert out['generated_share'] is None


def test_share_metrics_reports_generated_share_from_stage_log():
    log = [[record('generates', i % 4 == 0)] for i in range(8)]
    out = share_metrics(commands(8), commands(8), [True]*8, [False]*8, log)
    assert out['generated_share'] == .25 and out['network_share'] == 1.


@pytest.mark.skipif(not TRACE.exists(), reason='recorded phased_side_guard trace is not available')
def test_phased_side_guard_baseline_is_about_073():
    data = np.load(TRACE)
    share = network_share(data['network'], data['coordinated'])
    assert share == pytest.approx(.734, abs=.005)


def test_old_trace_analysis_starts_at_first_accepted_tick():
    coordinated = commands(10)
    coordinated[:3, 2] += 100                       # before AUTO: must not count
    accepted = commands(10)
    accepted[:3] = np.nan
    trace = dict(network=commands(), coordinated=coordinated, accepted=accepted)
    assert trace_network_share(trace) == 1.
    assert network_share(trace['network'], coordinated) == .7


def test_new_trace_uses_explicit_auto_and_stop_arrays():
    coordinated = commands(10)
    coordinated[0, 2] += 100                        # not AUTO
    coordinated[9, 2] += 100                        # STOP
    trace = dict(network=commands(), coordinated=coordinated, accepted=commands(10),
                 auto=np.arange(10) >= 1, stop_requested=np.arange(10) >= 9)
    assert trace_network_share(trace) == 1.


@pytest.mark.skipif(not TRACE.exists(), reason='recorded phased_side_guard trace is not available')
def test_recorded_trace_official_value_is_after_first_accepted_tick():
    assert trace_network_share(np.load(TRACE)) == pytest.approx(.7335, abs=.0005)
