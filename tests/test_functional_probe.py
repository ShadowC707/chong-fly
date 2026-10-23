import copy

import numpy as np
import pytest
import torch


def test_trace_matches_normal_policy_and_does_not_modify_it(tiny_model_meta):
    from simulation.policy import ChongFlyMSPPolicy
    from optimizer.functional_probe import trace_response
    from configs.flight_config import CONTROL_DT
    torch.manual_seed(3)
    policy = ChongFlyMSPPolicy.from_meta(str(tiny_model_meta)).eval()
    obs = torch.zeros(2, 12, 74)
    obs[..., 2:] = 1
    obs[1, 4:8, 2:34] = .1
    before = copy.deepcopy(policy.state_dict())
    expected, last = policy(obs, dt=CONTROL_DT)
    pwm, states = trace_response(policy, obs)
    torch.testing.assert_close(torch.from_numpy(pwm), expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.from_numpy(states[:, -1]), last, rtol=0, atol=0)
    for key, value in before.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, policy.state_dict()[key], rtol=0, atol=0)
        else:
            assert value == policy.state_dict()[key]
    assert not policy.cfc_network.cell._forward_hooks


def test_clamp_blocks_known_relay_and_is_removed_afterwards(tiny_model_meta):
    from simulation.policy import ChongFlyMSPPolicy
    from optimizer.functional_probe import trace_response
    policy = ChongFlyMSPPolicy.from_meta(str(tiny_model_meta)).eval()
    cell = policy.cfc_network.cell
    with torch.no_grad():
        cell.W_in.weight.zero_()
        cell.W_in.weight[1, 2] = 1
        cell.W_in.bias.zero_()
        cell.b.zero_()
        policy.dn_head.proj.weight.zero_()
        policy.dn_head.proj.weight[3, 7] = 1
    obs = torch.zeros(1, 30, 74)
    obs[..., 2:] = 1
    intact, _ = trace_response(policy, obs)
    blocked, states = trace_response(policy, obs, clamp_indices=[3])
    assert np.all(states[..., 3] == 0)
    assert intact[0, -1, 3] > blocked[0, -1, 3] + 1
    restored, _ = trace_response(policy, obs)
    np.testing.assert_array_equal(intact, restored)


def test_failed_trace_also_removes_hook(tiny_model_meta):
    from simulation.policy import ChongFlyMSPPolicy
    from optimizer.functional_probe import trace_response
    policy = ChongFlyMSPPolicy.from_meta(str(tiny_model_meta))
    original = policy.dn_head.forward
    def fail(_):
        raise RuntimeError('readout failed')
    policy.dn_head.forward = fail
    with pytest.raises(RuntimeError, match='readout failed'):
        trace_response(policy, torch.zeros(1, 3, 74), clamp_indices=[1])
    assert not policy.cfc_network.cell._forward_hooks
    assert policy.training
    policy.dn_head.forward = original


@pytest.mark.parametrize('indices', [[-1], [8], [1, 1], [True]])
def test_invalid_clamp_is_rejected(tiny_model_meta, indices):
    from simulation.policy import ChongFlyMSPPolicy
    from optimizer.functional_probe import trace_response
    with pytest.raises(ValueError, match='indices'):
        trace_response(ChongFlyMSPPolicy.from_meta(str(tiny_model_meta)),
                       torch.zeros(1, 3, 74), clamp_indices=indices)


def test_stimuli_are_mirrored_with_identical_clear_history():
    from optimizer.functional_probe import make_stimuli
    cases, obs = make_stimuli()
    assert cases[0]['name'] == 'clear'
    assert obs.shape == (9, 175, 74)
    for i in range(1, 9, 2):
        left = obs[i, 50:75, 2:66].reshape(25, 8, 8)
        right = obs[i+1, 50:75, 2:66].reshape(25, 8, 8)
        torch.testing.assert_close(left.flip(-1), right)
    torch.testing.assert_close(obs[:, :50], obs[0:1, :50].expand(9, -1, -1))
    assert torch.all(obs[:, 75:, 2:] == 1)


def test_metrics_subtract_each_interventions_own_clear_response():
    from optimizer.functional_probe import response_metrics
    pwm = np.full((2, 175, 4), 1600.)
    pwm[1, 50:75, 3] += 40
    rows = response_metrics(pwm)
    assert rows[1]['mean_evoked_yaw_pwm'] == 40
    assert rows[1]['clear_mean_yaw_pwm'] == 1600
    assert rows[1]['onset_s'] == 0
    assert rows[1]['recovery_settle_s'] == 0


def test_explicit_research_opt_in_before_file_access(tmp_path):
    from optimizer.functional_probe import run_probe
    with pytest.raises(ValueError, match='research-only'):
        run_probe(tmp_path/'missing', tmp_path/'config', tmp_path/'out')
    assert not (tmp_path/'out').exists()


def group_fixture():
    signatures = [dict(cell_type='target' if i == 0 else 'other',
                       sensor_roles=[], motor_roles=[], side='left', outgoing_sign_profile=1)
                  for i in range(8)]
    signatures[4]['side'] = 'right'
    signatures[5]['outgoing_sign_profile'] = -1
    signatures[6]['sensor_roles'] = ['lc_looming']
    meta = dict(k=8, provenance=dict(cluster_signatures=signatures, cluster_sizes=[1]*7+[2]))
    plan = dict(format_version='functional-probe-plan-v1', groups=[dict(name='target',
                field='cell_type', value='target', purpose='hypothesis')],
                matched_controls_for='target', matched_control_count=3, control_seed=42)
    return meta, plan


def test_controls_are_reproducible_and_match_side_sign_size_without_interfaces():
    from optimizer.functional_probe import resolve_groups
    meta, plan = group_fixture()
    groups = resolve_groups(meta, plan)
    assert resolve_groups(meta, plan) == groups
    assert {tuple(g['indices']) for g in groups[2:]} == {(1,), (2,), (3,)}


def test_missing_annotations_do_not_silently_produce_noop_intervention():
    from optimizer.functional_probe import resolve_groups
    meta, plan = group_fixture()
    plan['groups'][0]['value'] = 'missing'
    with pytest.raises(ValueError, match='no matching'):
        resolve_groups(meta, plan)


def test_controls_fail_if_matching_is_impossible():
    from optimizer.functional_probe import resolve_groups
    meta, plan = group_fixture()
    for s in meta['provenance']['cluster_signatures'][1:]:
        s['side'] = 'right'
    with pytest.raises(ValueError, match='Insufficient'):
        resolve_groups(meta, plan)


def test_no_response_is_not_reported_as_zero_latency():
    from optimizer.functional_probe import response_metrics
    result = response_metrics(np.full((2, 175, 4), 1500.))
    assert result[1]['onset_s'] is None
    assert result[1]['mean_evoked_yaw_pwm'] == 0
