import copy

import pytest
import torch


def test_scaling_is_copy_only_and_preserves_graph_and_neural_time_constants(tiny_model_meta):
    from simulation.policy import ChongFlyMSPPolicy
    from optimizer.probe_recurrent_dynamics import scaled_policy
    model = ChongFlyMSPPolicy.from_meta(str(tiny_model_meta), preserve_signs=True)
    original = copy.deepcopy(model.state_dict())
    changed = scaled_policy(model, tau_scale=.5, recurrent_scale=.75)
    torch.testing.assert_close(changed.cfc_network.cell.tau, model.cfc_network.cell.tau*.5)
    torch.testing.assert_close(changed.cfc_network.cell._effective_W(),
                               model.cfc_network.cell._effective_W()*.75)
    assert torch.equal(changed.cfc_network.cell._synapse_mask, model.cfc_network.cell._synapse_mask)
    assert torch.equal(changed.cfc_network.cell._synapse_sign, model.cfc_network.cell._synapse_sign)
    for key, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)
    identity = scaled_policy(model, tau_scale=1., recurrent_scale=1.)
    x = torch.ones(1, 5, 74)
    torch.testing.assert_close(identity(x)[0], model(x)[0], rtol=0, atol=0)


@pytest.mark.parametrize('tau,gain', [(0, 1), (1, 0), (float('nan'), 1), (1, -1), (.00001, 1)])
def test_invalid_or_below_floor_scaling_is_rejected(tiny_model_meta, tau, gain):
    from simulation.policy import ChongFlyMSPPolicy
    from optimizer.probe_recurrent_dynamics import scaled_policy
    with pytest.raises(ValueError):
        scaled_policy(ChongFlyMSPPolicy.from_meta(str(tiny_model_meta)),
                      tau_scale=tau, recurrent_scale=gain)


def test_selection_rejects_fast_but_weak_or_wrong_turns():
    from optimizer.probe_recurrent_dynamics import summarize_probe
    from optimizer.search_pipeline import GATE
    row = dict(active_columns=4, direction_correct=True, onset_s=.1,
               recovery_settle_s=.4, clear_abs_yaw_pwm=0,
               threat_yaw_pwm=1850, threat_target_yaw_pwm=1850)
    good = summarize_probe({'cases': [row]}, GATE)
    assert good['broad_joint_passed'] == 1
    weak = summarize_probe({'cases': [{**row, 'threat_yaw_pwm': 1510}]}, GATE)
    assert weak['recovery_passed'] == 1 and weak['broad_joint_passed'] == 0
    wrong = summarize_probe({'cases': [{**row, 'direction_correct': False}]}, GATE)
    assert wrong['broad_joint_passed'] == 0


def test_search_refuses_overwriting_results_before_reading_inputs(tmp_path):
    from optimizer.search_recurrent_dynamics import run_search
    destination = tmp_path/'existing'
    destination.mkdir()
    sentinel = destination/'keep.txt'
    sentinel.write_text('original')
    with pytest.raises(FileExistsError):
        run_search(tmp_path/'missing', tmp_path/'missing.pt', tmp_path/'missing.json', destination)
    assert sentinel.read_text() == 'original'


@pytest.mark.parametrize('change', ['parent', 'gate', 'no_shortlist'])
def test_search_requires_matching_unchanged_screen_before_loading_data(tmp_path, change):
    import json
    from optimizer.search_recurrent_dynamics import run_search
    from optimizer.search_pipeline import GATE
    parent = tmp_path/'parent'
    parent.mkdir()
    (parent/'preflight.json').write_text(json.dumps({'experiment_fingerprint': 'parent'}))
    screen = dict(parent_experiment_fingerprint='parent', gate=dict(GATE),
                  candidates=[{'shortlisted': [{'tau_scale': .5, 'recurrent_scale': .75}]}])
    if change == 'parent':
        screen['parent_experiment_fingerprint'] = 'different'
    elif change == 'gate':
        screen['gate']['recovery_max_s'] = 10
    else:
        screen['candidates'][0]['shortlisted'] = []
    path = tmp_path/'screen.json'
    path.write_text(json.dumps(screen))
    with pytest.raises(ValueError, match='matching sensitivity'):
        run_search(parent, tmp_path/'unread.pt', path, tmp_path/'out')
    assert not (tmp_path/'out').exists()
