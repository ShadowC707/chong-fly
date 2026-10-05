import pytest

from generator.role_reducer import RolePreservingReducer
from tests.test_role_reduction import graph


def test_audit_distinguishes_source_gap_from_reduction_loss_and_invented_route():
    from generator.audit_routes import audit_routes
    args = graph()
    # Disconnect the looming population from the otherwise directed ring.
    args['signed_weights'][4:7] = 0
    source = RolePreservingReducer(**args)
    model = source(16)
    report = audit_routes(source, [model])
    assert report['models'][0]['required_routes']['lc_looming->yaw'] == 'missing_in_source_or_mapping'
    # Deliberately fabricate a compressed edge: the audit must call this out.
    changed = model.W.tolil()
    changed[model.sensor_index_map['lc_looming'][0], model.motor_index_map['yaw'][0]] = 1
    model.W = changed.tocsr()
    report = audit_routes(source, [model])
    assert report['models'][0]['required_routes']['lc_looming->yaw'] == 'introduced_after_reduction'
    # A graph that had the requested flow route can also lose it on reduction.
    source = RolePreservingReducer(**graph())
    model = source(16)
    model.W.data[:] = 0
    report = audit_routes(source, [model])
    assert report['models'][0]['required_routes']['lptc_flow->pitch'] == 'lost_after_reduction'


def test_audit_refuses_to_compare_artifacts_from_a_different_source():
    from generator.audit_routes import audit_routes
    source = RolePreservingReducer(**graph())
    model = source(16)
    model.provenance['source_sha256'] = 'unrelated'
    with pytest.raises(ValueError, match='source'):
        audit_routes(source, [model])


@pytest.mark.parametrize('k', [128, 256])
def test_current_structured_yaw_is_invariant_to_tof_with_identical_flow_history(k):
    import torch
    from optimizer.evaluate import create_model
    torch.manual_seed(42)
    model = create_model({'k_clusters': k})
    x = torch.ones(2, 12, 74)
    x[..., :2] = 0
    x[1, :, 2:66] = .1
    outputs, _ = model(x)
    torch.testing.assert_close(outputs[0, :, 3], outputs[1, :, 3], rtol=0, atol=0)
    assert model.routing_diagnostics['sensor_motor_paths']['lc_looming']['minimum_hops']['yaw'] is None
