import json
import hashlib
import numpy as np
import pytest
import torch
from simulation.policy import SensorInputLayer, ChongFlyMSPPolicy
from edge.export_numpy import export_policy
from edge.runtime import NumpyPolicy
from tests.synthetic_graphs import write_tiny_model


def test_release_encoding_preserves_margin_without_changing_v1():
    x=torch.ones(2,74); x[:,:2]=0; x[0,2:66]=.85/3.5; x[1,2:66]=1.05/3.5
    old=SensorInputLayer(encoding='proximity-v1',learnable_scale=False)(x)
    new=SensorInputLayer(encoding='proximity-release-v1',learnable_scale=False)(x)
    assert torch.equal(old[0],old[1])
    torch.testing.assert_close(new[0,2:66],torch.full((64,),.15))
    assert torch.count_nonzero(new[1])==0


def test_release_checkpoint_rejects_old_contract_and_exports_exactly(tmp_path):
    meta=str(write_tiny_model(tmp_path/'graph'))
    kw=dict(mode='masked',preserve_signs=True,dt=.02,tau_init=.025,neutral_origin=True)
    old=ChongFlyMSPPolicy.from_meta(meta,sensor_encoding='proximity-v1',**kw)
    from optimizer.release_encoding import migrate_release_encoding
    transferred=migrate_release_encoding(old)
    assert old.sensor_layer.encoding=='proximity-v1'
    assert transferred.sensor_layer.encoding=='proximity-release-v1'
    for name,param in old.named_parameters():
        assert torch.equal(param,dict(transferred.named_parameters())[name])
    new=ChongFlyMSPPolicy.from_meta(meta,sensor_encoding='proximity-release-v1',**kw)
    with pytest.raises(RuntimeError,match='contract'):
        new.load_state_dict(old.state_dict())
    bundle=tmp_path/'bundle';export_policy(new,bundle,source={'candidate':'synthetic_test_only'})
    runtime=NumpyPolicy(bundle);rng=np.random.default_rng(1)
    for i in range(150):
        flow=rng.uniform(-.1,.1,2).astype(np.float32)
        tof=np.full(64,[.5,.85,.95,1.05,3.5][(i//30)%5]/3.5,np.float32)
        np.testing.assert_allclose(runtime.step_np(flow,tof),new.step_np(flow,tof),atol=.02,rtol=0)
    metadata=json.loads((bundle/'metadata.json').read_text());metadata['proximity_threshold_m']=.8
    (bundle/'metadata.json').write_text(json.dumps(metadata))
    (bundle/'metadata.sha256').write_text(hashlib.sha256((bundle/'metadata.json').read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='contract'):NumpyPolicy(bundle)
