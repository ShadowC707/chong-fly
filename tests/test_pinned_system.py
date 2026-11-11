import hashlib
import json
import numpy as np
import pytest
from tests.test_edge_runtime import exported
from edge.system import RunConfig, build_system
from edge.pinned_system import build_pinned_system
from edge.sensors import SensorSnapshot, Measurement


def test_pinned_runtime_retains_shared_types_and_rejects_other_drift(exported):
    _,bundle=exported
    # Distinct, checksummed runtime source, same numerical implementation.
    with (bundle/'runtime.py').open('a') as f:f.write('\n# historical fixture\n')
    meta=json.loads((bundle/'metadata.json').read_text())
    meta['files']['runtime.py']=hashlib.sha256((bundle/'runtime.py').read_bytes()).hexdigest()
    (bundle/'metadata.json').write_text(json.dumps(meta))
    digest=hashlib.sha256((bundle/'metadata.json').read_bytes()).hexdigest()
    (bundle/'metadata.sha256').write_text(digest)
    c=RunConfig(bundle_dir=str(bundle),expected_metadata_sha256=digest,sensor_mode='multi_rate_v1')
    with pytest.raises(ValueError,match='Executing code'): build_system(c)
    s=build_pinned_system(c)
    assert not (bundle/'__pycache__').exists()
    frame=SensorSnapshot(0,0.,Measurement(0,0.,(0.,0.)),Measurement(0,0.,np.ones(64)),
        Measurement(0,0.,(1.,0.,0.)),Measurement(0,0.,(0.,0.)))
    result=s.tick(frame,now_s=0.,request='SHADOW',manual_pwm=(1213,1500,1500,1500))
    assert result['fault'] is None
    assert s.manifest['executed_modules']['runtime.py']['sha256']==meta['files']['runtime.py']
    with (bundle/'sensors.py').open('a') as f:f.write('\n# changed\n')
    with pytest.raises(ValueError,match='Hash'):build_pinned_system(c)
