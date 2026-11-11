import numpy as np
import pytest

from simulation.sitl_rc_units import reference_axes_to_actual, actual_rate_dps


def test_inverse_preserves_legacy_angle_and_yaw_targets_over_operating_range():
    for offset in (-500,-275,-140,-50,-1,0,1,50,140,275,500):
        raw=np.array([1213.,1500+offset,1500-offset,1500+offset])
        mapped=reference_axes_to_actual(raw)
        assert mapped[0]==raw[0]
        rates=actual_rate_dps((mapped[1:]-1500)/500)
        np.testing.assert_allclose(rates[:2]/670*60,[offset/500*35,-offset/500*35],atol=1e-12)
        assert rates[2]==pytest.approx(offset/500*200,abs=1e-12)


def test_native_rc_has_different_meaning_and_mapping_does_not_clip_silently():
    assert actual_rate_dps(.1)==pytest.approx(13.)
    assert 60*actual_rate_dps(.1)/670==pytest.approx(1.1641791)
    with pytest.raises(ValueError):reference_axes_to_actual([1213,2100,1500,1500])


def test_adapter_rejects_other_firmware_profile(tmp_path):
    import json
    from pathlib import Path
    from simulation.sitl_rc_units import validate_virtual_profile
    profile=Path('configs/sitl_virtual_v1.cli')
    params=json.loads(Path('configs/ready_sim_reference_v1.json').read_text())['uncertainty_study']['nominal_dynamics']
    validate_virtual_profile(profile,params)
    changed=tmp_path/'profile.cli'
    changed.write_text(profile.read_text().replace('set yaw_srate = 67','set yaw_srate = 70'))
    with pytest.raises(ValueError,match='ACTUAL'):validate_virtual_profile(changed,params)
