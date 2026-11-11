import json

import numpy as np
import pytest

from evaluate.render_sitl_replay import replay_data, html_document


def fixture_run(folder):
    state=np.zeros((5,12));state[:,0]=np.arange(5);state[:,2]=1
    np.savez(folder/'trace.npz',time_s=np.arange(5)*.01,state=state,
        control_time=[0.,.02],clearance=[.5,.4,.3,.2],flow=[[1,0],[2,0]],
        tof=np.stack([np.full(64,.4),np.full(64,.8)]),pwm=np.full((4,4),1500.))
    rows=[dict(tick_time_s=t,network_pwm=[1200,1500,1500,1600],
        accepted_pwm=None if t==0 else [1200,1500,1500,1500],
        selected=dict(source='fallback' if t==0 else 'auto',applied_pwm=[1200,1500,1500,1500]),
        sensor_channels=dict(tof=dict(timestamp_s=0.))) for t in (0.,.02)]
    for name,obj in [('manifest',dict(scene={},horizon_s=.04)),('report',dict(completed=True)),
        ('decisions',[dict(time_s=t,phase='CRUISE') for t in (0.,.02)]),('command_trace',rows)]:
        (folder/(name+'.json')).write_text(json.dumps(obj))


def test_replay_preserves_recorded_pose_and_held_sensor_not_fresh_truth(tmp_path):
    fixture_run(tmp_path)
    data=replay_data(tmp_path)
    assert [f['t'] for f in data['frames']]==[0.,.02,.04]
    assert [f['p'][0] for f in data['frames']]==[0.,2.,4.]
    assert data['frames'][0]['trace']['accepted_pwm'] is None
    assert data['frames'][1]['tof']==[.4]*64  # NOT fresh .8 at .02s
    assert data['frames'][2]['clearance']==.2
    assert data['frames'][2]['trace']['tick_time_s']==.02
    assert data['frames'][2]['tof_age_s']==.04


def test_replay_rejects_missing_capture_frame_and_misaligned_commands(tmp_path):
    fixture_run(tmp_path)
    p=tmp_path/'command_trace.json';rows=json.loads(p.read_text())
    rows[1]['sensor_channels']['tof']['timestamp_s']=.01
    p.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='capture'):replay_data(tmp_path)
    rows[1]['sensor_channels']['tof']['timestamp_s']=0.
    rows[1]['tick_time_s']=.03;p.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='command'):replay_data(tmp_path)


def test_embedded_data_cannot_close_script_tag():
    html=html_document(dict(title='</script><script>alert(1)</script>'))
    assert '</script><script>alert' not in html
    assert '\\u003c/script' in html
