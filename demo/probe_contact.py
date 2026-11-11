"""Replay the saved corner false positive with the demo sphere contact model."""
import json
import sys

from .contacts import contact_and_clearance
from .recording import load_recording,save_recording,plain
from .runner import DemoRunner,DT


def probe(path,recording_path=None):
    data=load_recording(path); m=data['manifest']
    r=DemoRunner(scene=m['scene'],seed=m['seed'],start=m['start']['position_m'],
        yaw=m['start']['yaw_rad'],scene_options=m.get('scene_options'),
        collision_radius_m=m['geometry']['collision_radius_m'],
        max_duration_s=m.get('max_duration_s',180))
    schedule=data['input_schedule']; pending=0
    end=round(data['end_time_s']/DT)
    for step in range(end):
        while pending<len(schedule) and schedule[pending]['step']==step:
            r.event(schedule[pending]['event']); pending+=1
        if r.paused or r.terminal: break
        r.advance()
    old=data['frames'][-1]
    kind,gap=contact_and_clearance(r.env.physics.pos,r.env.physics.pos,r.env.room,
        r.env.boxes,r.env.cylinders,r.env.collision_radius_m)
    comparison=dict(time_s=r.time_s,position_m=plain(r.env.physics.pos),
        position_exactly_matches_old=plain(r.env.physics.pos)==old['position_m'],
        previous_detector=old['collision'],sphere_contact=kind,real_gap_m=gap,
        new_run_terminal=r.terminal)
    r.advance(500)
    result=dict(source=str(path),scene=m['scene'],seed=m['seed'],same_tick=comparison,
        after_continuing=dict(time_s=r.time_s,terminal=r.terminal,
            collision=r.rows[-1]['collision'],clearance_m=r.rows[-1]['clearance_m']))
    if recording_path:
        save_recording(r.recording(),recording_path)
        result['corrected_recording']=str(recording_path)
    r.env.close()
    return result


if __name__=='__main__':
    print(json.dumps(probe(*sys.argv[1:]),ensure_ascii=False,indent=2))
