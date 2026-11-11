"""Versioned recordings. Legacy observations remain explicitly unavailable."""
from bisect import bisect_right
import gzip
import json
import math
from pathlib import Path

SCHEMA = 'chong-fly-demo-v1'
STAGES = ('network_pwm', 'coordinated_pwm', 'supervised_pwm', 'issued_pwm', 'accepted_pwm')


def plain(value):
    if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if hasattr(value, 'tolist'): return plain(value.tolist())
    return value


def validate_recording(data):
    try:
        if data['schema'] != SCHEMA: raise ValueError('unsupported schema')
        if not isinstance(data['manifest'], dict): raise ValueError('missing manifest')
        rows = data['frames']
        if not isinstance(rows, list) or not rows: raise ValueError('empty frames')
        previous = -1.
        for row in rows:
            t = row['time_s']
            if type(t) not in (int, float) or not math.isfinite(t) or t <= previous:
                raise ValueError('frame times must increase')
            previous = t
            p = row['position_m']
            if len(p) != 3 or not all(math.isfinite(x) for x in p):
                raise ValueError('invalid position')
            for pwm in [row.get('manual_pwm'), row.get('plant_input_pwm'), row.get('fc_rc_pwm'),
                        *(row['trace'].get(s) for s in STAGES)]:
                if pwm is not None and (len(pwm) != 4 or
                    not all(math.isfinite(x) and 1000 <= x <= 2000 for x in pwm)):
                    raise ValueError('invalid PWM')
            sensors = row.get('sensors')
            if sensors is not None:
                for name, size in [('tof',64), ('flow',2), ('altitude',3), ('attitude',2)]:
                    m = sensors[name]
                    if len(m['value']) != size or not all(math.isfinite(x) for x in m['value']):
                        raise ValueError('invalid sensor payload')
                    if type(m['valid']) is not bool or not 0 <= m['timestamp_s'] <= t+1e-8:
                        raise ValueError('invalid sensor time/validity')
                if any(not 0 <= x <= 1 for x in sensors['tof']['value']):
                    raise ValueError('invalid normalized ToF')
        json.dumps(data, allow_nan=False)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid recording: '+str(exc)) from exc
    return data


def legacy_recording(data):
    if data.get('version') not in ('standalone-mvp-viewer-v1', 'standalone-mvp-viewer-v2'):
        raise ValueError('Invalid recording: unsupported schema/version')
    controls = data['trace']['control']; times = [r['time_s'] for r in controls]
    frames = []
    for p in data['trace']['physics']:
        i = bisect_right(times, p['time_s'])-1
        c = controls[i] if i >= 0 else {}
        frames.append(dict(time_s=p['time_s'], position_m=p['position_m'],
            attitude_rad=[None,None,p.get('yaw_rad')], velocity_mps=None,
            body_velocity_mps=p.get('body_velocity_mps'), clearance_m=p.get('clearance_m'),
            sensors=None, flow_xy=c.get('flow_xy'), min_tof_m=c.get('min_tof_m'),
            manual_pwm=None, plant_input_pwm=c.get('rc_pwm'), plant_source='recorded_direct',
            authority='UNAVAILABLE', request=None, reason='legacy: no receiver',
            trace={s: c.get(s) if s == 'network_pwm' else None for s in STAGES},
            collision=None, events=[]))
    return dict(schema=SCHEMA, manifest=dict(kind='legacy_replay',
        original={k:v for k,v in data.items() if k != 'trace'}, geometry=data.get('geometry',{}),
        candidate=data.get('candidate'), target_height_m=data.get('target_height_m'),
        limitations=['Legacy: full ToF, roll/pitch, receiver and stage evidence unavailable.']),
        frames=frames, events=[], input_schedule=[])


def load_recording(path):
    try:
        path = Path(path)
        raw = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
        data = json.loads(raw)
        if 'schema' not in data: data = legacy_recording(data)
        return validate_recording(data)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError('Cannot load recording: '+str(exc)) from exc


def save_recording(data, path):
    validate_recording(data)
    raw = json.dumps(plain(data), ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
    path = Path(path)
    if path.suffix == '.gz': raw = gzip.compress(raw, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f: f.write(raw)


def replay_frame(recording, time_s):
    rows = recording['frames']
    index = max(0, min(len(rows)-1, bisect_right([r['time_s'] for r in rows], time_s)-1))
    return rows[index]
