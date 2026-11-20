"""Live, display-only summary for the browser: who controls now, running network share, memory points.

Nothing here reaches the controller and nothing is written into recorded frames (recordings and
their verification stay unchanged). Memory points are shown in the controller's own estimate: flow
odometry plus measured heading, anchored at the start position (the heading reference is the world
yaw zero and odometry starts at the start pose), together with the controller's estimate of its own
position. After a memory reset that anchor is unknown to the controller; the points are then placed
with the true pose and flagged.
"""

import numpy as np

TOLERANCE_PWM = 10.  # the network_share tolerance of the plan, section 2
TOF_FOV_DEG = 45.
TOF_RANGE_M = 3.5
MAX_MEMORY_POINTS = 400
MEMORY_RADIUS_M = 3.

STAGE_LABELS = {
    'brake_no_rollback': 'Гальмо без відкату', 'side_veto': 'Бічне вето', 'path_limit': 'Перевірка шляху',
    'path_brake': 'Гальмо перед точкою пам\'яті', 'velocity_hold': 'Утримання швидкості',
    'turn_latch': 'Фіксація повороту', 'anti_rollback': 'Анти-відкат', 'yaw_shaping': 'Згладжування повороту',
    'stop_hover': 'STOP: зависання', 'side_clearance_yaw': 'Бічний захист: обмеження повороту',
    'phased_brake': 'Фаза: гальмування', 'phased_turn': 'Фаза: поворот', 'phased_recover': 'Фаза: відновлення',
    'phased_failed': 'Фаза: FAILED (утримання після таймауту)'}
PHASE_LABELS = {
    'BRAKE': 'Фаза: гальмування', 'TURN': 'Фаза: поворот', 'RECOVER': 'Фаза: відновлення',
    'FAILED': 'Фаза: FAILED (утримання після таймауту)', 'CRITICAL': 'Фаза: критична близькість',
    'WAIT_CLEAR': 'Фаза: очікування вільного фронту', 'STOP': 'STOP'}


def _deviates(a, b):
    return a is not None and b is not None and max(abs(a[i]-b[i]) for i in (1, 2, 3)) > TOLERANCE_PWM


def controller_of(frame, decision):
    """Who sets the R/P/Y command on this frame: the network, a pipeline stage or a phase."""
    if frame is None:
        return dict(kind='idle', label='Очікування')
    if frame.get('plant_source') != 'auto':
        return dict(kind='manual', label='Ручний канал' if frame.get('request') != 'AUTO' else 'Очікування допуску AUTO')
    if frame.get('stop_requested'):
        return dict(kind='stop', label='STOP')
    trace = frame.get('trace') or {}
    network, coordinated = trace.get('network_pwm'), trace.get('coordinated_pwm')
    if network is None or coordinated is None:
        return dict(kind='idle', label='Очікування')
    if not _deviates(network, coordinated):
        return dict(kind='network', label='Мережа')
    decision = decision or {}
    for record in decision.get('stages') or []:
        if _deviates(network, record.get('after')):
            return dict(kind='stage', label=STAGE_LABELS.get(record['stage'], record['stage']), stage=record['stage'],
                        reason=record.get('reason'))
    phase = decision.get('phased_phase') or (frame.get('decision') or {}).get('phase')
    if phase and phase != 'CRUISE':
        return dict(kind='phase', label=PHASE_LABELS.get(phase, 'Фаза: '+phase), phase=phase)
    reasons = [r for r in decision.get('reasons', []) if r not in ('brake_no_rollback', 'side_veto')]
    if reasons:
        return dict(kind='rule', label=STAGE_LABELS.get(reasons[0], reasons[0]), reason=reasons[0])
    return dict(kind='coordinator', label='Координатор')


def running_share(runner):
    """Share of analysed control ticks (AUTO, no STOP) where the network command passed (±10 PWM)."""
    cache = getattr(runner, '_live_share_cache', None)
    if cache is None or cache['rows_id'] != id(runner.rows) or cache['index'] > len(runner.rows):
        cache = dict(rows_id=id(runner.rows), index=0, analysed=0, network=0, last_tick=None)
    for row in runner.rows[cache['index']:]:
        trace = row.get('trace') or {}
        tick = trace.get('tick_time_s', row.get('time_s'))
        if (row.get('plant_source') != 'auto' or row.get('stop_requested') or tick == cache['last_tick']
                or trace.get('network_pwm') is None or trace.get('coordinated_pwm') is None):
            continue
        cache['last_tick'] = tick
        cache['analysed'] += 1
        cache['network'] += not _deviates(trace['network_pwm'], trace['coordinated_pwm'])
    cache['index'] = len(runner.rows)
    runner._live_share_cache = cache
    return (cache['network']/cache['analysed'] if cache['analysed'] else None), cache['analysed']


def memory_points(runner, frame):
    """Remembered hazard points (pipeline profiles only) in the controller's estimate, nearest first.

    Returns (points, estimated_position, frame_name); (None, None, None) without a memory."""
    from edge.survey_memory import rotation
    navigation = getattr(runner.system, 'navigation', None)
    memory = getattr(navigation, 'memory', None)
    if memory is None or not hasattr(memory, 'remembered') or frame is None:
        return None, None, None
    now = getattr(navigation, 'now_s', None)
    remembered = memory.remembered(now) if now is not None else None
    report = getattr(memory, 'report', {}) or {}
    if getattr(memory, 'resets', 0) == 0 and 'heading_rad' in report:
        # odometry origin = start pose, heading reference = world yaw 0
        estimate = np.asarray(runner.start, float)+np.asarray(memory.observer.position, float)
        heading = report['heading_rad']
        to_world = lambda local: estimate+local @ rotation(0., 0., heading).T
        name = 'controller_odometry'
    else:
        estimate = None
        yaw = float(frame['attitude_rad'][2])
        to_world = lambda local: np.asarray(frame['position_m'], float)+local @ rotation(0., 0., yaw).T
        name = 'true_pose_after_memory_reset'
    estimate_out = None if estimate is None else [round(float(v), 3) for v in estimate]
    if remembered is None:
        return [], estimate_out, name
    local = np.asarray(remembered[0], float)
    near = np.linalg.norm(local[:, :2], axis=1) <= MEMORY_RADIUS_M
    local = local[near]
    if len(local):  # thin on a 10 cm grid so the map shows coverage, not one dense surface
        _, keep = np.unique(np.floor(local/.1).astype(np.int64), axis=0, return_index=True)
        local = local[keep]
    local = local[np.argsort(np.linalg.norm(local[:, :2], axis=1))][:MAX_MEMORY_POINTS]
    points = [[round(float(v), 3) for v in p] for p in to_world(local)] if len(local) else []
    return points, estimate_out, name


def live_view(runner):
    frame = runner.rows[-1] if runner.rows else None
    coordinator = getattr(getattr(runner.system, 'runtime', None), 'coordinator', None)
    decision = getattr(coordinator, 'last_decision', None)
    share, ticks = running_share(runner)
    points, estimate, memory_frame = memory_points(runner, frame)
    return dict(controller=controller_of(frame, decision), network_share=share, share_ticks=ticks,
                share_tolerance_pwm=TOLERANCE_PWM, tof_fov_deg=TOF_FOV_DEG, tof_range_m=TOF_RANGE_M,
                memory_points=points, estimated_position=estimate, memory_frame=memory_frame)
