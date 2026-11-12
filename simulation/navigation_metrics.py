"""Offline navigation metrics shared by Standalone and SITL evaluations."""
import numpy as np


def repeated_motion_windows(t, xy):
    """Find 30 s windows of >1.5 m travel inside a box at most .8 m wide per axis."""
    hits = []
    for end in np.arange(30., float(t[-1])+1e-8, 5.):
        window = xy[(t >= end-30.-1e-8) & (t <= end+1e-8)]
        span = np.ptp(window, axis=0)
        path = np.linalg.norm(np.diff(window, axis=0), axis=1).sum()
        if np.max(span) <= .8 and path > 1.5:
            hits.append(dict(start_s=float(end-30), end_s=float(end), path_m=float(path), span_m=span.tolist()))
    return hits


def waiting_metrics(details, dt=.02):
    """Measure waits independently of contact and position coverage."""
    total = longest = current = 0.0
    blocked = False
    for row in details:
        decision = row['decision']
        if decision.get('phased_phase') in ('WAIT_CLEAR', 'FAILED'):
            total += dt
            current += dt
            longest = max(longest, current)
        else:
            current = 0.0
        blocked = blocked or decision.get('navigation_blocked', False)
    return dict(wait_or_failed_s=total, longest_wait_or_failed_s=longest,
                navigation_blocked=blocked, unresolved_wait=longest > 5.+1e-9 or blocked)


def progress_metrics(time, xy, samples, end_s):
    """Count first visits to 25 cm centre-position cells, including the final gap."""
    cells = {}
    for index in samples:
        cell = tuple(np.floor(xy[index]/.25).astype(int))
        cells.setdefault(cell, index)
    visits = np.array(list(cells.values()))
    gaps = np.diff(np.r_[time[visits], end_s])
    longest = int(np.argmax(gaps))
    start = int(visits[longest])
    end = int(visits[longest+1]) if longest+1 < len(visits) else len(time)-1
    path = float(np.linalg.norm(np.diff(xy[start:end+1], axis=0), axis=1).sum())
    return dict(visited_xy_cells_25cm=len(cells),
                new_cells_last_60s=int(np.count_nonzero(time[visits] >= end_s-60.)) if end_s >= 60. else None,
                longest_no_new_cell=dict(start_s=float(time[start]), duration_s=float(gaps[longest]), path_m=path),
                repeat_motion_windows=repeated_motion_windows(time[samples], xy[samples]))


# Trace layout of the Standalone stand: action_order is throttle, roll, pitch, yaw.
# Throttle belongs to the altitude controller and is not part of network_share.
NAVIGATION_COLUMNS = (1, 2, 3)
NETWORK_TOLERANCE_PWM = 10.
GENERATES = 'generates'


def network_share(network, coordinated, analyzed=None, tolerance_pwm=NETWORK_TOLERANCE_PWM):
    """Share of ticks where roll, pitch and yaw all stay within tolerance of the network command.

    `network` and `coordinated` are (ticks, 4) PWM arrays in action order. `analyzed` is a boolean
    mask of ticks to count (AUTO, without requested STOP); None counts every tick. A NaN inside
    the analysed ticks is an error: the caller must decide which ticks belong to the analysis.
    """
    network, coordinated = np.asarray(network, float), np.asarray(coordinated, float)
    if network.shape != coordinated.shape or network.ndim != 2 or network.shape[1] != 4:
        raise ValueError('network and coordinated must both have shape (ticks, 4)')
    mask = np.ones(len(network), bool) if analyzed is None else np.asarray(analyzed, bool)
    if mask.shape != (len(network),):
        raise ValueError('analyzed must have one flag per tick')
    if not mask.any():
        return None
    columns = list(NAVIGATION_COLUMNS)
    network, coordinated = network[mask][:, columns], coordinated[mask][:, columns]
    if np.isnan(network).any() or np.isnan(coordinated).any():
        raise ValueError('NaN command inside analysed ticks')
    return float(np.mean(np.all(np.abs(coordinated-network) <= tolerance_pwm, axis=1)))


def trace_network_share(trace):
    """network_share of a saved trace.npz (a mapping with network, coordinated and optionally more).

    New traces carry explicit `auto` and `stop_requested` arrays: analysed ticks are AUTO ticks
    without requested STOP. Old traces have neither; by agreement (plan step 1) the analysis then
    starts at the first tick whose `accepted` command is not NaN, which stands for entering AUTO.
    The old live-room traces contain no STOP, so nothing is excluded for it.
    """
    network = np.asarray(trace['network'], float)
    if 'auto' in trace and 'stop_requested' in trace:
        analyzed = np.asarray(trace['auto'], bool) & ~np.asarray(trace['stop_requested'], bool)
    else:
        accepted = ~np.isnan(np.asarray(trace['accepted'], float)).any(axis=1)
        analyzed = np.arange(len(network)) >= int(np.argmax(accepted)) if accepted.any() \
            else np.zeros(len(network), bool)
    return network_share(network, trace['coordinated'], analyzed)


def generated_share(stage_log, analyzed=None):
    """Share of ticks where at least one active stage of kind 'generates' issued its own command.

    `stage_log` has one entry per tick: a list of records with at least `kind` and `active`.
    """
    mask = np.ones(len(stage_log), bool) if analyzed is None else np.asarray(analyzed, bool)
    if mask.shape != (len(stage_log),):
        raise ValueError('analyzed must have one flag per tick')
    if not mask.any():
        return None
    hits = [any(_generated(r, records) for r in records) for records, keep in zip(stage_log, mask) if keep]
    return float(np.mean(hits))


def _generated(record, records, tolerance_pwm=NETWORK_TOLERANCE_PWM):
    """Author's rule (decision after round 4): an active 'generates' stage counts only when its OWN
    contribution (after - before) exceeds the network_share tolerance on roll, pitch or yaw.
    Records without before/after (older logs) count whenever active."""
    if record['kind'] != GENERATES or not record['active']:
        return False
    if record.get('before') is None or record.get('after') is None or records[0].get('before') is None:
        return True
    before, after = np.asarray(record['before'], float), np.asarray(record['after'], float)
    columns = list(NAVIGATION_COLUMNS)
    return bool(np.any(np.abs(after[columns]-before[columns]) > tolerance_pwm))


def generated_active_share(stage_log, analyzed=None):
    """Raw share of ticks with any active 'generates' stage, whatever the size of its command."""
    mask = np.ones(len(stage_log), bool) if analyzed is None else np.asarray(analyzed, bool)
    if not mask.any():
        return None
    return float(np.mean([any(r['kind'] == GENERATES and r['active'] for r in records)
                          for records, keep in zip(stage_log, mask) if keep]))


def share_metrics(network, coordinated, auto, stop_requested, stage_log=None):
    """Report fields for the stage-1 acceptance group; STOP ticks are excluded and counted apart."""
    auto, stop = np.asarray(auto, bool), np.asarray(stop_requested, bool)
    analyzed = auto & ~stop
    return dict(network_share=network_share(network, coordinated, analyzed),
                generated_share=None if stage_log is None else generated_share(stage_log, analyzed),
                generated_active_share=None if stage_log is None else generated_active_share(stage_log, analyzed),
                share_analyzed_ticks=int(analyzed.sum()), share_stop_ticks=int((auto & stop).sum()))


def share_loss(network, coordinated, stage_log, analyzed=None, tolerance_pwm=NETWORK_TOLERANCE_PWM):
    """Exclusive attribution of the network_share loss, as shares of the analysed ticks.

    For every analysed tick outside tolerance the loss goes to the first stage (in pipeline order)
    after which the command leaves the tolerance band around the network on roll, pitch or yaw; if
    no stage does, to 'xy_base' (turn_latch and the rest of the XY coordinator after the stages).
    The parts sum to 1 - network_share.
    """
    network, coordinated = np.asarray(network, float), np.asarray(coordinated, float)
    mask = np.ones(len(network), bool) if analyzed is None else np.asarray(analyzed, bool)
    if len(stage_log) != len(network) or mask.shape != (len(network),):
        raise ValueError('one stage record list and one flag per tick are required')
    if not mask.any():
        return None
    columns = list(NAVIGATION_COLUMNS)
    counts = {}
    for net, out, records, keep in zip(network, coordinated, stage_log, mask):
        if not keep or np.all(np.abs(out[columns]-net[columns]) <= tolerance_pwm):
            continue
        owner = 'xy_base'
        for r in records:
            if np.any(np.abs(np.asarray(r['after'], float)[columns]-net[columns]) > tolerance_pwm):
                owner = r['stage']
                break
        counts[owner] = counts.get(owner, 0)+1
    return {k: v/int(mask.sum()) for k, v in sorted(counts.items())}


def reverse_outside_corridor(time, position, velocity, yaw, auto, corridor_s=7., tolerance_m=.05,
                             reverse_mps=.03):
    """Author's redefinition (round 3): body-frame rollback counted only outside the own corridor.

    A tick reverses when the forward body speed is below -reverse_mps under AUTO (as the old metric).
    For each contiguous reverse episode starting at t0 the corridor is the set of true positions of
    the last `corridor_s` before t0; reverse distance counts only while the body is farther than
    `tolerance_m` from every corridor position. Returns (outside_m, total_m).
    """
    t, p, v = np.asarray(time, float), np.asarray(position, float), np.asarray(velocity, float)
    yaw, auto = np.asarray(yaw, float), np.asarray(auto, bool)
    dt = np.diff(t, append=t[-1])
    forward = v[:, 0]*np.cos(yaw)+v[:, 1]*np.sin(yaw)
    reverse = auto & (forward < -reverse_mps)
    outside, start, trail = 0., None, None
    for i in np.flatnonzero(reverse):
        if start is None or not reverse[i-1] or i == 0:
            start = i
            keep = (t >= t[i]-corridor_s) & (t <= t[i])
            trail = p[keep][::5] if keep.any() else p[i:i+1]
        if np.min(np.linalg.norm(trail-p[i], axis=1)) > tolerance_m:
            outside += -forward[i]*dt[i]
    return float(outside), float((-forward[reverse]*dt[reverse]).sum())
