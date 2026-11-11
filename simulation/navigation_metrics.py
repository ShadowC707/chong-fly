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
    hits = [any(r['kind'] == GENERATES and r['active'] for r in records)
            for records, keep in zip(stage_log, mask) if keep]
    return float(np.mean(hits))


def share_metrics(network, coordinated, auto, stop_requested, stage_log=None):
    """Report fields for the stage-1 acceptance group; STOP ticks are excluded and counted apart."""
    auto, stop = np.asarray(auto, bool), np.asarray(stop_requested, bool)
    analyzed = auto & ~stop
    return dict(network_share=network_share(network, coordinated, analyzed),
                generated_share=None if stage_log is None else generated_share(stage_log, analyzed),
                share_analyzed_ticks=int(analyzed.sum()), share_stop_ticks=int((auto & stop).sum()))
