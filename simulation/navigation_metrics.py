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
