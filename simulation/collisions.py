"""Conservative swept body-envelope tests for the standalone benchmark.

The envelope radius must be set from the assembled aircraft dimensions. Boxes
and cylinders are expanded on each axis; this overestimates contact near edges.
These tests detect contact, not rigid-body impact dynamics.
"""
import math

import numpy as np


def _slab_interval(start, delta, lower, upper):
    enter, leave = 0.0, 1.0
    for p, d, lo, hi in zip(start, delta, lower, upper):
        if abs(d) < 1e-12:
            if p < lo or p > hi:
                return None
        else:
            a, b = sorted(((lo - p) / d, (hi - p) / d))
            enter, leave = max(enter, a), min(leave, b)
            if enter > leave:
                return None
    return enter, leave


def contact_and_clearance(start, end, room, boxes, cylinders, radius):
    """Return contact kind and endpoint clearance in metres (zero on contact).

    Sweeps the straight centre trajectory over one physics tick. Clearance is
    the distance from the spherical body envelope to the nearest real surface.
    """
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    if not (np.isfinite(start).all() and np.isfinite(end).all()):
        return "invalid_state", 0.0
    delta = end - start
    lower = np.array([room.x_min, room.y_min, room.z_min]) + radius
    upper = np.array([room.x_max, room.y_max, room.z_max]) - radius
    clearance = float(min(np.min(end - lower), np.min(upper - end)))
    if np.any(start <= lower) or np.any(start >= upper) or np.any(end <= lower) or np.any(end >= upper):
        return "room", 0.0
    for box in boxes:
        lo = np.array([box.x_min, box.y_min, box.z_min])
        hi = np.array([box.x_max, box.y_max, box.z_max])
        if _slab_interval(start, delta, lo - radius, hi + radius) is not None:
            return "box", 0.0
        distance = np.linalg.norm(np.maximum(np.maximum(lo - end, end - hi), 0.0))
        clearance = min(clearance, float(distance - radius))
    for cyl in cylinders:
        offset = start[:2] - [cyl.center_x, cyl.center_y]
        a = float(delta[:2] @ delta[:2])
        b = 2.0 * float(offset @ delta[:2])
        c = float(offset @ offset) - (cyl.radius + radius) ** 2
        radial = None
        if a < 1e-24:
            if c <= 0:
                radial = (0.0, 1.0)
        else:
            discriminant = b*b - 4*a*c
            if discriminant >= 0:
                root = math.sqrt(discriminant)
                radial = (max(0.0, (-b-root)/(2*a)), min(1.0, (-b+root)/(2*a)))
        vertical = _slab_interval(start[2:], delta[2:], [-radius], [cyl.height + radius])
        if radial is not None and vertical is not None:
            if max(radial[0], vertical[0]) <= min(radial[1], vertical[1]):
                return "cylinder", 0.0
        radial_distance = max(0.0, np.linalg.norm(end[:2] - [cyl.center_x, cyl.center_y]) - cyl.radius)
        vertical_distance = max(0.0, -end[2], end[2] - cyl.height)
        clearance = min(clearance, math.hypot(radial_distance, vertical_distance) - radius)
    return None, max(0.0, clearance)
