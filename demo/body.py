"""Single shared visual geometry, bounded by the actual existing collision sphere."""
from copy import deepcopy
from functools import lru_cache
import json
import math
from pathlib import Path


@lru_cache(maxsize=1)
def _unit_geometry():
    return json.loads(Path(__file__).with_suffix('.json').read_text(encoding='utf-8'))


def visual_contract(collision_radius_m):
    radius=float(collision_radius_m)
    if not math.isfinite(radius) or radius<=0:
        raise ValueError('Visual collision radius must be positive and finite')
    result=dict(collision_radius_m=radius,unit_geometry=deepcopy(_unit_geometry()))
    if visual_bound(result)>radius:
        raise ValueError('Visible vehicle extends outside its physical collision sphere')
    return result


def visual_bound(contract):
    """Analytic conservative bound including disks and world-scaled stroke width."""
    g=contract['unit_geometry']
    points=g['hull']+g['nose']+[p for arm in g['arms'] for p in arm]
    bound=max(math.sqrt(sum(x*x for x in p)) for p in points)
    for x,y,z in g['rotor_centers']:
        bound=max(bound,math.hypot(math.hypot(x,y)+g['rotor_radius'],z))
    return contract['collision_radius_m']*(bound+g['stroke_width']/2)
