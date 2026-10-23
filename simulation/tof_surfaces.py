"""Analytic surfaces for STATIC ToF research, not flight/collision geometry.

Rays and surfaces share one Cartesian frame, with distances in metres.
No return is capped at max_range. No material, optical, or noise model is implied.
"""
from dataclasses import dataclass

import numpy as np

GEOMETRY_VERSION = 'analytic-static-surfaces-v1'


def _vector(value):
    x = np.asarray(value, dtype=np.float64)
    if x.shape != (3,) or not np.isfinite(x).all():
        raise ValueError('Expected a finite 3-vector')
    return tuple(float(v) for v in x)


def _positive(value):
    if not np.isfinite(value) or value <= 0:
        raise ValueError('Expected a finite positive length')


@dataclass(frozen=True)
class Sphere:
    center: tuple
    radius: float

    def __post_init__(self):
        object.__setattr__(self, 'center', _vector(self.center))
        _positive(self.radius)


@dataclass(frozen=True)
class PlanePatch:
    """Two-sided rectangle: center +/- half_width*u +/- half_height*v."""
    center: tuple
    u: tuple
    v: tuple
    half_width: float
    half_height: float

    def __post_init__(self):
        for name in ('center', 'u', 'v'):
            object.__setattr__(self, name, _vector(getattr(self, name)))
        _positive(self.half_width)
        _positive(self.half_height)
        if (not np.isclose(np.linalg.norm(self.u), 1., rtol=0, atol=1e-8)
                or not np.isclose(np.linalg.norm(self.v), 1., rtol=0, atol=1e-8)
                or abs(np.dot(self.u, self.v)) > 1e-8):
            raise ValueError('Patch axes must be orthonormal')


def cast_surfaces(origin, directions, surfaces, max_range=3.5):
    """Nearest positive slant range for each unit ray, excluding t<=1e-9.

    Sphere rays starting inside use the exit root. Parallel/coplanar plane rays
    have no isolated intersection. Rectangle boundaries are inclusive.
    """
    origin = np.array(_vector(origin))
    rays = np.asarray(directions, dtype=np.float64)
    _positive(max_range)
    if (rays.ndim != 2 or rays.shape[1] != 3 or not np.isfinite(rays).all()
            or not np.allclose(np.linalg.norm(rays, axis=1), 1., rtol=0, atol=1e-6)):
        raise ValueError('Expected finite unit rays shaped (N, 3)')
    result = np.full(len(rays), float(max_range))
    for surface in surfaces:
        if isinstance(surface, Sphere):
            offset = origin - surface.center
            a = np.sum(rays*rays, axis=1)
            b = np.sum(rays*offset, axis=1)
            c = float(np.dot(offset, offset) - surface.radius**2)
            disc = b*b - a*c
            # Only absorb roundoff near tangency, not genuinely missed spheres.
            tolerance = 1e-12*np.maximum(1., np.maximum(b*b, abs(a*c)))
            hit = disc >= -tolerance
            root = np.sqrt(np.maximum(disc, 0.))
            near, far = (-b-root)/a, (-b+root)/a
            distance = np.where(near > 1e-9, near, far)
            hit &= distance > 1e-9
        elif isinstance(surface, PlanePatch):
            normal = np.cross(surface.u, surface.v)
            denominator = np.sum(rays*normal, axis=1)
            nonparallel = np.abs(denominator) > 1e-12
            distance = np.dot(np.array(surface.center)-origin, normal)/np.where(nonparallel, denominator, 1.)
            local = origin + distance[:, None]*rays - surface.center
            hit = (nonparallel & (distance > 1e-9)
                   & (np.abs(np.sum(local*surface.u, axis=1)) <= surface.half_width+1e-10)
                   & (np.abs(np.sum(local*surface.v, axis=1)) <= surface.half_height+1e-10))
        else:
            raise TypeError('Unsupported analytic surface')
        result = np.where(hit, np.minimum(result, distance), result)
    return result
