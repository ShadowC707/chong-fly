"""Exact swept sphere contacts for the interactive demo, outside benchmark code.

The existing expanded primitives remain the broad phase and legacy replay
contract. Narrow phase removes their square/cap corner false positives.
"""
import math

import numpy as np

from simulation.collisions import _slab_interval


def _box_distance2(point,lo,hi):
    offset=np.maximum(np.maximum(lo-point,point-hi),0.)
    return float(offset@offset)


def _swept_box_distance2(start,delta,lo,hi):
    # Distance squared to an AABB is quadratic between each slab crossing.
    cuts={0.,1.}
    for p,d,a,b in zip(start,delta,lo,hi):
        if abs(d)>1e-15:
            cuts.update(t for t in ((a-p)/d,(b-p)/d) if 0<t<1)
    cuts=sorted(cuts)
    minimum=min(_box_distance2(start,lo,hi),_box_distance2(start+delta,lo,hi))
    for left,right in zip(cuts,cuts[1:]):
        middle=start+delta*((left+right)/2)
        outside=(middle<lo)|(middle>hi)
        offset=np.where(middle<lo,start-lo,start-hi)[outside]
        velocity=delta[outside]
        a=float(velocity@velocity)
        t=left if a==0 else min(right,max(left,-float(offset@velocity)/a))
        minimum=min(minimum,_box_distance2(start+delta*t,lo,hi))
    return minimum


def _cylinder_distance2_derivative(point,delta,cylinder):
    x,y=point[:2]-[cylinder.center_x,cylinder.center_y]
    radial=math.hypot(x,y)
    dr=max(0.,radial-cylinder.radius)
    dz=float(point[2]-min(cylinder.height,max(0.,point[2])))
    derivative=2*dz*float(delta[2])
    if dr: derivative+=2*dr*(x*delta[0]+y*delta[1])/radial
    return dr*dr+dz*dz,float(derivative)


def _swept_cylinder_distance2(start,delta,cylinder):
    # Squared distance to a convex cylinder is convex along the segment.
    d0,g0=_cylinder_distance2_derivative(start,delta,cylinder)
    d1,g1=_cylinder_distance2_derivative(start+delta,delta,cylinder)
    if g0>=0: return d0
    if g1<=0: return d1
    left,right=0.,1.
    for _ in range(60):
        middle=(left+right)/2
        _,gradient=_cylinder_distance2_derivative(start+delta*middle,delta,cylinder)
        if gradient<0: left=middle
        else: right=middle
    return min(d0,d1,_cylinder_distance2_derivative(start+delta*((left+right)/2),delta,cylinder)[0])


def contact_and_clearance(start,end,room,boxes,cylinders,radius):
    """Continuous sphere sweep; return true contact and endpoint surface gap."""
    start,end=np.asarray(start,dtype=float),np.asarray(end,dtype=float)
    if not (np.isfinite(start).all() and np.isfinite(end).all()):
        return 'invalid_state',0.
    delta=end-start
    lower=np.array([room.x_min,room.y_min,room.z_min])+radius
    upper=np.array([room.x_max,room.y_max,room.z_max])-radius
    clearance=float(min(np.min(end-lower),np.min(upper-end)))
    if np.any(start<=lower) or np.any(start>=upper) or np.any(end<=lower) or np.any(end>=upper):
        return 'room',0.
    threshold=radius*radius+1e-12  # sub-nanometre roundoff, not a visible margin
    for box in boxes:
        lo=np.array([box.x_min,box.y_min,box.z_min])
        hi=np.array([box.x_max,box.y_max,box.z_max])
        if (_slab_interval(start,delta,lo-radius,hi+radius) is not None
                and _swept_box_distance2(start,delta,lo,hi)<=threshold):
            return 'box',0.
        clearance=min(clearance,math.sqrt(_box_distance2(end,lo,hi))-radius)
    for cylinder in cylinders:
        lo=np.array([cylinder.center_x-cylinder.radius,cylinder.center_y-cylinder.radius,0.])
        hi=np.array([cylinder.center_x+cylinder.radius,cylinder.center_y+cylinder.radius,cylinder.height])
        if (_slab_interval(start,delta,lo-radius,hi+radius) is not None
                and _swept_cylinder_distance2(start,delta,cylinder)<=threshold):
            return 'cylinder',0.
        distance2,_=_cylinder_distance2_derivative(end,delta,cylinder)
        clearance=min(clearance,math.sqrt(distance2)-radius)
    return None,max(0.,clearance)


def step_with_contact(env,action):
    """Retain the existing integrator/sensors and refine only contact results."""
    previous=env.physics.pos.copy()
    obs,reward,done,info=env.step(action)
    if info['invalid_state']:
        return obs,reward,done,info
    kind,gap=contact_and_clearance(previous,env.physics.pos,env.room,env.boxes,env.cylinders,env.collision_radius_m)
    crashed=bool(kind is not None or info['excessive_tilt'] or info['invalid_state'])
    reward+=10*(int(info['crashed'])-int(crashed))
    info=dict(info,collision_kind=kind,clearance_m=gap,crashed=crashed)
    env.is_crashed=crashed
    return obs,reward,bool(crashed or info['user_closed']),info
