"""Demo geometry only; all rays, contacts and dynamics use existing primitives."""
from dataclasses import asdict
import math

import numpy as np

from simulation.benchmark_scenes import NavigationScene
from simulation.drone_env import DroneSimulationEnv, RoomBoundaries, BoxObstacle, CylinderObstacle

LEGACY_SCENES=('clear','obstacle_left','obstacle_right','obstacle_center')
ROOM_SCENES=('room','random_room','wall_room')
SCENES=LEGACY_SCENES+ROOM_SCENES
VERSION='demo-room-geometry-v1'
DENSITY_VERSION='demo-room-density-v2'
DENSITY_GAP_M=.28


def _footprint(obstacle):
    if isinstance(obstacle,BoxObstacle):
        return obstacle.x_min,obstacle.x_max,obstacle.y_min,obstacle.y_max
    return (obstacle.center_x-obstacle.radius,obstacle.center_x+obstacle.radius,
            obstacle.center_y-obstacle.radius,obstacle.center_y+obstacle.radius)


def _gap(a,b):
    return math.hypot(max(0.,a[0]-b[1],b[0]-a[1]),max(0.,a[2]-b[3],b[2]-a[3]))


def _area(item):
    if isinstance(item,BoxObstacle):
        return (item.x_max-item.x_min)*(item.y_max-item.y_min)
    return math.pi*item.radius**2


def _density_layout(seed,start,density,style,boxes,cylinders,footprints,gap):
    """Fill irregular, separated cells in clustered or shuffled order.

    Unlike random sequential placement, free-cell subdivision does not jam at
    the same sparse coverage for every high density setting. Actual coverage
    remains explicit when a custom wall or start pose reduces available space.
    """
    rng=np.random.default_rng(seed)
    target=64.*density/100
    filled=0.
    centers=rng.uniform(-2.8,2.8,size=(int(rng.integers(3,6)),2))
    cells=[(-4+gap,4-gap,-4+gap,4-gap)]
    exclusions=[(start[0]-1.35,start[0]+1.35,start[1]-1.35,start[1]+1.35)]
    exclusions.extend((a-gap,b+gap,c-gap,d+gap)
                      for a,b,c,d in footprints)
    for ex in exclusions:
        free=[]
        for a,b,c,d in cells:
            left,right=max(a,ex[0]),min(b,ex[1])
            bottom,top=max(c,ex[2]),min(d,ex[3])
            if left>=right or bottom>=top:
                free.append((a,b,c,d)); continue
            free.extend([(a,left,c,d),(right,b,c,d),(left,right,c,bottom),(left,right,top,d)])
        cells=[v for v in free if v[1]-v[0]>.45 and v[3]-v[2]>.45]
    pending=cells; cells=[]
    while pending:
        cell=pending.pop(); a,b,c,d=cell
        width,depth=b-a,d-c
        if max(width,depth)>rng.uniform(1.6,2.5):
            if width>=depth:
                cut=a+width*rng.uniform(.35,.65)
                pending.extend([(a,cut,c,d),(cut,b,c,d)])
            else:
                cut=c+depth*rng.uniform(.35,.65)
                pending.extend([(a,b,c,cut),(a,b,cut,d)])
        else:
            inset=gap/2
            cell=(a+inset,b-inset,c+inset,d-inset)
            if min(cell[1]-cell[0],cell[3]-cell[2])>=.18: cells.append(cell)
    def score(cell):
        center=np.array([(cell[0]+cell[1])/2,(cell[2]+cell[3])/2])
        cluster_distance=float(np.min(np.linalg.norm(centers-center,axis=1)))
        return cluster_distance+rng.uniform(-.7,.7) if style=='clustered' else rng.random()
    cells=sorted(cells,key=score)
    capacities=[(b-a)*(d-c) for a,b,c,d in cells]
    future=sum(capacities)
    accepted=0
    for (a,b,c,d),capacity in zip(cells,capacities):
        remaining=target-filled
        if remaining<=.064: break  # at most 0.1 percentage point below target
        future-=capacity
        x,y=(a+b)/2,(c+d)/2
        dx,dy=b-a,d-c
        height=float(rng.uniform(1.2,2.7))
        kind=int(rng.integers(4))
        min_required=min(capacity,max(0.,remaining-future))
        if kind==0:
            radius=min(dx,dy)/2*rng.uniform(.75,1.)
            radius=min(radius,math.sqrt(remaining/math.pi))
            if math.pi*radius**2+1e-9>=min_required:
                item=CylinderObstacle(float(x),float(y),float(radius),height)
            else: kind=2
        if kind!=0:
            # A narrow partition or a broad block, with independent proportions.
            if kind==1:
                if rng.integers(2): dx*=rng.uniform(.2,.4)
                else: dy*=rng.uniform(.2,.4)
            else:
                dx*=rng.uniform(.65,1.); dy*=rng.uniform(.65,1.)
            desired=min(remaining,max(dx*dy,min_required))
            if desired>dx*dy:
                dx,dy=b-a,d-c
            shrink=math.sqrt(desired/(dx*dy))
            dx*=shrink; dy*=shrink
            item=BoxObstacle(float(x-dx/2),float(x+dx/2),float(y-dy/2),float(y+dy/2),0.,height)
        fp=_footprint(item)
        footprints.append(fp)
        (boxes if isinstance(item,BoxObstacle) else cylinders).append(item)
        filled+=_area(item); accepted+=1
    return dict(requested_density_pct=density,actual_density_pct=100*filled/64.,
                density_basis='sum of random obstacle XY footprint area / 64 m2 room floor; custom wall excluded',
                target_reached=target-filled<=.064,random_obstacle_count=accepted,
                available_cell_count=len(cells),layout_style=style,
                cluster_centers_m=centers.tolist() if style=='clustered' else [])


def room_geometry(scene, *, seed, start, options=None, collision_radius_m=.1):
    if scene not in ROOM_SCENES: raise ValueError('Unknown room scene')
    if options is not None and not isinstance(options,dict): raise ValueError('Invalid scene options')
    options=dict(options or {})
    defaults=dict(obstacle_count=10,wall_enabled=False,wall_x=1.5,wall_y=0.,
                  wall_length_m=3.,wall_axis='y')
    if set(options)-set(defaults)-{'obstacle_density_pct','layout_style'}: raise ValueError('Unknown scene options')
    if 'obstacle_count' in options and 'obstacle_density_pct' in options:
        raise ValueError('Choose density or legacy obstacle count, not both')
    density=options.get('obstacle_density_pct',20.)
    style=options.get('layout_style','clustered')
    if type(density) not in (int,float) or not math.isfinite(density) or not 0<=density<=35:
        raise ValueError('Obstacle density must be from 0 to 35 percent')
    if style not in ('clustered','uniform'): raise ValueError('Invalid obstacle layout style')
    density_mode=scene=='random_room' and 'obstacle_count' not in options
    settings=dict(defaults,**options)
    count=settings['obstacle_count']
    if type(count) is not int or not 0<=count<=18:
        raise ValueError('Obstacle count must be an integer from 0 to 18')
    if type(settings['wall_enabled']) is not bool or settings['wall_axis'] not in ('x','y'):
        raise ValueError('Invalid wall orientation/enabled flag')
    if any(type(settings[k]) not in (int,float) or not math.isfinite(settings[k])
           for k in ('wall_x','wall_y','wall_length_m')):
        raise ValueError('Invalid wall dimensions')
    room=RoomBoundaries(-4.,4.,-4.,4.,0.,3.)
    boxes,cylinders,footprints=[],[],[]
    if scene=='wall_room' or settings['wall_enabled']:
        x,y,length=(settings[k] for k in ('wall_x','wall_y','wall_length_m'))
        if not .3<=length<=7.5: raise ValueError('Invalid wall length: expected 0.3–7.5 m')
        dx,dy=(length,.15) if settings['wall_axis']=='x' else (.15,length)
        wall=BoxObstacle(x-dx/2,x+dx/2,y-dy/2,y+dy/2,0.,2.7)
        if not (-4<=wall.x_min<wall.x_max<=4 and -4<=wall.y_min<wall.y_max<=4):
            raise ValueError('The wall must fit inside the room; geometry is not clipped')
        boxes.append(wall); footprints.append(_footprint(wall))
    density_info=None
    density_gap=max(DENSITY_GAP_M,2*collision_radius_m+.08)
    if density_mode:
        settings.pop('obstacle_count')
        settings.update(obstacle_density_pct=density,layout_style=style)
        density_info=_density_layout(seed,start,density,style,boxes,cylinders,footprints,density_gap)
    elif scene=='random_room':
        rng=np.random.default_rng(seed)  # independent of the environment sensor RNG
        spawn=(start[0],start[0],start[1],start[1])
        for i in range(count):
            for _ in range(10000):
                x,y=rng.uniform(-3.1,3.1,size=2)
                height=float(rng.uniform(1.3,2.6))
                if i%3==0:
                    item=CylinderObstacle(float(x),float(y),float(rng.uniform(.18,.4)),height)
                else:
                    length,width=float(rng.uniform(.55,1.35)),float(rng.uniform(.18,.5))
                    dx,dy=(length,width) if i%3==1 else (width,length)
                    item=BoxObstacle(float(x-dx/2),float(x+dx/2),float(y-dy/2),float(y+dy/2),0.,height)
                fp=_footprint(item)
                if min(fp[0]+4,4-fp[1],fp[2]+4,4-fp[3])<.5: continue
                if _gap(fp,spawn)<1.35 or any(_gap(fp,old)<.55 for old in footprints): continue
                footprints.append(fp)
                (boxes if isinstance(item,BoxObstacle) else cylinders).append(item)
                break
            else:
                raise ValueError('Cannot fit this obstacle count with the clearance constraints; use fewer or another seed')
    version=('demo-room-density-v3' if density_gap>DENSITY_GAP_M else DENSITY_VERSION) if density_mode else VERSION
    details=dict(version=version,scene=scene,seed=seed,settings=settings,
        actual_obstacle_count=len(boxes)+len(cylinders),room=asdict(room),
        primitive_orientation='axis-aligned boxes (0/90 degrees) and vertical cylinders',
        minimum_random_surface_gap_m=density_gap if density_mode else .55,minimum_random_spawn_xy_clearance_m=1.35,
        random_layout_is_not_a_navigation_success_guarantee=True)
    if density_info: details.update(density_info)
    return room, boxes, cylinders, details


def make_environment(scene, *, seed, start, options=None, collision_radius_m=.1):
    if scene not in SCENES: raise ValueError('Unknown demo scene')
    if scene in LEGACY_SCENES:
        if options or (options is not None and not isinstance(options,dict)):
            raise ValueError('Scene options require a room scene')
        env=NavigationScene(scene)
        env.collision_radius_m=float(collision_radius_m)
        return env
    room, boxes, cylinders, details = room_geometry(
        scene, seed=seed, start=start, options=options, collision_radius_m=collision_radius_m)
    env=DroneSimulationEnv(engine='standalone',headless=True,dt=.004,room=room,collision_radius_m=collision_radius_m)
    env.boxes,env.cylinders=boxes,cylinders
    env.demo_geometry_spec=details
    return env
