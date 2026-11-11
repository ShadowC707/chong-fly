import math
from dataclasses import asdict

import pytest

from demo.scenes import make_environment
from simulation.collisions import contact_and_clearance


def footprint(item):
    if hasattr(item,'x_min'):
        return item.x_min,item.x_max,item.y_min,item.y_max
    return item.center_x-item.radius,item.center_x+item.radius,item.center_y-item.radius,item.center_y+item.radius


def occupied_area(env):
    return sum((b.x_max-b.x_min)*(b.y_max-b.y_min) for b in env.boxes)+sum(math.pi*c.radius**2 for c in env.cylinders)


def geometry(env):
    return [asdict(x) for x in env.boxes],[asdict(x) for x in env.cylinders]


@pytest.mark.parametrize('density',[0,8,20,35])
@pytest.mark.parametrize('style',['clustered','uniform'])
def test_density_controls_real_occupied_area_with_safe_geometry(density,style):
    for seed in (0,2,20261021):
        env=make_environment('random_room',seed=seed,start=(-.7,0,1),options={
            'obstacle_density_pct':density,'layout_style':style})
        spec=env.demo_geometry_spec
        actual=occupied_area(env)/64*100
        assert density-.1000001<=actual<=density+1e-9
        assert spec['actual_density_pct']==pytest.approx(actual)
        assert spec['target_reached']
        assert spec['random_obstacle_count']==len(env.boxes)+len(env.cylinders)
        hit,clearance=contact_and_clearance([-.7,0,1],[-.7,0,1],env.room,env.boxes,env.cylinders,.1)
        assert hit is None and clearance>=.8
        bounds=[footprint(o) for o in env.boxes+env.cylinders]
        for i,a in enumerate(bounds):
            assert min(a[0]+4,4-a[1],a[2]+4,4-a[3])>=.28-1e-9
            for b in bounds[:i]:
                gap=math.hypot(max(0,a[0]-b[1],b[0]-a[1]),max(0,a[2]-b[3],b[2]-a[3]))
                assert gap>=.28-1e-9
        env.close()


def test_count_is_automatic_without_eighteen_object_limit_and_patterns_differ():
    options={'obstacle_density_pct':20,'layout_style':'clustered'}
    a=make_environment('random_room',seed=1,start=(-.7,0,1),options=options)
    b=make_environment('random_room',seed=1,start=(-.7,0,1),options=options)
    dense=make_environment('random_room',seed=1,start=(-.7,0,1),options={**options,'obstacle_density_pct':35})
    uniform=make_environment('random_room',seed=1,start=(-.7,0,1),options={**options,'layout_style':'uniform'})
    assert geometry(a)==geometry(b) and geometry(a)!=geometry(uniform)
    assert len(a.boxes)+len(a.cylinders)>18
    assert occupied_area(dense)>1.7*occupied_area(a)
    areas=[(b.x_max-b.x_min)*(b.y_max-b.y_min) for b in a.boxes]
    assert max(areas)>3*min(areas) and a.cylinders
    for env in (a,b,dense,uniform): env.close()


def test_custom_wall_is_excluded_from_density_and_reduces_available_space_honestly():
    env=make_environment('random_room',seed=7,start=(-.7,0,1),options={
        'obstacle_density_pct':35,'wall_enabled':True,'wall_length_m':7.5,'wall_x':1.5})
    spec=env.demo_geometry_spec
    wall=env.boxes[0]
    random_area=occupied_area(env)-(wall.x_max-wall.x_min)*(wall.y_max-wall.y_min)
    assert spec['actual_density_pct']==pytest.approx(random_area/64*100)
    assert spec['target_reached']==(35-spec['actual_density_pct']<=.1+1e-9)
    a=footprint(wall)
    for other in env.boxes[1:]+env.cylinders:
        b=footprint(other)
        assert math.hypot(max(0,a[0]-b[1],b[0]-a[1]),max(0,a[2]-b[3],b[2]-a[3]))>=.28-1e-9
    env.close()


def test_density_recording_recomputes_identical_physics_and_geometry():
    from demo.runner import DemoRunner
    from demo.examples import recompute
    r=DemoRunner(scene='random_room',seed=42,scene_options={
        'obstacle_density_pct':20,'layout_style':'clustered'})
    original=geometry(r.env)
    r.resume(); r.event({'type':'mode','mode':'AUTO'}); r.advance(200)
    data=r.recording(); rebuilt=recompute(data)
    assert data['frames']==rebuilt['frames']
    assert data['manifest']['geometry']==rebuilt['manifest']['geometry']
    r.reset()
    assert geometry(r.env)==original
    r.env.close()


@pytest.mark.parametrize('options',[
    {'obstacle_density_pct':-1},{'obstacle_density_pct':36},
    {'obstacle_density_pct':float('nan')},{'obstacle_density_pct':True},
    {'obstacle_density_pct':'20'},{'obstacle_density_pct':20,'obstacle_count':10},
    {'layout_style':'unknown'},
])
def test_invalid_density_options_are_rejected(options):
    with pytest.raises(ValueError):
        make_environment('random_room',seed=1,start=(-.7,0,1),options=options)
