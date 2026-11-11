import numpy as np

from evaluate.diagnose_maneuver_failures import box_ranges,nearest_surface,nearest_point
from simulation.drone_env import RoomBoundaries,BoxObstacle,CylinderObstacle


def test_box_probe_distinguishes_front_hit_parallel_miss_and_backward():
    box=BoxObstacle(1,2,-.5,.5,0,2)
    distances=box_ranges(np.array([0.,0.,1.]),np.array([[1.,0.,0.],[0.,1.,0.],[-1.,0.,0.]]),box)
    assert distances[0]==1.
    assert np.isinf(distances[1:]).all()


def test_near_wall_event_is_not_attributed_to_a_distant_box():
    pos=np.array([3.75,0.,1.]);room=RoomBoundaries(-4,4,-4,4,0,3)
    kind,index,target=nearest_surface(pos,room,[BoxObstacle(0,1,1,2,0,2)],[])
    assert kind=='wall'
    np.testing.assert_array_equal(nearest_point(pos,kind,target),[4.,0.,1.])


def test_box_corner_and_cylinder_surface_are_compared_geometrically():
    room=RoomBoundaries(-4,4,-4,4,0,3)
    pos=np.array([0.,0.,1.]);box=BoxObstacle(.15,.4,.15,.4,0,2)
    kind,_,obj=nearest_surface(pos,room,[box],[])
    assert kind=='box'
    assert np.isclose(np.linalg.norm(nearest_point(pos,kind,obj)-pos),np.sqrt(2)*.15)
    kind,_,obj=nearest_surface(pos,room,[box],[CylinderObstacle(.3,0,.2,2)])
    assert kind=='cylinder'
    np.testing.assert_allclose(nearest_point(pos,kind,obj),[.1,0.,1.])
