import pytest
from evaluate.describe_xy_passages import crossing_events


def rows(points):
    return [dict(time_s=t,position_m=[x,y,1.]) for t,x,y in points]


def test_gate_crossing_interpolates_time_and_does_not_count_twice_at_vertex():
    r=crossing_events(rows([(0,0,0),(1,1,1),(2,2,2)]),[0,2],[2,0])
    assert len(r)==1
    assert r[0]['time_s']==pytest.approx(1.)
    assert r[0]['xy_m']==pytest.approx([1,1])


def test_infinite_line_crossing_outside_gate_or_during_stop_is_not_passage():
    assert crossing_events(rows([(0,3,-2),(1,3,2)]),[0,2],[2,0])==[]
    assert crossing_events(rows([(35,0,0),(36,2,2)]),[0,2],[2,0])==[]
