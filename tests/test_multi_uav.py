from shapely.geometry import LineString, box

from planner.coverage import Segment
from planner.models import FlightConfig, PlanConfig
from planner.multi_uav import assign_segments, simulate_multi_uav
from planner.transit import TransitRouter


def _segs(lengths):
    return [Segment(i, LineString([(0, 10 * i), (L, 10 * i)]), float(L), i, "forward") for i, L in enumerate(lengths)]


def _router():
    return TransitRouter(box(-10, -10, 500, 200))


def test_deterministic_greedy_assignment():
    groups = assign_segments(_segs([100, 90, 80, 70, 60, 50]), 3)
    assert [[s.id for s in g] for g in groups] == [[0, 5], [1, 4], [2, 3]]
    assert [sum(s.length for s in g) for g in groups] == [150, 150, 150]
    assert groups == assign_segments(_segs([100, 90, 80, 70, 60, 50]), 3)


def test_workload_and_feasibility():
    cfg = PlanConfig(num_uavs=3, flight=FlightConfig(battery_wh=1000))
    res = simulate_multi_uav((0, 0), _segs([100, 90, 80, 70, 60, 50]), cfg, _router())
    assert res.feasible and not res.unassigned_segment_ids
    assert sum(u.coverage_distance_m for u in res.uavs) == 450


def test_per_uav_battery_infeasible():
    cfg = PlanConfig(num_uavs=2, flight=FlightConfig(battery_wh=5.0, wh_per_m=0.05))
    res = simulate_multi_uav((0, 0), _segs([200, 20]), cfg, _router())
    assert not res.feasible and res.infeasible_segment_ids and not res.uavs[0].feasible


def test_more_uavs_than_segments():
    cfg = PlanConfig(num_uavs=4)
    res = simulate_multi_uav((0, 0), _segs([100]), cfg, _router())
    assert sum(1 for u in res.uavs if not u.segments) == 3 and res.feasible