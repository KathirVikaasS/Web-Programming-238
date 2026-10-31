import math

import pytest
from shapely.geometry import LineString, box

from planner.coverage import Segment
from planner.transit import TransitRouter, plan_transits

REGION = box(0, 0, 100, 100).difference(box(40, 0, 60, 80))
OBST = box(40, 0, 60, 80)


def test_direct_route_valid():
    r = TransitRouter(REGION, OBST).route((10, 90), (90, 90))
    assert r.feasible and r.direct and r.length == pytest.approx(80)


def test_direct_route_through_obstacle_detected():
    router = TransitRouter(REGION, OBST)
    assert router.violates(LineString([(10, 50), (90, 50)]))
    assert not router.edge_valid((10, 50), (90, 50))


def test_obstacle_avoiding_route():
    r = TransitRouter(REGION, OBST).route((10, 50), (90, 50))
    expected = 2 * math.hypot(30, 30) + 20
    assert r.feasible and not r.direct and r.length == pytest.approx(expected, rel=1e-6)
    router = TransitRouter(REGION, OBST)
    assert all(router.edge_valid(p, q) for p, q in zip(r.path, r.path[1:]))


def test_infeasible_transition_when_blocked():
    region = box(0, 0, 100, 100).difference(box(40, 0, 60, 100))
    r = TransitRouter(region, box(40, 0, 60, 100)).route((10, 50), (90, 50))
    assert not r.feasible and r.reason


def test_endpoint_outside_region():
    assert not TransitRouter(REGION, OBST).route((50, 40), (90, 90)).feasible


def test_boundary_contact_is_allowed():
    assert TransitRouter(REGION, OBST).edge_valid((40, 80), (60, 80))


def test_plan_transits_flags_failure():
    region = box(0, 0, 100, 100).difference(box(40, 0, 60, 100))
    segs = [Segment(0, LineString([(10, 50), (20, 50)]), 10, 0, "forward"),
            Segment(1, LineString([(80, 50), (90, 50)]), 10, 1, "forward")]
    plan = plan_transits((10, 10), segs, TransitRouter(region, box(40, 0, 60, 100)))
    assert not plan.feasible and plan.failures