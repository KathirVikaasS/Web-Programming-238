import pytest

from planner.battery import assess, energy_required_wh, mission_duration_s, usable_battery_wh, validate_flight
from planner.models import FlightConfig


def test_formulas():
    assert usable_battery_wh(100, 20) == pytest.approx(80)
    assert energy_required_wh(1000, 0.05) == pytest.approx(50)
    assert mission_duration_s(800, 8, 4, 10) == pytest.approx(140)


def test_feasible_and_infeasible():
    f = FlightConfig(battery_wh=100, reserve_pct=20, wh_per_m=0.05)
    assert assess(1000, 5, f).feasible
    r = assess(2000, 5, f)
    assert not r.feasible and not r.battery_ok and r.reasons


def test_reserve_changes_outcome():
    assert assess(1700, 1, FlightConfig(battery_wh=100, reserve_pct=10, wh_per_m=0.05)).battery_ok
    assert not assess(1700, 1, FlightConfig(battery_wh=100, reserve_pct=20, wh_per_m=0.05)).battery_ok


def test_budget_constraints():
    f = FlightConfig(battery_wh=1000)
    assert not assess(1000, 1, f, max_distance_m=500).feasible
    assert not assess(1000, 1, f, max_energy_wh=1.0).feasible


@pytest.mark.parametrize("kw", [dict(cruise_speed_mps=0), dict(battery_wh=-1), dict(reserve_pct=100),
                                dict(wh_per_m=-0.1), dict(turn_overhead_s=-1)])
def test_invalid(kw):
    assert validate_flight(FlightConfig(**kw))
    with pytest.raises(ValueError):
        assess(100, 1, FlightConfig(**kw))