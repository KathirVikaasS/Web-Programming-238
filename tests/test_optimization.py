import pytest
from dataclasses import replace

from planner.models import FlightConfig, STRATEGY_MAX_COVERAGE, STRATEGY_MIN_DISTANCE
from planner.optimization import ConfigError, optimize, validate_config


def test_max_coverage_selection(rect_farm, base_cfg):
    res = optimize(rect_farm, replace(base_cfg, strategy=STRATEGY_MAX_COVERAGE))
    feas = [c for c in res.candidates if c.feasible]
    assert res.selected and res.selected.coverage_pct == pytest.approx(max(c.coverage_pct for c in feas))


def test_min_distance_with_coverage_floor(rect_farm, base_cfg):
    cfg = replace(base_cfg, strategy=STRATEGY_MIN_DISTANCE, min_coverage_pct=99.0)
    res = optimize(rect_farm, cfg)
    feas = [c for c in res.candidates if c.feasible]
    assert res.selected.coverage_pct >= 99.0
    assert res.selected.total_distance_m == pytest.approx(min(c.total_distance_m for c in feas))
    assert all(not c.feasible for c in res.candidates if c.spacing_m >= 35)  # gaps -> rejected
    assert any("below the required" in " ".join(c.reasons) for c in res.candidates)


def test_no_feasible_candidate(rect_farm, base_cfg):
    cfg = replace(base_cfg, flight=FlightConfig(battery_wh=1.0, safety_margin_m=5.0))
    res = optimize(rect_farm, cfg)
    assert res.selected is None and res.n_rejected == len(res.candidates) and res.message
    assert all("battery" in " ".join(c.reasons) for c in res.candidates)


def test_deterministic(rect_farm, base_cfg):
    def sig(r):
        return [(c.spacing_m, c.coverage_pct, c.total_distance_m, c.feasible) for c in r.candidates], r.selected.spacing_m
    assert sig(optimize(rect_farm, base_cfg)) == sig(optimize(rect_farm, base_cfg))


def test_invalid_config_rejected(rect_farm, base_cfg):
    bad = replace(base_cfg, min_coverage_pct=150)
    assert validate_config(bad)
    with pytest.raises(ConfigError):
        optimize(rect_farm, bad)


def test_margin_mismatch_rejected(rect_farm, base_cfg):
    with pytest.raises(ConfigError):
        optimize(rect_farm, replace(base_cfg, flight=replace(base_cfg.flight, safety_margin_m=9.0)))