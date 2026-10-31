from dataclasses import replace

import pytest

from planner.models import (STRATEGY_MAX_COVERAGE, STRATEGY_MIN_DISTANCE, PlanConfig, apply_preset,
                            compute_signature)
from planner.optimization import optimize, validate_config
from planner.sample import rect_farm_input


def test_preset_mapping():
    std = apply_preset(PlanConfig(), "Standard Coverage")
    high = apply_preset(PlanConfig(), "High Coverage")
    assert std.strategy == STRATEGY_MIN_DISTANCE and std.min_coverage_pct == 90.0
    assert high.strategy == STRATEGY_MAX_COVERAGE and high.preset == "High Coverage"
    with pytest.raises(KeyError):
        apply_preset(PlanConfig(), "Nope")


def test_simple_and_advanced_use_same_engine(rect_farm, base_cfg):
    simple = apply_preset(base_cfg, "High Coverage")
    advanced = replace(base_cfg, strategy=STRATEGY_MAX_COVERAGE, min_coverage_pct=0.0)
    a, b = optimize(rect_farm, simple), optimize(rect_farm, advanced)
    assert a.selected.spacing_m == b.selected.spacing_m
    assert a.selected.total_distance_m == pytest.approx(b.selected.total_distance_m)


def test_stale_result_signature():
    fi, cfg = rect_farm_input(), PlanConfig()
    s0 = compute_signature(fi, cfg)
    assert s0 == compute_signature(rect_farm_input(), PlanConfig())
    assert s0 != compute_signature(fi, replace(cfg, min_coverage_pct=50.0))
    assert s0 != compute_signature(rect_farm_input(width_m=400), cfg)


def test_advanced_validation():
    assert validate_config(PlanConfig())  == []
    assert validate_config(replace(PlanConfig(), num_uavs=0))
    assert validate_config(replace(PlanConfig(), search_min_m=50.0, search_max_m=10.0))
    assert validate_config(replace(PlanConfig(), search_min_m=1.0, search_max_m=400.0, search_step_m=1.0))