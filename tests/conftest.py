import pytest

from planner.farm_validation import build_farm
from planner.models import CameraConfig, FlightConfig, PlanConfig
from planner.sample import rect_farm_input


@pytest.fixture
def rect_farm():
    farm, rep = build_farm(rect_farm_input(300, 200), margin_m=5.0)
    assert farm is not None, rep.errors
    return farm


@pytest.fixture
def base_cfg():
    return PlanConfig(camera=CameraConfig(20, 90, 20),
                      flight=FlightConfig(battery_wh=1000.0, safety_margin_m=5.0),
                      search_min_m=15.0, search_max_m=45.0, search_step_m=5.0, min_coverage_pct=0.0)