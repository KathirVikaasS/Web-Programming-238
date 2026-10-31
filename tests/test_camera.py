import pytest

from planner.camera import (CameraError, compute_swath, effective_swath_m, footprint_width_m, to_metres,
                            validate_camera)
from planner.models import CameraConfig


def test_formula_example():
    fp = footprint_width_m(20, 90)
    assert fp == pytest.approx(40.0)
    assert effective_swath_m(fp, 20) == pytest.approx(32.0)


def test_radians_conversion():
    assert footprint_width_m(10, 60) == pytest.approx(2 * 10 * 0.5773502691896257)


def test_compute_swath():
    s = compute_swath(CameraConfig(20, 90, 20))
    assert s.footprint_m == pytest.approx(40) and s.effective_m == pytest.approx(32)


@pytest.mark.parametrize("alt,fov,ov", [(0, 90, 20), (-5, 90, 20), (20, 0, 20), (20, 180, 20),
                                        (20, 90, -1), (20, 90, 100), (float("nan"), 90, 20)])
def test_invalid(alt, fov, ov):
    assert validate_camera(alt, fov, ov)
    with pytest.raises(CameraError):
        compute_swath(CameraConfig(alt, fov, ov))


def test_units():
    assert to_metres(100, "ft") == pytest.approx(30.48)
    assert to_metres(1, "km") == 1000
    with pytest.raises(CameraError):
        to_metres(1, "furlong")