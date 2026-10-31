import pytest

from planner.coordinates import CoordinateError, choose_projection, utm_epsg, validate_lonlat


def test_roundtrip():
    proj = choose_projection([(80.20, 13.00), (80.21, 13.01), (80.20, 13.01)])
    pairs = [(80.205, 13.004), (80.2, 13.0)]
    back = proj.to_lonlat_pairs(proj.to_xy_pairs(pairs))
    for (a, b), (c, d) in zip(pairs, back):
        assert a == pytest.approx(c, abs=1e-8) and b == pytest.approx(d, abs=1e-8)


def test_distance_in_metres():
    proj = choose_projection([(80.0, 13.0), (80.0, 13.001)])
    (x0, y0), (x1, y1) = proj.to_xy_pairs([(80.0, 13.0), (80.0, 13.001)])
    assert ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 == pytest.approx(110.6, abs=1.0)


def test_zone_selection():
    assert utm_epsg(80.2, 13.0) == 32644
    assert utm_epsg(151.2, -33.9) == 32756


def test_polar_fallback():
    proj = choose_projection([(10.0, 85.0), (10.1, 85.01), (10.0, 85.02)])
    assert "azimuthal" in proj.label.lower()


@pytest.mark.parametrize("lon,lat", [(0, 95), (200, 10), (float("nan"), 1), ("a", 2), (None, 1)])
def test_invalid_coordinates(lon, lat):
    with pytest.raises(CoordinateError):
        validate_lonlat(lon, lat)