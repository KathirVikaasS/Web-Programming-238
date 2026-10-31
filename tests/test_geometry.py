import copy

from planner.farm_validation import build_farm, parse_geojson
from planner.models import FarmInput
from planner.sample import (farm_input_to_geojson, local_to_lonlat, rect_farm_input, sample_farm_input,
                            sample_geojson_dict)


def _ll(pts, lon0=80.0, lat0=13.0):
    return [local_to_lonlat(x, y, lon0, lat0) for x, y in pts]


def test_valid_farm_and_area():
    farm, rep = build_farm(rect_farm_input(300, 200), 5.0)
    assert rep.ok and farm.field_area_m2 == __import__("pytest").approx(60000, rel=0.01)


def test_sample_farm_valid():
    farm, rep = build_farm(sample_farm_input(), 5.0)
    assert farm is not None and len(farm.nfz) == 2 and farm.usable_area_m2 < farm.field_area_m2


def test_self_intersecting_rejected():
    fi = FarmInput(boundary=_ll([(0, 0), (100, 100), (100, 0), (0, 100)]), depot=local_to_lonlat(10, 5, 80, 13))
    farm, rep = build_farm(fi)
    assert farm is None and any("invalid" in e.lower() for e in rep.errors)


def test_too_few_points():
    farm, rep = build_farm(FarmInput(boundary=_ll([(0, 0), (10, 0)])))
    assert farm is None and rep.errors


def test_depot_inside_restricted_zone():
    fi = rect_farm_input(300, 200, nfz_boxes=[(0, 0, 50, 50)], depot_m=(10, 10))
    farm, rep = build_farm(fi, 5.0)
    assert farm is None and any("depot" in e.lower() for e in rep.errors)


def test_depot_outside_field():
    fi = rect_farm_input(300, 200, depot_m=(-50, -50))
    farm, rep = build_farm(fi)
    assert farm is None and any("depot" in e.lower() for e in rep.errors)


def test_empty_usable_region():
    fi = rect_farm_input(100, 100, nfz_boxes=[(-10, -10, 110, 110)], depot_m=(50, 50))
    farm, rep = build_farm(fi, 5.0)
    assert farm is None and rep.errors


def test_invalid_restricted_zone():
    fi = rect_farm_input(300, 200)
    fi.nfz.append(_ll([(10, 10), (60, 60), (60, 10), (10, 60)]))
    farm, rep = build_farm(fi)
    assert farm is None and any("Restricted zone 1" in e for e in rep.errors)


def test_geojson_valid_roundtrip():
    fi, rep = parse_geojson(sample_geojson_dict())
    assert rep.ok and len(fi.nfz) == 2 and fi.depot is not None


def test_geojson_errors():
    good = sample_geojson_dict()
    assert parse_geojson("not json")[0] is None
    assert parse_geojson('{"type":"Feature"}')[0] is None
    no_field = copy.deepcopy(good)
    no_field["features"] = [f for f in no_field["features"] if f["properties"]["role"] != "field"]
    assert any("Exactly one" in e for e in parse_geojson(no_field)[1].errors)
    multi = copy.deepcopy(good)
    multi["features"][0]["geometry"] = {"type": "MultiPolygon", "coordinates": []}
    assert any("unsupported" in e for e in parse_geojson(multi)[1].errors)
    swapped = farm_input_to_geojson(FarmInput(boundary=[(-33.9, 151.2), (-33.9, 151.3), (-33.8, 151.3)]))
    assert not parse_geojson(swapped)[1].ok