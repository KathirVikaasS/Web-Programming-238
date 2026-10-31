import pytest
from shapely.geometry import box

from planner.coverage import CoverageError, estimate_coverage, generate_segments


def test_track_generation_and_alternation():
    segs = generate_segments(box(0, 0, 200, 100), 20.0, 0.0)
    assert len(segs) == 5
    assert all(s.length == pytest.approx(200) for s in segs)
    assert segs[0].start[0] == pytest.approx(0) and segs[1].start[0] == pytest.approx(200)


def test_full_coverage_bounds():
    usable = box(0, 0, 200, 100)
    segs = generate_segments(usable, 20.0)
    plan = estimate_coverage(usable, segs, 20.0, 20.0)
    assert 99.99 <= plan.coverage_pct <= 100.0
    assert plan.covered_area_m2 + plan.uncovered_area_m2 == pytest.approx(usable.area)


def test_gaps_when_spacing_exceeds_swath():
    usable = box(0, 0, 200, 100)
    plan = estimate_coverage(usable, generate_segments(usable, 25.0), 10.0, 25.0)
    assert 0 < plan.coverage_pct < 60


def test_no_double_counting_with_overlap():
    usable = box(0, 0, 200, 100)
    plan = estimate_coverage(usable, generate_segments(usable, 10.0), 40.0, 10.0)
    assert plan.coverage_pct <= 100.0


def test_disconnected_regions():
    usable = box(0, 0, 200, 100).difference(box(80, -1, 120, 101))
    segs = generate_segments(usable, 20.0)
    assert len(segs) == 10
    assert all(not s.line.intersects(box(80.01, 0, 119.99, 100)) for s in segs)


def test_obstacle_creates_gap_in_tracks():
    usable = box(0, 0, 200, 100).difference(box(80, 25, 120, 75))
    segs = generate_segments(usable, 20.0)
    assert len(segs) == 8
    plan = estimate_coverage(usable, segs, 20.0, 20.0)
    assert plan.usable_area_m2 == pytest.approx(200 * 100 - 40 * 50)


def test_rotated_tracks_stay_inside():
    usable = box(0, 0, 200, 100)
    segs = generate_segments(usable, 15.0, 30.0)
    assert segs and all(usable.buffer(1e-6).covers(s.line) for s in segs)


def test_zero_usable_area_and_bad_spacing():
    assert generate_segments(box(0, 0, 1, 1).difference(box(-1, -1, 2, 2)), 5.0) == []
    with pytest.raises(CoverageError):
        generate_segments(box(0, 0, 10, 10), 0)
    with pytest.raises(CoverageError):
        generate_segments(box(0, 0, 10000, 10000), 1.0)