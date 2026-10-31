"""Alternating scanline (boustrophedon-style) coverage planning and coverage diagnostics.

This is a simple parallel-sweep planner, NOT full Boustrophedon Cellular Decomposition.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from shapely import affinity
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .geometry import extract_lines

MAX_LINES = 3000
MIN_SEGMENT_LEN_M = 0.5


class CoverageError(ValueError):
    pass


@dataclass
class Segment:
    id: int
    line: LineString
    length: float
    scan_index: int
    direction: str  # "forward" / "reverse" relative to the sweep axis

    @property
    def start(self) -> tuple[float, float]:
        return self.line.coords[0]

    @property
    def end(self) -> tuple[float, float]:
        return self.line.coords[-1]


@dataclass
class CoveragePlan:
    spacing_m: float
    swath_m: float
    segments: list[Segment]
    usable_area_m2: float
    covered_area_m2: float
    uncovered_area_m2: float
    coverage_pct: float
    covered_geom: BaseGeometry
    uncovered_geom: BaseGeometry
    coverage_distance_m: float


def generate_segments(usable: BaseGeometry, spacing_m: float, orientation_deg: float = 0.0,
                      min_segment_len: float = MIN_SEGMENT_LEN_M) -> list[Segment]:
    """Rotate the usable region so sweeps are horizontal, cut parallel lines through it,
    clip them to the region (one line may yield several segments), alternate directions
    line by line, and rotate back."""
    if not (spacing_m > 0 and math.isfinite(spacing_m)):
        raise CoverageError("Track spacing must be positive.")
    if usable is None or usable.is_empty:
        return []
    c = usable.centroid
    origin = (c.x, c.y)
    rotated = affinity.rotate(usable, -orientation_deg, origin=origin)
    minx, miny, maxx, maxy = rotated.bounds
    n_lines = int(math.ceil((maxy - miny) / spacing_m))
    if n_lines > MAX_LINES:
        raise CoverageError(f"Spacing {spacing_m:g} m would need {n_lines} scan lines (limit {MAX_LINES}).")
    segments: list[Segment] = []
    for i in range(n_lines):
        y = miny + spacing_m * (i + 0.5)  # first track half a spacing from the edge
        scan = LineString([(minx - 1.0, y), (maxx + 1.0, y)])
        pieces = [p for p in extract_lines(rotated.intersection(scan)) if p.length >= min_segment_len]
        pieces.sort(key=lambda p: p.bounds[0])
        forward = i % 2 == 0
        if not forward:
            pieces.reverse()
        for p in pieces:
            a, b = sorted([p.coords[0], p.coords[-1]])  # left, right
            if not forward:
                a, b = b, a
            line = affinity.rotate(LineString([a, b]), orientation_deg, origin=origin)
            segments.append(Segment(len(segments), line, line.length, i, "forward" if forward else "reverse"))
    return segments


def estimate_coverage(usable: BaseGeometry, segments: list[Segment], swath_m: float,
                      spacing_m: float = float("nan")) -> CoveragePlan:
    """Buffer each track by half the swath (flat caps), union (no double counting),
    clip to the usable area, and measure."""
    usable_area = usable.area if usable is not None and not usable.is_empty else 0.0
    cov_dist = sum(s.length for s in segments)
    if usable_area <= 0 or not segments:
        empty = usable.difference(usable) if usable is not None else usable
        return CoveragePlan(spacing_m, swath_m, segments, usable_area, 0.0, usable_area, 0.0,
                            empty, usable, cov_dist)
    half = swath_m / 2.0
    swaths = unary_union([s.line.buffer(half, cap_style="flat") for s in segments])
    covered = swaths.intersection(usable)
    uncovered = usable.difference(covered)
    covered_area = min(covered.area, usable_area)
    pct = max(0.0, min(100.0, 100.0 * covered_area / usable_area))
    return CoveragePlan(spacing_m, swath_m, segments, usable_area, covered_area,
                        max(0.0, usable_area - covered_area), pct, covered, uncovered, cov_dist)