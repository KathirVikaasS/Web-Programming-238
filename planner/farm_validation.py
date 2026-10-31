"""GeoJSON parsing, farm validation and construction of the projected Farm model."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Optional

from shapely.geometry import GeometryCollection, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.validation import explain_validity

from .coordinates import CoordinateError, Projection, choose_projection, validate_lonlat
from .geometry import iter_polygons
from .models import FarmInput

MIN_AREA_M2 = 1.0
QUAD_SEGS = 8


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class Farm:
    projection: Projection
    field: Polygon
    nfz: list[Polygon]
    nfz_buffered: BaseGeometry   # union of safety-buffered zones (may be empty)
    usable: BaseGeometry         # field minus buffered zones (coverage region)
    operational: BaseGeometry    # region transit routes may use
    depot: Point
    margin_m: float
    outside_margin_m: float

    @property
    def depot_xy(self) -> tuple[float, float]:
        return (self.depot.x, self.depot.y)

    @property
    def field_area_m2(self) -> float:
        return self.field.area

    @property
    def usable_area_m2(self) -> float:
        return self.usable.area

    @property
    def excluded_area_m2(self) -> float:
        return max(0.0, self.field.area - self.usable.area)

    @property
    def nfz_area_in_field_m2(self) -> float:
        if not self.nfz:
            return 0.0
        return self.field.intersection(unary_union(self.nfz)).area


def _parse_ring(coords, label: str, rep: ValidationReport) -> Optional[list[tuple[float, float]]]:
    if not isinstance(coords, list) or len(coords) < 4:
        rep.errors.append(f"{label}: a Polygon ring needs at least 4 positions (closed ring).")
        return None
    ring: list[tuple[float, float]] = []
    for j, pos in enumerate(coords):
        if not isinstance(pos, (list, tuple)) or len(pos) < 2:
            rep.errors.append(f"{label}: position {j + 1} must be [longitude, latitude].")
            return None
        try:
            ring.append(validate_lonlat(pos[0], pos[1], f"{label}, position {j + 1}"))
        except CoordinateError as exc:
            rep.errors.append(str(exc))
            return None
    if ring[0] == ring[-1]:
        ring = ring[:-1]
    else:
        rep.warnings.append(f"{label}: ring was not closed; it was closed automatically.")
    if len(set(ring)) < 3:
        rep.errors.append(f"{label}: polygon needs at least 3 distinct vertices.")
        return None
    return ring


def parse_geojson(data) -> tuple[Optional[FarmInput], ValidationReport]:
    """Strictly parse a FeatureCollection with roles field / nfz / depot."""
    rep = ValidationReport()
    try:
        obj = json.loads(data) if isinstance(data, (str, bytes, bytearray)) else data
    except ValueError as exc:
        rep.errors.append(f"The file is not valid JSON: {exc}")
        return None, rep
    if not isinstance(obj, dict) or obj.get("type") != "FeatureCollection":
        rep.errors.append("The file must be a GeoJSON FeatureCollection.")
        return None, rep
    feats = obj.get("features")
    if not isinstance(feats, list):
        rep.errors.append("FeatureCollection has no 'features' list.")
        return None, rep

    fields, zones, depots = [], [], []
    for i, feat in enumerate(feats):
        label = f"Feature {i + 1}"
        if not isinstance(feat, dict) or feat.get("type") != "Feature":
            rep.errors.append(f"{label}: not a GeoJSON Feature.")
            continue
        props = feat.get("properties") or {}
        role = props.get("role") if isinstance(props, dict) else None
        if role not in ("field", "nfz", "depot"):
            rep.errors.append(f"{label}: properties.role must be 'field', 'nfz' or 'depot' (got {role!r}).")
            continue
        geom = feat.get("geometry")
        if not isinstance(geom, dict):
            rep.errors.append(f"{label} ({role}): missing geometry.")
            continue
        gtype, coords = geom.get("type"), geom.get("coordinates")
        if role in ("field", "nfz"):
            if gtype != "Polygon":
                rep.errors.append(f"{label} ({role}): geometry type '{gtype}' is unsupported; only Polygon is accepted.")
                continue
            if not isinstance(coords, list) or not coords:
                rep.errors.append(f"{label} ({role}): empty coordinates.")
                continue
            if len(coords) > 1:
                rep.errors.append(f"{label} ({role}): polygons with holes are unsupported; "
                                  "model holes as separate 'nfz' polygons.")
                continue
            ring = _parse_ring(coords[0], f"{label} ({role})", rep)
            if ring is not None:
                (fields if role == "field" else zones).append(ring)
        else:
            if gtype != "Point":
                rep.errors.append(f"{label} (depot): geometry type '{gtype}' is unsupported; only Point is accepted.")
                continue
            if not isinstance(coords, list) or len(coords) < 2:
                rep.errors.append(f"{label} (depot): coordinates must be [longitude, latitude].")
                continue
            try:
                depots.append(validate_lonlat(coords[0], coords[1], f"{label} (depot)"))
            except CoordinateError as exc:
                rep.errors.append(str(exc))

    if len(fields) != 1:
        rep.errors.append(f"Exactly one Polygon with role 'field' is required (found {len(fields)}).")
    if len(depots) > 1:
        rep.errors.append("At most one Point with role 'depot' is allowed.")
    if not depots:
        rep.warnings.append("No depot found in the file; set one in Farm Setup.")
    if rep.errors:
        return None, rep
    return FarmInput(boundary=fields[0], nfz=zones, depot=depots[0] if depots else None), rep


def build_farm(farm_input: FarmInput, margin_m: float = 5.0,
               outside_margin_m: float = 0.0) -> tuple[Optional[Farm], ValidationReport]:
    """Validate geometry and build the projected Farm. Returns (None, report) on any error."""
    rep = ValidationReport()
    if not (math.isfinite(margin_m) and margin_m >= 0):
        rep.errors.append("Safety margin must be a finite number >= 0.")
    if not (math.isfinite(outside_margin_m) and outside_margin_m >= 0):
        rep.errors.append("Transit-outside margin must be a finite number >= 0.")
    if len(farm_input.boundary) < 3:
        rep.errors.append("The field boundary needs at least 3 points.")
    if rep.errors:
        return None, rep

    try:
        proj = choose_projection(farm_input.boundary)
        pts = proj.to_xy_pairs(farm_input.boundary)
    except CoordinateError as exc:
        rep.errors.append(str(exc))
        return None, rep
    try:
        poly = Polygon(pts)
    except ValueError as exc:
        rep.errors.append(f"Field boundary is degenerate: {exc}")
        return None, rep
    if not poly.is_valid:
        rep.errors.append(f"Field boundary is invalid: {explain_validity(poly)} (check for self-intersections).")
        return None, rep
    if poly.area < MIN_AREA_M2:
        rep.errors.append("Field boundary is degenerate (area under 1 m²).")
        return None, rep

    zones: list[Polygon] = []
    for i, ring in enumerate(farm_input.nfz):
        label = f"Restricted zone {i + 1}"
        if len(ring) < 3:
            rep.errors.append(f"{label}: needs at least 3 points.")
            continue
        try:
            zpts = proj.to_xy_pairs([validate_lonlat(lo, la, label) for lo, la in ring])
            zp = Polygon(zpts)
        except (CoordinateError, ValueError) as exc:
            rep.errors.append(f"{label}: {exc}")
            continue
        if not zp.is_valid:
            rep.errors.append(f"{label} is invalid: {explain_validity(zp)}.")
            continue
        if zp.area < MIN_AREA_M2:
            rep.errors.append(f"{label} is degenerate (area under 1 m²).")
            continue
        if not poly.intersects(zp):
            rep.warnings.append(f"{label} lies outside the field and has no effect.")
        elif not poly.covers(zp):
            rep.warnings.append(f"{label} extends beyond the field; only the overlapping part matters.")
        zones.append(zp)
    if rep.errors:
        return None, rep

    buffered: BaseGeometry = (unary_union([z.buffer(margin_m, quad_segs=QUAD_SEGS) for z in zones])
                              if zones else GeometryCollection())
    usable_raw = poly.difference(buffered)
    big = [p for p in iter_polygons(usable_raw) if p.area >= MIN_AREA_M2]
    if not big:
        rep.errors.append("No usable area remains after excluding restricted zones and the safety margin.")
        return None, rep
    usable = unary_union(big)

    base = poly.buffer(outside_margin_m, quad_segs=QUAD_SEGS) if outside_margin_m > 0 else poly
    operational = base.difference(buffered)

    if farm_input.depot is None:
        rep.errors.append("A depot (start/return location) is required.")
        return None, rep
    try:
        dlon, dlat = validate_lonlat(*farm_input.depot, "Depot")
        dx, dy = proj.to_xy_pairs([(dlon, dlat)])[0]
    except CoordinateError as exc:
        rep.errors.append(str(exc))
        return None, rep
    depot = Point(dx, dy)
    if not poly.buffer(1e-6).covers(depot):
        rep.errors.append("The depot must be inside the field boundary.")
    elif not buffered.is_empty and buffered.contains(depot):
        rep.errors.append("The depot lies inside a restricted zone or its safety margin.")
    elif not operational.buffer(1e-6).covers(depot):
        rep.errors.append("The depot is not inside the allowed operational region.")
    if rep.errors:
        return None, rep

    if usable.area < 0.05 * poly.area:
        rep.warnings.append("Less than 5% of the field remains usable after exclusions.")
    if len(big) > 1:
        rep.warnings.append(f"Restricted zones split the usable area into {len(big)} separate regions. "
                            "Transit between them is only possible if a safe route exists inside the "
                            "operational region (see 'transit outside margin' in Advanced Mode).")
    return Farm(proj, poly, zones, buffered, usable, operational, depot, margin_m, outside_margin_m), rep