"""Small geometry helpers shared across modules."""
from __future__ import annotations

from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry


def iter_polygons(geom: BaseGeometry) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if hasattr(geom, "geoms"):
        out: list[Polygon] = []
        for g in geom.geoms:
            out.extend(iter_polygons(g))
        return out
    return []


def extract_lines(geom: BaseGeometry, min_len: float = 1e-6) -> list[LineString]:
    """Pull LineStrings out of any intersection result (ignores stray points)."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom] if geom.length > min_len else []
    if hasattr(geom, "geoms"):
        out: list[LineString] = []
        for g in geom.geoms:
            out.extend(extract_lines(g, min_len))
        return out
    return []