"""WGS84 longitude/latitude <-> local projected metres (PyProj)."""
from __future__ import annotations

import math

import numpy as np
from pyproj import CRS, Transformer
from pyproj.exceptions import ProjError

WGS84 = CRS.from_epsg(4326)


class CoordinateError(ValueError):
    pass


def validate_lonlat(lon, lat, label: str = "point") -> tuple[float, float]:
    try:
        lon_f, lat_f = float(lon), float(lat)
    except (TypeError, ValueError):
        raise CoordinateError(f"{label}: coordinates must be numbers.") from None
    if not (math.isfinite(lon_f) and math.isfinite(lat_f)):
        raise CoordinateError(f"{label}: coordinates must be finite numbers.")
    if not -90.0 <= lat_f <= 90.0:
        raise CoordinateError(
            f"{label}: latitude {lat_f} is outside [-90, 90]. GeoJSON order is [longitude, latitude] - "
            "are the values swapped?")
    if not -180.0 <= lon_f <= 180.0:
        raise CoordinateError(f"{label}: longitude {lon_f} is outside [-180, 180].")
    return lon_f, lat_f


def utm_epsg(lon: float, lat: float) -> int:
    zone = min(60, max(1, int((lon + 180.0) // 6) + 1))
    return (32600 if lat >= 0 else 32700) + zone


class Projection:
    """Forward/inverse transformer between WGS84 (always lon, lat) and a metric CRS."""

    def __init__(self, crs: CRS, label: str):
        self.crs = crs
        self.label = label
        try:
            self._fwd = Transformer.from_crs(WGS84, crs, always_xy=True)
            self._inv = Transformer.from_crs(crs, WGS84, always_xy=True)
        except ProjError as exc:
            raise CoordinateError(f"Could not create coordinate transformation: {exc}") from None

    def to_xy(self, lon, lat):
        try:
            x, y = self._fwd.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
        except ProjError as exc:
            raise CoordinateError(f"Projection failed: {exc}") from None
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise CoordinateError("Projection produced non-finite values.")
        return x, y

    def to_lonlat(self, x, y):
        try:
            lon, lat = self._inv.transform(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        except ProjError as exc:
            raise CoordinateError(f"Inverse projection failed: {exc}") from None
        return lon, lat

    def to_xy_pairs(self, lonlat_pairs) -> list[tuple[float, float]]:
        pairs = list(lonlat_pairs)
        if not pairs:
            return []
        lons, lats = zip(*pairs)
        x, y = self.to_xy(np.array(lons, float), np.array(lats, float))
        return [(float(a), float(b)) for a, b in zip(np.atleast_1d(x), np.atleast_1d(y))]

    def to_lonlat_pairs(self, xy_pairs) -> list[tuple[float, float]]:
        pairs = list(xy_pairs)
        if not pairs:
            return []
        xs, ys = zip(*pairs)
        lon, lat = self.to_lonlat(np.array(xs, float), np.array(ys, float))
        return [(float(a), float(b)) for a, b in zip(np.atleast_1d(lon), np.atleast_1d(lat))]


def choose_projection(lonlat_pairs) -> Projection:
    """Pick a local metric CRS for the farm: its own UTM zone, or a local azimuthal
    equidistant projection near the poles (UTM is undefined there)."""
    pairs = [validate_lonlat(lo, la, f"vertex {i + 1}") for i, (lo, la) in enumerate(lonlat_pairs)]
    if not pairs:
        raise CoordinateError("No coordinates supplied.")
    lons = [p[0] for p in pairs]
    lats = [p[1] for p in pairs]
    if max(lons) - min(lons) > 180.0:
        raise CoordinateError("Farms spanning the antimeridian (or >180 degrees of longitude) are unsupported.")
    lon0, lat0 = sum(lons) / len(lons), sum(lats) / len(lats)
    if lat0 > 84.0 or lat0 < -80.0:
        crs = CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs")
        return Projection(crs, f"Local azimuthal equidistant (lat0={lat0:.4f}, lon0={lon0:.4f})")
    epsg = utm_epsg(lon0, lat0)
    return Projection(CRS.from_epsg(epsg), f"UTM EPSG:{epsg}")