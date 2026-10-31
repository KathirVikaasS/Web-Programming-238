"""Synthetic demonstration data and test helpers. Locations are arbitrary, not a real farm."""
from __future__ import annotations

import math

from .models import FarmInput

SAMPLE_ORIGIN = (76.50, 12.30)  # (lon, lat) of local (0, 0) - arbitrary
_SAMPLE_FIELD = [(0, 40), (120, 0), (300, 15), (480, 0), (540, 110), (520, 260), (470, 380),
                 (300, 350), (150, 390), (30, 330), (-20, 190)]
_SAMPLE_NFZ = [
    [(180, 140), (260, 130), (280, 200), (200, 230), (160, 190)],   # synthetic pond
    [(380, 240), (430, 240), (430, 300), (380, 300)],               # synthetic pylon strip
]
_SAMPLE_DEPOT = (30, 70)


def local_to_lonlat(x: float, y: float, lon0: float, lat0: float) -> tuple[float, float]:
    """Approximate local metres -> degrees (good enough for demo/test data)."""
    return (lon0 + x / (111_320.0 * math.cos(math.radians(lat0))), lat0 + y / 111_320.0)


def _conv(pts, origin):
    return [local_to_lonlat(x, y, *origin) for x, y in pts]


def sample_farm_input() -> FarmInput:
    return FarmInput(boundary=_conv(_SAMPLE_FIELD, SAMPLE_ORIGIN),
                     nfz=[_conv(z, SAMPLE_ORIGIN) for z in _SAMPLE_NFZ],
                     depot=local_to_lonlat(*_SAMPLE_DEPOT, *SAMPLE_ORIGIN))


def farm_input_to_geojson(fi: FarmInput, label: str = "") -> dict:
    def ring(pts):
        r = [list(p) for p in pts]
        return [r + [r[0]]]
    feats = [{"type": "Feature", "properties": {"role": "field", "name": label},
              "geometry": {"type": "Polygon", "coordinates": ring(fi.boundary)}}]
    for i, z in enumerate(fi.nfz):
        feats.append({"type": "Feature", "properties": {"role": "nfz", "name": f"restricted {i + 1}"},
                      "geometry": {"type": "Polygon", "coordinates": ring(z)}})
    if fi.depot:
        feats.append({"type": "Feature", "properties": {"role": "depot"},
                      "geometry": {"type": "Point", "coordinates": list(fi.depot)}})
    return {"type": "FeatureCollection", "features": feats}


def sample_geojson_dict() -> dict:
    return farm_input_to_geojson(sample_farm_input(), "SYNTHETIC DEMONSTRATION FARM")


def rect_farm_input(width_m=300.0, height_m=200.0, lon0=80.0, lat0=13.0,
                    nfz_boxes=(), depot_m=(10.0, 10.0)) -> FarmInput:
    """Rectangular farm with optional rectangular restricted zones (x0, y0, x1, y1) in local metres."""
    def box(x0, y0, x1, y1):
        return _conv([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], (lon0, lat0))
    return FarmInput(boundary=box(0, 0, width_m, height_m), nfz=[box(*b) for b in nfz_boxes],
                     depot=local_to_lonlat(*depot_m, lon0, lat0))