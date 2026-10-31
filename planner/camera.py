"""Idealised camera footprint model.

footprint_width = 2 * altitude * tan(fov / 2)          (altitude = height above GROUND)
effective_swath = footprint_width * (1 - overlap / 100)

Simplified: real coverage depends on terrain, camera tilt, lens distortion and capture pattern.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

MAX_OVERLAP_PCT = 95.0
_UNITS = {"m": 1.0, "ft": 0.3048, "km": 1000.0}


class CameraError(ValueError):
    pass


@dataclass(frozen=True)
class SwathInfo:
    footprint_m: float
    effective_m: float


def to_metres(value: float, unit: str = "m") -> float:
    if unit not in _UNITS:
        raise CameraError(f"Unsupported unit '{unit}'. Use one of {sorted(_UNITS)}.")
    return value * _UNITS[unit]


def validate_camera(altitude_m: float, fov_deg: float, overlap_pct: float) -> list[str]:
    errs: list[str] = []
    if not (isinstance(altitude_m, (int, float)) and math.isfinite(altitude_m) and altitude_m > 0):
        errs.append("Camera altitude above ground level must be a positive number of metres.")
    if not (isinstance(fov_deg, (int, float)) and math.isfinite(fov_deg) and 0 < fov_deg < 180):
        errs.append("Field of view must be strictly between 0 and 180 degrees.")
    if not (isinstance(overlap_pct, (int, float)) and math.isfinite(overlap_pct)
            and 0 <= overlap_pct <= MAX_OVERLAP_PCT):
        errs.append(f"Image overlap must be between 0 and {MAX_OVERLAP_PCT:g} percent.")
    return errs


def footprint_width_m(altitude_m: float, fov_deg: float) -> float:
    errs = validate_camera(altitude_m, fov_deg, 0.0)
    if errs:
        raise CameraError(" ".join(errs))
    width = 2.0 * altitude_m * math.tan(math.radians(fov_deg) / 2.0)  # degrees -> radians
    if not (math.isfinite(width) and width > 0):
        raise CameraError("Computed footprint width is not positive.")
    return width


def effective_swath_m(footprint: float, overlap_pct: float) -> float:
    if not (0 <= overlap_pct <= MAX_OVERLAP_PCT):
        raise CameraError(f"Image overlap must be between 0 and {MAX_OVERLAP_PCT:g} percent.")
    swath = footprint * (1.0 - overlap_pct / 100.0)
    if swath <= 0:
        raise CameraError("Effective swath must be positive.")
    return swath


def compute_swath(cam) -> SwathInfo:
    errs = validate_camera(cam.altitude_agl_m, cam.fov_deg, cam.overlap_pct)
    if errs:
        raise CameraError(" ".join(errs))
    fp = footprint_width_m(cam.altitude_agl_m, cam.fov_deg)
    return SwathInfo(fp, effective_swath_m(fp, cam.overlap_pct))