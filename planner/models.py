"""Shared configuration data classes, presets and result signatures."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from typing import Optional

LonLat = tuple[float, float]  # (longitude, latitude) - GeoJSON order

STRATEGY_MAX_COVERAGE = "max_coverage"
STRATEGY_MIN_DISTANCE = "min_distance"
SPACING_CAMERA = "camera"
SPACING_MANUAL = "manual"


@dataclass(frozen=True)
class CameraConfig:
    altitude_agl_m: float = 20.0   # altitude ABOVE GROUND LEVEL, metres
    fov_deg: float = 90.0          # field of view, degrees (0 < fov < 180)
    overlap_pct: float = 20.0      # image overlap, percent


@dataclass(frozen=True)
class FlightConfig:
    cruise_speed_mps: float = 8.0
    battery_wh: float = 150.0
    reserve_pct: float = 20.0
    wh_per_m: float = 0.02               # energy per metre, Wh/m
    turn_overhead_s: float = 4.0         # seconds added per coverage segment
    safety_margin_m: float = 5.0         # buffer around restricted zones
    orientation_deg: float = 0.0         # sweep direction, degrees CCW from east
    transit_outside_margin_m: float = 0.0  # how far transit may leave the field


@dataclass(frozen=True)
class PlanConfig:
    spacing_mode: str = SPACING_CAMERA
    manual_spacing_m: float = 30.0
    search_min_m: Optional[float] = None    # None -> automatic
    search_max_m: Optional[float] = None
    search_step_m: Optional[float] = None
    strategy: str = STRATEGY_MIN_DISTANCE
    min_coverage_pct: float = 90.0
    max_distance_m: Optional[float] = None
    max_energy_wh: Optional[float] = None
    num_uavs: int = 3
    camera: CameraConfig = field(default_factory=CameraConfig)
    flight: FlightConfig = field(default_factory=FlightConfig)
    preset: Optional[str] = None


@dataclass
class FarmInput:
    """Raw user geometry in WGS84 (longitude, latitude)."""
    boundary: list[LonLat] = field(default_factory=list)
    nfz: list[list[LonLat]] = field(default_factory=list)
    depot: Optional[LonLat] = None


# Presets map to real configuration values; Simple Mode calls the same engine as Advanced Mode.
PRESETS: dict[str, dict] = {
    "Standard Coverage": dict(strategy=STRATEGY_MIN_DISTANCE, min_coverage_pct=90.0,
                              spacing_mode=SPACING_CAMERA),
    "High Coverage": dict(strategy=STRATEGY_MAX_COVERAGE, min_coverage_pct=0.0,
                          spacing_mode=SPACING_CAMERA),
}
PRESET_DESCRIPTIONS = {
    "Standard Coverage": "Balances coverage and distance: picks the shortest feasible mission that "
                         "still reaches at least 90% estimated coverage.",
    "High Coverage": "Picks the feasible mission with the highest estimated coverage "
                     "(ties are broken by shorter distance).",
}


def apply_preset(cfg: PlanConfig, name: str) -> PlanConfig:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset: {name}")
    return replace(cfg, preset=name, **PRESETS[name])


def compute_signature(*objs) -> str:
    """Stable hash of inputs; used to detect stale results."""
    def conv(o):
        return asdict(o) if is_dataclass(o) and not isinstance(o, type) else o
    payload = json.dumps([conv(o) for o in objs], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()