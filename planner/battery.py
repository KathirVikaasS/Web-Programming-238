"""Simple energy/duration estimates (simulation only, not real-world battery guarantees).

duration = distance / speed + turn_overhead * segments
energy   = distance * Wh_per_m
usable   = capacity * (1 - reserve/100)
Ignores wind, payload, acceleration and detailed manoeuvres.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


def validate_flight(flight) -> list[str]:
    errs: list[str] = []
    def ok(v): return isinstance(v, (int, float)) and math.isfinite(v)
    if not (ok(flight.cruise_speed_mps) and flight.cruise_speed_mps > 0):
        errs.append("Cruise speed must be positive (m/s).")
    if not (ok(flight.battery_wh) and flight.battery_wh > 0):
        errs.append("Battery capacity must be positive (Wh).")
    if not (ok(flight.reserve_pct) and 0 <= flight.reserve_pct < 100):
        errs.append("Battery reserve must be at least 0 and below 100 percent.")
    if not (ok(flight.wh_per_m) and flight.wh_per_m >= 0):
        errs.append("Energy consumption must be >= 0 (Wh/m).")
    if not (ok(flight.turn_overhead_s) and flight.turn_overhead_s >= 0):
        errs.append("Turn overhead must be >= 0 (seconds).")
    if not (ok(flight.safety_margin_m) and flight.safety_margin_m >= 0):
        errs.append("Safety margin must be >= 0 (m).")
    if not (ok(flight.transit_outside_margin_m) and flight.transit_outside_margin_m >= 0):
        errs.append("Transit-outside margin must be >= 0 (m).")
    if not ok(flight.orientation_deg):
        errs.append("Sweep orientation must be a finite number of degrees.")
    return errs


def usable_battery_wh(capacity_wh: float, reserve_pct: float) -> float:
    return capacity_wh * (1.0 - reserve_pct / 100.0)


def energy_required_wh(distance_m: float, wh_per_m: float) -> float:
    return distance_m * wh_per_m


def mission_duration_s(distance_m: float, speed_mps: float, turn_overhead_s: float, n_segments: int) -> float:
    return distance_m / speed_mps + turn_overhead_s * n_segments


@dataclass
class BatteryAssessment:
    energy_wh: float
    usable_wh: float
    duration_s: float
    battery_ok: bool
    feasible: bool
    reasons: list[str] = field(default_factory=list)


def assess(total_distance_m: float, n_segments: int, flight,
           max_distance_m: Optional[float] = None, max_energy_wh: Optional[float] = None) -> BatteryAssessment:
    errs = validate_flight(flight)
    if errs:
        raise ValueError(" ".join(errs))
    energy = energy_required_wh(total_distance_m, flight.wh_per_m)
    usable = usable_battery_wh(flight.battery_wh, flight.reserve_pct)
    duration = mission_duration_s(total_distance_m, flight.cruise_speed_mps, flight.turn_overhead_s, n_segments)
    reasons: list[str] = []
    battery_ok = energy <= usable + 1e-9
    if not battery_ok:
        reasons.append(f"Estimated energy {energy:.1f} Wh exceeds usable battery {usable:.1f} Wh.")
    if max_distance_m is not None and total_distance_m > max_distance_m + 1e-9:
        reasons.append(f"Mission distance {total_distance_m:.0f} m exceeds the maximum {max_distance_m:.0f} m.")
    if max_energy_wh is not None and energy > max_energy_wh + 1e-9:
        reasons.append(f"Estimated energy {energy:.1f} Wh exceeds the energy budget {max_energy_wh:.1f} Wh.")
    return BatteryAssessment(energy, usable, duration, battery_ok, not reasons, reasons)