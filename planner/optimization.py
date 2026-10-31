"""Deterministic candidate-spacing search using the real planning engine."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd
from shapely.geometry import LineString

from .battery import assess, validate_flight
from .camera import CameraError, SwathInfo, compute_swath, validate_camera
from .coverage import CoverageError, CoveragePlan, estimate_coverage, generate_segments
from .models import (SPACING_CAMERA, SPACING_MANUAL, STRATEGY_MAX_COVERAGE, STRATEGY_MIN_DISTANCE,
                     PlanConfig)
from .transit import TransitPlan, TransitRouter, plan_transits

MIN_SPACING_M = 0.5
MAX_SPACING_M = 500.0
MAX_CANDIDATES = 40


class ConfigError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class Candidate:
    spacing_m: float
    coverage_pct: float = 0.0
    covered_area_m2: float = 0.0
    uncovered_area_m2: float = 0.0
    n_segments: int = 0
    coverage_distance_m: float = 0.0
    transit_distance_m: Optional[float] = None   # None when a transition is infeasible
    total_distance_m: Optional[float] = None
    duration_s: Optional[float] = None
    energy_wh: Optional[float] = None
    usable_battery_wh: Optional[float] = None
    battery_ok: Optional[bool] = None
    violations: int = 0
    transit_ok: bool = False
    feasible: bool = False
    reasons: list[str] = field(default_factory=list)
    selected: bool = False
    plan: Optional[CoveragePlan] = None
    transit: Optional[TransitPlan] = None


@dataclass
class OptimizationResult:
    candidates: list[Candidate]
    selected: Optional[Candidate]
    swath: SwathInfo
    base_spacing_m: float
    message: str
    explanation: list[str]
    router: TransitRouter

    @property
    def n_rejected(self) -> int:
        return sum(1 for c in self.candidates if not c.feasible)


def validate_config(cfg: PlanConfig) -> list[str]:
    errs = validate_camera(cfg.camera.altitude_agl_m, cfg.camera.fov_deg, cfg.camera.overlap_pct)
    errs += validate_flight(cfg.flight)
    if cfg.strategy not in (STRATEGY_MAX_COVERAGE, STRATEGY_MIN_DISTANCE):
        errs.append(f"Unknown optimization strategy '{cfg.strategy}'.")
    if cfg.spacing_mode not in (SPACING_CAMERA, SPACING_MANUAL):
        errs.append(f"Unknown spacing mode '{cfg.spacing_mode}'.")
    if cfg.spacing_mode == SPACING_MANUAL and not (MIN_SPACING_M <= cfg.manual_spacing_m <= MAX_SPACING_M):
        errs.append(f"Manual track spacing must be between {MIN_SPACING_M} and {MAX_SPACING_M} m.")
    if not (0 <= cfg.min_coverage_pct <= 100):
        errs.append("Minimum coverage must be between 0 and 100 percent.")
    if cfg.max_distance_m is not None and cfg.max_distance_m <= 0:
        errs.append("Maximum mission distance must be positive (or left unset).")
    if cfg.max_energy_wh is not None and cfg.max_energy_wh <= 0:
        errs.append("Maximum energy budget must be positive (or left unset).")
    if not (1 <= int(cfg.num_uavs) <= 20):
        errs.append("Number of UAVs must be between 1 and 20.")
    for name, v in (("search_min_m", cfg.search_min_m), ("search_max_m", cfg.search_max_m),
                    ("search_step_m", cfg.search_step_m)):
        if v is not None and not (math.isfinite(v) and v > 0):
            errs.append(f"{name} must be positive (or left automatic).")
    if not errs:
        try:
            swath = compute_swath(cfg.camera)
            base = swath.effective_m if cfg.spacing_mode == SPACING_CAMERA else cfg.manual_spacing_m
            lo, hi, step = _search_range(cfg, base)
            if hi < lo:
                errs.append("Candidate spacing maximum must be >= minimum.")
            elif lo < MIN_SPACING_M or hi > MAX_SPACING_M:
                errs.append(f"Candidate spacings must stay within {MIN_SPACING_M}-{MAX_SPACING_M} m "
                            f"(got {lo:g}-{hi:g} m).")
            elif (hi - lo) / step + 1 > MAX_CANDIDATES:
                errs.append(f"The search would test more than {MAX_CANDIDATES} candidates; increase the step.")
        except CameraError as exc:
            errs.append(str(exc))
    return errs


def _search_range(cfg: PlanConfig, base: float) -> tuple[float, float, float]:
    lo = cfg.search_min_m if cfg.search_min_m else 0.6 * base
    hi = cfg.search_max_m if cfg.search_max_m else 1.2 * base
    step = cfg.search_step_m if cfg.search_step_m else max((hi - lo) / 8.0, 0.01)
    return lo, hi, step


def generate_candidates(cfg: PlanConfig, swath: SwathInfo) -> list[float]:
    base = swath.effective_m if cfg.spacing_mode == SPACING_CAMERA else cfg.manual_spacing_m
    lo, hi, step = _search_range(cfg, base)
    n = int(math.floor((hi - lo) / step + 1e-9)) + 1
    values = {round(lo + i * step, 3) for i in range(max(n, 1))}
    values.add(round(base, 3))  # always test the camera-derived / manual spacing itself
    return sorted(values)


def evaluate_candidate(farm, cfg: PlanConfig, spacing: float, router: TransitRouter,
                       swath: SwathInfo) -> Candidate:
    cand = Candidate(spacing_m=spacing)
    try:
        segments = generate_segments(farm.usable, spacing, cfg.flight.orientation_deg)
    except CoverageError as exc:
        cand.reasons.append(str(exc))
        return cand
    plan = estimate_coverage(farm.usable, segments, swath.effective_m, spacing)
    cand.plan = plan
    cand.coverage_pct, cand.covered_area_m2 = plan.coverage_pct, plan.covered_area_m2
    cand.uncovered_area_m2, cand.n_segments = plan.uncovered_area_m2, len(segments)
    cand.coverage_distance_m = plan.coverage_distance_m
    if not segments:
        cand.reasons.append("No coverage tracks could be generated at this spacing.")
        return cand
    if plan.coverage_pct + 1e-9 < cfg.min_coverage_pct:
        cand.reasons.append(f"Coverage {plan.coverage_pct:.1f}% is below the required {cfg.min_coverage_pct:.1f}%.")

    transit = plan_transits(farm.depot_xy, segments, router)
    cand.transit = transit
    cand.transit_ok = transit.feasible
    lines = [s.line for s in segments] + [LineString(r.path) for r in transit.routes
                                          if r.feasible and len(r.path) >= 2]
    cand.violations = sum(1 for ln in lines if router.violates(ln))
    if cand.violations:
        cand.reasons.append(f"{cand.violations} route piece(s) violate restricted zones or the operational region.")
    if transit.feasible:
        cand.transit_distance_m = transit.total_length
        cand.total_distance_m = cand.coverage_distance_m + transit.total_length
        bat = assess(cand.total_distance_m, len(segments), cfg.flight, cfg.max_distance_m, cfg.max_energy_wh)
        cand.duration_s, cand.energy_wh = bat.duration_s, bat.energy_wh
        cand.usable_battery_wh, cand.battery_ok = bat.usable_wh, bat.battery_ok
        cand.reasons.extend(bat.reasons)
    else:
        n = len(transit.failures)
        cand.reasons.append(f"{n} required transit route(s) are infeasible. First: {transit.failures[0]}")
    cand.feasible = not cand.reasons
    return cand


def select_best(cands: list[Candidate], strategy: str) -> Optional[Candidate]:
    feas = [c for c in cands if c.feasible]
    if not feas:
        return None
    if strategy == STRATEGY_MAX_COVERAGE:
        key = lambda c: (-round(c.coverage_pct, 6), round(c.total_distance_m, 3), c.spacing_m)
    else:
        key = lambda c: (round(c.total_distance_m, 3), -round(c.coverage_pct, 6), c.spacing_m)
    return min(feas, key=key)


def optimize(farm, cfg: PlanConfig,
             progress: Optional[Callable[[int, int, float], None]] = None) -> OptimizationResult:
    errs = validate_config(cfg)
    if errs:
        raise ConfigError(errs)
    if abs(farm.margin_m - cfg.flight.safety_margin_m) > 1e-9 or \
            abs(farm.outside_margin_m - cfg.flight.transit_outside_margin_m) > 1e-9:
        raise ConfigError(["The farm was built with different margins than the current configuration."])
    swath = compute_swath(cfg.camera)
    base = swath.effective_m if cfg.spacing_mode == SPACING_CAMERA else cfg.manual_spacing_m
    spacings = generate_candidates(cfg, swath)
    router = TransitRouter.from_farm(farm)
    cands: list[Candidate] = []
    for i, s in enumerate(spacings):
        if progress:
            progress(i, len(spacings), s)
        cands.append(evaluate_candidate(farm, cfg, s, router, swath))
    best = select_best(cands, cfg.strategy)
    expl: list[str] = []
    if best is None:
        msg = "No candidate satisfied all mandatory constraints, so no mission was selected."
        expl.append(msg)
    else:
        best.selected = True
        n_feas = sum(c.feasible for c in cands)
        if cfg.strategy == STRATEGY_MAX_COVERAGE:
            why = "it has the highest estimated coverage among feasible candidates (ties: shorter distance, then smaller spacing)"
        else:
            why = (f"it has the shortest total distance among feasible candidates that reach at least "
                   f"{cfg.min_coverage_pct:.0f}% coverage (ties: higher coverage, then smaller spacing)")
        msg = (f"Selected spacing {best.spacing_m:g} m: {why}. "
               f"{n_feas} of {len(cands)} candidates were feasible.")
        expl.append(msg)
        for c in cands:
            if c.feasible and not c.selected:
                expl.append(f"Spacing {c.spacing_m:g} m was feasible but not selected "
                            f"(coverage {c.coverage_pct:.1f}%, distance {c.total_distance_m:.0f} m).")
    for c in cands:
        if not c.feasible:
            expl.append(f"Spacing {c.spacing_m:g} m rejected: " + " ".join(c.reasons))
    return OptimizationResult(cands, best, swath, base, msg, expl, router)


def _f(v):
    return float("nan") if v is None else v


def candidates_dataframe(cands: list[Candidate]) -> pd.DataFrame:
    rows = []
    for c in cands:
        rows.append({
            "Spacing (m)": c.spacing_m, "Coverage (%)": round(c.coverage_pct, 2),
            "Covered (m²)": round(c.covered_area_m2, 1), "Uncovered (m²)": round(c.uncovered_area_m2, 1),
            "Segments": c.n_segments, "Coverage dist (m)": round(c.coverage_distance_m, 1),
            "Transit dist (m)": None if c.transit_distance_m is None else round(c.transit_distance_m, 1),
            "Total dist (m)": None if c.total_distance_m is None else round(c.total_distance_m, 1),
            "Duration (min)": None if c.duration_s is None else round(c.duration_s / 60.0, 2),
            "Energy (Wh)": None if c.energy_wh is None else round(c.energy_wh, 2),
            "Battery OK": c.battery_ok, "Violations": c.violations, "Transit OK": c.transit_ok,
            "Feasible": c.feasible,
            "Status": "SELECTED" if c.selected else ("Feasible" if c.feasible else "Rejected"),
            "Reasons": " | ".join(c.reasons),
        })
    return pd.DataFrame(rows)