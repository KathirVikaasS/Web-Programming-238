"""Greedy workload-balancing multi-UAV simulation.

NOT a time-synchronised scheduler: there are no departure times or trajectories, so this module
does NOT verify drone-to-drone collision avoidance. A geometric crossing is not a collision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .battery import assess
from .models import PlanConfig
from .transit import TransitPlan, TransitRouter, plan_transits


@dataclass
class UavResult:
    uav_id: int
    segments: list
    coverage_distance_m: float = 0.0
    transit_distance_m: Optional[float] = 0.0
    total_distance_m: Optional[float] = 0.0
    duration_s: Optional[float] = 0.0
    energy_wh: Optional[float] = 0.0
    usable_battery_wh: float = 0.0
    battery_ok: Optional[bool] = True
    transit_ok: bool = True
    feasible: bool = True
    reasons: list[str] = field(default_factory=list)
    transit: Optional[TransitPlan] = None


@dataclass
class MultiUavResult:
    uavs: list[UavResult]
    unassigned_segment_ids: list[int]
    infeasible_segment_ids: list[int]
    notes: list[str]

    @property
    def feasible(self) -> bool:
        return not self.unassigned_segment_ids and all(u.feasible for u in self.uavs)


def assign_segments(segments, n_uavs: int) -> list[list]:
    """Sort by descending length (ties by id); give each to the least-loaded UAV (ties by index)."""
    if n_uavs < 1:
        raise ValueError("At least one UAV is required.")
    loads = [0.0] * n_uavs
    groups: list[list] = [[] for _ in range(n_uavs)]
    for seg in sorted(segments, key=lambda s: (-s.length, s.id)):
        k = min(range(n_uavs), key=lambda i: (loads[i], i))
        groups[k].append(seg)
        loads[k] += seg.length
    for g in groups:
        g.sort(key=lambda s: s.id)  # fly assigned tracks in original sweep order
    return groups


def simulate_multi_uav(depot, segments, cfg: PlanConfig, router: TransitRouter,
                       n_uavs: Optional[int] = None) -> MultiUavResult:
    n = int(n_uavs if n_uavs is not None else cfg.num_uavs)
    groups = assign_segments(segments, n)
    assigned = {s.id for g in groups for s in g}
    unassigned = sorted(s.id for s in segments if s.id not in assigned)
    uavs: list[UavResult] = []
    infeasible_ids: list[int] = []
    for k, segs in enumerate(groups):
        u = UavResult(uav_id=k + 1, segments=segs)
        u.coverage_distance_m = sum(s.length for s in segs)
        u.usable_battery_wh = cfg.flight.battery_wh * (1 - cfg.flight.reserve_pct / 100.0)
        if not segs:
            u.transit = TransitPlan()
            uavs.append(u)
            continue
        tp = plan_transits(depot, segs, router)
        u.transit, u.transit_ok = tp, tp.feasible
        if tp.feasible:
            u.transit_distance_m = tp.total_length
            u.total_distance_m = u.coverage_distance_m + tp.total_length
            b = assess(u.total_distance_m, len(segs), cfg.flight, cfg.max_distance_m, cfg.max_energy_wh)
            u.duration_s, u.energy_wh, u.battery_ok = b.duration_s, b.energy_wh, b.battery_ok
            u.reasons.extend(b.reasons)
        else:
            u.transit_distance_m = u.total_distance_m = u.duration_s = u.energy_wh = None
            u.battery_ok = None
            u.reasons.extend(tp.failures)
        u.feasible = not u.reasons
        if not u.feasible:
            infeasible_ids.extend(s.id for s in segs)
        uavs.append(u)
    notes = ["Workload is balanced by coverage-track length only; transit distance is calculated afterwards.",
             "No synchronised departure times or trajectories are modelled: drone-to-drone collision "
             "avoidance is NOT verified, and a geometric route crossing is not by itself a collision."]
    return MultiUavResult(uavs, unassigned, sorted(infeasible_ids), notes)


def uav_dataframe(res: MultiUavResult) -> pd.DataFrame:
    def r(v, nd=1):
        return None if v is None else round(v, nd)
    return pd.DataFrame([{
        "UAV": u.uav_id, "Segments": len(u.segments),
        "Segment IDs": ",".join(str(s.id) for s in u.segments),
        "Coverage dist (m)": r(u.coverage_distance_m), "Transit dist (m)": r(u.transit_distance_m),
        "Total dist (m)": r(u.total_distance_m),
        "Duration (min)": None if u.duration_s is None else round(u.duration_s / 60, 2),
        "Energy (Wh)": r(u.energy_wh, 2), "Usable battery (Wh)": r(u.usable_battery_wh, 2),
        "Battery OK": u.battery_ok, "Transit OK": u.transit_ok, "Feasible": u.feasible,
        "Reasons": " | ".join(u.reasons),
    } for u in res.uavs])