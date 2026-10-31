"""Exports built from real computed results: Markdown, JSON, CSV, GeoJSON."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

from .multi_uav import uav_dataframe
from .optimization import candidates_dataframe

DISCLAIMER = ("This is a software planning simulation. It is not a certified aviation system, does not control "
              "drones, and gives no airspace authorisation.")
LIMITATIONS = [
    "Coverage is an idealised geometric estimate (flat swath rectangles); real coverage depends on terrain, "
    "camera orientation, lens distortion and capture pattern.",
    "Energy and duration are simple linear estimates: no wind, payload, acceleration or detailed manoeuvres.",
    "Camera altitude is treated as height above ground level (AGL); terrain following is not modelled.",
    "Transit routes stay inside the operational region and outside safety-buffered restricted zones; "
    "airspace rules and obstacles that were not entered are not considered.",
    "The planner is deterministic search plus computational geometry; it is not a trained machine-learning model.",
    "Multi-UAV simulation balances track length only and does not verify drone-to-drone collision avoidance.",
]


def _cand_summary(c):
    if c is None:
        return None
    return {"spacing_m": c.spacing_m, "coverage_pct": c.coverage_pct, "covered_area_m2": c.covered_area_m2,
            "uncovered_area_m2": c.uncovered_area_m2, "segments": c.n_segments,
            "coverage_distance_m": c.coverage_distance_m, "transit_distance_m": c.transit_distance_m,
            "total_distance_m": c.total_distance_m, "duration_s": c.duration_s, "energy_wh": c.energy_wh,
            "usable_battery_wh": c.usable_battery_wh, "battery_ok": c.battery_ok,
            "violations": c.violations, "transit_ok": c.transit_ok, "feasible": c.feasible,
            "selected": c.selected, "reasons": c.reasons}


def results_dict(farm, cfg, opt, multi) -> dict:
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "disclaimer": DISCLAIMER,
        "farm": {"projection": farm.projection.label, "field_area_m2": farm.field_area_m2,
                 "restricted_area_in_field_m2": farm.nfz_area_in_field_m2,
                 "excluded_area_m2": farm.excluded_area_m2, "usable_area_m2": farm.usable_area_m2,
                 "n_restricted_zones": len(farm.nfz)},
        "config": asdict(cfg),
        "camera": {"footprint_width_m": opt.swath.footprint_m, "effective_swath_m": opt.swath.effective_m},
        "mission_feasible": opt.selected is not None,
        "message": opt.message,
        "selected": _cand_summary(opt.selected),
        "candidates": [_cand_summary(c) for c in opt.candidates],
        "multi_uav": None if multi is None else {
            "feasible": multi.feasible, "notes": multi.notes,
            "unassigned_segment_ids": multi.unassigned_segment_ids,
            "infeasible_segment_ids": multi.infeasible_segment_ids,
            "uavs": json.loads(uav_dataframe(multi).to_json(orient="records"))},
        "limitations": LIMITATIONS,
    }


def results_json(farm, cfg, opt, multi) -> str:
    return json.dumps(results_dict(farm, cfg, opt, multi), indent=2, default=str)


def candidates_csv(opt) -> str:
    return candidates_dataframe(opt.candidates).to_csv(index=False)


def uav_csv(multi) -> str:
    return uav_dataframe(multi).to_csv(index=False)


def _line_feature(farm, coords, props):
    ll = farm.projection.to_lonlat_pairs(coords)
    return {"type": "Feature", "properties": props,
            "geometry": {"type": "LineString", "coordinates": [[round(a, 8), round(b, 8)] for a, b in ll]}}


def routes_geojson(farm, opt, multi=None) -> str:
    """Selected plan (single UAV) plus per-UAV routes if available. Coordinates are [lon, lat]."""
    feats = []
    sel = opt.selected
    if sel is not None and sel.plan is not None:
        for s in sel.plan.segments:
            feats.append(_line_feature(farm, list(s.line.coords),
                                       {"kind": "coverage", "id": s.id, "length_m": round(s.length, 3)}))
        for i, r in enumerate(sel.transit.routes):
            if r.feasible and len(r.path) >= 2:
                feats.append(_line_feature(farm, r.path, {"kind": "transit", "leg": i, "length_m": round(r.length, 3)}))
    if multi is not None:
        for u in multi.uavs:
            for s in u.segments:
                feats.append(_line_feature(farm, list(s.line.coords),
                                           {"kind": "uav_coverage", "uav": u.uav_id, "id": s.id}))
            if u.transit:
                for i, r in enumerate(u.transit.routes):
                    if r.feasible and len(r.path) >= 2:
                        feats.append(_line_feature(farm, r.path, {"kind": "uav_transit", "uav": u.uav_id, "leg": i}))
    return json.dumps({"type": "FeatureCollection", "mission_feasible": sel is not None,
                       "note": DISCLAIMER, "features": feats}, indent=2)


def _fmt(v, nd=1):
    return "n/a" if v is None else f"{v:,.{nd}f}"


def _md_table(df) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("n/a" if (v is None or v != v) else str(v) for v in row) + " |")
    return "\n".join(lines)


def markdown_report(farm, cfg, opt, multi) -> str:
    sel = opt.selected
    L = ["# UAV Coverage Mission Report", "",
         f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} (UTC)_", "",
         f"> {DISCLAIMER}", "",
         f"## Status: {'FEASIBLE' if sel else 'NO FEASIBLE MISSION'}", "", opt.message, "",
         "## Farm", "",
         f"- Projection: {farm.projection.label}",
         f"- Field area: {farm.field_area_m2:,.0f} m² ({farm.field_area_m2 / 1e4:.2f} ha)",
         f"- Excluded area (restricted zones + {farm.margin_m:g} m margin): {farm.excluded_area_m2:,.0f} m²",
         f"- Usable area: {farm.usable_area_m2:,.0f} m² ({farm.usable_area_m2 / 1e4:.2f} ha)",
         f"- Restricted zones: {len(farm.nfz)}", "",
         "## Parameters", "",
         f"- Preset: {cfg.preset or 'none (Advanced configuration)'}",
         f"- Strategy: {cfg.strategy}; minimum coverage {cfg.min_coverage_pct:g}%",
         f"- Camera: {cfg.camera.altitude_agl_m:g} m AGL, FOV {cfg.camera.fov_deg:g}°, overlap {cfg.camera.overlap_pct:g}%",
         f"- Footprint width {opt.swath.footprint_m:.2f} m; effective swath {opt.swath.effective_m:.2f} m",
         f"- Cruise speed {cfg.flight.cruise_speed_mps:g} m/s; battery {cfg.flight.battery_wh:g} Wh "
         f"(reserve {cfg.flight.reserve_pct:g}%); {cfg.flight.wh_per_m:g} Wh/m; turn overhead {cfg.flight.turn_overhead_s:g} s",
         f"- Sweep orientation {cfg.flight.orientation_deg:g}°; safety margin {farm.margin_m:g} m", ""]
    if sel:
        L += ["## Selected mission", "",
              f"- Track spacing: {sel.spacing_m:g} m",
              f"- Coverage: {sel.coverage_pct:.2f}% (covered {sel.covered_area_m2:,.0f} m², uncovered {sel.uncovered_area_m2:,.0f} m²)",
              f"- Coverage segments: {sel.n_segments}",
              f"- Coverage distance: {sel.coverage_distance_m:,.0f} m; transit distance: {sel.transit_distance_m:,.0f} m; "
              f"total: {sel.total_distance_m:,.0f} m",
              f"- Estimated duration: {sel.duration_s / 60:.1f} min",
              f"- Estimated energy: {sel.energy_wh:.1f} Wh vs usable battery {sel.usable_battery_wh:.1f} Wh "
              f"({'OK' if sel.battery_ok else 'EXCEEDED'})",
              f"- Restricted-zone/region violations: {sel.violations}; detour transits: {sel.transit.n_detours}", ""]
    L += [f"## Candidate comparison ({opt.n_rejected} rejected)", "",
          _md_table(candidates_dataframe(opt.candidates).drop(columns=["Covered (m²)", "Uncovered (m²)"])), "",
          "### Explanation", ""] + [f"- {e}" for e in opt.explanation] + [""]
    if multi is not None:
        L += [f"## Multi-UAV simulation ({len(multi.uavs)} UAVs) - {'all feasible' if multi.feasible else 'NOT all feasible'}", "",
              _md_table(uav_dataframe(multi)), ""] + [f"- {n}" for n in multi.notes] + [""]
    L += ["## Assumptions and limitations", ""] + [f"- {x}" for x in LIMITATIONS]
    return "\n".join(L)