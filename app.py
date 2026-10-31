"""Streamlit UI: AI-Based Agricultural UAV Coverage Planning and Optimization."""
from __future__ import annotations

import json
import logging
from dataclasses import replace

import pandas as pd
import streamlit as st
from shapely.geometry import Polygon

from planner.camera import CameraError, compute_swath
from planner.farm_validation import ValidationReport, build_farm, parse_geojson
from planner.models import (PRESET_DESCRIPTIONS, PRESETS, SPACING_CAMERA, SPACING_MANUAL,
                            STRATEGY_MAX_COVERAGE, STRATEGY_MIN_DISTANCE, CameraConfig, FarmInput,
                            FlightConfig, PlanConfig, apply_preset, compute_signature)
from planner.multi_uav import simulate_multi_uav, uav_dataframe
from planner.optimization import ConfigError, candidates_dataframe, optimize, validate_config
from planner.reporting import (DISCLAIMER, LIMITATIONS, candidates_csv, markdown_report, results_json,
                               routes_geojson, uav_csv)
from planner.sample import sample_farm_input, sample_geojson_dict
from planner.visualization import comparison_charts, figure_farm, figure_plan, figure_uavs

log = logging.getLogger("uav-planner")
st.set_page_config(page_title="AI-Based Agricultural UAV Coverage Planning", page_icon="🛰️", layout="wide")
SS = st.session_state
PAGES = ["Overview", "Farm Setup", "Mission Planning", "Optimization Comparison",
         "Multi-UAV Simulation", "Mission Report", "Help"]
STRATEGY_LABELS = {"Minimum distance": STRATEGY_MIN_DISTANCE, "Maximum coverage": STRATEGY_MAX_COVERAGE}
DEFAULTS = dict(altitude=20.0, fov=90.0, overlap=20.0, spacing_mode="Camera-derived", manual_spacing=30.0,
                orientation=0.0, margin=5.0, outside_margin=0.0, speed=8.0, battery=150.0, reserve=20.0,
                wh_per_m=0.02, turn_s=4.0, search_min=0.0, search_max=0.0, search_step=0.0, min_cov=90.0,
                max_dist=0.0, max_energy=0.0, strategy="Minimum distance", n_uavs=3,
                preset="Standard Coverage")


# ----------------------------------------------------------------------------- helpers
def plot(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_df(df):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True)


def fmt(v, nd=1, unit=""):
    return "n/a" if v is None else f"{v:,.{nd}f}{unit}"


def kept(name):
    return SS.get(f"keep_{name}", DEFAULTS[name])


def _prep(name):
    k = f"w_{name}"
    if k not in SS:
        SS[k] = SS.get(f"keep_{name}", DEFAULTS[name])
    return k


def w_number(label, name, **kw):
    v = st.number_input(label, key=_prep(name), **kw)
    SS[f"keep_{name}"] = v
    return v


def w_select(label, name, options, **kw):
    v = st.selectbox(label, options, key=_prep(name), **kw)
    SS[f"keep_{name}"] = v
    return v


def w_radio(label, name, options, **kw):
    v = st.radio(label, options, key=_prep(name), **kw)
    SS[f"keep_{name}"] = v
    return v


def goto(page):
    SS["page"] = page


# ----------------------------------------------------------------------------- farm state
def _empty_pts():
    return pd.DataFrame({"lat": pd.Series(dtype="float64"), "lon": pd.Series(dtype="float64")})


def _pts_df(pairs):  # pairs are (lon, lat)
    if not pairs:
        return _empty_pts()
    return pd.DataFrame([{"lat": float(la), "lon": float(lo)} for lo, la in pairs], columns=["lat", "lon"])


def _df_pairs(df):
    return [(float(r.lon), float(r.lat)) for r in df.dropna().itertuples(index=False)]


def init_state():
    if "ver" not in SS:
        SS.update(ver=0, dver=0, boundary_src=_empty_pts(), boundary_cur=_empty_pts(), nfz_src=[],
                  nfz_cur=[], depot_cur=None, farm_source="none")
    SS.setdefault("mode", "Simple")
    SS.setdefault("page", "Overview")


def set_farm_input(fi: FarmInput, source: str):
    SS["boundary_src"] = _pts_df(fi.boundary)
    SS["boundary_cur"] = SS["boundary_src"].copy()
    SS["nfz_src"] = [_pts_df(z) for z in fi.nfz]
    SS["nfz_cur"] = [d.copy() for d in SS["nfz_src"]]
    SS["depot_cur"] = fi.depot
    SS["farm_source"] = source
    SS["ver"] += 1
    SS["dver"] += 1
    SS.pop("result", None)


def load_sample():
    set_farm_input(sample_farm_input(), "Sample farm (synthetic demonstration data)")


def new_mission():
    set_farm_input(FarmInput(), "none")
    SS["page"] = "Farm Setup"


def reset_settings():
    for k in [k for k in SS.keys() if k.startswith(("keep_", "w_"))]:
        del SS[k]
    SS.pop("result", None)


def _sync_src():
    SS["boundary_src"] = SS["boundary_cur"].copy()
    SS["nfz_src"] = [d.copy() for d in SS["nfz_cur"]]


def add_zone():
    _sync_src()
    b = SS["boundary_cur"].dropna()
    c_lat, c_lon = (float(b.lat.mean()), float(b.lon.mean())) if len(b) else (0.0, 0.0)
    d = 0.0003
    df = _pts_df([(c_lon - d, c_lat - d), (c_lon + d, c_lat - d), (c_lon + d, c_lat + d), (c_lon - d, c_lat + d)])
    SS["nfz_src"].append(df)
    SS["nfz_cur"].append(df.copy())
    SS["ver"] += 1


def remove_zone(i):
    _sync_src()
    SS["nfz_src"].pop(i)
    SS["nfz_cur"].pop(i)
    SS["ver"] += 1


def _representative_depot():
    pts = _df_pairs(SS["boundary_cur"])
    if len(pts) >= 3:
        try:
            p = Polygon(pts).representative_point()
            return (p.x, p.y)
        except Exception:
            pass
    return None


def depot_to_field_point():
    d = _representative_depot()
    if d:
        SS["depot_cur"] = d
        SS["dver"] += 1


def current_input() -> FarmInput:
    return FarmInput(boundary=_df_pairs(SS["boundary_cur"]),
                     nfz=[p for p in (_df_pairs(d) for d in SS["nfz_cur"])],
                     depot=SS["depot_cur"])


# ----------------------------------------------------------------------------- config / farm / results
def build_config() -> PlanConfig:
    g = kept
    cfg = PlanConfig(
        spacing_mode=SPACING_CAMERA if g("spacing_mode") == "Camera-derived" else SPACING_MANUAL,
        manual_spacing_m=g("manual_spacing"),
        search_min_m=g("search_min") or None, search_max_m=g("search_max") or None,
        search_step_m=g("search_step") or None,
        strategy=STRATEGY_LABELS[g("strategy")], min_coverage_pct=g("min_cov"),
        max_distance_m=g("max_dist") or None, max_energy_wh=g("max_energy") or None,
        num_uavs=int(g("n_uavs")),
        camera=CameraConfig(g("altitude"), g("fov"), g("overlap")),
        flight=FlightConfig(g("speed"), g("battery"), g("reserve"), g("wh_per_m"), g("turn_s"),
                            g("margin"), g("orientation"), g("outside_margin")))
    if SS["mode"] == "Simple":
        cfg = apply_preset(cfg, g("preset"))  # same engine, preset -> real config values
    return cfg


def get_farm(cfg):
    fi = current_input()
    key = compute_signature(fi, cfg.flight.safety_margin_m, cfg.flight.transit_outside_margin_m)
    if SS.get("farm_key") != key:
        SS["farm_cache"] = build_farm(fi, cfg.flight.safety_margin_m, cfg.flight.transit_outside_margin_m)
        SS["farm_key"] = key
    return SS["farm_cache"]


def context():
    cfg = build_config()
    farm, rep = get_farm(cfg)
    sig = compute_signature(current_input(), cfg)
    return cfg, farm, rep, sig


def valid_result(sig):
    res = SS.get("result")
    return res if res and res["sig"] == sig else None


def show_report(rep: ValidationReport):
    for e in rep.errors:
        st.error(e)
    for w in rep.warnings:
        st.warning(w)


def generate(farm, cfg, sig):
    prog = st.progress(0.0, text="Evaluating candidate spacings...")
    try:
        opt = optimize(farm, cfg, progress=lambda i, n, s: prog.progress(
            i / n, text=f"Evaluating spacing {s:g} m ({i + 1}/{n})"))
        multi = simulate_multi_uav(farm.depot_xy, opt.selected.plan.segments, cfg, opt.router) if opt.selected else None
    except ConfigError as exc:
        prog.empty()
        for e in exc.errors:
            st.error(e)
        return
    except Exception as exc:  # noqa: BLE001 - keep stack traces away from end users
        prog.empty()
        log.exception("Planning failed")
        st.error(f"Planning failed unexpectedly ({type(exc).__name__}: {exc}). Please check the inputs.")
        return
    SS["result"] = dict(sig=sig, opt=opt, multi=multi, cfg=cfg, farm=farm)
    prog.empty()


# ----------------------------------------------------------------------------- pages
def page_overview():
    st.title("AI-Based Agricultural UAV Coverage Planning and Optimization")
    st.caption("Portfolio project - planning simulation using computational geometry and deterministic "
               "optimization (no trained machine-learning model).")
    st.info(DISCLAIMER)
    c1, c2 = st.columns(2)
    c1.button("➕ Create New Mission", type="primary", on_click=new_mission)
    c2.button("Try the sample farm", on_click=lambda: (load_sample(), goto("Mission Planning")))
    st.markdown("""
**Workflow:** 1) Create a mission → 2) provide the farm boundary (sample, GeoJSON or GPS table) →
3) review the map → 4) add restricted areas and the depot → 5) choose a preset →
6) **Generate Best Mission** → 7) review results → 8) download the report.""")
    st.subheader("Status")
    cfg, farm, rep, sig = context()
    st.write(f"- Farm loaded: **{'yes' if current_input().boundary else 'no'}** ({SS['farm_source']})")
    st.write(f"- Farm valid: **{'yes' if farm else 'no'}**")
    st.write(f"- Current results: **{'up to date' if valid_result(sig) else 'none / outdated'}**")


def page_farm():
    st.header("Farm Setup")
    st.caption("Manual coordinates here are **latitude, longitude**. GeoJSON files use **[longitude, latitude]**. "
               "Interactive map drawing is not included; edit the coordinate tables or upload GeoJSON.")
    t1, t2, t3 = st.tabs(["Sample farm", "Upload GeoJSON", "Manual GPS entry"])
    with t1:
        st.write("A synthetic farm with an irregular boundary, two restricted zones and a depot. Works offline.")
        st.button("Load sample farm", on_click=load_sample)
        st.download_button("Download sample as GeoJSON", json.dumps(sample_geojson_dict(), indent=2),
                           "sample_farm.geojson", "application/geo+json")
    with t2:
        st.write("FeatureCollection with one Polygon `role: field`, optional Polygons `role: nfz`, "
                 "optional Point `role: depot`.")
        up = st.file_uploader("GeoJSON file", type=["geojson", "json"])
        if up is not None:
            fi, rep = parse_geojson(up.getvalue())
            show_report(rep)
            if fi is not None:
                st.button("Use this farm", on_click=set_farm_input, args=(fi, f"Uploaded file: {up.name}"))
    with t3:
        st.write("Type boundary points in the table below (add or delete rows), define restricted zones, "
                 "and set the depot.")
        st.button("Start a blank farm", on_click=lambda: set_farm_input(FarmInput(), "Manual entry"))
    st.divider()

    ver, dver = SS["ver"], SS["dver"]
    cc = {"lat": st.column_config.NumberColumn("Latitude (°)", format="%.6f", min_value=-90.0, max_value=90.0),
          "lon": st.column_config.NumberColumn("Longitude (°)", format="%.6f", min_value=-180.0, max_value=180.0)}
    left, right = st.columns([1, 2])
    with left:
        with st.expander("Field boundary (lat, lon points)", expanded=SS["mode"] == "Advanced" or not len(SS["boundary_cur"])):
            SS["boundary_cur"] = st.data_editor(SS["boundary_src"], num_rows="dynamic", key=f"b_ed_{ver}",
                                                column_config=cc, hide_index=True)
        st.markdown("**Restricted areas**")
        for i in range(len(SS["nfz_src"])):
            with st.expander(f"Restricted zone {i + 1}"):
                SS["nfz_cur"][i] = st.data_editor(SS["nfz_src"][i], num_rows="dynamic", key=f"z_ed_{ver}_{i}",
                                                  column_config=cc, hide_index=True)
                st.button("Remove this zone", key=f"rm_{ver}_{i}", on_click=remove_zone, args=(i,))
        st.button("Add restricted zone", on_click=add_zone)
        st.markdown("**Depot (start/return point)**")
        has = st.checkbox("Depot is set", value=SS["depot_cur"] is not None, key=f"hasdep_{dver}")
        if has:
            cur = SS["depot_cur"] or _representative_depot() or (0.0, 0.0)
            lat = st.number_input("Depot latitude (°)", value=float(cur[1]), format="%.6f", step=1e-5,
                                  min_value=-90.0, max_value=90.0, key=f"dlat_{dver}")
            lon = st.number_input("Depot longitude (°)", value=float(cur[0]), format="%.6f", step=1e-5,
                                  min_value=-180.0, max_value=180.0, key=f"dlon_{dver}")
            SS["depot_cur"] = (lon, lat)
        else:
            SS["depot_cur"] = None
        st.button("Place depot at a point inside the field", on_click=depot_to_field_point)
    cfg = build_config()
    farm, rep = get_farm(cfg)
    with right:
        show_report(rep)
        if farm:
            m = st.columns(3)
            m[0].metric("Field area", f"{farm.field_area_m2:,.0f} m²", f"{farm.field_area_m2 / 1e4:.2f} ha", delta_color="off")
            m[1].metric("Excluded (zones + margin)", f"{farm.excluded_area_m2:,.0f} m²")
            m[2].metric("Usable area", f"{farm.usable_area_m2:,.0f} m²", f"{farm.usable_area_m2 / 1e4:.2f} ha", delta_color="off")
            if SS["mode"] == "Advanced":
                st.caption(f"Projection: {farm.projection.label}. Safety margin {farm.margin_m:g} m.")
            plot(figure_farm(farm))
            st.button("Continue to Mission Planning →", on_click=goto, args=("Mission Planning",))
        else:
            st.info("Provide a valid farm boundary and depot to see the map.")


def _settings_simple():
    st.subheader("Choose a mission preset")
    preset = w_radio("Preset", "preset", list(PRESETS), label_visibility="collapsed")
    st.caption(PRESET_DESCRIPTIONS[preset])
    with st.expander("Camera and flight defaults used (change them in Advanced Mode)"):
        c = build_config()
        try:
            sw = compute_swath(c.camera)
            st.write(f"Camera {c.camera.altitude_agl_m:g} m above ground, {c.camera.fov_deg:g}° FOV, "
                     f"{c.camera.overlap_pct:g}% overlap → footprint {sw.footprint_m:.1f} m, "
                     f"effective swath {sw.effective_m:.1f} m.")
        except CameraError as exc:
            st.error(str(exc))
        st.write(f"Speed {c.flight.cruise_speed_mps:g} m/s, battery {c.flight.battery_wh:g} Wh "
                 f"({c.flight.reserve_pct:g}% reserve), {c.flight.wh_per_m:g} Wh/m, "
                 f"safety margin {c.flight.safety_margin_m:g} m.")


def _settings_advanced():
    a, b = st.columns(2)
    with a:
        st.subheader("Camera")
        w_number("Camera altitude above ground level (m)", "altitude", min_value=0.1, step=1.0, format="%.2f")
        w_number("Field of view (°)", "fov", min_value=1.0, max_value=179.0, step=1.0, format="%.1f")
        w_number("Image overlap (%)", "overlap", min_value=0.0, max_value=95.0, step=1.0, format="%.1f")
        try:
            sw = compute_swath(CameraConfig(kept("altitude"), kept("fov"), kept("overlap")))
            st.info(f"Footprint width **{sw.footprint_m:.2f} m**; effective swath **{sw.effective_m:.2f} m** "
                    "(idealised model; recommended starting spacing, not guaranteed optimal).")
        except CameraError as exc:
            st.error(str(exc))
        st.subheader("Spacing and search")
        w_select("Spacing mode", "spacing_mode", ["Camera-derived", "Manual"])
        w_number("Manual track spacing (m)", "manual_spacing", min_value=0.5, max_value=500.0, step=1.0, format="%.2f")
        w_number("Sweep orientation (° CCW from east)", "orientation", min_value=-180.0, max_value=180.0, step=5.0)
        w_number("Candidate spacing minimum (m, 0 = automatic)", "search_min", min_value=0.0, step=1.0)
        w_number("Candidate spacing maximum (m, 0 = automatic)", "search_max", min_value=0.0, step=1.0)
        w_number("Candidate spacing step (m, 0 = automatic)", "search_step", min_value=0.0, step=0.5)
    with b:
        st.subheader("Flight and battery")
        w_number("Cruise speed (m/s)", "speed", min_value=0.1, step=0.5)
        w_number("Battery capacity (Wh)", "battery", min_value=0.1, step=10.0)
        w_number("Battery reserve (%)", "reserve", min_value=0.0, max_value=99.0, step=1.0)
        w_number("Energy consumption (Wh/m)", "wh_per_m", min_value=0.0, step=0.001, format="%.4f")
        w_number("Turn-time overhead per segment (s)", "turn_s", min_value=0.0, step=0.5)
        st.subheader("Safety")
        w_number("Safety margin around restricted zones (m)", "margin", min_value=0.0, step=1.0)
        w_number("Allow transit outside field boundary by (m)", "outside_margin", min_value=0.0, step=1.0)
        st.subheader("Optimization constraints")
        w_select("Optimization strategy", "strategy", list(STRATEGY_LABELS))
        w_number("Minimum coverage requirement (%)", "min_cov", min_value=0.0, max_value=100.0, step=1.0)
        w_number("Maximum mission distance (m, 0 = none)", "max_dist", min_value=0.0, step=100.0)
        w_number("Maximum energy budget (Wh, 0 = none)", "max_energy", min_value=0.0, step=5.0)
        w_number("Number of simulated UAVs", "n_uavs", min_value=1, max_value=20, step=1)
        st.button("Reset all settings to defaults", on_click=reset_settings)


def metric_row(items):
    cols = st.columns(len(items))
    for c, (label, val) in zip(cols, items):
        c.metric(label, val)


def render_summary(res, advanced):
    opt, farm, cfg = res["opt"], res["farm"], res["cfg"]
    sel = opt.selected
    if sel is None:
        st.error("❌ No feasible mission. " + opt.message)
        st.caption("This is an infeasible plan, not invalid input. See the reasons below and in the comparison page.")
        for line in opt.explanation[1:6]:
            st.write("- " + line)
        return
    st.success(f"✅ Feasible mission selected - spacing {sel.spacing_m:g} m "
               f"({cfg.preset or STRATEGY_LABELS and cfg.strategy}).")
    metric_row([("Estimated coverage", f"{sel.coverage_pct:.1f}%"),
                ("Uncovered area", f"{sel.uncovered_area_m2 / 1e4:.2f} ha"),
                ("Mission distance", f"{sel.total_distance_m / 1000:.2f} km"),
                ("Duration", f"{sel.duration_s / 60:.1f} min")])
    metric_row([("Energy (est.)", f"{sel.energy_wh:.1f} Wh"), ("Usable battery", f"{sel.usable_battery_wh:.1f} Wh"),
                ("Battery status", "OK" if sel.battery_ok else "EXCEEDED"),
                ("Rejected candidates", f"{opt.n_rejected} of {len(opt.candidates)}")])
    if not advanced:
        st.caption("**Coverage:** estimated percentage of the usable farm area covered by the camera footprint. "
                   "**Uncovered area:** usable farm area that the estimated camera swaths do not cover. "
                   "**Mission distance:** coverage tracks plus validated transit routes. "
                   "**Battery status:** estimated energy demand compared with usable battery capacity.")
    else:
        metric_row([("Footprint width", f"{opt.swath.footprint_m:.2f} m"),
                    ("Effective swath", f"{opt.swath.effective_m:.2f} m"),
                    ("Coverage segments", str(sel.n_segments)), ("Covered area", f"{sel.covered_area_m2:,.0f} m²")])
        metric_row([("Coverage distance", f"{sel.coverage_distance_m:,.0f} m"),
                    ("Transit distance", f"{sel.transit_distance_m:,.0f} m"),
                    ("Detour transits", str(sel.transit.n_detours)), ("Safety violations", str(sel.violations))])
    st.caption("Safety validation: every coverage track and transit leg was checked as a full line against the "
               f"buffered restricted zones and the operational region ({sel.violations} violations).")
    plot(figure_plan(farm, sel.plan.segments, sel.transit, sel.plan.uncovered_geom,
                     f"Selected mission - spacing {sel.spacing_m:g} m"))
    if advanced:
        with st.expander("Routing diagnostics"):
            st.write(f"{len(sel.transit.routes)} transit legs; {sel.transit.n_detours} required obstacle-avoiding "
                     "detours; all legs validated segment by segment.")
        with st.expander("Selection explanation"):
            for line in opt.explanation:
                st.write("- " + line)
        with st.expander("Configuration used"):
            st.json(json.loads(results_json(farm, cfg, opt, res["multi"]))["config"])


def page_plan():
    st.header("Mission Planning")
    if SS["mode"] == "Simple":
        _settings_simple()
    else:
        _settings_advanced()
    cfg, farm, rep, sig = context()
    st.divider()
    if farm is None:
        st.warning("The farm is not valid yet. Fix these issues in Farm Setup:")
        show_report(rep)
        st.button("← Go to Farm Setup", on_click=goto, args=("Farm Setup",))
        return
    cfg_errors = validate_config(cfg)
    for e in cfg_errors:
        st.error(e)
    if st.button("🚀 Generate Best Mission", type="primary", disabled=bool(cfg_errors)):
        generate(farm, cfg, sig)
    res = valid_result(sig)
    if res:
        render_summary(res, SS["mode"] == "Advanced")
    elif SS.get("result"):
        st.warning("Inputs or settings changed since the last mission was generated. "
                   "Previous results are hidden - click Generate Best Mission again.")
    else:
        st.info("No mission generated yet.")


def page_compare():
    st.header("Optimization Comparison")
    cfg, farm, rep, sig = context()
    res = valid_result(sig)
    if not res:
        st.info("Generate a mission first (results must match the current inputs).")
        return
    opt = res["opt"]
    df = candidates_dataframe(opt.candidates)
    st.write(opt.message)
    simple_cols = ["Spacing (m)", "Coverage (%)", "Total dist (m)", "Energy (Wh)", "Status"]
    show_df(df if SS["mode"] == "Advanced" else df[simple_cols])
    usable = opt.candidates[0].usable_battery_wh if opt.candidates and opt.candidates[0].usable_battery_wh else \
        next((c.usable_battery_wh for c in opt.candidates if c.usable_battery_wh), None)
    f1, f2 = comparison_charts(df, usable)
    plot(f1)
    plot(f2)
    st.subheader("Why this result?")
    for line in opt.explanation:
        st.write("- " + line)


def page_multi():
    st.header("Multi-UAV Simulation")
    cfg, farm, rep, sig = context()
    res = valid_result(sig)
    if not res or res["multi"] is None:
        st.info("Generate a feasible mission first. The simulation distributes the selected mission's tracks.")
        return
    multi = res["multi"]
    st.warning(multi.notes[1])
    st.success("All UAV plans feasible.") if multi.feasible else st.error(
        f"Some UAV plans are infeasible (segments: {multi.infeasible_segment_ids}).")
    if multi.unassigned_segment_ids:
        st.error(f"Unassigned segments: {multi.unassigned_segment_ids}")
    show_df(uav_dataframe(multi))
    plot(figure_uavs(farm, multi, f"{len(multi.uavs)} UAVs (selected spacing {res['opt'].selected.spacing_m:g} m)"))
    st.caption(multi.notes[0] + " Change the number of UAVs in Advanced Mode (Mission Planning).")


def page_report():
    st.header("Mission Report")
    cfg, farm, rep, sig = context()
    res = valid_result(sig)
    if not res:
        st.info("Generate a mission first. Reports are only available for results that match current inputs.")
        return
    opt, multi, farm, cfg = res["opt"], res["multi"], res["farm"], res["cfg"]
    c = st.columns(5)
    c[0].download_button("Report (.md)", markdown_report(farm, cfg, opt, multi), "mission_report.md", "text/markdown")
    c[1].download_button("Results (.json)", results_json(farm, cfg, opt, multi), "mission_results.json", "application/json")
    c[2].download_button("Candidates (.csv)", candidates_csv(opt), "optimization_comparison.csv", "text/csv")
    if multi is not None:
        c[3].download_button("UAV workload (.csv)", uav_csv(multi), "uav_workload.csv", "text/csv")
        c[4].download_button("Routes (.geojson)", routes_geojson(farm, opt, multi), "routes.geojson", "application/geo+json")
    st.markdown(markdown_report(farm, cfg, opt, multi))


def page_help():
    st.header("Help")
    st.markdown(f"""
**{DISCLAIMER}**

### Creating a mission
1. **Farm Setup**: load the sample farm, upload GeoJSON, or type latitude/longitude points.
2. Add **restricted zones** (areas the drone must avoid) and set the **depot**.
3. **Mission Planning**: choose a preset (Simple) or tune settings (Advanced), then **Generate Best Mission**.
4. Review results, compare candidates, check the multi-UAV split, download the report.

### Input formats
- **GeoJSON**: FeatureCollection; one Polygon with `properties.role = "field"`, optional Polygons with `"nfz"`,
  optional Point with `"depot"`. Coordinates are **[longitude, latitude]**. Multi-polygons and polygons with holes are rejected.
- **Manual GPS**: table columns are **latitude, longitude** (the opposite order from GeoJSON).
- **Map drawing**: not implemented in this version.

### Camera, swath and spacing
- *Altitude* means height above **ground level**. *Field of view* is the camera's angle of view.
- **Footprint width** = 2 × altitude × tan(FOV / 2). **Effective swath** = footprint × (1 − overlap/100).
- **Track spacing** is the distance between neighbouring flight lines. Camera swath is how wide the camera sees.
  If spacing > swath, strips are left uncovered. The camera-derived value is only a starting point.
- This is an idealised model; terrain, camera tilt and lens distortion change real coverage.

### Optimization
The planner tries several spacings, runs the full coverage + routing + battery pipeline for each,
rejects infeasible ones, then picks: **Minimum distance** (shortest feasible mission meeting the coverage requirement)
or **Maximum coverage** (highest coverage among feasible). Ties are broken deterministically. It is search + geometry, not machine learning.

### Reading results
- **Coverage**: share of the usable area inside the estimated swaths. **Uncovered area**: usable area left out.
- **Battery**: estimated energy = distance × Wh/m compared with capacity × (1 − reserve).
- **Multi-UAV**: tracks sorted longest-first and assigned to the least-loaded UAV.

### What this cannot guarantee
""" + "\n".join(f"- {x}" for x in LIMITATIONS))


# ----------------------------------------------------------------------------- main
init_state()
st.sidebar.title("UAV Coverage Planner")
st.sidebar.radio("Mode", ["Simple", "Advanced"], key="mode", horizontal=True)
st.sidebar.radio("Navigate", PAGES, key="page")
{"Overview": page_overview, "Farm Setup": page_farm, "Mission Planning": page_plan,
 "Optimization Comparison": page_compare, "Multi-UAV Simulation": page_multi,
 "Mission Report": page_report, "Help": page_help}[SS["page"]]()
_cfg, _farm, _rep, _sig = context()
st.sidebar.divider()
st.sidebar.write(f"{'✅' if _farm else '⬜'} Farm valid")
st.sidebar.write(f"{'✅' if valid_result(_sig) else '⬜'} Mission generated (current)")
st.sidebar.button("Create another mission", on_click=new_mission)
st.sidebar.caption("Simulation only - not a certified aviation system.")