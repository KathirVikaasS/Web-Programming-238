"""Plotly figures built from the actual computed geometries (local geometry, not satellite imagery)."""
from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

from .geometry import iter_polygons

COVERAGE_COLOR = "#1f6feb"
TRANSIT_COLOR = "#f08c00"
UAV_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728", "#17becf", "#8c564b", "#e377c2", "#bcbd22"]
NOTE = "Locally generated geometry - not satellite imagery."


def _ll(farm, coords):
    xs, ys = zip(*coords)
    lon, lat = farm.projection.to_lonlat(np.asarray(xs, float), np.asarray(ys, float))
    return [float(v) for v in np.atleast_1d(lon)], [float(v) for v in np.atleast_1d(lat)]


def _poly_trace(farm, geom, name, fill, line_color, dash=None, width=1.5, showlegend=True):
    xs, ys = [], []
    for poly in iter_polygons(geom):
        for ring in [poly.exterior, *poly.interiors]:
            lo, la = _ll(farm, list(ring.coords))
            xs += lo + [None]
            ys += la + [None]
    if not xs:
        return None
    return go.Scatter(x=xs, y=ys, mode="lines", fill="toself", fillcolor=fill, name=name,
                      line=dict(color=line_color, width=width, dash=dash), hoverinfo="name",
                      showlegend=showlegend)


def _line_trace(farm, items, name, color, dash=None, width=2.5, legendgroup=None):
    """items: list of (coords, hover_text)."""
    xs, ys, hov = [], [], []
    for coords, text in items:
        lo, la = _ll(farm, coords)
        xs += lo + [None]
        ys += la + [None]
        hov += [text] * len(lo) + [None]
    if not xs:
        return None
    return go.Scatter(x=xs, y=ys, mode="lines", name=name, text=hov, hoverinfo="text",
                      line=dict(color=color, width=width, dash=dash), legendgroup=legendgroup)


def _arrow_trace(farm, segments, color, name, legendgroup=None):
    xs, ys, sizes = [], [], []
    for s in segments:
        (x0, y0), (x1, y1) = s.start, s.end
        if s.length <= 0:
            continue
        ux, uy = (x1 - x0) / s.length, (y1 - y0) / s.length
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        off = 0.05 * s.length
        lo, la = _ll(farm, [(mx - ux * off, my - uy * off), (mx, my)])
        xs += lo + [None]
        ys += la + [None]
        sizes += [0, 9, 0]
    if not xs:
        return None
    return go.Scatter(x=xs, y=ys, mode="markers", name=name, hoverinfo="skip", legendgroup=legendgroup,
                      marker=dict(symbol="arrow", size=sizes, color=color, angleref="previous"))


def _base_traces(farm):
    traces = [
        _poly_trace(farm, farm.field, "Farm boundary", "rgba(46,160,67,0.08)", "#2e7d32", width=2.5),
        _poly_trace(farm, farm.nfz_buffered, f"Safety buffer ({farm.margin_m:g} m)",
                    "rgba(255,140,0,0.12)", "#ff8c00", dash="dash"),
        _poly_trace(farm, farm.nfz and __import__("shapely.ops", fromlist=["unary_union"]).unary_union(farm.nfz),
                    "Restricted zones", "rgba(220,20,60,0.35)", "#b00020"),
    ]
    lo, la = _ll(farm, [farm.depot_xy])
    traces.append(go.Scatter(x=lo, y=la, mode="markers", name="Depot", hoverinfo="name",
                             marker=dict(symbol="star", size=16, color="#111", line=dict(color="white", width=1))))
    return traces


def _layout(fig, farm, title):
    lat0 = _ll(farm, [(farm.field.centroid.x, farm.field.centroid.y)])[1][0]
    fig.update_layout(
        title=f"{title}  ({NOTE})", height=680, margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        xaxis=dict(title="Longitude (°)", constrain="domain", tickformat=".4f"),
        yaxis=dict(title="Latitude (°)", scaleanchor="x", scaleratio=1.0 / max(math.cos(math.radians(lat0)), 1e-6),
                   tickformat=".4f"),
        hovermode="closest")
    return fig


def figure_farm(farm, title="Farm preview"):
    fig = go.Figure([t for t in _base_traces(farm) if t is not None])
    return _layout(fig, farm, title)


def figure_plan(farm, segments, transit=None, uncovered=None, title="Mission plan"):
    fig = go.Figure([t for t in _base_traces(farm) if t is not None])
    if uncovered is not None and not uncovered.is_empty:
        t = _poly_trace(farm, uncovered, "Uncovered area", "rgba(120,120,120,0.45)", "rgba(80,80,80,0.6)", width=0.5)
        if t:
            fig.add_trace(t)
    if transit is not None:
        ok_items = [(r.path, f"Transit route<br>{r.length:.1f} m" + ("" if r.direct else "<br>detour"))
                    for r in transit.routes if r.feasible and len(r.path) >= 2]
        t = _line_trace(farm, ok_items, "Transit route", TRANSIT_COLOR, dash="dash", width=2)
        if t:
            fig.add_trace(t)
        bad = [([r.start, r.end], f"BLOCKED: {r.reason}") for r in transit.routes if not r.feasible]
        t = _line_trace(farm, bad, "BLOCKED transition (infeasible)", "#d00000", dash="dot", width=3)
        if t:
            fig.add_trace(t)
    items = [(list(s.line.coords), f"Track {s.id}<br>{s.length:.1f} m<br>{s.direction}") for s in segments]
    t = _line_trace(farm, items, "Coverage track", COVERAGE_COLOR, width=2.5)
    if t:
        fig.add_trace(t)
    a = _arrow_trace(farm, segments, COVERAGE_COLOR, "Track direction")
    if a:
        fig.add_trace(a)
    return _layout(fig, farm, title)


def figure_uavs(farm, multi, title="Multi-UAV simulation"):
    fig = go.Figure([t for t in _base_traces(farm) if t is not None])
    for u in multi.uavs:
        color = UAV_COLORS[(u.uav_id - 1) % len(UAV_COLORS)]
        group = f"uav{u.uav_id}"
        items = [(list(s.line.coords), f"UAV {u.uav_id} - track {s.id}<br>{s.length:.1f} m") for s in u.segments]
        t = _line_trace(farm, items, f"UAV {u.uav_id} tracks", color, width=3, legendgroup=group)
        if t:
            fig.add_trace(t)
        if u.transit:
            tr = [(r.path, f"UAV {u.uav_id} transit {r.length:.1f} m") for r in u.transit.routes
                  if r.feasible and len(r.path) >= 2]
            t = _line_trace(farm, tr, f"UAV {u.uav_id} transit", color, dash="dot", width=1.5, legendgroup=group)
            if t:
                fig.add_trace(t)
            bad = [([r.start, r.end], f"UAV {u.uav_id} BLOCKED: {r.reason}") for r in u.transit.routes if not r.feasible]
            t = _line_trace(farm, bad, f"UAV {u.uav_id} BLOCKED", "#d00000", dash="dot", width=3, legendgroup=group)
            if t:
                fig.add_trace(t)
    return _layout(fig, farm, title)


def comparison_charts(df, usable_battery_wh=None):
    colors = {"SELECTED": "#2ea043", "Feasible": "#1f6feb", "Rejected": "#d1242f"}
    f1 = go.Figure()
    for status, g in df.groupby("Status"):
        f1.add_trace(go.Scatter(x=g["Total dist (m)"], y=g["Coverage (%)"], mode="markers+text",
                                text=[f"{s:g} m" for s in g["Spacing (m)"]], textposition="top center",
                                name=status, marker=dict(size=12, color=colors.get(status, "#888"))))
    f1.update_layout(title="Coverage vs total mission distance (rejected transit-infeasible candidates have no distance)",
                     xaxis_title="Total mission distance (m)", yaxis_title="Estimated coverage (%)", height=420)
    f2 = go.Figure(go.Bar(x=df["Spacing (m)"].astype(str), y=df["Energy (Wh)"],
                          marker_color=[colors.get(s, "#888") for s in df["Status"]]))
    if usable_battery_wh is not None:
        f2.add_hline(y=usable_battery_wh, line_dash="dash", annotation_text="Usable battery")
    f2.update_layout(title="Estimated energy per candidate", xaxis_title="Track spacing (m)",
                     yaxis_title="Energy (Wh)", height=420)
    return f1, f2