"""Safe transit routing: direct validation + visibility graph + Dijkstra.

The routable region is the operational region (field, optionally widened, minus safety-buffered
restricted zones). A route is accepted only if the WHOLE polyline lies in that region.
Boundary contact is allowed (clearance equals the margin); a tiny numerical tolerance (0.1 mm)
is used. Shortest paths in a polygon with holes bend only at reflex vertices, so only those
are graph nodes.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Optional

from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.prepared import prep

from .geometry import iter_polygons

TOL_M = 1e-4
XY = tuple[float, float]


@dataclass
class RouteResult:
    feasible: bool
    path: list[XY]
    length: float
    reason: Optional[str] = None
    direct: bool = False
    start: Optional[XY] = None
    end: Optional[XY] = None


@dataclass
class TransitPlan:
    routes: list[RouteResult] = field(default_factory=list)
    feasible: bool = True
    total_length: float = 0.0
    failures: list[str] = field(default_factory=list)
    n_detours: int = 0


class TransitRouter:
    def __init__(self, region: BaseGeometry, obstacles: Optional[BaseGeometry] = None):
        self.region = region
        self._prepared = prep(region.buffer(TOL_M))
        self._core = None
        if obstacles is not None and not obstacles.is_empty:
            core = obstacles.buffer(-TOL_M)  # eroded: penetration deeper than TOL counts
            if not core.is_empty:
                self._core = prep(core)
        self.nodes = self._reflex_vertices(region)
        self._adj: Optional[list[list[tuple[int, float]]]] = None
        self._cache: dict = {}

    @classmethod
    def from_farm(cls, farm) -> "TransitRouter":
        return cls(farm.operational, farm.nfz_buffered)

    @staticmethod
    def _reflex_vertices(region: BaseGeometry) -> list[XY]:
        nodes: list[XY] = []
        seen = set()
        for poly in iter_polygons(region):
            poly = orient(poly, 1.0)  # exterior CCW, holes CW -> interior always on the left
            for ring in [poly.exterior, *poly.interiors]:
                pts = list(ring.coords)[:-1]
                n = len(pts)
                for i in range(n):
                    p0, p1, p2 = pts[i - 1], pts[i], pts[(i + 1) % n]
                    cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
                    if cross < -1e-9:  # right turn => reflex w.r.t. the region
                        key = (round(p1[0], 6), round(p1[1], 6))
                        if key not in seen:
                            seen.add(key)
                            nodes.append((p1[0], p1[1]))
        return nodes

    def edge_valid(self, a: XY, b: XY) -> bool:
        if a == b:
            return True
        line = LineString([a, b])
        return self._prepared.covers(line) and not (self._core is not None and self._core.intersects(line))

    def violates(self, line: LineString) -> bool:
        """Independent check: leaves the operational region or penetrates a restricted buffer."""
        if not self._prepared.covers(line):
            return True
        return self._core is not None and self._core.intersects(line)

    def _graph(self) -> list[list[tuple[int, float]]]:
        if self._adj is None:
            n = len(self.nodes)
            adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if self.edge_valid(self.nodes[i], self.nodes[j]):
                        w = math.dist(self.nodes[i], self.nodes[j])
                        adj[i].append((j, w))
                        adj[j].append((i, w))
            self._adj = adj
        return self._adj

    def route(self, a: XY, b: XY) -> RouteResult:
        a = (float(a[0]), float(a[1]))
        b = (float(b[0]), float(b[1]))
        key = (round(a[0], 4), round(a[1], 4), round(b[0], 4), round(b[1], 4))
        if key not in self._cache:
            self._cache[key] = self._route(a, b)
        return self._cache[key]

    def _route(self, a: XY, b: XY) -> RouteResult:
        for p, name in ((a, "start"), (b, "end")):
            if not self._prepared.covers(Point(p)):
                return RouteResult(False, [], math.inf, f"Route {name} point is outside the allowed region.", False, a, b)
        d = math.dist(a, b)
        if d < 1e-9:
            return RouteResult(True, [a], 0.0, None, True, a, b)
        if self.edge_valid(a, b):
            return RouteResult(True, [a, b], d, None, True, a, b)

        adj = self._graph()
        n = len(self.nodes)
        S, T = n, n + 1
        start_edges = {i: math.dist(a, nd) for i, nd in enumerate(self.nodes) if self.edge_valid(a, nd)}
        end_edges = {i: math.dist(nd, b) for i, nd in enumerate(self.nodes) if self.edge_valid(nd, b)}
        dist = {S: 0.0}
        prev: dict[int, int] = {}
        heap = [(0.0, S)]
        while heap:
            du, u = heapq.heappop(heap)
            if du > dist.get(u, math.inf):
                continue
            if u == T:
                break
            nbrs = list(start_edges.items()) if u == S else list(adj[u])
            if u != S and u in end_edges:
                nbrs.append((T, end_edges[u]))
            for v, w in nbrs:
                nd_ = du + w
                if nd_ < dist.get(v, math.inf) - 1e-12:
                    dist[v] = nd_
                    prev[v] = u
                    heapq.heappush(heap, (nd_, v))
        if T not in dist:
            return RouteResult(False, [], math.inf,
                               "No feasible obstacle-avoiding path exists inside the operational region.",
                               False, a, b)
        order = [T]
        while order[-1] != S:
            order.append(prev[order[-1]])
        order.reverse()
        path = [a] + [self.nodes[i] for i in order[1:-1]] + [b]
        if not all(self.edge_valid(p, q) for p, q in zip(path, path[1:])):
            return RouteResult(False, [], math.inf, "Computed route failed segment validation.", False, a, b)
        length = sum(math.dist(p, q) for p, q in zip(path, path[1:]))
        return RouteResult(True, path, length, None, False, a, b)


def plan_transits(depot: XY, segments, router: TransitRouter) -> TransitPlan:
    """depot -> first track, each track end -> next track start, last track -> depot."""
    if not segments:
        return TransitPlan()
    legs = [(depot, segments[0].start, "Depot -> first track")]
    for s, t in zip(segments, segments[1:]):
        legs.append((s.end, t.start, f"Track {s.id} -> track {t.id}"))
    legs.append((segments[-1].end, depot, f"Track {segments[-1].id} -> depot"))
    plan = TransitPlan()
    for a, b, label in legs:
        r = router.route(a, b)
        plan.routes.append(r)
        if r.feasible:
            plan.total_length += r.length
            if not r.direct:
                plan.n_detours += 1
        else:
            plan.failures.append(f"{label}: {r.reason}")
    plan.feasible = not plan.failures
    return plan