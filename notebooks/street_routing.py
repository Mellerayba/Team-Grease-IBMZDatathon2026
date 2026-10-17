"""
street_routing.py - real street route and drive time for a move the model recommends.

    router = StreetRouter()
    r = router.route((40.8244, -73.9100), (40.6446, -73.9560))     # (lat, lon) -> (lat, lon)
    r.minutes, r.km, r.coords                                       # drive time, distance, polyline for a map
    moves = router.route_moves(moves_df, zips_table)                # adds street_min / street_km / route columns

It uses the same method as Mapping.ipynb (OSMnx drivable network, speeds from the road type, shortest path by travel time) but needs
no pre-built matrix: for each requested pair it downloads only the streets around that trip (cached on disk by OSMnx, so a repeat
is instant). Pass `graph=` with a full-city graph (for example loaded with ox.load_graphml) to skip downloads entirely.

Times are FREE-FLOW at posted or imputed speed limits: no traffic lights, no congestion. `speed_factor` scales them (above 1 = faster,
for example a siren allowance; below 1 = slower, for congestion). Leave it at 1.0 unless it has been calibrated against real travel times.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import osmnx as ox

ox.settings.use_cache = True


@dataclass
class Route:
    minutes: float
    km: float
    coords: list            # [(lat, lon), ...] along the streets, ready for a map polyline
    source: str = "street"  # "street" or "straight-line fallback"


def _haversine_km(a, b) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(2 * 6371.0088 * np.arcsin(np.sqrt(h)))


class StreetRouter:
    def __init__(self, graph=None, speed_factor: float = 1.0, pad_m: float = 2500, fallback_kmh: float = 30.0, fallback_detour: float = 1.35):
        self.G = graph
        if graph is not None and "travel_time" not in next(iter(graph.edges(data=True)))[2]:
            self.G = ox.add_edge_travel_times(ox.add_edge_speeds(graph))
        self.speed_factor, self.pad_m = speed_factor, pad_m
        self.fallback_kmh, self.fallback_detour = fallback_kmh, fallback_detour
        self._local: dict = {}

    def _graph_for(self, o, d, pad_m=None):
        if self.G is not None:
            return self.G
        pad = self.pad_m if pad_m is None else pad_m
        key = (round(o[0], 3), round(o[1], 3), round(d[0], 3), round(d[1], 3), pad)
        if key not in self._local:
            centre = ((o[0] + d[0]) / 2, (o[1] + d[1]) / 2)
            dist = _haversine_km(o, d) * 1000 / 2 + pad                  # half the trip plus a margin, so both ends and the detours fit
            g = ox.graph_from_point(centre, dist=dist, network_type="drive")
            self._local[key] = ox.add_edge_travel_times(ox.add_edge_speeds(g))
        return self._local[key]

    def route(self, origin, destination) -> Route:
        """origin / destination are (lat, lon). Falls back to a straight-line estimate (flagged) if no street path is found."""
        pads = [None] if self.G is not None else [self.pad_m, self.pad_m * 3]     # if the first area cuts off a bridge, retry with a wider one
        for pad in pads:
            try:
                G = self._graph_for(origin, destination, pad)
                o = ox.distance.nearest_nodes(G, X=origin[1], Y=origin[0])
                d = ox.distance.nearest_nodes(G, X=destination[1], Y=destination[0])
                if o == d:
                    return Route(0.5, 0.0, [tuple(origin), tuple(destination)])
                nodes = nx.shortest_path(G, o, d, weight="travel_time")
                seconds = length_m = 0.0
                for u, v in zip(nodes[:-1], nodes[1:]):
                    e = min(G[u][v].values(), key=lambda x: x["travel_time"])     # the fastest of any parallel edges
                    seconds += e["travel_time"]
                    length_m += e.get("length", 0.0)
                coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in nodes]
                return Route(seconds / 60 / self.speed_factor, length_m / 1000, coords)
            except Exception:                                                     # no path inside the area, or a network error
                continue
        km = _haversine_km(origin, destination)                                   # last resort, clearly flagged in the result
        return Route(km * self.fallback_detour / self.fallback_kmh * 60, km, [tuple(origin), tuple(destination)], "straight-line fallback")

    def route_moves(self, moves, zips, n: int | None = None, from_col="from_zip", to_col="to_zip"):
        """Add real street routes to a moves table. `zips` is indexed by ZIP with lat and lon columns (the replay's ZIP table)."""
        m = moves.head(n).copy() if n else moves.copy()
        routes = [self.route((zips.loc[a, "lat"], zips.loc[a, "lon"]), (zips.loc[b, "lat"], zips.loc[b, "lon"]))
                  for a, b in zip(m[from_col], m[to_col])]
        m["street_min"] = [r.minutes for r in routes]
        m["street_km"] = [r.km for r in routes]
        m["route"] = [r.coords for r in routes]
        m["route_source"] = [r.source for r in routes]
        return m


def build_city_graph(path="data/raw/nyc_drive.graphml", places=None):
    """One-time download of the five-borough drivable network (takes several minutes), saved so routing is instant afterwards.
    data/raw is git-ignored, so the file is never committed. Then:  StreetRouter(graph=ox.load_graphml(path))"""
    places = places or ["Manhattan, New York, USA", "Brooklyn, New York, USA", "Queens, New York, USA",
                        "Bronx, New York, USA", "Staten Island, New York, USA"]
    G = ox.add_edge_travel_times(ox.add_edge_speeds(ox.graph_from_place(places, network_type="drive")))
    ox.save_graphml(G, path)
    return G
