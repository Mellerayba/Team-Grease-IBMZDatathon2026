"""
street_routing.py - real street route and drive time for a move the model recommends.

    router = StreetRouter()                                             # nothing to install: asks a public routing server
    r = router.route((40.8244, -73.9100), (40.6446, -73.9560))          # (lat, lon) -> (lat, lon)
    r.minutes, r.km, r.coords, r.source                                 # drive time, distance, polyline for a map, where it came from
    moves = router.route_moves(moves_df, zips_table, n=6)               # adds street_min / street_km / route columns to a table of moves
    bike = StreetRouter(profile="bike")                                 # same, for e-bike units

Backends, tried in this order:
  1. OSRM over HTTP (default): one small request per route (about 0.2 s), only needs `requests`, so it works on the LinuxONE VM where OSMnx cannot
     be installed (OSMnx needs shapely, which has no s390x wheel). Uses the public demo servers: fair use is about one request per second, so
     routes are cached (in memory and, if `cache_path` is given, in a JSON file) and requests are spaced out.
     Times use OSRM's typical road speeds and include signal and turn penalties, but NO live traffic.
  2. OSMnx, if it is installed (free-flow at speed limits, no signals). Pass `graph=` to use a pre-built city graph.
  3. A straight-line estimate, clearly flagged in `source` so it is never mistaken for a street route.
`speed_factor` scales the result (above 1 = faster, e.g. a siren allowance); leave it at 1.0 until calibrated against real travel times.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

OSRM_URLS = {"car": "https://router.project-osrm.org/route/v1/driving",
             "bike": "https://routing.openstreetmap.de/routed-bike/route/v1/driving"}     # the demo server only has the car profile; this one has bike


@dataclass
class Route:
    minutes: float
    km: float
    coords: list            # [(lat, lon), ...] along the streets, ready for a map polyline
    source: str = "osrm"    # "osrm", "osmnx" or "straight-line fallback"


def _haversine_km(a, b) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(2 * 6371.0088 * np.arcsin(np.sqrt(h)))


class StreetRouter:
    def __init__(self, profile: str = "car", backend: str = "auto", url: str | None = None, graph=None, cache_path=None,
                 speed_factor: float = 1.0, min_interval_s: float = 1.0, timeout_s: float = 15.0, pad_m: float = 2500,
                 fallback_kmh: float = 30.0, fallback_detour: float = 1.35):
        if profile not in OSRM_URLS:
            raise ValueError(f"profile must be one of {list(OSRM_URLS)}")
        self.profile, self.backend, self.url = profile, backend, url or OSRM_URLS[profile]
        self.G, self.speed_factor, self.pad_m = graph, speed_factor, pad_m
        self.min_interval_s, self.timeout_s = min_interval_s, timeout_s
        self.fallback_kmh, self.fallback_detour = fallback_kmh, fallback_detour
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict = {}
        self._last_request = 0.0
        self._local: dict = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self._cache = {k: Route(**{**v, "coords": [tuple(c) for c in v["coords"]]}) for k, v in json.loads(self.cache_path.read_text()).items()}
            except (OSError, ValueError, TypeError):
                self._cache = {}

    # ------------------------------------------------------------------ backends
    def _osrm(self, o, d) -> Route:
        wait = self.min_interval_s - (time.time() - self._last_request)
        if wait > 0:
            time.sleep(wait)                                           # be polite to the public server
        self._last_request = time.time()
        r = requests.get(f"{self.url}/{o[1]},{o[0]};{d[1]},{d[0]}", params={"overview": "full", "geometries": "geojson"}, timeout=self.timeout_s)
        r.raise_for_status()
        j = r.json()
        if j.get("code") != "Ok" or not j.get("routes"):
            raise RuntimeError(f"OSRM: {j.get('code')} {j.get('message', '')}")
        rt = j["routes"][0]
        return Route(rt["duration"] / 60 / self.speed_factor, rt["distance"] / 1000, [(lat, lon) for lon, lat in rt["geometry"]["coordinates"]], "osrm")

    def _osmnx(self, o, d) -> Route:
        import networkx as nx
        import osmnx as ox                                             # optional: not installable on the VM
        G = self.G
        if G is None:
            key = (round(o[0], 3), round(o[1], 3), round(d[0], 3), round(d[1], 3))
            if key not in self._local:
                centre = ((o[0] + d[0]) / 2, (o[1] + d[1]) / 2)
                g = ox.graph_from_point(centre, dist=_haversine_km(o, d) * 1000 / 2 + self.pad_m, network_type="drive" if self.profile == "car" else "bike")
                self._local[key] = ox.add_edge_travel_times(ox.add_edge_speeds(g))
            G = self._local[key]
        a = ox.distance.nearest_nodes(G, X=o[1], Y=o[0])
        b = ox.distance.nearest_nodes(G, X=d[1], Y=d[0])
        nodes = nx.shortest_path(G, a, b, weight="travel_time")
        seconds = length_m = 0.0
        for u, v in zip(nodes[:-1], nodes[1:]):
            e = min(G[u][v].values(), key=lambda x: x["travel_time"])
            seconds += e["travel_time"]; length_m += e.get("length", 0.0)
        return Route(seconds / 60 / self.speed_factor, length_m / 1000, [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in nodes], "osmnx")

    def _fallback(self, o, d) -> Route:
        km = _haversine_km(o, d)
        return Route(km * self.fallback_detour / self.fallback_kmh * 60, km, [tuple(o), tuple(d)], "straight-line fallback")

    # ------------------------------------------------------------------ public API
    def route(self, origin, destination) -> Route:
        """origin / destination are (lat, lon)."""
        key = f"{self.profile}|{origin[0]:.4f},{origin[1]:.4f}|{destination[0]:.4f},{destination[1]:.4f}"
        if key in self._cache:
            return self._cache[key]
        order = {"auto": ["osrm", "osmnx"], "osrm": ["osrm"], "osmnx": ["osmnx"]}[self.backend]
        if self.G is not None and self.backend == "auto":
            order = ["osmnx", "osrm"]                                  # a pre-built city graph is instant and offline: prefer it
        for name in order:
            try:
                res = self._osrm(origin, destination) if name == "osrm" else self._osmnx(origin, destination)
                self._cache[key] = res
                self._save()
                return res
            except Exception:                                          # no internet, server busy, no path, osmnx missing: try the next backend
                continue
        return self._fallback(origin, destination)                    # not cached: a later retry may succeed

    def _save(self):
        if self.cache_path:
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(json.dumps({k: vars(v) for k, v in self._cache.items()}))
            except OSError:
                pass

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
    """Optional, needs OSMnx (not installable on the VM): one-time download of the five-borough drivable network for offline routing.
    data/raw is git-ignored. Then:  StreetRouter(graph=ox.load_graphml(path))"""
    import osmnx as ox
    places = places or ["Manhattan, New York, USA", "Brooklyn, New York, USA", "Queens, New York, USA",
                        "Bronx, New York, USA", "Staten Island, New York, USA"]
    G = ox.add_edge_travel_times(ox.add_edge_speeds(ox.graph_from_place(places, network_type="drive")))
    ox.save_graphml(G, path)
    return G
