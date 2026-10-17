"""Small pure-numpy geometry helpers (no geopandas/shapely needed)."""
import json
from pathlib import Path

import numpy as np


def load_geojson(path: Path) -> list[dict]:
    """Return features as [{'props': {...}, 'polys': [ring, ...]}] where each ring is an (n, 2) lon/lat array.

    Only exterior rings are kept (holes are ignored; fine at zone scale).
    """
    out = []
    for f in json.loads(Path(path).read_text())["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        out.append({"props": f["properties"], "polys": [np.asarray(p[0], dtype=float) for p in polys]})
    return out


def ring_centroid(ring: np.ndarray) -> tuple[float, float, float]:
    """Shoelace centroid of a lon/lat ring -> (lon, lat, signed area)."""
    x, y = ring[:, 0], ring[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = cross.sum() / 2
    if abs(a) < 1e-15:
        return float(x.mean()), float(y.mean()), 0.0
    return float(((x + x1) * cross).sum() / (6 * a)), float(((y + y1) * cross).sum() / (6 * a)), float(a)


def feature_centroid(polys: list[np.ndarray]) -> tuple[float, float]:
    """Area-weighted centroid over all exterior rings."""
    parts = [ring_centroid(r) for r in polys]
    w = np.array([abs(a) for _, _, a in parts])
    if w.sum() == 0:
        return float(np.mean([p[0] for p in parts])), float(np.mean([p[1] for p in parts]))
    return float(np.average([p[0] for p in parts], weights=w)), float(np.average([p[1] for p in parts], weights=w))


def point_in_ring(lon: float, lat: float, ring: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test."""
    x, y = ring[:, 0], ring[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    crosses = (y > lat) != (y1 > lat)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = (x1 - x) * (lat - y) / (y1 - y) + x
    return bool(np.count_nonzero(crosses & (lon < xint)) % 2)


def locate(lon: float, lat: float, features: list[dict], key: str):
    """Return props[key] of the first feature containing the point, else None."""
    for f in features:
        if any(point_in_ring(lon, lat, r) for r in f["polys"]):
            return f["props"][key]
    return None


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))
