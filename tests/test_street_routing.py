import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import street_routing as SR                                                    # noqa: E402

A, B = (40.8244, -73.9100), (40.7900, -73.9500)


class _Resp:
    def __init__(self, payload, status=200):
        self.payload, self.status = payload, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self.payload


OK = {"code": "Ok", "routes": [{"duration": 600.0, "distance": 5000.0, "geometry": {"coordinates": [[-73.91, 40.8244], [-73.93, 40.81], [-73.95, 40.79]]}}]}


def _patch(monkeypatch_calls, payload=OK, status=200, boom=False):
    def fake_get(url, params=None, timeout=None):
        monkeypatch_calls.append(url)
        if boom:
            raise ConnectionError("no internet")
        return _Resp(payload, status)
    SR.requests.get = fake_get


def test_parses_a_route_and_orders_lon_lat_correctly():
    calls, real = [], SR.requests.get
    try:
        _patch(calls)
        r = SR.StreetRouter(min_interval_s=0).route(A, B)
    finally:
        SR.requests.get = real
    assert r.source == "osrm" and abs(r.minutes - 10.0) < 1e-9 and abs(r.km - 5.0) < 1e-9
    assert r.coords[0] == (40.8244, -73.91) and r.coords[-1] == (40.79, -73.95)         # (lat, lon), not (lon, lat)
    assert f"{A[1]},{A[0]};{B[1]},{B[0]}" in calls[0]                                    # the server wants lon,lat


def test_repeat_is_cached_and_the_cache_survives_a_restart(tmp_path=None):
    import tempfile
    cache = Path(tempfile.mkdtemp()) / "c.json"
    calls, real = [], SR.requests.get
    try:
        _patch(calls)
        r1 = SR.StreetRouter(cache_path=cache, min_interval_s=0)
        r1.route(A, B); r1.route(A, B)
        assert len(calls) == 1                                                           # second call came from memory
        SR.StreetRouter(cache_path=cache, min_interval_s=0).route(A, B)
        assert len(calls) == 1                                                           # a new router read the file
    finally:
        SR.requests.get = real


def test_failures_fall_back_and_are_flagged_and_not_cached():
    real = SR.requests.get
    try:
        for kwargs in (dict(boom=True), dict(payload={"code": "NoRoute", "message": "x"}), dict(status=500)):
            _patch([], **kwargs)
            r = SR.StreetRouter(min_interval_s=0, backend="osrm")
            res = r.route(A, B)
            assert res.source == "straight-line fallback" and res.minutes > 0 and not r._cache
    finally:
        SR.requests.get = real


def test_works_without_osmnx_installed_and_profiles_pick_the_right_server():
    calls, real, saved = [], SR.requests.get, sys.modules.get("osmnx")
    try:
        sys.modules["osmnx"] = None                                                      # as on the LinuxONE VM
        _patch(calls)
        SR.StreetRouter(min_interval_s=0).route(A, B)
        SR.StreetRouter(profile="bike", min_interval_s=0).route(A, B)
    finally:
        SR.requests.get = real
        if saved is None:
            sys.modules.pop("osmnx", None)
        else:
            sys.modules["osmnx"] = saved
    assert "router.project-osrm.org" in calls[0] and "routed-bike" in calls[1]


def test_route_moves_adds_columns():
    calls, real = [], SR.requests.get
    zips = pd.DataFrame({"lat": [A[0], B[0]], "lon": [A[1], B[1]]}, index=["10001", "10002"])
    moves = pd.DataFrame({"from_zip": ["10001"], "to_zip": ["10002"], "drive_min": [9.0]})
    try:
        _patch(calls)
        out = SR.StreetRouter(min_interval_s=0).route_moves(moves, zips)
    finally:
        SR.requests.get = real
    assert {"street_min", "street_km", "route", "route_source"} <= set(out.columns) and out.street_min.iloc[0] == 10.0
