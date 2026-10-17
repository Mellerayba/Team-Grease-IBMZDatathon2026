"""Export the demo data: replay one real day under two policies and write what the React demo plays back.

    python scripts/export_demo.py                         # heat-wave day 2022-07-19, 14:00-20:00, 330 units
    python scripts/export_demo.py --street-routes         # also fetch real street routes for the relocation moves (public OSRM, cached)

Needs notebooks/data/processed/forecast_zone_test.csv.gz (the forecaster's hand-off table) and notebooks/data/processed/incidents.csv.gz.
Writes pulsenyc-source/pulsenyc/public/data/{demo.json, zones.geojson}. Everything shown in the demo comes from the model's own
forecast and the replay: no number is typed in. Response times are MODELLED (see ems_replay.py).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "notebooks"), str(ROOT / "src")]
import ems_replay as R                                   # noqa: E402
from grease import risk                                  # noqa: E402

DATA = ROOT / "notebooks" / "data"
OUT = ROOT / "pulsenyc-source" / "pulsenyc" / "public" / "data"
KIND = {"relocation": 0, "respond": 1, "scene": 2, "transport": 3, "hospital": 4, "return": 5}


def r5(x):
    return round(float(x), 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="2022-07-19")
    ap.add_argument("--from-hour", type=int, default=14)
    ap.add_argument("--hours", type=int, default=6)
    ap.add_argument("--units", type=int, default=330)
    ap.add_argument("--street-routes", action="store_true", help="fetch street routes (OSRM) for relocation moves")
    args = ap.parse_args()

    zone_ctx = pd.read_csv(DATA / "interim" / "zone_context.csv", dtype={"zone": str}).set_index("zone")
    zone_ctx["label"] = zone_ctx["label"].fillna("Other / unassigned ZIPs")          # zone 99999 collects rare ZIPs
    vuln = risk.vulnerability(zone_ctx)
    zips = zone_ctx[["lat", "lon", "borough"]].copy()
    posts = pd.read_csv(DATA / "interim" / "posts.csv", dtype={"zone": str, "zipcode": str})
    hosp = posts[(posts.post_type == "hospital") & posts.zone.isin(zips.index)].reset_index(drop=True)
    hospital_zones = sorted(set(hosp.zone))

    ho = pd.read_csv(DATA / "processed" / "forecast_zone_test.csv.gz", dtype={"zone": str}, parse_dates=["bucket"])
    day = pd.Timestamp(args.day)
    day_end = day + pd.Timedelta("1D")
    win_start = day + pd.Timedelta(hours=args.from_hour)
    win_end = win_start + pd.Timedelta(hours=args.hours)

    inc = pd.read_csv(DATA / "processed" / "incidents.csv.gz", usecols=["incident_datetime", "zone", "acuity"],
                      dtype={"zone": str}, parse_dates=["incident_datetime"])
    inc = inc.rename(columns={"incident_datetime": "time", "zone": "zipcode"}).dropna(subset=["zipcode"])
    inc["acuity"] = np.where(inc.acuity.str.startswith("critical"), "critical", "other")
    train = inc[(inc.time >= "2021-01-01") & (inc.time < "2021-10-01")]
    # static layout input: average hourly demand per zone over the training months (never sees 2022)
    n_buckets = (pd.Timestamp("2021-10-01") - pd.Timestamp("2021-01-01")) / pd.Timedelta("30min")
    g = train.groupby("zipcode").acuity.value_counts().unstack(fill_value=0).reindex(zips.index, fill_value=0)
    static_fc = pd.DataFrame({"pred_critical_1h": g.get("critical", 0) / n_buckets * 2, "pred_other_1h": g.get("other", 0) / n_buckets * 2})
    incidents = inc[(inc.time >= day) & (inc.time < day_end)].sort_values("time").reset_index(drop=True)
    print(f"{day.date()}: {len(incidents):,} incidents, {len(hospital_zones)} hospital zones, {args.units} units")

    # forecast per 30-minute cycle in the form the policy reads it
    hd = ho[(ho.bucket >= day - pd.Timedelta("30min")) & (ho.bucket < day_end)]
    fc_by_bucket = {b: d.set_index("zone")[["pred_critical_1h", "pred_other_1h"]] for b, d in hd.groupby("bucket")}
    # oracle and scheduled baselines from the same table
    fc_oracle = {b: pd.DataFrame({"pred_critical_1h": d.set_index("zone").actual_critical_1h, "pred_other_1h": d.set_index("zone").actual_other_1h})
                 for b, d in hd.groupby("bucket")}

    kw = dict(n_units=args.units, start=day, end=day_end, hospital_zips=hospital_zones)
    hxy = {z: [(float(r.lat), float(r.lon)) for r in d.itertuples()] for z, d in hosp.groupby("zone")}
    runs = {
        "static": R.run_replay(zips, {}, incidents, static_fc, vuln, dynamic=False, label="static", trace=True, hospital_xy=hxy, **kw),
        "forecast": R.run_replay(zips, fc_by_bucket, incidents, static_fc, vuln, dynamic=True, label="dynamic: forecast", trace=True, hospital_xy=hxy, **kw),
        "oracle": R.run_replay(zips, fc_oracle, incidents, static_fc, vuln, dynamic=True, label="dynamic: oracle", **kw),
    }
    overall, by_borough = R.summarize(list(runs.values()), zips)
    print(overall.to_string())
    print(by_borough.to_string())

    # ---- zones: polygons (simplified) + per-cycle model output
    gj = json.load(open(DATA / "raw" / "modzcta_boundaries.geojson"))
    from shapely.geometry import shape, mapping
    feats = []
    for f in gj["features"]:
        zid = f["properties"]["modzcta"]
        if zid not in zips.index:
            continue
        geom = shape(f["geometry"]).simplify(0.0004, preserve_topology=True)
        m = mapping(geom)
        round_c = lambda c: [[round(x, 4), round(y, 4)] for x, y in c]
        if m["type"] == "Polygon":
            m = {"type": "Polygon", "coordinates": [round_c(r) for r in m["coordinates"]]}
        else:
            m = {"type": "MultiPolygon", "coordinates": [[round_c(r) for r in p] for p in m["coordinates"]]}
        feats.append({"type": "Feature", "properties": {"id": zid, "label": f["properties"]["label"]}, "geometry": m})
    zone_ids = [f["properties"]["id"] for f in feats]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "zones.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":")))

    cycles = []
    for t in pd.date_range(win_start, win_end, freq="30min", inclusive="left"):
        b = ho[ho.bucket == t - pd.Timedelta("30min")].set_index("zone").reindex(zone_ids)
        cycles.append(dict(
            t=int((t - win_start) / pd.Timedelta("1min")),
            crit=[r5(v) for v in b.pred_critical_1h.fillna(0)], other=[r5(v) for v in b.pred_other_1h.fillna(0)],
            risk=[round(float(v), 1) for v in b.risk_index_1h.fillna(0)],
            actual=[int(v) for v in (b.actual_critical_1h.fillna(0) + b.actual_other_1h.fillna(0))]))

    # ---- calls (same for every policy) and per-policy timelines
    base = runs["static"]["trace"]
    w0, w1 = (win_start - day) / pd.Timedelta("1min"), (win_end - day) / pd.Timedelta("1min")
    call_ids = [c["i"] for c in base["calls"] if w0 - 1 <= c["minute"] < w1]
    remap = {i: k for k, i in enumerate(call_ids)}
    calls = [dict(t=base["calls"][i]["minute"] - w0, zone=base["calls"][i]["zone"], acuity=base["calls"][i]["acuity"][0],
                  lat=r5(base["calls"][i]["xy"][0]), lon=r5(base["calls"][i]["xy"][1])) for i in call_ids]

    routes = {}                                          # (a, b) -> polyline, filled below when --street-routes is set
    router = None
    if args.street_routes:
        from street_routing import StreetRouter
        router = StreetRouter(cache_path=DATA / "raw" / "route_cache.json")

    def path_for(a, b):
        key = f"{a[0]:.4f},{a[1]:.4f}|{b[0]:.4f},{b[1]:.4f}"
        if key not in routes:
            rt = router.route(a, b)
            pts = rt.coords[::max(1, len(rt.coords) // 40)] + [rt.coords[-1]]
            routes[key] = ([[r5(p[0]), r5(p[1])] for p in pts] if rt.source == "osrm" else None)
        return routes[key]

    out_pol = {}
    for key, label in (("static", "Static layout"), ("forecast", "Model-guided")):
        tr = runs[key]["trace"]
        cinfo = {}
        for c in tr["calls"]:
            if c["i"] in remap and "dispatched" in c:
                cinfo[remap[c["i"]]] = [round(c["dispatched"] - w0, 1), round(c["arrived"] - w0, 1), round(c["done"] - w0, 1), int(c["unit"][1:])]
        units, why, moves = [], [], []
        for uid, u in tr["units"].items():
            legs = []
            for g in u["legs"]:
                if g["t1"] < w0 or g["t0"] > w1:
                    continue
                row = [round(g["t0"] - w0, 1), round(g["t1"] - w0, 1), KIND[g["kind"]], r5(g["a"][0]), r5(g["a"][1]), r5(g["b"][0]), r5(g["b"][1]),
                       remap.get(g.get("call"), -1)]
                if "why" in g:                                   # a move the policy chose (idle unit relocated, or a freed unit sent to a post)
                    why.append(g["why"])
                    row.append(len(why) - 1)
                    if router and g["t1"] > g["t0"]:
                        row.append(path_for(g["a"], g["b"]))
                    moves.append(dict(t=row[0], unit=int(uid[1:]), why=len(why) - 1, to=g["to_zip"], kind=g["kind"]))
                legs.append(row)
            # start-of-window position: end of the last leg before the window, else home
            before = [g for g in u["legs"] if g["t1"] < w0]
            start_pos = before[-1]["b"] if before else u["home"]
            units.append(dict(id=int(uid[1:]), lat=r5(start_pos[0]), lon=r5(start_pos[1]), legs=legs))
        out_pol[key] = dict(label=label, units=units, calls=cinfo, why=why, moves=moves)

    summ = overall.reset_index().astype(object).where(overall.reset_index().notna(), None).to_dict("records")
    doc = dict(
        meta=dict(day=str(day.date()), start=win_start.isoformat(), minutes=int((win_end - win_start) / pd.Timedelta("1min")),
                  units=args.units, cycle_min=30, target_min=8, day_incidents=len(incidents), window_incidents=len(calls),
                  note="Real 2022 incidents (NYC Open Data), model forecast from the held-out test year. Fleet, scene time, hospital time and "
                       "travel times are modelled assumptions; response times are modelled, not measured."),
        zone_ids=zone_ids,
        zones={z: dict(label=zone_ctx.loc[z, "label"], borough=zone_ctx.loc[z, "borough"], lat=r5(zone_ctx.loc[z, "lat"]),
                       lon=r5(zone_ctx.loc[z, "lon"]), vuln=round(float(vuln[z]), 2)) for z in zone_ids},
        hospitals=[dict(name=r["name"], borough=r["borough"], lat=r5(r["lat"]), lon=r5(r["lon"])) for _, r in hosp.iterrows()],
        cycles=cycles, calls=calls, policies=out_pol,
        day_summary=[{k: (round(float(v), 2) if isinstance(v, (int, float, np.floating, np.integer)) and v is not None else v) for k, v in row.items()} for row in summ],
        by_borough=json.loads(by_borough.round(1).to_json()))
    (OUT / "demo.json").write_text(json.dumps(doc, separators=(",", ":"), default=str, allow_nan=False))
    kb = (OUT / "demo.json").stat().st_size / 1e3
    print(f"wrote {OUT}/demo.json ({kb:,.0f} kB), zones.geojson | window {win_start} -> {win_end}: {len(calls)} calls, "
          f"{sum(len(p['moves']) for p in out_pol.values())} relocation moves, {len(routes)} street routes")


if __name__ == "__main__":
    main()
