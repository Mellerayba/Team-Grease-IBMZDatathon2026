"""
ems_replay.py - glue between your trained forecaster and routing_policy.py, at ZIP-code level.

  load_zip_centroids / build_zip_table : ZIP -> borough, lat, lon (the policy's `zones` table)
  forecast_by_bucket                   : model `pred` -> {issue bucket: ZIP forecast table}
  static_forecast                      : training-period average demand, for the fixed-layout baseline
  load_incidents                       : real incident timestamps for the replay window
  run_replay                           : step through real incidents minute by minute under one policy
  summarize                            : response-time and equity comparison tables

Everything the data does not contain (fleet size, service times, hospital ZIPs, traffic) is a
parameter. Response times are MODELLED from the travel-time matrix, not measured.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from routing_policy import LiveState, PolicyConfig, RoutingPolicy, Unit


# ----------------------------------------------------------------------------- ZIP table
def norm_zip(s: pd.Series) -> pd.Series:
    """Five-digit ZIP strings ('10025.0' / 10025 / '10025-1234' -> '10025'); NaN if none found."""
    return s.astype(str).str.extract(r"(\d{5})", expand=False)


def load_zip_centroids(path_or_url: str) -> pd.DataFrame:
    """Census ZCTA gazetteer (tab-separated; GEOID, INTPTLAT, INTPTLONG) -> DataFrame(lat, lon) by ZIP."""
    g = pd.read_csv(path_or_url, sep="\t", dtype={"GEOID": str})
    g.columns = g.columns.str.strip()
    g["zipcode"] = norm_zip(g["GEOID"])
    return g.dropna(subset=["zipcode"]).set_index("zipcode")[["INTPTLAT", "INTPTLONG"]].rename(
        columns={"INTPTLAT": "lat", "INTPTLONG": "lon"}).astype(float)


def build_zip_table(zips_in_data, centroids: pd.DataFrame, zip_to_zone: pd.Series | None = None,
                    zone_ctx: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per ZIP with lat, lon and (if available) borough. ZIPs without a centroid are dropped."""
    z = pd.Index(sorted(set(pd.Series(list(zips_in_data)).pipe(norm_zip).dropna())), name="zipcode")
    t = centroids.reindex(z)
    missing = t.lat.isna()
    if missing.any():
        print(f"dropping {int(missing.sum())} ZIPs with no centroid: {list(z[missing])[:10]}")
    t = t[~missing].copy()
    if zip_to_zone is not None and zone_ctx is not None and "borough" in zone_ctx:
        zmap = zip_to_zone.copy()
        zmap.index = norm_zip(pd.Series(zmap.index.astype(str))).to_numpy()
        t["zone"] = t.index.map(zmap)
        t["borough"] = t["zone"].map(zone_ctx["borough"])
        print(f"borough known for {t.borough.notna().mean():.1%} of ZIPs")
    return t


# ----------------------------------------------------------------------------- forecast tables
def forecast_by_bucket(pred: pd.DataFrame, test: pd.DataFrame, start, end,
                       cols=("pred_critical_1h", "pred_other_1h"), issue_offset=pd.Timedelta("30min")):
    """{bucket: DataFrame indexed by ZIP}. Only buckets whose forecast is used inside [start, end)."""
    cols = list(cols)
    b = test.loc[pred.index, "bucket"]
    keep = ((b >= pd.Timestamp(start) - issue_offset) & (b < pd.Timestamp(end) - issue_offset)).to_numpy()
    df = pred.loc[keep, cols].copy()
    df["zipcode"] = norm_zip(test.loc[df.index, "zipcode"]).to_numpy()
    df["bucket"] = b[keep].to_numpy()
    return {k: g.set_index("zipcode")[cols] for k, g in df.dropna(subset=["zipcode"]).groupby("bucket")}


def static_forecast(train_df: pd.DataFrame, per_hour_buckets: float = 2.0) -> pd.DataFrame:
    """Average hourly demand per ZIP from training rows (needs zipcode, y, y_critical per 30-min bucket).
    Fixed-layout baseline: it must not see the test period."""
    d = train_df[["zipcode", "y", "y_critical"]].copy()
    d["zipcode"] = norm_zip(d["zipcode"])
    g = d.dropna(subset=["zipcode"]).groupby("zipcode")[["y", "y_critical"]].mean()
    return pd.DataFrame({"pred_critical_1h": g.y_critical * per_hour_buckets,
                         "pred_other_1h": (g.y - g.y_critical).clip(lower=0) * per_hour_buckets})


def load_incidents(paths, start, end, critical_codes=(1, 2, 3)) -> pd.DataFrame:
    """Real incident times from your ems_raw CSVs: columns time, zipcode, acuity."""
    parts = []
    for p in paths:
        d = pd.read_csv(p, usecols=["incident_datetime", "zipcode", "initial_severity_level_code"],
                        dtype={"zipcode": str})
        d["time"] = pd.to_datetime(d.pop("incident_datetime"), errors="coerce")
        parts.append(d[(d.time >= start) & (d.time < end)])
    inc = pd.concat(parts, ignore_index=True)
    inc["zipcode"] = norm_zip(inc["zipcode"])
    sev = pd.to_numeric(inc.pop("initial_severity_level_code"), errors="coerce")
    inc["acuity"] = np.where(sev.isin(critical_codes), "critical", "other")
    return inc.dropna(subset=["time", "zipcode"]).sort_values("time").reset_index(drop=True)


def default_traffic(ts: pd.Timestamp) -> float:
    """Assumed citywide slowdown by hour. Replace with a live feed or a fitted profile."""
    return 1.4 if (7 <= ts.hour < 10 or 16 <= ts.hour < 19) else 1.0


# ----------------------------------------------------------------------------- replay
def run_replay(zips: pd.DataFrame, fc_by_bucket: dict, incidents: pd.DataFrame, static_fc: pd.DataFrame,
               vulnerability: pd.Series | None, *, dynamic: bool, n_units: int, start, end,
               hospital_zips=(), cfg: PolicyConfig | None = None, issue_offset=pd.Timedelta("30min"),
               cycle_min: int = 30, scene_min: float = 15.0, traffic_fn=default_traffic, seed: int = 0,
               label: str | None = None):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    assert start.minute % cycle_min == 0 and start.second == 0, "start must sit on a cycle boundary"
    rng = np.random.default_rng(seed)
    pol = RoutingPolicy(zips, cfg or PolicyConfig())

    inc = incidents[(incidents.time >= start) & (incidents.time < end)]
    known = inc.zipcode.isin(pol.ix)
    dropped = int((~known).sum())
    inc = inc[known].reset_index(drop=True)
    n = len(inc)
    inc_m = ((inc.time - start).dt.total_seconds() // 60).astype(int).to_numpy()
    inc_z = inc.zipcode.to_numpy()
    inc_ac = inc.acuity.to_numpy()
    inc_zi = np.array([pol.ix[z] for z in inc_z], dtype=int)
    svc = rng.uniform(10, 25, n)                      # minutes at the hospital; drawn up front so every
                                                      # policy replays identical incidents
    hz = [z for z in hospital_zips if z in pol.ix]
    if hz:
        hidx = np.array([pol.ix[z] for z in hz])
        inc_h = hidx[np.argmin(pol.tt_base[inc_zi][:, hidx], axis=1)]
    else:
        inc_h = inc_zi                                # no hospital list: unit frees up where it was sent
    by_min = defaultdict(list)
    for i, m in enumerate(inc_m):
        by_min[m].append(i)

    home = pol.optimal_layout(static_fc, n_units, LiveState(start, [], 1.0), vulnerability)
    U = {f"U{i}": dict(zone=home[i], home=home[i], status="idle", free_at=0.0, free_zone=home[i],
                       moves=[], freed=False, job=False, home_at=None, move_to=None, arrive_at=None) for i in range(n_units)}

    def mk_state(m, t, planning=False):
        # A repositioning unit stays IDLE and dispatchable while it drives (as in real life). The planner is told where it is
        # heading (planning=True) so it does not move it again; dispatch is told where it still is (planning=False, conservative).
        return LiveState(t, [Unit(k, (u["move_to"] if planning and u["move_to"] else u["zone"]), u["status"],
                                  max(u["free_at"] - m, 0.0), u["free_zone"], len(u["moves"]), u["freed"])
                             for k, u in U.items()], traffic_fn(t), 0.0)

    queue, resp, cycles, moves_log = [], [], [], []
    for m in range(int((end - start).total_seconds() // 60)):
        t = start + pd.Timedelta(minutes=m)
        tm = traffic_fn(t)
        for u in U.values():
            if u["move_to"] is not None and u["arrive_at"] <= m:      # a repositioning unit has arrived
                u["zone"], u["move_to"], u["arrive_at"] = u["move_to"], None, None
            if not dynamic and u["status"] == "idle" and u["home_at"] is not None and u["home_at"] <= m:
                u["zone"], u["home_at"] = u["home"], None        # static unit has driven home
            if u["status"] == "busy" and u["free_at"] <= m:
                u["status"], u["zone"], u["freed"], u["job"] = "idle", u["free_zone"], u["job"], False
            u["moves"] = [x for x in u["moves"] if x > m - 60]

        if dynamic and m % cycle_min == 0:
            fc = fc_by_bucket.get(t - issue_offset)
            if fc is None:
                cycles.append(dict(time=t, note="no forecast for this cycle: held positions"))
            else:
                dec = pol.reposition(fc, mk_state(m, t, planning=True), vulnerability)
                for mv in dec.moves:
                    u = U[mv.unit_id]
                    u.update(move_to=mv.to_zone, arrive_at=m + mv.travel_min, freed=False)
                    if mv.kind == "relocation":
                        u["moves"].append(m)
                    moves_log.append(dict(time=t, unit=mv.unit_id, kind=mv.kind, from_zip=mv.from_zone,
                                          to_zip=mv.to_zone, drive_min=mv.travel_min, gain=mv.gain, reason=mv.reason))
                for u in U.values():
                    if u["status"] == "idle":
                        u["freed"] = False
                cycles.append(dict(time=t, moves=len(dec.moves), latency_ms=dec.latency_ms,
                                   crit_covered_before=dec.critical_covered_before,
                                   crit_covered_after=dec.critical_covered_after, note="; ".join(dec.notes)))

        queue += by_min.get(m, [])
        if queue and any(u["status"] == "idle" for u in U.values()):
            queue.sort(key=lambda i: (inc_ac[i] != "critical", inc_m[i]))
            st = mk_state(m, t)
            su = {u.id: u for u in st.units}
            rest = []
            for i in queue:
                ch = pol.dispatch(inc_z[i], inc_ac[i], st, smart=dynamic) \
                    if any(x.status == "idle" for x in st.units) else None
                if ch is None:
                    rest.append(i)
                    continue
                u = U[ch.unit_id]
                hzi = inc_h[i]
                busy = ch.eta_min + scene_min + svc[i] + pol.tt_base[inc_zi[i], hzi] * tm
                u.update(status="busy", free_at=m + busy, free_zone=pol.ids[hzi], job=True, freed=False, move_to=None, arrive_at=None)
                if not dynamic:                               # static units drive home but stay dispatchable
                    u["home_at"] = m + busy + pol.tt_base[hzi, pol.ix[u["home"]]] * tm
                su[ch.unit_id].status, su[ch.unit_id].free_in_min = "busy", busy
                su[ch.unit_id].free_zone = pol.ids[hzi]
                resp.append(dict(time=inc.time[i], zipcode=inc_z[i], acuity=inc_ac[i],
                                 wait=m - inc_m[i], eta=ch.eta_min, minutes=(m - inc_m[i]) + ch.eta_min))
            queue = rest

    return dict(policy=label or ("dynamic" if dynamic else "static"), responses=pd.DataFrame(resp),
                cycles=pd.DataFrame(cycles), moves=pd.DataFrame(moves_log), unserved=len(queue),
                dropped_incidents=dropped, n_incidents=n)


def summarize(results: list[dict], zips: pd.DataFrame | None = None, target_min: float = 8.0):
    """(overall table, critical-calls-within-target by borough). Modelled response times."""
    rows, boro = [], {}
    for r in results:
        d = r["responses"]
        c = d[d.acuity == "critical"]
        cy = r["cycles"]
        rows.append(dict(policy=r["policy"], incidents=r["n_incidents"], dropped=r["dropped_incidents"], unserved=r["unserved"],
                         crit_within_target_pct=100 * (c.minutes <= target_min).mean(),
                         crit_mean_min=c.minutes.mean(), crit_p90_min=c.minutes.quantile(.9),
                         all_mean_min=d.minutes.mean(), moves=len(r["moves"]),
                         median_cycle_ms=cy["latency_ms"].median() if "latency_ms" in cy else np.nan))
        if zips is not None and "borough" in zips:
            cb = c.assign(borough=c.zipcode.map(zips["borough"]))
            boro[r["policy"]] = cb.groupby("borough").minutes.apply(lambda s: 100 * (s <= target_min).mean())
    return pd.DataFrame(rows).set_index("policy").round(2), pd.DataFrame(boro).round(1)
