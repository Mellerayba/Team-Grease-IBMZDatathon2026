"""
demo_policy.py - checks routing_policy.py end to end on SYNTHETIC data.
Replace make_world()/forecast_at() with your zones, zone_ctx and model predictions.
Numbers from this script show the code works; they say nothing about real NYC performance.
"""
import numpy as np
import pandas as pd
from routing_policy import RoutingPolicy, PolicyConfig, LiveState, Unit

N_UNITS, SCENE_MIN = 30, 15
HOTSPOT = 2.5          # strength of the moving demand hotspot; raise it to stress-test dynamic deployment


def make_world(seed=1):
    rng = np.random.default_rng(seed)
    names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
    centers = [(40.85, -73.88), (40.65, -73.95), (40.78, -73.97), (40.72, -73.80), (40.58, -74.15)]
    codes = ["BX", "BK", "MN", "QN", "SI"]
    rows = []
    for code, b, (la, lo) in zip(codes, names, centers):
        for i in range(8):
            rows.append((f"{code}{i}", b, la + rng.normal(0, .03), lo + rng.normal(0, .04)))
    z = pd.DataFrame(rows, columns=["zone", "borough", "lat", "lon"]).set_index("zone")
    base = rng.lognormal(0, .6, len(z)) * np.where(z.borough == "Manhattan", 2.0,
                                                    np.where(z.borough == "Staten Island", .5, 1.0))
    vuln = pd.Series(rng.uniform(0, 1, len(z)), index=z.index)
    return z, base, vuln


def rate_30(zones, base, minute, heat=1.0):
    """True expected incidents per 30 min per zone: diurnal x moving hotspot x heat factor."""
    hr = (minute / 60) % 24
    diurnal = 0.7 + 0.5 * np.sin((hr - 9) / 24 * 2 * np.pi)
    f = np.clip(minute / (24 * 60), 0, 1)                      # hotspot drifts Bronx -> Queens
    c = np.array([40.85, -73.88]) * (1 - f) + np.array([40.72, -73.80]) * f
    d2 = (zones.lat - c[0]) ** 2 + (zones.lon - c[1]) ** 2
    return base * diurnal * (1 + HOTSPOT * np.exp(-d2 / (2 * 0.05 ** 2))) * heat * 0.07


def forecast_at(zones, base, minute, rng, heat, noise=0.2):
    """Stand-in for the LightGBM output: noisy truth for the next hour, split critical/other."""
    r = 2 * rate_30(zones, base, minute + 30, heat) * rng.lognormal(0, noise, len(zones))
    return pd.DataFrame({"pred_critical_1h": 0.2 * r, "pred_other_1h": 0.8 * r}, index=zones.index)


def traffic(minute):
    h = (minute / 60) % 24
    return 1.4 if (7 <= h < 10 or 16 <= h < 19) else 1.0


def make_incidents(zones, base, hours, heat, seed):
    """Same incidents, hospitals and service times for every policy (common random numbers)."""
    rng = np.random.default_rng(1000 + seed)
    hosp = [z for z in zones.index if z.endswith("0")]
    out = []
    for m in range(hours * 60):
        for z, n in enumerate(rng.poisson(rate_30(zones, base, m, heat) / 30)):
            for _ in range(n):
                out.append((m, zones.index[z], "critical" if rng.random() < .2 else "other",
                            hosp[rng.integers(len(hosp))], rng.uniform(10, 25)))
    return out


def run(dynamic, seed=0, hours=24, heat=1.3):
    rng = np.random.default_rng(seed)
    zones, base, vuln = make_world()
    pol = RoutingPolicy(zones, PolicyConfig(max_moves_per_hour=3))
    incidents = make_incidents(zones, base, hours, heat, seed)
    by_min = {}
    for inc in incidents:
        by_min.setdefault(inc[0], []).append(inc)
    avg_fc = sum(forecast_at(zones, base, m, np.random.default_rng(0), heat, 0) for m in range(0, 1440, 60)) / 24
    home = pol.optimal_layout(avg_fc, N_UNITS, LiveState(pd.Timestamp("2022-07-19"), [], 1.0), vuln)
    U = {f"U{i}": dict(zone=home[i], home=home[i], status="idle", free_at=0.0, free_zone=home[i], moves=[])
         for i in range(N_UNITS)}
    queue, resp, n_moves = [], [], 0
    if not dynamic:
        pol._last = None

    def state(m):
        return LiveState(pd.Timestamp("2022-07-19") + pd.Timedelta(minutes=m), [
            Unit(k, u["zone"], u["status"], max(u["free_at"] - m, 0), u["free_zone"],
                 sum(t > m - 60 for t in u["moves"]), u.get("freed", False)) for k, u in U.items()],
            traffic(m), 0.0)

    for m in range(hours * 60):
        for u in U.values():
            if not dynamic and u["status"] == "idle" and u.get("home_at") and u["home_at"] <= m:
                u["zone"], u["home_at"] = u["home"], None            # static unit finished driving home
            if u["status"] == "busy" and u["free_at"] <= m:
                u["status"], u["zone"], u["freed"] = "idle", u["free_zone"], u.get("job", False)
                u["job"] = False
        if dynamic and m % 15 == 0:
            dec = pol.reposition(forecast_at(zones, base, m, rng, heat), state(m), vuln)
            for mv in dec.moves:
                u = U[mv.unit_id]
                u.update(status="busy", free_at=m + mv.travel_min, free_zone=mv.to_zone, freed=False, job=False)
                if "relocation" in mv.reason:
                    u["moves"].append(m)
                n_moves += 1
            for u in U.values():
                if u["status"] == "idle":
                    u["freed"] = False
        queue += [(t, z, ac, h, sv) for t, z, ac, h, sv in by_min.get(m, [])]
        queue.sort(key=lambda x: (x[2] != "critical", x[0]))
        rest = []
        for t, z, ac, hosp, svc in queue:
            ch = pol.dispatch(z, ac, state(m), smart=dynamic)
            if ch is None:
                rest.append((t, z, ac, hosp, svc)); continue
            u = U[ch.unit_id]
            busy = ch.eta_min + SCENE_MIN + svc
            if dynamic:
                u.update(status="busy", free_at=m + busy, free_zone=hosp, job=True, freed=False)
            else:
                back = pol.tt_base[pol.ix[hosp], pol.ix[u["home"]]] * traffic(m)
                # static unit is dispatchable from the hospital while it drives home (fair baseline)
                u.update(status="busy", free_at=m + busy, free_zone=hosp, home_at=m + busy + back)
            resp.append(dict(ac=ac, minutes=(m - t) + ch.eta_min))
        queue = rest
    r = pd.DataFrame(resp)
    crit = r[r.ac == "critical"].minutes
    return dict(policy="dynamic" if dynamic else "static", seed=seed, calls=len(r), unserved=len(queue),
                crit_within_8=100 * (crit <= 8).mean(), crit_mean=crit.mean(),
                crit_p90=crit.quantile(.9), all_mean=r.minutes.mean(), moves=n_moves)


if __name__ == "__main__":
    zones, base, vuln = make_world()
    pol = RoutingPolicy(zones)
    rng = np.random.default_rng(0)
    units = [Unit(f"U{i}", zones.index[i * 3]) for i in range(8)] + \
            [Unit("U8", zones.index[5], "busy", 25, zones.index[8])]
    st = LiveState(pd.Timestamp("2022-07-19 17:00"), units, traffic_multiplier=1.4)
    dec = pol.reposition(forecast_at(zones, base, 17 * 60, rng, 1.3), st, vuln)
    print(f"One cycle: {len(dec.moves)} moves in {dec.latency_ms:.1f} ms | "
          f"critical covered {dec.critical_covered_before:.0%} -> {dec.critical_covered_after:.0%}")
    for mv in dec.moves:
        print(" -", mv.reason)
    print("\nStale-forecast failsafe:", pol.reposition(forecast_at(zones, base, 0, rng, 1.3),
          LiveState(st.now, units, 1.0, forecast_age_min=90), vuln).notes)
    print("\n24 h replay (synthetic):")
    res = pd.DataFrame([run(d, sd) for sd in range(5) for d in (False, True)])
    print(res.groupby("policy").mean(numeric_only=True).drop(columns="seed").round(1).to_string())
    print("\nper-seed critical mean response (min):")
    print(res.pivot(index="seed", columns="policy", values="crit_mean").round(1).to_string())
