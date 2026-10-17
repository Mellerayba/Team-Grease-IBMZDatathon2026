"""
routing_policy.py - live repositioning + dispatch policy on top of the demand forecaster.

Inputs each cycle
  forecast : DataFrame indexed by zone with pred_critical_<h> and pred_other_<h>
             (your `zones_now` from the notebook, i.e. ZIP forecasts summed to zones)
  state    : LiveState - unit positions/status, traffic multiplier, forecast age, recent counts
Outputs
  Decision : list of Moves (with plain-language reasons), before/after coverage, latency
  dispatch : which idle unit to send to a new incident

Method
  Objective = expected weighted demand covered, MEXCLP-style: a zone covered by n units counts
  w * (1 - q**n), where q = live busy fraction of the fleet. Extra units on the same zone have
  diminishing value, which is what spreads units out when the fleet is stretched.
  Greedy marginal-gain selection of staging zones -> Hungarian assignment of idle units to them
  -> apply only the moves whose individual gain clears a threshold (stops thrashing).
Live data enters through: unit status, traffic multiplier, busy fraction q, an optional
nowcast correction of the forecast, and a forecast-staleness failsafe (hold positions).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# ----------------------------------------------------------------------------- data classes
@dataclass
class Unit:
    id: str
    zone: str                       # current zone (idle) or zone where it will become free (busy)
    status: str = "idle"            # "idle" | "busy"
    free_in_min: float = 0.0        # minutes until available, if busy
    free_zone: str | None = None    # where it becomes free (e.g. hospital zone); defaults to zone
    moves_last_hour: int = 0
    just_freed: bool = False        # became idle after a job: needs a destination, so moves are uncapped


@dataclass
class LiveState:
    now: pd.Timestamp
    units: list[Unit]
    traffic_multiplier: float = 1.0          # 1.0 = free-flow; 1.4 = 40% slower, from a live feed
    forecast_age_min: float = 0.0            # minutes since the forecast was produced


@dataclass
class PolicyConfig:
    horizon: str = "1h"                      # which forecast columns to plan on: 30m|1h|2h|3h
    w_critical: float = 3.0                  # value of covering a critical incident vs. other
    w_other: float = 1.0
    cover_critical_min: float = 8.0          # "covered" = a unit could arrive within this many minutes
    cover_other_min: float = 15.0
    max_move_min: float = 20.0               # never send a unit further than this to reposition
    max_moves_per_cycle: int = 3             # discretionary relocations of already-idle units
    max_moves_per_hour: int = 2              # per unit; crew fatigue / realism. Does not apply to just_freed units
    min_gain_frac: float = 0.003             # a move must add >= this share of total weighted demand
    free_soon_min: float = 10.0              # busy units freeing within this count as coverage
    min_units_per_borough: int = 1           # equity floor (needs a `borough` column in zones)
    equity_alpha: float = 0.5                # demand weight x (1 + alpha * vulnerability in 0..1)
    nowcast_prior: float = 5.0               # pseudo-incidents of prior belief in the nowcast
    nowcast_clip: tuple = (0.5, 2.5)
    nowcast_damping: float = 0.5             # exponent on the multiplier (forecast lags already hold most signal)
    max_forecast_age_min: float = 45.0       # older than this -> hold positions
    q_clip: tuple = (0.1, 0.9)
    dispatch_slack_min: float = 1.0          # non-critical: accept a slightly slower unit to protect coverage


@dataclass
class Move:
    unit_id: str
    from_zone: str
    to_zone: str
    travel_min: float
    gain: float
    reason: str
    kind: str = "relocation"        # "return" (unit back from a job) | "relocation" (idle unit moved)


@dataclass
class Decision:
    moves: list[Move]
    score_before: float
    score_after: float
    critical_covered_before: float           # share of expected critical demand within the target time
    critical_covered_after: float
    latency_ms: float
    notes: list[str] = field(default_factory=list)


@dataclass
class DispatchChoice:
    unit_id: str
    eta_min: float
    reason: str


# ----------------------------------------------------------------------------- travel times
def travel_time_matrix(zones: pd.DataFrame, speed_kmh: float = 30.0, detour: float = 1.35,
                       within_zone_min: float = 2.0) -> pd.DataFrame:
    """Free-flow minutes between zone centroids (zones needs lat, lon). Straight-line x detour is a
    placeholder: swap in an OSRM/OSMnx matrix for the real thing (same shape: from x to)."""
    lat = np.radians(zones["lat"].to_numpy(float))[:, None]
    lon = np.radians(zones["lon"].to_numpy(float))[:, None]
    a = (np.sin((lat - lat.T) / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin((lon - lon.T) / 2) ** 2)
    km = 2 * 6371.0 * np.arcsin(np.sqrt(a))
    mins = km * detour / speed_kmh * 60
    np.fill_diagonal(mins, within_zone_min)
    return pd.DataFrame(mins, index=zones.index, columns=zones.index)


# ----------------------------------------------------------------------------- policy
class RoutingPolicy:
    def __init__(self, zones: pd.DataFrame, cfg: PolicyConfig | None = None,
                 tt_base: pd.DataFrame | None = None):
        self.cfg = cfg or PolicyConfig()
        if not zones.index.is_unique:
            raise ValueError("zone ids must be unique")
        self.ids = list(zones.index)
        self.ix = {z: i for i, z in enumerate(self.ids)}
        self.borough = (zones["borough"].fillna("?").to_numpy() if "borough" in zones
                        else np.array(["all"] * len(self.ids)))
        tt = tt_base if tt_base is not None else travel_time_matrix(zones)
        self.tt_base = tt.loc[self.ids, self.ids].to_numpy(float)       # [from_zone, to_zone] minutes
        self._last: tuple | None = None                                  # (weights, q) from last cycle

    # ---- live-data layers -------------------------------------------------------------------
    def nowcast_multiplier(self, expected_recent: pd.Series, observed_recent: pd.Series) -> pd.Series:
        """Gamma-Poisson update of the forecast from what actually arrived in the last window.
        Prior mean = citywide ratio, so quiet zones are pulled toward the city trend, not to 1.
        expected_recent = what the model predicted for that window; observed_recent = actual counts."""
        c = self.cfg
        e = expected_recent.reindex(self.ids).fillna(0.0)
        o = observed_recent.reindex(self.ids).fillna(0.0)
        city = float(np.clip((o.sum() + 1) / (e.sum() + 1), *c.nowcast_clip))
        m = (c.nowcast_prior * city + o) / (c.nowcast_prior + e)
        return (m.clip(*c.nowcast_clip) ** c.nowcast_damping)

    def demand_weights(self, forecast: pd.DataFrame, vulnerability: pd.Series | None = None,
                       nowcast_mult: pd.Series | None = None) -> dict[str, np.ndarray]:
        c, h = self.cfg, self.cfg.horizon
        for col in (f"pred_critical_{h}", f"pred_other_{h}"):
            if col not in forecast:
                raise KeyError(f"forecast is missing '{col}'")
        crit = forecast[f"pred_critical_{h}"].reindex(self.ids).fillna(0.0).to_numpy(float)
        oth = forecast[f"pred_other_{h}"].reindex(self.ids).fillna(0.0).to_numpy(float)
        if nowcast_mult is not None:
            m = nowcast_mult.reindex(self.ids).fillna(1.0).to_numpy(float)
            crit, oth = crit * m, oth * m
        vm = np.ones(len(self.ids))
        if vulnerability is not None:
            v = vulnerability.reindex(self.ids).fillna(vulnerability.median()).to_numpy(float)
            if v.max() > 1.0 or v.min() < 0.0:
                v = (v - v.min()) / max(v.max() - v.min(), 1e-9)
            vm = 1.0 + c.equity_alpha * v
        return {"critical": c.w_critical * crit * vm, "other": c.w_other * oth * vm}

    def _busy_prob(self, state: LiveState) -> float:
        n = max(len(state.units), 1)
        return float(np.clip(sum(u.status == "busy" for u in state.units) / n, *self.cfg.q_clip))

    # ---- core maths -------------------------------------------------------------------------
    def _tt(self, state: LiveState) -> np.ndarray:
        return self.tt_base * state.traffic_multiplier

    def _cover(self, tt: np.ndarray) -> dict[str, np.ndarray]:
        c = self.cfg   # cover[z, p] = 1 if a unit staged at p reaches zone z in time
        return {"critical": (tt.T <= c.cover_critical_min).astype(np.float32),
                "other": (tt.T <= c.cover_other_min).astype(np.float32)}

    def _score(self, pos, w, q, cover) -> float:
        pos = np.asarray(pos, int)
        total = 0.0
        for k in cover:
            n = cover[k][:, pos].sum(axis=1) if len(pos) else np.zeros(len(self.ids))
            total += float((w[k] * (1 - q ** n)).sum())
        return total

    def _crit_share(self, pos, w, cover) -> float:
        pos = np.asarray(pos, int)
        hit = (cover["critical"][:, pos].sum(axis=1) > 0) if len(pos) else np.zeros(len(self.ids), bool)
        return float((w["critical"] * hit).sum() / max(w["critical"].sum(), 1e-9))

    def _greedy(self, k, fixed, w, q, cover) -> list[int]:
        """Pick k staging zones by marginal expected coverage; seed the borough equity floor first."""
        c = self.cfg
        n = {m: (cover[m][:, fixed].sum(axis=1) if len(fixed) else np.zeros(len(self.ids))) for m in cover}
        chosen: list[int] = []

        def gain_vec():
            return sum((w[m] * (1 - q) * q ** n[m]) @ cover[m] for m in cover)

        def add(p):
            chosen.append(p)
            for m in cover:
                n[m] = n[m] + cover[m][:, p]

        if c.min_units_per_borough > 0 and k > 0:
            demand = w["critical"] + w["other"]
            boroughs = sorted(set(self.borough), key=lambda b: -demand[self.borough == b].sum())
            have = {b: sum(self.borough[p] == b for p in list(fixed)) for b in boroughs}
            for b in boroughs:
                while have[b] < c.min_units_per_borough and len(chosen) < k:
                    g = gain_vec()
                    g[self.borough != b] = -np.inf
                    add(int(np.argmax(g)))
                    have[b] += 1
        while len(chosen) < k:
            add(int(np.argmax(gain_vec())))
        return chosen

    def optimal_layout(self, forecast, k, state: LiveState, vulnerability=None, q: float = 0.3) -> list[str]:
        """Best fixed layout for k units under a given forecast (use the daily average for a strong
        *static* baseline, or the current forecast for a one-off plan)."""
        tt = self._tt(state)
        w = self.demand_weights(forecast, vulnerability)
        return [self.ids[p] for p in self._greedy(k, [], w, q, self._cover(tt))]

    # ---- repositioning ----------------------------------------------------------------------
    def reposition(self, forecast: pd.DataFrame, state: LiveState, vulnerability: pd.Series | None = None,
                   nowcast_mult: pd.Series | None = None) -> Decision:
        t0, c, notes = time.perf_counter(), self.cfg, []

        def done(moves, sb=0.0, sa=0.0, cb=0.0, ca=0.0):
            return Decision(moves, sb, sa, cb, ca, (time.perf_counter() - t0) * 1000, notes)

        if state.forecast_age_min > c.max_forecast_age_min:
            notes.append(f"Forecast is {state.forecast_age_min:.0f} min old (limit {c.max_forecast_age_min:.0f}): "
                         "holding all units in place.")
            return done([])
        idle = [u for u in state.units if u.status == "idle"]
        if not idle:
            notes.append("No idle units to reposition.")
            return done([])

        tt = self._tt(state)
        cover = self._cover(tt)
        w = self.demand_weights(forecast, vulnerability, nowcast_mult)
        q = self._busy_prob(state)
        self._last = (w, q)
        total_w = sum(v.sum() for v in w.values())
        if total_w <= 0:
            notes.append("Forecast has no demand: holding positions.")
            return done([])

        fixed = [self.ix[u.free_zone or u.zone] for u in state.units
                 if u.status == "busy" and u.free_in_min <= c.free_soon_min]
        cur = {u.id: self.ix[u.zone] for u in idle}
        before_pos = fixed + list(cur.values())
        score0, share0 = self._score(before_pos, w, q, cover), self._crit_share(before_pos, w, cover)

        targets = self._greedy(len(idle), fixed, w, q, cover)
        cost = np.array([[tt[cur[u.id], p] for p in targets] for u in idle])
        freed = np.array([u.just_freed for u in idle])
        forbidden = (cost > c.max_move_min) & ~freed[:, None]       # freed units may go anywhere
        r, col = linear_sum_assignment(np.where(forbidden, 1e4, cost))
        cand_free, cand_disc = [], []
        for i, j in zip(r, col):
            u, p = idle[i], targets[j]
            if forbidden[i, j] or p == cur[u.id]:
                continue
            if u.just_freed:
                cand_free.append((u, p))
            elif u.moves_last_hour < c.max_moves_per_hour:
                cand_disc.append((u, p))

        layout, moves = dict(cur), []
        min_gain = c.min_gain_frac * total_w

        def apply(cands, limit, threshold, kind):
            while cands and len(moves) < limit:
                base = self._score(fixed + list(layout.values()), w, q, cover)
                gains = [self._score(fixed + list({**layout, u.id: p}.values()), w, q, cover) - base
                         for u, p in cands]
                i = int(np.argmax(gains))
                if gains[i] <= threshold:
                    if kind == "discretionary":
                        notes.append("Remaining moves fall below the minimum-gain threshold: not worth the disruption.")
                    break
                u, p = cands.pop(i)
                drive = tt[layout[u.id], p]
                why = ("back into service" if kind == "freed" else "relocation")
                moves.append(Move(u.id, self.ids[layout[u.id]], self.ids[p], float(drive), float(gains[i]),
                                  kind=("return" if kind == "freed" else "relocation"),
                                  reason=
                                  f"{u.id} {why}: {self.ids[layout[u.id]]} -> {self.ids[p]} "
                                  f"({drive:.0f} min drive), +{gains[i]:.2f} weighted incidents covered over the "
                                  f"next {c.horizon} (fleet busy {q:.0%}, traffic x{state.traffic_multiplier:.2f})."))
                layout[u.id] = p

        apply(cand_free, len(cand_free), 1e-9, "freed")
        apply(cand_disc, len(moves) + c.max_moves_per_cycle, min_gain, "discretionary")
        after_pos = fixed + list(layout.values())
        return done(moves, score0, self._score(after_pos, w, q, cover),
                    share0, self._crit_share(after_pos, w, cover))

    # ---- dispatch ---------------------------------------------------------------------------
    def dispatch(self, zone: str, acuity: str, state: LiveState, smart: bool = True) -> DispatchChoice | None:
        """Pick an idle unit for a new incident. Critical: fastest unit, no trade-offs. Other: among
        units within `dispatch_slack_min` of the fastest, send the one whose absence hurts coverage least."""
        idle = [u for u in state.units if u.status == "idle"]
        if not idle:
            return None
        tt, z = self._tt(state), self.ix[zone]
        eta = {u.id: float(tt[self.ix[u.zone], z]) for u in idle}
        best = min(eta.values())
        slack = 0.0 if acuity == "critical" else self.cfg.dispatch_slack_min
        pool = [u for u in idle if eta[u.id] <= best + slack]
        if smart and len(pool) > 1 and self._last is not None:
            w, q = self._last
            cover = self._cover(tt)
            others = [self.ix[(u.free_zone or u.zone)] for u in state.units
                      if u.status == "busy" and u.free_in_min <= self.cfg.free_soon_min]
            def loss(u):
                rest = others + [self.ix[v.zone] for v in idle if v.id != u.id]
                return -self._score(rest, w, q, cover)
            pick = min(pool, key=loss)
            why = "fastest unit within tolerance whose departure leaves the best coverage"
        else:
            pick = min(pool, key=lambda u: eta[u.id])
            why = "fastest available unit"
        return DispatchChoice(pick.id, eta[pick.id], why)
