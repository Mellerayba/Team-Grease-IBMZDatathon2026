"""Panel, history features, future-window targets and feature selection for the demand forecaster (decisions D8, M1, M2, M5).

Framing: a row is an *issue time* = the end of the 30-minute bucket `bucket` for one ZIP code. Everything in the row is known by then
(counts up to and including this bucket, the weather in it, the calendar). Targets are what happens AFTER it: the number of incidents
in the next 30 min / 1 h / 2 h / 3 h, split into critical and other calls.
"""
import numpy as np
import pandas as pd

HORIZONS = {"30m": 1, "1h": 2, "2h": 4, "3h": 6}        # name -> number of 30-minute buckets ahead
ACUITY = ["critical", "other"]
ACUITY_COL = {"critical": "y_critical", "other": "y_other"}
CRITICAL_MAX_SEVERITY = 3                                # ASSUMPTION (decision F9): severity code 1 = most critical; 1-3 counts as critical
LAGS = [1, 2, 3, 6, 48, 336]                             # 30 min, 1 h, 1.5 h, 3 h, 1 day, 1 week
ROLLS = {"roll_2h": 4, "roll_6h": 12, "roll_24h": 48}    # sums over the last 4 / 12 / 48 buckets, INCLUDING the current one
HISTORY_BUCKETS = max(LAGS) + 1                          # history needed to build features for a single time

FEATURE_GROUPS = {
    "id": ["zipcode"],
    "history": ["y", "y_critical", "lag_1", "lag_2", "lag_3", "lag_6", "lag_48", "lag_336",
                "roll_2h", "roll_6h", "roll_24h", "crit_roll_2h"],
    "time": ["hour", "dow", "month", "is_weekend"],
    "weather": ["temp_c", "feels_like_c", "precip_mm", "rain_3h_mm", "wind_kmh"],                       # decision F5 starting set
    "weather_extra": ["humidity_pct", "is_raining", "rain_24h_mm", "cloud_pct", "temp_max_24h_c",
                      "is_hot", "is_heat_stress", "is_cold"],
    "calendar": ["is_holiday", "is_day_before_holiday", "is_day_after_holiday"],
    "zone_basic": ["lon", "lat", "pop_est", "borough"],                                                  # decision F7 starting set
    "zone_census": ["population", "median_income", "pct_no_vehicle", "pct_poverty"],
    "zone_health_access": ["asthma_rate_per_10k", "dist_nearest_hospital_km", "n_hospitals", "n_firehouses"],
}
DEFAULT_GROUPS = ["id", "history", "time", "weather", "calendar", "zone_basic"]


def target_name(acuity: str, horizon: str) -> str:
    return f"t_{acuity}_{horizon}"


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Incident rows -> one row per (zipcode, 30-minute bucket), zero-filled, with y (all), y_critical and y_other counts."""
    df = df.dropna(subset=["incident_datetime", "zipcode"])
    bucket = df["incident_datetime"].dt.floor("30min")
    severity = pd.to_numeric(df["initial_severity_level_code"], errors="coerce")
    tmp = pd.DataFrame({"zipcode": df["zipcode"].astype(str).values, "bucket": bucket.values,
                        "crit": (severity <= CRITICAL_MAX_SEVERITY).astype("int8").values})
    g = tmp.groupby(["zipcode", "bucket"])
    counts = pd.concat([g.size().rename("y"), g["crit"].sum().rename("y_critical")], axis=1)
    idx = pd.MultiIndex.from_product([sorted(tmp.zipcode.unique()), pd.date_range(tmp.bucket.min(), tmp.bucket.max(), freq="30min")],
                                     names=["zipcode", "bucket"])
    panel = counts.reindex(idx, fill_value=0).reset_index()          # sorted by zipcode, then bucket
    panel["y"] = panel["y"].astype("int16")
    panel["y_critical"] = panel["y_critical"].astype("int16")
    panel["y_other"] = (panel["y"] - panel["y_critical"]).astype("int16")
    panel["zipcode"] = panel["zipcode"].astype("category")
    return panel


def add_time_features(panel: pd.DataFrame) -> pd.DataFrame:
    t = panel["bucket"]
    panel["hour"] = (t.dt.hour + t.dt.minute / 60).astype("float32")
    panel["dow"] = t.dt.dayofweek.astype("int8")
    panel["month"] = t.dt.month.astype("int8")
    panel["is_weekend"] = (panel["dow"] >= 5).astype("int8")
    return panel


def add_history_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Lags and rolling sums per ZIP, as of the issue bucket (rolling sums include the current bucket)."""
    g = panel.groupby("zipcode", observed=True)
    for k in LAGS:
        panel[f"lag_{k}"] = g["y"].shift(k).astype("float32")
    for name, w in ROLLS.items():
        panel[name] = g["y"].transform(lambda s, w=w: s.rolling(w).sum()).astype("float32")
    panel["crit_roll_2h"] = g["y_critical"].transform(lambda s: s.rolling(ROLLS["roll_2h"]).sum()).astype("float32")
    return add_time_features(panel)


def add_targets(panel: pd.DataFrame) -> pd.DataFrame:
    """Future-window targets: incidents in the next k buckets (t+1 .. t+k), per acuity. NaN for the last k buckets of each ZIP."""
    zips = panel["zipcode"]
    for acuity, col in ACUITY_COL.items():
        cum = panel.groupby(zips, observed=True)[col].cumsum()
        cum_by_zip = cum.groupby(zips, observed=True)
        for name, k in HORIZONS.items():
            panel[target_name(acuity, name)] = (cum_by_zip.shift(-k) - cum).astype("float32")
    for name in HORIZONS:                                   # critical + other, used to score the "all calls" forecast and the baselines
        panel[target_name("all", name)] = panel[target_name("critical", name)] + panel[target_name("other", name)]
    return panel


def select_features(columns, groups=DEFAULT_GROUPS) -> list[str]:
    """Explicit allow-list (decision M2). Never returns targets, `y_other`, `bucket`, `eval_*` or the duplicate `zone` column."""
    cols = [c for g in groups for c in FEATURE_GROUPS[g]]
    missing = [c for c in cols if c not in set(columns)]
    if missing:
        raise KeyError(f"features not in the table: {missing} (did the linking cells run? is the group name right?)")
    assert not [c for c in cols if c.startswith(("t_", "eval_")) or c in ("y_other", "bucket", "zone")]
    return cols


def features_at(history: pd.DataFrame, now: pd.Timestamp, weather: pd.DataFrame | None = None,
                zone_ctx: pd.DataFrame | None = None, zip_to_zone: pd.Series | None = None) -> pd.DataFrame:
    """Feature rows (one per ZIP) for a single issue time, built from recent history, for use by the simulator (decision M8).

    `history`: columns zipcode, bucket, y, y_critical covering at least the previous HISTORY_BUCKETS buckets up to and including `now`
    (`now` is the START of the last completed bucket). `weather`: the 30-minute weather/calendar table indexed by bucket (optional).
    `zone_ctx`: zone context indexed by zone (optional); needs `zip_to_zone` (zipcode -> zone) to join.
    """
    now = pd.Timestamp(now).floor("30min")
    start = now - pd.Timedelta(minutes=30 * (HISTORY_BUCKETS - 1))
    h = history[(history["bucket"] >= start) & (history["bucket"] <= now)]
    full_index = pd.date_range(start, now, freq="30min")
    y = h.pivot_table(index="bucket", columns="zipcode", values="y", aggfunc="sum", observed=True).reindex(full_index).fillna(0.0)
    yc = h.pivot_table(index="bucket", columns="zipcode", values="y_critical", aggfunc="sum", observed=True).reindex(full_index).fillna(0.0)
    yc = yc.reindex(columns=y.columns, fill_value=0.0)
    out = pd.DataFrame(index=y.columns)
    out["y"], out["y_critical"] = y.iloc[-1], yc.iloc[-1]
    for k in LAGS:
        out[f"lag_{k}"] = y.iloc[-1 - k]
    for name, w in ROLLS.items():
        out[name] = y.iloc[-w:].sum()
    out["crit_roll_2h"] = yc.iloc[-ROLLS["roll_2h"]:].sum()
    out["hour"] = now.hour + now.minute / 60
    out["dow"], out["month"], out["is_weekend"] = now.dayofweek, now.month, int(now.dayofweek >= 5)
    out.index.name = "zipcode"
    out = out.reset_index()
    out["bucket"] = now
    if weather is not None:
        out = out.join(weather.loc[[now]].reset_index(drop=True).reindex(out.index, method="ffill"))
    if zone_ctx is not None and zip_to_zone is not None:
        out["zone"] = out["zipcode"].astype(str).map(zip_to_zone)
        out = out.join(zone_ctx.reindex(out["zone"]).reset_index(drop=True))
    return out
