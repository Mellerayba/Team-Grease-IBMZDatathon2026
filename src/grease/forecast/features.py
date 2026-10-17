"""Panel, history features, future-window targets, context linking and feature selection for the demand forecaster (decisions D8, M1, M2, M5).

Framing: a row is an *issue time* = the end of the 30-minute bucket `bucket` for one ZIP code. Everything in the row is known by then
(counts up to and including this bucket, the weather in it, the calendar). Targets are what happens AFTER it: the number of incidents
in the next 30 min / 1 h / 2 h / 3 h, split into critical and other calls.

Memory: the panel is a complete regular grid (every ZIP x every bucket, ZIP-major), which lets everything be done with array reshapes and
lookups instead of copies. 7 years x 235 ZIPs is about 29 million rows, so nothing here materialises per-row strings or repeated context.
"""
import warnings

import numpy as np
import pandas as pd

HORIZONS = {"30m": 1, "1h": 2, "2h": 4, "3h": 6}        # name -> number of 30-minute buckets ahead
ACUITY = ["critical", "other"]
ACUITY_COL = {"critical": "y_critical", "other": "y_other", "all": "y"}
CRITICAL_MAX_SEVERITY = 3                                # ASSUMPTION (decision F9): severity code 1 = most critical; 1-3 counts as critical
LAGS = [1, 2, 3, 6, 48, 336]                             # 30 min, 1 h, 1.5 h, 3 h, 1 day, 1 week
ROLLS = {"roll_2h": 4, "roll_6h": 12, "roll_24h": 48,   # sums over the last 4 / 12 / 48 buckets, INCLUDING the current one
         "roll_7d": 336, "roll_28d": 1344}               # long-run level (7 and 28 days): lets the model track year-on-year drift (decision F16)
HISTORY_BUCKETS = max(max(LAGS), max(ROLLS.values())) + 1   # history needed to build features for a single time
CSV_COLUMNS = ["incident_datetime", "zipcode", "initial_severity_level_code"]

FEATURE_GROUPS = {
    "id": ["zipcode"],
    "history": ["y", "y_critical", "lag_1", "lag_2", "lag_3", "lag_6", "lag_48", "lag_336",
                "roll_2h", "roll_6h", "roll_24h", "roll_7d", "roll_28d", "crit_roll_2h"],
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


# ------------------------------------------------------------------ panel

def _counts(df: pd.DataFrame) -> dict:
    """Incident rows -> compact (zip code, bucket) counts. The ZIP is kept as small integer codes, never as repeated strings."""
    df = df.dropna(subset=["incident_datetime", "zipcode"])
    severity = pd.to_numeric(df["initial_severity_level_code"], errors="coerce")
    zc = pd.Categorical(df["zipcode"].astype(str))
    tmp = pd.DataFrame({"z": zc.codes, "b": df["incident_datetime"].dt.floor("30min").to_numpy(),
                        "c": (severity <= CRITICAL_MAX_SEVERITY).to_numpy().astype("int32")})
    out = tmp.groupby(["z", "b"], sort=False).agg(y=("c", "size"), yc=("c", "sum")).reset_index()
    return {"zips": [str(c) for c in zc.categories], "z": out["z"].to_numpy(), "b": out["b"].to_numpy(),
            "y": out["y"].to_numpy(), "yc": out["yc"].to_numpy()}


def _panel_from_counts(parts: list[dict]) -> pd.DataFrame:
    zips = sorted(set().union(*[p["zips"] for p in parts]))
    zmap = {z: i for i, z in enumerate(zips)}
    bmin = min(p["b"].min() for p in parts)
    bmax = max(p["b"].max() for p in parts)
    step = np.timedelta64(30, "m")
    nb, nz = int((bmax - bmin) // step) + 1, len(zips)
    y, yc = np.zeros(nz * nb, np.int32), np.zeros(nz * nb, np.int32)
    for p in parts:
        remap = np.array([zmap[z] for z in p["zips"]], dtype=np.int64)
        pos = remap[p["z"]] * nb + ((p["b"] - bmin) // step).astype(np.int64)
        y += np.bincount(pos, weights=p["y"], minlength=nz * nb).astype(np.int32)
        yc += np.bincount(pos, weights=p["yc"], minlength=nz * nb).astype(np.int32)
    buckets = pd.date_range(pd.Timestamp(bmin), periods=nb, freq="30min")

    per_day = pd.Series(y.reshape(nz, nb).sum(axis=0), index=buckets).resample("D").sum()
    if (per_day == 0).any():
        warnings.warn(f"{int((per_day == 0).sum())} days between {buckets[0].date()} and {buckets[-1].date()} have no incidents at all "
                      f"(missing data is zero-filled, which would bias the model): {sorted({d.strftime('%Y-%m') for d in per_day[per_day == 0].index})}")
    panel = pd.DataFrame({"zipcode": pd.Categorical.from_codes(np.repeat(np.arange(nz, dtype=np.int16), nb), zips),
                          "bucket": np.tile(buckets.to_numpy(), nz),
                          "y": y.astype(np.int16), "y_critical": yc.astype(np.int16)})
    panel["y_other"] = (panel["y"] - panel["y_critical"]).astype(np.int16)
    return panel


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Incident rows -> one row per (zipcode, 30-minute bucket), zero-filled, with y (all), y_critical and y_other counts."""
    return _panel_from_counts([_counts(df)])


def build_panel_from_csvs(paths) -> tuple[pd.DataFrame, dict]:
    """Same as build_panel but streams one file at a time, so the raw rows (about 10 million for 2016-2022) are never all in memory.
    Returns (panel, {path: rows read}) so the caller can check each year is complete."""
    parts, rows = [], {}
    for path in paths:
        d = pd.read_csv(path, usecols=CSV_COLUMNS, dtype={"zipcode": str})
        d["incident_datetime"] = pd.to_datetime(d["incident_datetime"], errors="coerce")
        rows[str(path)] = len(d)
        parts.append(_counts(d))
        del d
    return _panel_from_counts(parts), rows


def grid_shape(panel: pd.DataFrame) -> tuple[int, int]:
    """(number of ZIPs, number of buckets). Verifies the panel is a complete ZIP-major grid, which every array shortcut below relies on."""
    nz = len(panel["zipcode"].cat.categories)
    nb, rem = divmod(len(panel), nz)
    codes = panel["zipcode"].cat.codes.to_numpy()
    if rem or not np.array_equal(codes[::nb], np.arange(nz)) or panel["bucket"].iloc[0] != panel["bucket"].iloc[nb * (nz - 1)]:
        raise ValueError("panel is not a complete ZIP-major grid (every ZIP x every bucket); build it with build_panel / build_panel_from_csvs")
    return nz, nb


# ------------------------------------------------------------------ features and targets

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


def future_window(panel: pd.DataFrame, col: str, k: int) -> np.ndarray:
    """Incidents in the next k buckets (t+1 .. t+k) for every row; NaN for the last k buckets of each ZIP. Float32, aligned to panel rows."""
    nz, nb = grid_shape(panel)
    cum = np.cumsum(panel[col].to_numpy().reshape(nz, nb), axis=1, dtype=np.int32)
    out = np.full((nz, nb), np.nan, dtype=np.float32)
    out[:, :nb - k] = cum[:, k:] - cum[:, :nb - k]
    return out.ravel()


def target_array(panel: pd.DataFrame, acuity: str, horizon: str) -> np.ndarray:
    """One target as a float32 array aligned to the panel rows (acuity: 'critical', 'other' or 'all')."""
    return future_window(panel, ACUITY_COL[acuity], HORIZONS[horizon])


def complete_window_mask(panel: pd.DataFrame) -> np.ndarray:
    """True where the longest horizon has a full window (drops the last 3 h of each ZIP, whose targets are incomplete)."""
    nz, nb = grid_shape(panel)
    return np.tile(np.arange(nb) < nb - max(HORIZONS.values()), nz)


def add_targets(panel: pd.DataFrame) -> pd.DataFrame:
    """Materialise every target as a column (t_<acuity>_<horizon>). Convenient for small data and tests; the notebook uses target_array instead (less memory)."""
    for acuity in ("critical", "other"):
        for name in HORIZONS:
            panel[target_name(acuity, name)] = target_array(panel, acuity, name)
    for name in HORIZONS:
        panel[target_name("all", name)] = panel[target_name("critical", name)] + panel[target_name("other", name)]
    return panel


# ------------------------------------------------------------------ linking other data

def link_context(panel: pd.DataFrame, weather: pd.DataFrame, zip_to_zone: pd.Series, zone_ctx: pd.DataFrame,
                 groups=DEFAULT_GROUPS) -> pd.DataFrame:
    """Add the weather / calendar / zone columns that `groups` need, plus the `eval_*` flags, as lean columns.

    `weather`: the 30-minute table indexed by bucket (notebooks/data/weather_calendar_30min.csv.gz). `zip_to_zone`: zipcode -> zone.
    `zone_ctx`: zone context indexed by zone. Each value is built from one small lookup repeated over the grid (np.tile / np.repeat),
    so there are no per-row joins and no per-row strings. 0/1 flags are int8, other numbers float32, text columns categories.
    """
    nz, nb = grid_shape(panel)
    wanted = [c for g in groups for c in FEATURE_GROUPS[g]]
    eval_cols = [c for c in weather.columns if c.startswith("eval_")]
    new = {}

    # weather and calendar: one lookup per bucket (not per row), tiled over the ZIPs
    bucket_pos = weather.index.get_indexer(panel["bucket"].iloc[:nb])             # -1 where the weather table has no row
    n_missing = int((bucket_pos < 0).sum())
    for c in [c for c in wanted if c in weather.columns] + eval_cols:
        v = weather[c].to_numpy(dtype="float32")[np.clip(bucket_pos, 0, None)]
        v[bucket_pos < 0] = np.nan
        flag = c.startswith(("is_", "eval_")) and c != "is_weekend"
        new[c] = np.tile(np.nan_to_num(v).astype("int8") if flag else v, nz)
    if n_missing:
        warnings.warn(f"{n_missing} buckets have no weather row; weather features are NaN there (flags are 0)")

    # zone context: one lookup per ZIP, repeated over the buckets
    zips = pd.Series(panel["zipcode"].cat.categories.astype(str))
    zone_of_zip = zips.map(zip_to_zone)
    ctx = zone_ctx.reindex(zone_of_zip.to_numpy())
    for c in [c for c in wanted if c in zone_ctx.columns]:
        if pd.api.types.is_numeric_dtype(ctx[c]):
            new[c] = np.repeat(ctx[c].to_numpy(dtype="float32"), nb)
        else:
            cats = sorted(ctx[c].dropna().unique())
            codes = pd.Categorical(ctx[c].to_numpy(), categories=cats).codes
            new[c] = pd.Categorical.from_codes(np.repeat(codes, nb), cats)
    return pd.concat([panel, pd.DataFrame(new, index=panel.index)], axis=1)


def select_features(columns, groups=DEFAULT_GROUPS) -> list[str]:
    """Explicit allow-list (decision M2). Never returns targets, `y_other`, `bucket`, `eval_*` or the duplicate `zone` column."""
    cols = [c for g in groups for c in FEATURE_GROUPS[g]]
    missing = [c for c in cols if c not in set(columns)]
    if missing:
        raise KeyError(f"features not in the table: {missing} (was link_context run with these groups? is the group name right?)")
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
