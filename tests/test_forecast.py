import numpy as np
import pandas as pd

from grease.forecast import baseline, evaluate, features as F
from grease.forecast.zones import aggregate_to_zones
from grease.risk import RiskConfig, risk_index, vulnerability

START = pd.Timestamp("2022-03-01")
N = 420        # buckets, enough for the 336-bucket (one week) lag


def _incidents():
    """Zip 10001 has i % 3 incidents in bucket i, zip 10002 has (7*i) % 5. First incident of a bucket is critical (severity 1), the rest are not."""
    rows = []
    for zipcode, rule in (("10001", lambda i: i % 3), ("10002", lambda i: (7 * i) % 5)):
        for i in range(N):
            for j in range(rule(i)):
                rows.append((zipcode, START + pd.Timedelta(minutes=30 * i + 5 * j), 1 if j == 0 else 5))
    return pd.DataFrame(rows, columns=["zipcode", "incident_datetime", "initial_severity_level_code"])


def _panel():
    p = F.build_panel(_incidents())
    return F.add_targets(F.add_history_features(p))


def test_panel_counts_and_zero_fill():
    p = F.build_panel(_incidents())
    a = p[p.zipcode == "10001"].set_index("bucket")
    at = lambda col, i: int(a.loc[START + pd.Timedelta(minutes=30 * i), col])    # the panel starts at the first bucket WITH an incident (i = 1)
    assert [at("y", i) for i in range(1, 7)] == [1, 2, 0, 1, 2, 0]
    assert [at("y_critical", i) for i in range(1, 7)] == [1, 1, 0, 1, 1, 0]      # one critical call per busy bucket
    a = a.reset_index()
    assert (a.y_other == a.y - a.y_critical).all()


def test_targets_are_future_windows_and_do_not_cross_zips():
    p = _panel()
    a = p[p.zipcode == "10001"].reset_index(drop=True)
    for i in (5, 100):
        assert a.t_other_30m[i] == a.y_other[i + 1]
        assert a.t_critical_1h[i] == a.y_critical[i + 1] + a.y_critical[i + 2]
        assert a.t_other_3h[i] == a.y_other[i + 1:i + 7].sum()
        assert a.t_all_1h[i] == a.y[i + 1] + a.y[i + 2]
    # the last k buckets of each zip have no complete window -> NaN, they must not borrow the next zip's data
    assert a.t_critical_3h.iloc[-6:].isna().all() and a.t_critical_30m.iloc[-1:].isna().all()
    assert a.t_critical_3h.iloc[-7] == a.y_critical.iloc[-6:].sum()


def test_history_features_only_use_the_past_and_include_current_bucket():
    p = _panel()
    a = p[p.zipcode == "10002"].reset_index(drop=True)
    i = 50
    assert a.lag_1[i] == a.y[i - 1] and a.lag_336[400] == a.y[64]
    assert a.roll_2h[i] == a.y[i - 3:i + 1].sum()                                # includes the current bucket
    assert a.crit_roll_2h[i] == a.y_critical[i - 3:i + 1].sum()


def test_features_at_matches_training_features():
    p = _panel()
    now = p.bucket.iloc[380]
    live = F.features_at(p[["zipcode", "bucket", "y", "y_critical"]], now).set_index("zipcode")
    train = p[p.bucket == now].set_index("zipcode")
    cols = F.FEATURE_GROUPS["history"] + F.FEATURE_GROUPS["time"]
    for z in ("10001", "10002"):
        assert np.allclose(live.loc[z, cols].astype(float).to_numpy(), train.loc[z, cols].astype(float).to_numpy()), z


def test_select_features_never_leaks():
    p = _panel()
    cols = F.select_features(p.columns, ["id", "history", "time"])
    assert not [c for c in cols if c.startswith(("t_", "eval_")) or c in ("y_other", "bucket")]
    try:
        F.select_features(p.columns, ["weather"])
    except KeyError:
        pass
    else:
        raise AssertionError("missing weather columns should raise, not be silently skipped")


def test_baseline_uses_training_rows_only():
    p = _panel()
    train, test = p[p.bucket < START + pd.Timedelta(days=7)], p[p.bucket >= START + pd.Timedelta(days=7)]
    pred = baseline.historical_mean(train, train.t_other_30m.to_numpy(), test)
    a_train = train[train.zipcode == "10001"]
    row = test[test.zipcode == "10001"].iloc[1]                                # a test row whose weekday x time cell exists in training
    expect = a_train[(a_train.dow == row.dow) & (a_train.hour == row.hour)].t_other_30m.mean()
    assert np.isclose(pred[test.index.get_loc(row.name)], expect)
    assert np.isfinite(pred).all()
    # a weekday x time cell absent from training falls back to the zipcode mean instead of predicting 0 or NaN
    short_train = p[p.bucket < START + pd.Timedelta(days=1)]
    assert np.isfinite(baseline.historical_mean(short_train, short_train.t_other_30m.to_numpy(), p[p.bucket >= START + pd.Timedelta(days=3)])).all()
    assert (baseline.surge_multiplier(train, test) > 0).all()
    lw = baseline.last_week(p, p.t_other_30m.to_numpy())
    a = np.flatnonzero((p.zipcode == "10001").to_numpy())
    assert lw[a[400]] == p.t_other_30m.to_numpy()[a[400 - 336]] and np.isnan(lw[a[10]])


def test_scoring():
    y = np.array([0, 1, 2, 3, 0, 1])
    assert evaluate.poisson_deviance(y, y.astype(float).clip(1e-6)) < 1e-4
    assert evaluate.mae(y, y) == 0
    t = evaluate.calibration_table(y.repeat(50), (y + 0.5).repeat(50), bins=3)
    assert len(t) == 3


def test_zone_aggregation_is_an_exact_sum():
    f = pd.DataFrame({"zipcode": pd.Categorical(["10001", "10002", "10003"]), "bucket": [START] * 3, "pred_all_1h": [1.0, 2.0, 4.0]})
    out = aggregate_to_zones(f, pd.Series({"10001": "A", "10002": "A", "10003": "B"}), ["pred_all_1h"]).set_index("zone")
    assert out.loc["A", "pred_all_1h"] == 3.0 and out.loc["B", "pred_all_1h"] == 4.0


def test_risk_index():
    zones = ["A", "B", "C"]
    crit, oth = pd.Series([1.0, 1.0, 0.0], index=zones), pd.Series([0.0, 0.0, 4.0], index=zones)
    ctx = pd.DataFrame({"pct_poverty": [0.1, 0.1, 0.1], "pct_no_vehicle": [0.1, 0.1, 0.1], "asthma_rate_per_10k": [1, 1, 1]}, index=zones)
    idx = risk_index(crit, oth, vulnerability(ctx))
    assert idx.between(0, 100).all() and idx["C"] > idx["A"]                      # 4 other calls (4) > 1 critical (3)
    vuln_ctx = ctx.assign(pct_poverty=[0.9, 0.1, 0.1], pct_no_vehicle=[0.9, 0.1, 0.1], asthma_rate_per_10k=[9, 1, 1])
    assert risk_index(crit, oth, vulnerability(vuln_ctx))["A"] > risk_index(crit, oth, vulnerability(ctx))["A"] - 1e-9
    assert risk_index(crit, oth, vulnerability(vuln_ctx))["A"] >= idx["B"]        # same demand, more vulnerable -> not lower


def test_saved_model_predicts_from_live_features(tmp_path):
    """Train a small model, save it, reload it, and predict for one live time from `features_at`: must equal predicting from the training rows."""
    from grease.forecast.lgbm import ForecastModel
    p = _panel()
    feats = F.select_features(p.columns, ["id", "history", "time"])
    ok = F.complete_window_mask(p)
    split, end = p.bucket.iloc[300], p.bucket.iloc[380]
    train_rows = np.flatnonzero(((p.bucket < split) & ok).to_numpy())
    valid_rows = np.flatnonzero(((p.bucket >= split) & (p.bucket < end) & ok).to_numpy())
    m = ForecastModel(feats, {"n_estimators": 15, "min_child_samples": 5, "num_leaves": 7}).fit(p, train_rows, valid_rows, log=lambda *_: None)
    m.save(tmp_path / "m")
    m2 = ForecastModel.load(tmp_path / "m")
    now = p.bucket.iloc[380]
    live = F.features_at(p[["zipcode", "bucket", "y", "y_critical"]], now)
    from_live = m2.predict(live).set_index(live["zipcode"].astype(str))
    rows = np.flatnonzero((p.bucket == now).to_numpy())
    from_train = m.predict(p, rows).set_index(p.zipcode.astype(str).to_numpy()[rows])
    assert set(from_live.columns) == set(from_train.columns) and "pred_all_3h" in from_live
    assert np.allclose(from_live.loc[from_train.index, from_train.columns].to_numpy(), from_train.to_numpy(), atol=1e-6)
    assert (from_live >= 0).all().all()
    assert np.allclose(from_live["pred_all_1h"], from_live["pred_critical_1h"] + from_live["pred_other_1h"])


def test_missing_targets_are_refused():
    from grease.forecast.lgbm import ForecastModel
    p = _panel()
    feats = F.select_features(p.columns, ["id", "history"])
    rows = np.arange(len(p))                                                   # includes the last buckets of each ZIP, whose windows are incomplete
    try:
        ForecastModel(feats, {"n_estimators": 3}).fit(p, rows, rows, log=lambda *_: None)
    except ValueError as e:
        assert "complete_window_mask" in str(e)
    else:
        raise AssertionError("training on rows with a missing target should be refused, not silently dropped")


def test_future_window_matches_a_straightforward_reference():
    p = F.add_history_features(F.build_panel(_incidents()))
    for col, k in (("y", 1), ("y_critical", 4), ("y_other", 6)):
        fast = F.future_window(p, col, k)
        ref = sum(p.groupby("zipcode", observed=True)[col].shift(-j) for j in range(1, k + 1)).to_numpy(dtype="float32")
        assert np.array_equal(np.isnan(fast), np.isnan(ref)) and np.allclose(fast[~np.isnan(fast)], ref[~np.isnan(ref)])


def test_streaming_from_csv_matches_loading_everything(tmp_path):
    inc = _incidents()
    inc["incident_datetime"] = inc["incident_datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S.000")
    paths = []
    for i, (_, chunk) in enumerate(inc.groupby(pd.to_datetime(inc["incident_datetime"]).dt.dayofyear)):
        pth = tmp_path / f"{i}.csv"; chunk.assign(valid_incident_rspns_time_indc="Y").to_csv(pth, index=False); paths.append(pth)
    streamed, rows = F.build_panel_from_csvs(paths)
    whole = F.build_panel(_incidents())
    assert sum(rows.values()) == len(inc)
    for c in ("zipcode", "bucket", "y", "y_critical", "y_other"):
        assert (streamed[c].astype(str).to_numpy() == whole[c].astype(str).to_numpy()).all(), c


def test_missing_months_warn_instead_of_silently_zero_filling():
    import warnings
    inc = _incidents()
    inc.loc[len(inc)] = ["10001", START + pd.Timedelta(days=40), 5]            # a gap of about 38 days with no incidents at all
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        F.build_panel(inc)
    assert any("no incidents at all" in str(x.message) for x in w)


def test_link_context_puts_the_right_values_in_the_right_rows():
    p = F.build_panel(_incidents())
    nz, nb = F.grid_shape(p)
    idx = pd.date_range(p.bucket.min() - pd.Timedelta(hours=2), periods=nb + 8, freq="30min", name="bucket")
    wx = pd.DataFrame({"temp_c": np.arange(len(idx), dtype=float), "is_holiday": (np.arange(len(idx)) % 2).astype(float),
                       "eval_is_heat_day": 1.0, "humidity_pct": 5.0}, index=idx)
    ctx = pd.DataFrame({"lon": [1.0, 2.0], "borough": ["BRONX", "QUEENS"], "population": [10.0, 20.0]}, index=pd.Index(["A", "B"], name="zone"))
    full = F.link_context(p, wx, pd.Series({"10001": "A", "10002": "B"}), ctx, ["id", "weather", "calendar", "zone_basic"][:3] + ["zone_basic"])
    assert "humidity_pct" not in full and "population" not in full                 # only the requested groups are built (memory)
    assert "eval_is_heat_day" in full and full.eval_is_heat_day.dtype == "int8"      # eval flags are always attached, as int8
    i = np.flatnonzero(((full.zipcode == "10002") & (full.bucket == p.bucket.iloc[5])).to_numpy())[0]
    assert full.temp_c.iloc[i] == 4 + 5                                              # 4 buckets of lead-in + bucket 5
    assert full.lon.iloc[i] == 2.0 and full.borough.iloc[i] == "QUEENS"
    assert full.is_holiday.dtype == "int8" and full.is_holiday.iloc[i] == (4 + 5) % 2
    # the panel's own columns are untouched and nothing was reordered
    assert (full.y.to_numpy() == p.y.to_numpy()).all()
