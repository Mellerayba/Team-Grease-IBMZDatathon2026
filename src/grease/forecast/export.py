"""Hand-off table for the optimizer and the simulator: one row per (zone, 30-minute bucket) of the test period.

Everything is expected or realised incidents in the window that STARTS when the bucket ends (the bucket is the issue time):
  pred_<acuity>_<horizon>    the model's forecast               (acuity: critical / other; horizon: 30m / 1h / 2h / 3h)
  hist_all_<horizon>         historical mean for that zone, weekday and time (all calls): the "scheduled" policy's demand
  actual_<acuity>_<horizon>  what really happened               (the "oracle" policy's demand, and for scoring)
  risk_index_1h / _3h        0-100 risk index (src/grease/risk.py), ranked across zones within each half-hour
Windows that run past the end of the data are NaN. See docs/forecast_handoff.md.
"""
import numpy as np
import pandas as pd

from grease import risk
from grease.forecast.features import ACUITY, HORIZONS
from grease.forecast.zones import aggregate_to_zones


def zone_forecast_table(keys: pd.DataFrame, pred: pd.DataFrame, actual: dict, baseline: dict, zip_to_zone: pd.Series,
                        zone_ctx: pd.DataFrame, cfg: risk.RiskConfig = risk.RiskConfig(), horizons=HORIZONS) -> pd.DataFrame:
    """`keys`: the test rows (columns zipcode, bucket). `pred`: ForecastModel.predict output aligned to them.
    `actual[(acuity, horizon)]` and `baseline[horizon]`: arrays aligned to the same rows. ZIP-level values are summed to zones (exact for counts)."""
    cols = {}
    for h in horizons:
        for a in ACUITY:
            cols[f"pred_{a}_{h}"] = pred[f"pred_{a}_{h}"].to_numpy(dtype="float32")
            cols[f"actual_{a}_{h}"] = np.asarray(actual[(a, h)], dtype="float32")
        cols[f"hist_all_{h}"] = np.asarray(baseline[h], dtype="float32")
    frame = pd.DataFrame(cols)
    frame["zipcode"] = keys["zipcode"].to_numpy()
    frame["bucket"] = keys["bucket"].to_numpy()
    table = aggregate_to_zones(frame, zip_to_zone, list(cols))
    vuln = risk.vulnerability(zone_ctx, cfg)
    for h in ("1h", "3h"):
        if h in horizons:
            table[f"risk_index_{h}"] = risk.risk_index_table(table, f"pred_critical_{h}", f"pred_other_{h}", vuln, cfg)
    value_cols = [c for c in table.columns if c not in ("zone", "bucket")]
    table[value_cols] = table[value_cols].astype("float32")
    return table
