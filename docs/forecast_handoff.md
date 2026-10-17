# Forecast hand-off: what the optimizer and simulator get from the forecaster

The forecaster answers one question: **for each zone, how many incidents do we expect in the next 30 min / 1 h / 2 h / 3 h, split into critical and other?**
It does not place units or route them. This page says exactly what it gives you and how to use it.

## 1. The table (`notebooks/data/processed/forecast_zone_test.csv.gz`)
Written by the last cells of `notebooks/HarryRowlands.ipynb` (git-ignored; rerun the notebook to regenerate). One row per **zone x 30-minute `bucket`** of the **test year (2022)**, about 3.1 million rows for 177 zones.

**Time convention.** `bucket` is the *start* of a half-hour that has just finished. The row is what a live system would know at the end of that half-hour (`bucket + 30 min`), and every value describes the window that **starts at `bucket + 30 min`**:
`_30m` = the next half-hour, `_1h` = the next hour, `_2h`, `_3h` likewise. Windows that run past the end of the data are NaN.

| Column | Meaning | Policy that uses it |
|---|---|---|
| `zone` | MODZCTA zone id (string), joins to `notebooks/data/interim/zone_context.csv` and `posts.csv` | all |
| `bucket` | see above | all |
| `pred_critical_<h>`, `pred_other_<h>` | **the model's** expected incidents in the window (`<h>` = `30m`, `1h`, `2h`, `3h`) | **Forecast** policy |
| `hist_all_<h>` | historical mean for that zone, weekday and time of day, all calls (no ML) | **Scheduled** policy |
| `actual_critical_<h>`, `actual_other_<h>` | what really happened in the window | **Oracle** policy (the ceiling) and scoring |
| `risk_index_1h`, `risk_index_3h` | 0-100 risk index (`src/grease/risk.py`): need-weighted expected demand x vulnerability, ranked across zones within that half-hour | dashboard map, equity floor |

Critical = initial severity code 1-3 (an assumption, decision F9). The risk index weights (critical x3, vulnerability boost up to 50%) are stated assumptions (D9).

## 2. The four policies we compare in the simulator
Run the same replayed days under each, same fleet, same travel times:

| Policy | Where idle units are staged using... |
|---|---|
| **Static** | fixed home posts, never moved |
| **Scheduled** | `hist_all_*` (demand by time of week) |
| **Forecast** | `pred_*` |
| **Oracle** | `actual_*` (the real future; not achievable, it is the upper bound) |

Reading the result: if Forecast is close to Scheduled, forecast accuracy barely matters for dispatch and we should say so. If Forecast is close to Oracle, a better forecaster could not add much either. A large Forecast-to-Oracle gap means improving the model is worth the time.

**Unit types.** The policy table above is written for generic units. Project Dispatch's headline unit is a lightweight first-response unit (`docs/decisions.md` D13 and section 7); the same demand columns (`pred_*`, `hist_all_*`, `actual_*`) feed it, and only the replay changes (speed, no hospital leg, stays until the ambulance arrives).

## 3. Using it
```python
import pandas as pd
t = pd.read_csv("notebooks/data/processed/forecast_zone_test.csv.gz", parse_dates=["bucket"], dtype={"zone": str})

now = pd.Timestamp("2022-07-20 16:00")                           # the half-hour that just finished
snap = t[t.bucket == now].set_index("zone")
demand_1h = snap.pred_critical_1h * 3 + snap.pred_other_1h       # whatever weighting the optimizer wants
top = snap.sort_values("risk_index_1h", ascending=False).head(10)
```
The three demand sources for the same decision are `snap.pred_*`, `snap.hist_all_*` and `snap.actual_*`.

For a live or replay loop that must not read the answer table, call the model itself:
```python
from grease.forecast.lgbm import ForecastModel
from grease.forecast import features as F, zones as Z
model = ForecastModel.load("notebooks/data/processed/forecast_models")
rows = F.features_at(history, now, weather=wx, zone_ctx=zone_ctx, zip_to_zone=zip_to_zone)   # `history`: zipcode, bucket, y, y_critical for the last 1345 buckets
pred = model.predict(rows)                                                                   # ZIP level
zone_pred = Z.aggregate_to_zones(pred.assign(zipcode=rows.zipcode, bucket=now), zip_to_zone, ["pred_critical_1h", "pred_other_1h"])
```
`features_at` builds exactly the features the model was trained on (a test checks this).

## 4. Know before you rely on it
- **The model is about 4% low on 2022** (predicted / actual = 0.96 in every slice and horizon; it was 0.94 before the 7-day / 28-day level features). It was trained on Jan-Sep 2021 only and 2022 had about 6% more incidents. The shortfall is nearly uniform across zones and across the demand range, so compare policies on *relative* demand and treat absolute counts (such as the busy-unit probability) as roughly 4% low.
- **Accuracy over the historical average is small** (0.0-1.6% lower MAE, 5-7% lower deviance at the ZIP-half-hour level; see `docs/decisions.md` section 3). The forecast is essentially last week's level for that ZIP times the usual shape for the time of day; weather adds almost nothing. Zone totals over an hour or more are far less noisy than the single cells the score table measures.
- **The `hist_all_*` and `pred_*` columns come from a model fitted before 2022; `actual_*` is 2022.** Do not tune policy parameters on the same year you report results for.
- **ZIP-level values are summed into zones**, which is exact for expected counts. About 0.1% of incidents are in ZIPs that belong to no zone and are dropped.
- Resolution is the 30-minute bucket. Locations are zone-level, so travel times must use zone centroids.
