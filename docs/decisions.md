# Model decisions log

A running record of what we decided, what is still open and why, so the pitch (and the next person to touch the model) can see the reasoning.
Add new entries at the bottom of each section; when a decision changes, mark the old one **Superseded** and link the new one instead of deleting it.

Status key: **Decided** (team agreed) · **Proposed** (recommended default, not yet confirmed) · **Open** (needs an experiment or a team call)

State below is as of **2026-10-17, afternoon**.

---

## 1. Decisions made so far

| ID | Decision | Status | Why |
|---|---|---|---|
| D1 | ~~The units we model are **ambulances** (not mobile clinics)~~ **Revised by D13 (2026-10-17):** ambulances stay as the transporting fleet; the lever we study is a second fleet of lightweight first-response units | Revised | Original reason still holds for the ambulance layer: gives a measurable response-time metric and fits the EMS data |
| D2 | **EMS data is fetched in `HarryRowlands.ipynb`** (API -> `notebooks/ems_raw/`). All other data is fetched and cleaned in `DarraghC.ipynb` and linked onto the EMS table in the Harry notebook | Decided | One owner per data source; keeps the model notebook readable |
| D3 | ~~**Do not change the model yet.**~~ **Superseded by D10 (2026-10-17).** Link all the data in first (the table `full`), then decide features from there | Decided | Avoid changing the model and the data at the same time; we need a clean before / after |
| D4 | ~~Other data is joined into a separate table `full`, never into `panel`~~ **Superseded (2026-10-17):** `full` is now the modelling table, and the model uses an explicit feature list (M2), so nothing is picked up by accident | Superseded | The model builds its feature list from every column of `panel`, so merging into it would silently change the model (and a text column would break LightGBM) |
| D5 | `eval_is_heat_day` and `eval_is_heavy_rain_day` are **never model inputs**; they only slice results | Decided | They use the whole day's weather, which is unknown when a forecast is made (leakage) |
| D6 | Weather comes from the **Open-Meteo API in New York local time** | Decided | The CSV in the repo is UTC (4-5 hours out of step with EMS timestamps) and blank for Oct 16 and Oct 20-25 2022 |
| D7 | Spatial unit for the simulator and optimizer is the **zone (MODZCTA)**; travel times use zone centroids | Proposed | 99.9% of incidents with a ZIP map to a zone; joins to Census and boundary data. The model still uses raw `zipcode` (see F2) |
| D8 | **The model's output is expected demand**: calibrated expected incidents per zone for the next 30 min / 1 h / 2 h / 3 h, split into critical and other acuity. The model inputs are the issue time plus what is known at that moment (recent counts, weather, calendar), not time of day alone | Implemented, awaiting full-data run | The optimizer needs additive rates, not an ordinal score. Rates can be validated against held-out data; a hand-weighted index cannot. Time of day alone is only the historical average, which is our "scheduled" comparison policy and what we must beat |
| D9 | **The risk index is a derived layer, not a model.** Risk = need-weighted expected demand (critical calls count more, e.g. 3:1, a stated assumption) x a vulnerability factor (poverty, no-vehicle households, asthma rate), shown as a 0-100 percentile map. Later it can add the coverage gap (expected response time minus the 8-minute target) once the simulator exists | Implemented, awaiting full-data run | Keeps the forecast testable and the policy weights explicit and adjustable without retraining. Vulnerability data is constant per zone, so it belongs in the weights (and the optimizer's equity floor), not in the forecast |
| D10 | **D3 (model frozen) ends** once the EMS download is complete and the unchanged model has been run once on real data and its results recorded here | Implemented, awaiting full-data run | We need an honest before / after, but the model has to change to produce D8's output |
| D11 | **Be honest about the IBM Z accelerator.** It does not fix our memory problem or speed up training. Right-size the VM first (event code, ask the SMEs), measure with `scripts/vm_check.py`, and only claim accelerator use if the VM shows the `nnpa` CPU flag and we actually run inference through it | Proposed | See section 6. Judges score *use of platform capabilities*, with screenshots as evidence, so the evidence has to be real |
| D12 | **Compare four policies in the simulator: static, scheduled (historical mean), forecast (our model) and oracle (the real future).** Report how close Forecast gets to Oracle and how far it is above Scheduled | Proposed | Tells us whether forecast accuracy matters for dispatch at all, and where to spend the remaining time. Hand-off format: `docs/forecast_handoff.md` |
| D13 | **Reframe the product: Project Dispatch adds a fleet of lightweight electric first-response units (e.g. e-bikes) carrying enough equipment to stabilise a critical patient until an ambulance arrives.** Ambulances still transport. The headline question for the city is "how many units should NYC buy, and where should they stand?", answered by replaying the same real incidents at many fleet sizes | Proposed | A cheap, fast layer is something a city can actually buy more of; the fleet-size curve turns our replay into a purchasing decision; it fits "Real-time AI for critical decisions". Detail, assumptions and what must change in the code: section 7 |
| D14 | **Headline metrics:** time to the first responder on scene (light unit or ambulance, whichever is first), share of critical calls reached within 4 and 8 minutes, and the borough gaps, all by fleet size. Any "lives saved" figure is an assumption-driven illustration, not a result | Proposed | Our data has no outcomes, so survival can only be inferred through a published rule of thumb (section 7) |

---

## 2. Data we have

| Dataset | Status | Coverage / size | Known issues |
|---|---|---|---|
| **EMS incidents** (`ems_raw/`, fetched by Harry's cell 0: time, ZIP, initial severity, response-valid flag) | **Partial** | 2017 complete (1,463,446 rows, matches the API exactly). 2016 file has 350,000 rows ending 2016-03-31 (API has 1,441,211). 2018 partly downloaded; 2019-2022 not yet. API totals: 2018 1.53M, 2019 1.53M, 2020 1.41M, 2021 1.49M, 2022 1.58M | **`2016.csv` is truncated and will never refetch** (the cell skips years whose file exists) - delete it and rerun. Only 4 columns, so no stage durations or real response times for the simulator |
| **Weather** (Open-Meteo, hourly, one point for all NYC) | Done | 2016-01-01 to 2022-12-31, 61,368 hours, no gaps; 122,736 rows on the 30-minute grid; 13 weather features | Citywide only: varies over time, not between zones |
| **Calendar** | Done | US federal holidays (72 in 2016-22) plus day before / after; 88 heat days (daily max >= 32 C), 239 heavy-rain days (>= 12.7 mm) | Heat days per year: 2016 17, 2017 7, 2018 9, 2019 7, 2020 11, 2021 13, 2022 24 (small samples) |
| **Zones (MODZCTA)** + ZIP crosswalk | Done | 178 zones, 219 ZIPs mapped | A few rare ZIPs (10121, 10000, 10281, ...) are in no zone: about 11% of panel rows but only about 0.1% of incidents that have a ZIP |
| **Hospitals, firehouses** (candidate staging posts) | Done | 60 hospitals, 219 firehouses, assigned to zones (2 firehouses unassigned) | True ambulance stations are not public |
| **FDNY monthly response times** | Done | Monthly by borough and incident class (fiscal-year summary rows dropped) | "Medical Emergencies" covers all first responders, not just ambulances: a rough benchmark only |
| **Asthma rates** (2017-19, 188 neighbourhoods) | Done, borough level | Borough asthma rate per 10,000 (as given in the source file; we have not confirmed whether it counts ED visits or hospitalisations): Bronx 190.7, Brooklyn 115.2, Staten Island 96.3, Manhattan 89.5, Queens 64.1 | No neighbourhood -> zone crosswalk, so borough only; constant within a borough |
| **Community health survey** | Done | 11 yearly rows, citywide only | No geography, so context only |
| **Zone context** (`zone_context.csv`: borough, location, population, asthma rate, distance to nearest hospital, hospitals / firehouses in zone) | Done | 178 zones | Borough comes from the community district containing the zone centroid |
| **Census ACS** (2022 5-year: population, median income, no-vehicle households, poverty) | Done | 177 of 178 zones; in `zone_context.csv` and so in `full` | Zone populations sum to about 8.7M vs roughly 8.3-8.5M for the city (ZIP areas extend past the boundary), so approximate. By borough, households without a vehicle: Manhattan 78%, Bronx 60%, Brooklyn 55%, Queens 34%, Staten Island 15% |
| **Road network (OpenStreetMap)** | Not fetched | - | Needed only for travel times; fast option is haversine distance x detour factor |
| Emergency call text | Not available | - | No public source; would need clearly labelled synthetic data |

Combined view: `full` (in `HarryRowlands.ipynb`, after the lag features) = `panel` + weather / calendar on the 30-minute bucket + zone context through ZIP -> zone.
About 4 GB in memory for two years, so roughly 14 GB for 2016-2022.

---

## 3. The model

### Current (rewritten 2026-10-17, implements D8-D10)
**Output:** expected incidents per ZIP for the next 30 min / 1 h / 2 h / 3 h, separately for critical and other calls (8 LightGBM Poisson models), plus a derived 0-100 risk index per zone.
**Code:** `src/grease/forecast/` (`features.py`, `lgbm.py`, `baseline.py`, `evaluate.py`, `zones.py`), `src/grease/risk.py`, driven from `HarryRowlands.ipynb`; 10 unit tests in `tests/test_forecast.py`.
**Default features (29):** ZIP id; current and recent counts (`y`, `y_critical`, lags 30 min to 1 week, rolling sums 2 h / 6 h / 24 h, recent critical sum); time (`hour`, `dow`, `month`, `is_weekend`); weather (temperature, feels-like, rain now and 3 h, wind); holiday flags; zone location, population estimate and borough. Change `FEATURE_GROUPS` in the notebook to run the ablations in section 4.
**Split:** train 2016 to Sep 2021, validation Oct-Dec 2021 (early stopping), test 2022, with a 3-hour gap before each boundary so no training target reaches into the next period.
**Baselines (same rows, same targets):** historical mean for that ZIP x weekday x time; that mean x a citywide recent-surge multiplier; the same window one week earlier.
**Reported:** MAE, Poisson deviance and bias, overall and on heat days, heavy-rain days, the 20 busiest ZIPs and each borough; calibration by prediction decile; feature importance.

**Status: two real runs done on the VM (2026-10-17).** Both trained on Jan-Sep 2021 (3.09M rows), validated on Oct-Dec 2021 (1.04M) and tested on all of 2022 (4.13M ZIP-half-hours per horizon). Run 1 used 29 features; run 2 added the 7-day and 28-day level features (31 features). Run 2, all test rows (the notebook export it came from was deleted; the numbers are kept here):

| Horizon | Model MAE | Historical-mean MAE | MAE gain | Deviance: model / hist mean / surge | Bias (pred / actual) |
|---|---|---|---|---|---|
| 30 min | 0.3914 | 0.3915 | 0.0% | 0.678 / 0.728 / 0.725 | 0.961 |
| 1 h | 0.5689 | 0.5731 | 0.7% | 0.805 / 0.855 / 0.849 | 0.961 |
| 2 h | 0.8186 | 0.8295 | 1.3% | 0.887 / 0.945 / 0.935 | 0.958 |
| 3 h | 1.0161 | 1.0329 | 1.6% | 0.920 / 0.986 / 0.972 | 0.958 |

Findings:
- **The model beats every baseline on deviance at every horizon (5-7% lower than the historical mean), but its MAE gain is tiny** (0.0% at 30 min up to 1.6% at 3 h). A ZIP gets about 0.38 calls per half-hour, so most of what is left is random noise, and the historical profile for that ZIP, weekday and time is already a strong predictor.
- **Level features (run 1 -> run 2):** bias went from about 6.2% low to about 4% low (0.94 -> 0.96). MAE gain at 2-3 h rose (1.2 -> 1.3%, 1.3 -> 1.6%) and at 30 min fell (0.4 -> 0.0%); deviance improved slightly at every horizon.
- **Calibration** (predicted vs actual by prediction decile): ratios of 0.94-0.97 in deciles 2-9 for both the 1 h and 3 h forecasts, and 0.82-0.94 in the lowest deciles (windows with fewer than about 0.02 expected calls). So the shortfall is a near-uniform ~4% level shift, and the ranking of zones is intact. Likely cause: 2022 had about 6% more incidents than 2021 and the model only trained on 2021.
- **Feature importance** (share of total gain, averaged over the 8 models): `roll_7d` 70.6%, `roll_24h` 10.4%, `zipcode` 8.5%, `hour` 7.3%, `roll_28d` 1.1%, everything else 0.4% or less (temperature 0.3%, feels-like, wind, rain, population about 0.1% each; the holiday flags are not in the top 15). The model is essentially "last week's level for this ZIP x the usual shape for this ZIP and time of day". **Weather adds almost nothing at this resolution, so we should not claim a weather-driven forecast.**
- **Baselines:** the "scale by recent citywide activity" baseline (`hist_x_surge`) has slightly *worse* MAE than the plain historical mean but slightly *better* deviance, and is clearly better on heat days (3 h deviance 0.983 vs 1.037); the same-window-last-week baseline is much worse on both. (An earlier version of this note said the surge baseline was worse everywhere; that was true for MAE only.)
- **Where the model helps:** longer horizons, the 20 busiest ZIPs (+2.1% MAE at 3 h), Manhattan (+2.2% at 3 h), heat days (+2.2% at 3 h). It is slightly worse than the historical mean at 30 min in Queens (-0.5%), Manhattan (-0.3%), Staten Island (-0.3%) and on heat days (-1.2%).
- **Risk snapshot:** the busiest half-hour of the test year (2022-03-30 11:00) ranks 10456, 11212, 10457 (Bronx and Brooklyn), 10029 (East Harlem) and 10467 highest: plausible high-need areas.
- **Known gap, one example:** at midnight on New Year's Day 2022, zone 10001 had 16 calls in the next 3 hours against 6.4 forecast and 7.2 from the historical mean. Neither handles special events; the slim EMS download drops `special_event_indicator`.
- **Hand-off table written:** 3,101,040 rows, 177 zones, 24 columns (`docs/forecast_handoff.md`).

Run 1 for comparison (no level features): MAE gain 0.4 / 0.9 / 1.2 / 1.3% and bias 0.94 at 30 min / 1 / 2 / 3 h.

### Previous model (original, replaced)
Predicted incidents in the *same* bucket from lags (one-step ahead only), all columns of `panel` as features, 60 / 15 / 25 percent split by time fraction, and a baseline that predicted zero for some cells. Its saved results came from a 33-day 2005 sample and should not be quoted. Limitations L1-L7 below are what the rewrite fixes.

**Task:** predict incidents per **ZIP code per 30-minute bucket** (`y`).
**Model:** LightGBM, Poisson objective (`n_estimators` 2000, learning rate 0.03, 63 leaves, min 50 samples per leaf, 80% row and column subsampling, early stopping after 100 rounds).
**Features (14):** `zipcode` (category), lags of `y` at 30 min / 1 h / 1.5 h / 3 h / 1 day / 1 week, rolling sums over 2 h / 6 h / 24 h, `hour`, `dow`, `month`, `is_weekend`.
**Split:** by time fraction of whatever data is loaded: train up to 60%, validation 60-75%, test the last 25%.
**Baseline:** mean `y` for each zipcode x weekday x hour in the training period.
**Metrics:** MAE and Poisson deviance.

**Saved results are not usable.** The stored outputs (MAE 0.341 vs baseline 0.356; deviance 0.644 vs 2.242) were produced on a 33-day sample from January 2005, not on 2016-2022. Do not quote them.

Limitations of the previous model (all addressed by the rewrite, see section 5):

| # | Limitation | Why it matters |
|---|---|---|
| L1 | **Only forecasts 30 minutes ahead** (shortest lags are 1-3 buckets and nothing is shifted by a horizon) | The optimizer needs forecasts 30-180 minutes ahead; one-step accuracy is optimistic |
| L2 | Baseline predicts exactly 0 for some zipcode-hours that then have incidents | The deviance comparison is distorted; MAE is the fairer number. The baseline also lacks the planned recent-surge multiplier |
| L3 | Split by time fraction, not by date | Test period shifts as data is added; cannot isolate heat days or a clean final year |
| L4 | No weather, holiday or zone features | The data is now linked in `full` but the model does not use it |
| L5 | Target is all incidents; severity is ignored | The plan targets low / mid-acuity calls |
| L6 | `fit(eval_X=..., eval_y=...)` uses a newer LightGBM API | May fail on the VM's LightGBM version |
| L7 | A few dead / duplicated cells (first `train_end`, duplicated baseline cell) | Housekeeping |

---

## 4. Feature decisions

Plan: change one group at a time and compare against the same baseline, so each choice has evidence. "Ablation" = train with and without the group and compare MAE, Poisson deviance, and error on heat days.

| ID | Question | Options | Recommended default | Status | How we decide |
|---|---|---|---|---|---|
| F1 | **Forecast horizon(s)** | one model per horizon (h = 1, 2, 4, 6 buckets = 0.5 / 1 / 2 / 3 h) vs one model with a horizon feature | Separate models per horizon; every lag, rolling and weather feature shifted back at least h buckets | Open | Check which horizons the optimizer actually uses (plan: 30-180 min) |
| F2 | **Spatial key:** raw `zipcode` vs `zone` (MODZCTA) | keep ZIP; switch to zone | Keep ZIP while the model is frozen (D3); switch to zone when we integrate with the simulator, since 99.9% of incidents map | Open | Compare error with each; zone also needed for Census and travel times |
| F3 | **Lags and rolling sums** | keep as is; add same-hour-last-week average; drop the 30-minute lags for longer horizons | Keep, but only lags >= horizon (F1) | Proposed | Ablation per horizon |
| F4 | **Time features** (`hour`, `dow`, `month`, `is_weekend`) | keep; add day-of-year or cyclical hour | Keep as is | Proposed | Only revisit if residuals show a pattern |
| F5 | **Weather features** | none; current temperature and rain only; the full set (temperature, feels-like, humidity, rain now / 3 h / 24 h, wind, cloud, 24 h max, hot / cold / heat-stress flags) | Start with temperature, feels-like, rain now and 3 h, wind; add the rest only if they help. Always use values from `h` buckets earlier, never future weather | Open | Ablation overall **and on heat days** (the case for weather is surges) |
| F6 | **Calendar features** (holiday, day before / after) | add / skip | Add | Proposed | Ablation; only 72 holidays in the data, so expect a small effect |
| F7 | **Zone context** (borough, lon / lat, population, distance to nearest hospital, hospital / firehouse counts) | add all; add location and population only; skip | Test location, population and borough; **skip asthma rate and survey data as model inputs** (constant within a borough, so redundant with the zone identity) | Open | Ablation; the main value of the health data is the fairness analysis and the pitch, not forecast accuracy |
| F8 | **Target definition** | all incidents; only mid / low acuity; critical vs other | Per D8: two targets, **critical** and **other** (sum = all), each for the next 30 min / 1 h / 2 h / 3 h. Acuity bands need F9 confirmed | Proposed | Depends on the severity-code meaning (F9). The current 4-column download has `initial_severity_level_code`, which is enough to start |
| F9 | **Severity code meaning** | 1 = most critical ... 8 = least (assumed) | Assume as stated | Open | Confirm in the NYC data dictionary |
| F10 | **COVID year 2020** | include as is; include with a flag; drop from training | Include with a `year` / covid flag, then test dropping it | Open | Ablation; demand dropped about 8% in 2020 (1.41M vs 1.53M in 2019) |
| F11 | **Train / validation / test split** | by fraction (now); by date | Train 2016-2021 (2020 per F10), validate on the last months of 2021, **test on 2022** and report heat days separately. Weather now covers all of 2022 | Proposed | Needs the full EMS download |
| F12 | **Baseline** | zipcode x weekday x hour mean (now); the same x a recent-surge multiplier; "same bucket last week" | Report all three; the surge-multiplier version is the one in the plan | Proposed | Fixes L2; the model must beat the best of them to claim anything |
| F13 | **Evaluation** | MAE and deviance only; add decision-relevant metrics | Add error on the busiest zones, on heat days, and per borough (equity gap) | Proposed | Matches how the optimizer uses the forecast |
| F14 | **Whole-day weather flags** | - | Never as features (D5) | Decided | Leakage |
| F15 | **Surge baseline** | keep; drop; rework as a per-ZIP multiplier | Keep it as a comparison: worse than the plain historical mean on MAE, slightly better on deviance and clearly better on heat days | Proposed | Run 2 results above |
| F16 | **Long-run level features** (7-day and 28-day rolling counts per ZIP) | none; add | **Added** (`roll_7d`, `roll_28d`) to track year-on-year drift | Implemented, effect small | Smoke test (2017, train Jan-Mar, test Jun-Dec): bias moved from 0.970 to 0.981 at 3 h and from 0.987 to 1.002 at 1 h, error unchanged. Tree models cannot extrapolate beyond the levels seen in training, so this is unlikely to fully remove a 6% year-on-year gap. Re-test on the VM (train 2021, test 2022) |
| F17 | **Rate offset** | none; model the ratio to the recent level (LightGBM `init_score` = log of the 28-day average rate, prediction = exp(raw score + offset)) | The principled fix if F16 is not enough: the model learns multiplicative deviations from the current level, so a drifting level cannot cap the prediction | Open | Needs F16's VM result first |

---

## 5. Changes the model notebook needs to deliver D8 (to do)

What stays: LightGBM with the Poisson objective, the ZIP x 30-minute panel, the lag and rolling-sum features.

| # | Change | Where in `HarryRowlands.ipynb` | Priority | Status |
|---|---|---|---|---|
| M1 | **Predict a future window.** Add targets `y_next_30m`, `y_next_1h`, `y_next_2h`, `y_next_3h` = incidents in the next 1 / 2 / 4 / 6 buckets (`g.shift(-k)` summed forward). Features stay as of the issue bucket `t`, and now include `y` at `t` itself (`lag_0`); rolling sums include the current bucket. Train one model per horizon in a loop | Panel / lag cell and the model cell | Must | Done (targets are `t_<acuity>_<horizon>`, e.g. `t_critical_1h`) |
| M2 | **Use the linked data.** Build the feature list from `full` with an explicit allowlist: weather (F5), calendar (F6), zone context (F7). Exclude `y*` targets, `bucket`, every `eval_*` column and `zone` (duplicates `zipcode`). Pass text columns such as `borough` as categorical | `full` cell, `features = ...` line | Must | Done |
| M3 | **Dated split** (F11): train 2016-2021, validate late 2021, test 2022; plus a heat-day slice of the test set using `eval_is_heat_day` | Split lines | Must | Done (needs 2022 data to run) |
| M4 | **Fair baseline** (F12): same horizons, zipcode x weekday x hour mean x recent-surge multiplier; report the "same bucket last week" baseline too | Baseline cell | Must | Done |
| M5 | **Acuity targets** (F8): count critical and other incidents separately in the panel (from `initial_severity_level_code`), repeat M1 for each | Panel cell | Should | Done (critical = severity 1-3, assumption F9) |
| M6 | **Calibration check:** predicted vs actual mean by prediction decile, per horizon. The optimizer uses the magnitudes, so they must be right, not just the ranking | After predictions | Should | Done |
| M7 | **Aggregate to zones** by summing ZIP-level predictions through `zip_to_zone` (rates add, so no retraining is needed to move from ZIP to zone, which partly settles F2) | After predictions | Must for the optimizer | Done |
| M8 | **Make it callable:** move the logic into `src/grease/forecast/` (features at a given time from recent history, `fit`, `predict(now) -> expected demand per zone`), and save the trained models. The simulator calls it every 15-30 simulated minutes | New code | Must | Done: `ForecastModel.save/load` and `features_at`, tested; simulator integration pending |
| M9 | **Risk layer** (D9) in `src/grease/risk.py` with the weights in `config.py` as stated assumptions, and a unit test | New code | Should | Done (weights are stated assumptions) |
| M10 | Replace the `eval_X` / `eval_y` fit arguments with the standard `eval_set=[(X, y)]` so it also runs on the VM's LightGBM; remove the dead and duplicated cells | Model cell | Must | Done: uses `eval_X` / `eval_y` when LightGBM supports them, falls back to `eval_set` (deprecated in 4.7 but all that older versions have) |
| M11 | **Hand-off export**: zone x half-hour table with forecast, scheduled and oracle demand and the risk index; format in `docs/forecast_handoff.md` | New cell in the notebook, `src/grease/forecast/export.py` | Must | Done (tested; written when the notebook runs) |
| M12 | **Long-run level features** (F16) | `features.py` | Should | Done; effect small, see F16 |

All ten are implemented. We skipped the intermediate "unchanged model on real data" baseline run that D10 suggested, because the original model cannot produce D8's output; the honest before / after is now the new model against the three baselines.

**Next:** run the real split once the EMS download is complete, record the results here, then work through the feature questions in section 4 (F5-F7 ablations, F10 COVID, F1 horizons).

---

## 6. Platform notes (IBM LinuxONE)

What we know, and where it comes from:
- **Guidebook:** gives no hardware specification. Judging criterion 3 rewards AI/ML running on LinuxONE and "platform capabilities such as containerisation, security, scalability and compute resources, using screenshots as evidence". The two provided AI/ML labs are mandatory.
- **VM image** ([jupyter-lab-ml README](https://github.com/linuxone-community-cloud/jupyter-lab-ml)): Ubuntu 24.04 container with TensorFlow, PyTorch, XGBoost, scikit-learn, Keras and Llama.cpp. It says to use the `SSF2026` event code "to get an instance with sufficient resources for ML notebooks", and states no vCPU or memory figures. It mentions z17's Telum II on-chip AI inference accelerator. Snap ML and LightGBM are not in the list it describes (LightGBM does import on our VM).
- **What the accelerator does** ([IBM Z Accelerated for Snap ML](https://github.com/IBM/ibmz-accelerated-for-snapml), [Snap ML model import docs](https://snapml.readthedocs.io/en/latest/model_import.html)): on z16 and later, Snap ML can run **inference** of tree-ensemble models on the Integrated Accelerator for AI through zDNN. On older machines it runs on the CPU with no changes. It does not accelerate training or data preparation.
- **Could our LightGBM models use it?** The Snap ML import table lists LightGBM regressors (formats PMML, ONNX, text), but the same page says elsewhere that only PMML is supported, and it does not say whether the Poisson objective is supported. Unverified: needs a test on the VM. Pickled models are not portable between x86 and s390x (endianness); LightGBM text files are.
- **Installing Snap ML:** its GitHub page distributes it as a container image from `icr.io` that needs credentials; it does not mention a pip package. Unverified whether `pip install snapml` works on the VM.

What we do not know (run `python scripts/vm_check.py --years 2` on the VM): how much memory the instance really has available, whether a container limit applies, whether the CPU shows the `nnpa` flag, and which of Snap ML / ONNX Runtime / TensorFlow / PyTorch are installed.

Why this matters: the notebook kernel died while building the panel for the latest data, which needs about 1-2 GB, so the VM probably has very little memory free (or old kernels are holding it). The accelerator cannot fix that. Options, in order: (1) check `free -h`, shut down old kernels, and confirm the instance was created with the `SSF2026` event code; (2) ask the IBM SMEs on Discord for a larger instance; (3) train where there is memory and run inference and the simulator on the VM (the saved models are portable text files), which still shows AI running on LinuxONE; (4) load fewer years.

---

## 7. Product framing: lightweight first-response units (D13, D14)

**Concept.** A fleet of cheap, fast, lightweight electric units (e-bikes or similar), each carrying a stabilisation kit (contents to be specified with clinicians). They are positioned across the city by our forecast and risk model, sent to critical calls, and stay with the patient until an ambulance arrives. They do not transport anyone.

**Why it is a better lever than "more ambulances".** An ambulance costs far more, needs a crew that is in short supply (about 10 are lost a day to staffing shortages, per the sources in the README) and is slowed by traffic. A light unit is cheap enough to buy hundreds, which makes "how many?" a real decision, and the curve of benefit against fleet size is exactly what our replay can produce.

**Status.** Concept and framing only. The replay today models one generic fleet (ambulance-like units: they drive to the call, spend time on scene, drive to a hospital and return). The light-unit model below is **not built yet**.

**What stays the same:** the demand forecaster and its hand-off table; the risk index; the positioning and dispatch engine (expected-coverage objective, assignment of units to staging spots, minimum-gain threshold, per-borough floor, staleness failsafe); the replay structure; and the four-policy test with an oracle (D12).

**What must change in the model**

| Item | Today (ambulance-like unit) | Light unit (assumptions to confirm) |
|---|---|---|
| Speed and network | straight line x 1.35 at 30 km/h (placeholder), driving network planned | cycling network (OSMnx `network_type="bike"`) with an assumed average speed, to be checked against published figures |
| After arrival | 15 min on scene, then hospital leg and return | stays on scene until the ambulance arrives (or a time cap), then returns to a staging spot; **no hospital leg** |
| Calls attended | all calls | critical calls first (severity 1-3); whether it also attends others is open |
| Coverage target | 8 min critical, 15 min other | shorter target for critical (for example 4-5 min), since the aim is to beat the ambulance |
| Dispatch | nearest idle unit | send the nearest light unit **and** the nearest ambulance to the same call; the value of the light unit is the minutes between the two arrivals |
| Constraints | none | battery range and charging, bad weather, night, rider fatigue (all placeholders until researched) |
| Cost | none | a unit cost, so results can be shown per unit and per dollar |

**New outputs**
- Fleet-size curve: share of critical calls with a responder within 4 and 8 minutes, and the 90th-percentile time to first responder, for 0, 50, 100, 200, 300 ... units, with the ambulance fleet held fixed.
- Where each extra batch of units goes (by borough) and which boroughs gain most; the equity gap.
- Minutes of unaided waiting avoided = ambulance arrival time minus first-responder arrival time, summed over critical calls.
- Cost-effectiveness: extra critical calls reached quickly per unit bought (placeholder prices).

**Turning minutes into "lives saved" (D14, handle with care).** The README cites the American Heart Association rule of thumb that survival falls roughly 7-10% for every minute without CPR or defibrillation. That applies to *cardiac arrest*. Our slim EMS download has no call type, so we cannot yet say what share of critical calls are arrests; the full API has `initial_call_type`, which we dropped and could re-fetch. Until then, any survival figure must be shown as an **illustration with its assumption stated** ("if X% of critical calls are arrests, each minute saved is worth about 7-10% survival for those"), never as a measured result.

**Open questions (need a decision or research)**
- Clinical scope: what the kit can safely do, who rides the unit and their training.
- Regulation, insurance and liability for non-ambulance responders.
- Operating limits: range, charging, rain, snow, heat, night.
- Evidence: published results exist from bicycle first-responder schemes in other cities; we have not reviewed them and should cite one or two.
- Whether light units also help non-critical calls, or only ever critical ones.

**Replay results so far (ambulance-only positioning, the baseline the new layer builds on).** Modelled, two real 2022 days, 330 generic units, placeholder travel times; averages of both days.

| Policy | Critical calls reached in 8 min | 90th-percentile response (min) |
|---|---|---|
| Static layout | 76.4% | 17.1 |
| Scheduled (historical demand) | 84.4% | 12.7 |
| Forecast (our model) | 84.9% | 12.5 |
| Oracle (perfect foresight) | 86.2% | 11.6 |

Most of the gain comes from positioning units by expected demand at all; our forecast adds only a little over the historical average. The gain held at 300 and 370 units. **Equity flag:** the dynamic policies did worse than static in Staten Island on both days (97.7% to 89.5% on one day, 58.8% to 57.5% on the other); this needs a fix or an explicit note.

**Routing review (the street-network work).** `notebooks/Mapping.ipynb` downloads the Manhattan drivable network with OSMnx and computes one route (Times Square to Washington Square Park: 5.2 minutes free-flow). That is a proof of concept, not yet the travel-time matrix the policy needs: it covers Manhattan only, one pair, and nothing calls it from the policy or replay. What is still needed: all five boroughs, a zone-to-zone (or ZIP-to-ZIP) matrix passed to the policy as `tt_base` (the policy already accepts one; `run_replay` does not yet pass it through), a siren-speed assumption for ambulances, a bike network and speed for light units, and the traffic multiplier on top. **Consequence for results:** for that one pair our placeholder says 8.6 minutes against 5.2 on the street network, so the placeholder is probably too slow and the fleet calibration (330 units) must be redone once real travel times go in.

---

## 8. Changelog

| Date | Change |
|---|---|
| 2026-10-17 | Log created. Data linked into `full`; model deliberately left unchanged (D3, D4) |
| 2026-10-17 | Census fetched and joined into the zone context (affects F7: population, income, vehicle access and poverty are now candidate features) |
| 2026-10-17 | Added D8-D10 (model output = expected demand; risk index is a derived layer; D3 ends after the baseline run), revised F8, and listed the model changes needed (section 5) |
| 2026-10-17 | Model rewritten per D8-D10 and M1-M10 (code in `src/grease/forecast/`, notebook updated, 10 tests pass). Smoke-tested on 2017 only; real run pending the complete EMS download |
| 2026-10-17 | Memory rewrite after the VM kernel died (out of memory at the linking cell): EMS streamed one year at a time, only the needed columns are linked (built from small lookups), targets computed one at a time, 8 models share one float32 dataset. Measured peak for one year of data fell from 4.1 GB to 1.9 GB; 15 tests pass |
| 2026-10-17 | Added D11 and section 6 (platform notes) and `scripts/vm_check.py` after the VM kernel died again at the panel cell |
| 2026-10-17 | VM measured: 8.4 GB RAM (5.6 GB free), 3 CPUs, IBM z17-generation LinuxONE with the `nnpa` AI-accelerator flag exposed; two years of data build the panel in 6 s and 1.6 GB. The kernel died because the notebook still defaulted to all seven years, so the repo default is now 2021-2022 (train Jan-Sep 2021, validate Q4 2021, test 2022). The full 2016-2022 history needs a bigger machine, so F11 / F10 experiments over many years are limited on the VM |
| 2026-10-17 | First real run recorded (section 3). Added 7-day / 28-day level features (F16), the zone-level hand-off export and `docs/forecast_handoff.md` (M11), the four-policy comparison with an oracle (D12), and F15-F17 |
| 2026-10-17 | Run 2 (with level features) recorded, including calibration, feature importance (roll_7d 71%, weather about 1% in total) and the risk snapshot. Corrected an earlier claim: the surge baseline is worse on MAE but not on deviance |
| 2026-10-17 | Reframed the product (D13, D14): lightweight first-response units as a second fleet, headline question = how many units to buy. Added section 7 (assumptions, required model changes, replay results, routing review). D1 revised |
