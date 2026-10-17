# Model decisions log

A running record of what we decided, what is still open and why, so the pitch (and the next person to touch the model) can see the reasoning.
Add new entries at the bottom of each section; when a decision changes, mark the old one **Superseded** and link the new one instead of deleting it.

Status key: **Decided** (team agreed) · **Proposed** (recommended default, not yet confirmed) · **Open** (needs an experiment or a team call)

State below is as of **2026-10-17, afternoon**.

---

## 1. Decisions made so far

| ID | Decision | Status | Why |
|---|---|---|---|
| D1 | The units we model are **ambulances** (not mobile clinics). Mobile clinics can be mentioned later as a generalisation | Decided | Gives a measurable response-time metric and fits the EMS data; the guidebook rewards a working demo |
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

**Status: built and tested, not yet run on the real split.** Only a plumbing smoke test has run (2017 data only; trained on Jan-Mar and tested on Jun-Dec, so the numbers mean nothing). It showed the model about 0-3% better than the historical mean on MAE, that the surge multiplier did not help, and that weather had very low importance, but with 3 months of training and no 2020-2022 data none of that is a finding. The real run needs the complete EMS download.

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

## 7. Changelog

| Date | Change |
|---|---|
| 2026-10-17 | Log created. Data linked into `full`; model deliberately left unchanged (D3, D4) |
| 2026-10-17 | Census fetched and joined into the zone context (affects F7: population, income, vehicle access and poverty are now candidate features) |
| 2026-10-17 | Added D8-D10 (model output = expected demand; risk index is a derived layer; D3 ends after the baseline run), revised F8, and listed the model changes needed (section 5) |
| 2026-10-17 | Model rewritten per D8-D10 and M1-M10 (code in `src/grease/forecast/`, notebook updated, 10 tests pass). Smoke-tested on 2017 only; real run pending the complete EMS download |
| 2026-10-17 | Memory rewrite after the VM kernel died (out of memory at the linking cell): EMS streamed one year at a time, only the needed columns are linked (built from small lookups), targets computed one at a time, 8 models share one float32 dataset. Measured peak for one year of data fell from 4.1 GB to 1.9 GB; 15 tests pass |
| 2026-10-17 | Added D11 and section 6 (platform notes) and `scripts/vm_check.py` after the VM kernel died again at the panel cell |
