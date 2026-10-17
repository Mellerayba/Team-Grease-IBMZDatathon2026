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
| D3 | **Do not change the model yet.** Link all the data in first (the table `full`), then decide features from there | Decided | Avoid changing the model and the data at the same time; we need a clean before / after |
| D4 | Other data is joined into a **separate table `full`**, never into `panel` | Decided | The model builds its feature list from every column of `panel`, so merging into it would silently change the model (and a text column would break LightGBM) |
| D5 | `eval_is_heat_day` and `eval_is_heavy_rain_day` are **never model inputs**; they only slice results | Decided | They use the whole day's weather, which is unknown when a forecast is made (leakage) |
| D6 | Weather comes from the **Open-Meteo API in New York local time** | Decided | The CSV in the repo is UTC (4-5 hours out of step with EMS timestamps) and blank for Oct 16 and Oct 20-25 2022 |
| D7 | Spatial unit for the simulator and optimizer is the **zone (MODZCTA)**; travel times use zone centroids | Proposed | 99.9% of incidents with a ZIP map to a zone; joins to Census and boundary data. The model still uses raw `zipcode` (see F2) |

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
| **Census ACS** (population, income, no-vehicle households, poverty) | **Not fetched** | - | Needs a free API key (https://api.census.gov/data/key_signup.html) |
| **Road network (OpenStreetMap)** | Not fetched | - | Needed only for travel times; fast option is haversine distance x detour factor |
| Emergency call text | Not available | - | No public source; would need clearly labelled synthetic data |

Combined view: `full` (in `HarryRowlands.ipynb`, after the lag features) = `panel` + weather / calendar on the 30-minute bucket + zone context through ZIP -> zone.
About 4 GB in memory for two years, so roughly 14 GB for 2016-2022.

---

## 3. The model we have now (`HarryRowlands.ipynb`, unchanged)

**Task:** predict incidents per **ZIP code per 30-minute bucket** (`y`).
**Model:** LightGBM, Poisson objective (`n_estimators` 2000, learning rate 0.03, 63 leaves, min 50 samples per leaf, 80% row and column subsampling, early stopping after 100 rounds).
**Features (14):** `zipcode` (category), lags of `y` at 30 min / 1 h / 1.5 h / 3 h / 1 day / 1 week, rolling sums over 2 h / 6 h / 24 h, `hour`, `dow`, `month`, `is_weekend`.
**Split:** by time fraction of whatever data is loaded: train up to 60%, validation 60-75%, test the last 25%.
**Baseline:** mean `y` for each zipcode x weekday x hour in the training period.
**Metrics:** MAE and Poisson deviance.

**Saved results are not usable.** The stored outputs (MAE 0.341 vs baseline 0.356; deviance 0.644 vs 2.242) were produced on a 33-day sample from January 2005, not on 2016-2022. Do not quote them.

Known limitations to address once the data is complete:

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
| F8 | **Target definition** | all incidents; only mid / low acuity (what the plan targets); both | Keep all incidents for the forecaster; add a low / mid-acuity target once severity bands are confirmed | Open | Depends on the severity-code meaning (below) |
| F9 | **Severity code meaning** | 1 = most critical ... 8 = least (assumed) | Assume as stated | Open | Confirm in the NYC data dictionary |
| F10 | **COVID year 2020** | include as is; include with a flag; drop from training | Include with a `year` / covid flag, then test dropping it | Open | Ablation; demand dropped about 8% in 2020 (1.41M vs 1.53M in 2019) |
| F11 | **Train / validation / test split** | by fraction (now); by date | Train 2016-2021 (2020 per F10), validate on the last months of 2021, **test on 2022** and report heat days separately. Weather now covers all of 2022 | Proposed | Needs the full EMS download |
| F12 | **Baseline** | zipcode x weekday x hour mean (now); the same x a recent-surge multiplier; "same bucket last week" | Report all three; the surge-multiplier version is the one in the plan | Proposed | Fixes L2; the model must beat the best of them to claim anything |
| F13 | **Evaluation** | MAE and deviance only; add decision-relevant metrics | Add error on the busiest zones, on heat days, and per borough (equity gap) | Proposed | Matches how the optimizer uses the forecast |
| F14 | **Whole-day weather flags** | - | Never as features (D5) | Decided | Leakage |

---

## 5. Changelog

| Date | Change |
|---|---|
| 2026-10-17 | Log created. Data linked into `full`; model deliberately left unchanged (D3, D4) |
