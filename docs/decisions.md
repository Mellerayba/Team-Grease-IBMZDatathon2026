# Project Dispatch: decisions, research and evidence

The single record of what Team Grease decided, why, what we measured, and what we can and cannot claim. It replaces the earlier separate notes (plan, data, research, hand-off, non-technical timeline, original plan), whose content is merged below. Screenshots and VM output are in [`screenshot_examples.md`](screenshot_examples.md).

Status key: **Decided** (team agreed) · **Proposed** (recommended default, not yet confirmed) · **Open** (needs an experiment or a team call). When a decision changes, mark the old one **Superseded** and link the new one instead of deleting it. State as of **2026-10-17, evening** (IBM Z Datathon 2026, deadline 18 Oct, 10:00 UK).

## Contents
1. [Summary](#1-summary)
2. [How the idea evolved](#2-how-the-idea-evolved)
3. [The problem and the evidence for it](#3-the-problem-and-the-evidence-for-it)
4. [Plan, original design and submission checklist](#4-plan-original-design-and-submission-checklist)
5. [Decisions made](#5-decisions-made)
6. [Data](#6-data)
7. [The forecast model](#7-the-forecast-model)
8. [Feature decisions](#8-feature-decisions)
9. [Changes the model notebook needed (delivered)](#9-changes-the-model-notebook-needed-delivered)
10. [Positioning, dispatch, the replay and its results](#10-positioning-dispatch-the-replay-and-its-results)
11. [Product framing: lightweight first-response units](#11-product-framing-lightweight-first-response-units-d13-d14)
12. [The demo](#12-the-demo)
13. [IBM Z / LinuxONE platform notes](#13-platform-notes-ibm-linuxone)
14. [Assumptions, limitations and open items](#14-assumptions-limitations-and-open-items)
15. [Changelog](#15-changelog)
- [Appendix A: forecast hand-off format](#appendix-a-forecast-hand-off-format) · [B: original build plan](#appendix-b-the-original-build-plan-plan_10) · [C: earlier requirements and pitch material](#appendix-c-earlier-requirements-and-pitch-material)

---

## 1. Summary

**What it is.** Project Dispatch decides where emergency-response units should wait in New York City before calls arrive, and which unit to send. A LightGBM model forecasts expected critical and other 911 medical calls for every ZIP zone over the next 30 minutes to 3 hours; a positioning policy moves idle units toward the demand that is most likely and most vulnerable; a minute-by-minute replay of real 2022 incidents measures what that does to response times against a fixed layout. The product framing (D13) is a fleet of lightweight electric first-response units that stabilise a critical patient until an ambulance arrives; the question for the city is **how many units to buy and where they should stand**.

**What exists and works (verified):**
- Data pipeline from open data into one modelling table (EMS via the NYC Open Data API, weather from Open-Meteo, Census, health, hospitals and firehouses, zone boundaries). Section 6.
- Demand forecaster: 8 LightGBM Poisson models (critical / other × 30 min, 1 h, 2 h, 3 h), calibrated, scored against three baselines on a held-out year; risk index layer. Section 7.
- Positioning and dispatch policy (expected-coverage objective, assignment, equity floor, failsafe) and the replay simulator with four policies including an oracle. Section 10.
- Real street routing through a public OSRM server, with bike profile for the light units. Section 10.
- A React demo that replays the real run: live units, where they head after each decision, incoming calls, and ZIP zones shaded by the model's output. It runs on the LinuxONE VM. Section 12.
- IBM Z accelerator tested and benchmarked on the VM; a measured negative result, reported as such. Section 13.
- 33 automated tests pass (`PYTHONPATH=src python3 -m pytest tests`).

**Headline result (modelled, not measured).** On the 2022-07-19 heat-wave day, 330 units, real incidents: critical calls reached within 8 minutes 82.3% with a fixed layout, **91.2% with model-guided positioning**, 93.4% with perfect foresight. Most of the gain comes from positioning by expected demand at all; our ML forecast adds a little over a historical-average schedule (see section 10 for the honest breakdown).

**Not built / not yet evidenced:** the light-unit (e-bike) replay model and the fleet-size sweep for them; real-time travel times in the decisions (they use a placeholder matrix); the Granite briefing run on the VM; completion of the two mandatory labs; VM screenshots beyond the pasted terminal output. Section 14.

---

## 2. How the idea evolved

| Step | What we considered | Decision and why |
|---|---|---|
| 1. Original idea | A real-time AI model predicting crime locations to dispatch police before calls are placed. Fits Theme 1 (real-time AI for critical decisions) | **Dropped.** Predictive policing can worsen over-policing, increased surveillance can escalate local tension, and shaving seconds matters less for police than for medical emergencies |
| 2. New idea | Predict medical-emergency hotspots to place ambulances where they will be needed | **Adopted.** Ambulances are universally welcomed and avoid the ethical controversy of predictive policing |
| 3. City | New York City | Megacity with a large open EMS dataset and evidence that response times are worsening (below) |
| 4. Unit type (D1) | Ambulances vs mobile clinics. The original plan noted mobile clinics do not respond to calls, so "response time" would not be the metric | **Ambulances** first, because they give a measurable response-time metric and fit the data. Early pitch material (clinic vans, "healthcare deserts") was an earlier direction and is superseded |
| 5. Reframe (D13) | Early "dream feature" in the requirements notes: a fleet of mobile health professionals on bicycles or motorcycles who reach cardiac arrests before an ambulance. The Non-technical timeline proposed medics on motorcycles / e-bikes told where to patrol | **Lightweight electric first-response units** as a second fleet. Ambulances still transport. Cheap enough that "how many should the city buy?" is a real decision |

Wording we replaced after review: the README's earlier "clinics" language and any claim that the product predicts weather-driven incidents (section 7 shows weather adds almost nothing).

---

## 3. The problem and the evidence for it

In New York City the first minutes of a critical emergency decide whether someone survives, but ambulance response times are at record highs. Ambulances are large, expensive, staffed by a shrinking workforce and stuck in the same traffic as everyone else, so the city cannot simply add more of them quickly.

**Response times are getting worse.**
- NYC response times for life-threatening medical emergencies rose from 9:34 in 2021 to 13:09 in the latest fiscal year reported ([EMS1](https://www.ems1.com/response-times/nyc-ambulance-response-times-hit-record-high-amid-ems-staffing-shortage)). FDNY's EMS chief told the City Council the department loses about 10 in-service ambulances a day because of staffing shortages (same source).
- "Emergency Medical Services take 11 minutes and 21 seconds to arrive on the scene, a 29-second increase compared to last year and 1 minute and 4 seconds longer than in 2023." (The Midtown Gazette, September 2026). "The EMS system is collapsing," a union leader says (The Chief, 18 September 2026).
- "Just 16 percent of cardiac arrest patients were revived in FY26, down from 17 percent the year before and 20 percent in Fiscal Year 2024." (The Chief, September 2026).
- The standard in the USA is about 8 minutes for a cardiac arrest; these services take much longer.

**The burden is uneven across boroughs.** The NYC Independent Budget Office found the share of advanced life support calls answered within 10 minutes fell in every borough between 2014 and 2024. In 2024 it was about 82% in Staten Island and about 75% in the Bronx. For basic life support the Bronx was about 48% against about 68% in Staten Island ([IBO analysis summary](https://isoclivecivic.substack.com/p/exploring-an-independent-perspective)).

**Minutes decide survival in cardiac arrest.** The American Heart Association rule of thumb is that survival falls by 7-10% for each minute without CPR or defibrillation ([CARES registry report](https://mycares.net/sitepages/uploads/2024/2023_flipbook/inc/html/34.html)). Newer studies refine this, showing much better survival with defibrillation within about four minutes ([FireRescue1](https://www.firerescue1.com/fire-products/aeds/articles/ny-countys-cardiac-emergency-response-time-subpar-ISf1uNeBLRnlC7jS/)).

**Recognising the emergency on the phone is part of the problem.** Dispatchers fail to identify roughly 25% of out-of-hospital cardiac arrests ([study abstract](https://swov.nl/en/publicatie/machine-learning-supportive-tool-recognize-cardiac-arrest-emergency-calls)). In a Copenhagen study of 108,607 calls a machine-learning model recognised cardiac arrest more often and faster than dispatchers, with more false alarms (same source).

**Why decision support, not automation.** A randomised trial across 169,049 calls found that alerting dispatchers did not significantly improve their recognition of cardiac arrest, even though the model outperformed humans alone ([trial summary](https://busqueda.bvsalud.org/portal/resource/es/mdl-33404620)). We keep a human dispatcher in the loop and test by simulation.

**Why New York.** Traffic gridlock (about 490,000 vehicles and 300,000 pedestrians in Midtown daily), staffing shortages, a population of about 8.8 million, and the largest and most informative open EMS dataset.

**Data limitation.** NYC's public EMS dispatch data aggregates incident locations to protect privacy, so our analysis works at area level, not street level ([data.gov](https://catalog.data.gov/dataset/ems-incident-dispatch-data)).

### Research sources we gathered
- de Bock, J. (2026) 'Ambulances have slower response times and fewer EMTs', The Midtown Gazette, 25 September. https://themidtowngazette.com/2026/09/ambulances-face-slower-response-times-and-fewer-emts/ (accessed 17 October 2026). Used for the problem statement.
- Freeman, H. (2026) 'FDNY ambulance response times eclipse 13 minutes', The Chief, 18 September. https://thechiefleader.com/stories/ambulance-response-times-eclipses-13-minutes,57653 (accessed 17 October 2026). Used for the problem statement.
- U.S. Census Bureau, 'New York city, New York'. https://data.census.gov/profile/New_York_city,_New_York?g=160XX00US3651000 (accessed 17 October 2026). Population statistics.
- New York State Climate Impacts Assessment (2026) 'Climate Impact Spotlight: New York City'. https://nysclimateimpacts.org/explore-by-region/new-york-city/ (accessed 17 October 2026). Climate and weather context.

---

## 4. Plan, original design and submission checklist

**Build order we followed:** (1) clean and aggregate EMS data to zones × 30-minute buckets; (2) baseline forecaster, then LightGBM; (3) travel-time matrix and simulator with a static policy to get baseline response times (the part to protect time for); (4) optimizer and dynamic policy; (5) dashboard and surge-day case study; (6) the story: problem, method, results, limitations. The original design is in Appendix B.

**Submission checklist (IBM Z Datathon, 17-18 Oct 2026).**
- Project title must contain the team name: `Team Grease_IBM Z Datathon 2026 Project`.
- Video of at most 3 minutes: problem, data, solution and how it uses IBM Z (show the LinuxONE instance and the model).
- Code repository link; credit open-source libraries (README); event code `SSF2026`.
- Both provided AI/ML labs (hybrid AI; predictive and generative) must be completed first. **Completion not yet confirmed.**
- Screenshots as evidence of LinuxONE capabilities (containers, security, scalability, compute): `screenshot_examples.md`.
- Must be built within the hack window; optional flowchart and prototype.

**Open decision carried from the plan:** the zone unit. Decided as the MODZCTA zone (D7).

---

## 5. Decisions made

| ID | Decision | Status | Why |
|---|---|---|---|
| D1 | ~~The units we model are **ambulances** (not mobile clinics)~~ **Revised by D13 (2026-10-17):** ambulances stay as the transporting fleet; the lever we study is a second fleet of lightweight first-response units | Revised | Original reason still holds for the ambulance layer: gives a measurable response-time metric and fits the EMS data |
| D2 | **EMS data is fetched in `HarryRowlands.ipynb`** (API -> `notebooks/ems_raw/`). All other data is fetched and cleaned in `DarraghC.ipynb` and linked onto the EMS table in the Harry notebook | Decided | One owner per data source; keeps the model notebook readable |
| D3 | ~~**Do not change the model yet.**~~ **Superseded by D10 (2026-10-17).** Link all the data in first (the table `full`), then decide features from there | Decided | Avoid changing the model and the data at the same time; we need a clean before / after |
| D4 | ~~Other data is joined into a separate table `full`, never into `panel`~~ **Superseded (2026-10-17):** `full` is now the modelling table, and the model uses an explicit feature list (M2), so nothing is picked up by accident | Superseded | The model builds its feature list from every column of `panel`, so merging into it would silently change the model (and a text column would break LightGBM) |
| D5 | `eval_is_heat_day` and `eval_is_heavy_rain_day` are **never model inputs**; they only slice results | Decided | They use the whole day's weather, which is unknown when a forecast is made (leakage) |
| D6 | Weather comes from the **Open-Meteo API in New York local time** | Decided | The CSV in the repo is UTC (4-5 hours out of step with EMS timestamps) and blank for Oct 16 and Oct 20-25 2022 |
| D7 | Spatial unit for the simulator and optimizer is the **zone (MODZCTA)**; travel times use zone centroids | Decided (used by the policy, replay and demo) | 99.9% of incidents with a ZIP map to a zone; joins to Census and boundary data. The model still trains on raw `zipcode` and its output is summed into zones (M7) |
| D8 | **The model's output is expected demand**: calibrated expected incidents per zone for the next 30 min / 1 h / 2 h / 3 h, split into critical and other acuity. The model inputs are the issue time plus what is known at that moment (recent counts, weather, calendar), not time of day alone | Implemented and run on the VM (section 7) | The optimizer needs additive rates, not an ordinal score. Rates can be validated against held-out data; a hand-weighted index cannot. Time of day alone is only the historical average, which is our "scheduled" comparison policy and what we must beat |
| D9 | **The risk index is a derived layer, not a model.** Risk = need-weighted expected demand (critical calls count more, e.g. 3:1, a stated assumption) x a vulnerability factor (poverty, no-vehicle households, asthma rate), shown as a 0-100 percentile map. Later it can add the coverage gap (expected response time minus the 8-minute target) once the simulator exists | Implemented and run on the VM (section 7) | Keeps the forecast testable and the policy weights explicit and adjustable without retraining. Vulnerability data is constant per zone, so it belongs in the weights (and the optimizer's equity floor), not in the forecast |
| D10 | **D3 (model frozen) ends** once the EMS download is complete and the unchanged model has been run once on real data and its results recorded here | Implemented and run on the VM (section 7) | We need an honest before / after, but the model has to change to produce D8's output |
| D11 | **Be honest about the IBM Z accelerator.** Tested on the VM (2026-10-17): it is exposed and Snap ML drives it correctly for a small numeric model, but for tree ensembles it was **slower than CPU LightGBM (0.26-0.57x) and, for larger trees, slightly less accurate**, so production inference stays on the CPU. The honest claim is "verified and benchmarked, then chose the faster CPU path"; see section 13 | Decided | Judges score use of platform capabilities with evidence; a measured negative result is stronger than an unverified claim |
| D12 | **Compare four policies in the simulator: static, scheduled (historical mean), forecast (our model) and oracle (the real future).** Report how close Forecast gets to Oracle and how far it is above Scheduled | Implemented (section 10) | Tells us whether forecast accuracy matters for dispatch at all, and where to spend the remaining time. Hand-off format: Appendix A |
| D13 | **Reframe the product: Project Dispatch adds a fleet of lightweight electric first-response units (e.g. e-bikes) carrying enough equipment to stabilise a critical patient until an ambulance arrives.** Ambulances still transport. The headline question for the city is "how many units should NYC buy, and where should they stand?", answered by replaying the same real incidents at many fleet sizes | Proposed | A cheap, fast layer is something a city can actually buy more of; the fleet-size curve turns our replay into a purchasing decision; it fits "Real-time AI for critical decisions". Detail, assumptions and what must change in the code: section 11 |
| D14 | **Headline metrics:** time to the first responder on scene (light unit or ambulance, whichever is first), share of critical calls reached within 4 and 8 minutes, and the borough gaps, all by fleet size. Any "lives saved" figure is an assumption-driven illustration, not a result | Proposed | Our data has no outcomes, so survival can only be inferred through a published rule of thumb (section 11) |

---

## 6. Data

| Dataset | Status | Coverage / size | Known issues |
|---|---|---|---|
| **EMS incidents** (`ems_raw/`, fetched by the Harry notebook's first cell: time, ZIP, initial severity, response-valid flag) | **2017-2022 complete** | 2017-2022 match the API totals exactly (about 1.4-1.6M rows a year); 2016 is truncated (350,000 of 1,441,211 rows) | **`2016.csv` is truncated and will never refetch** (the cell skips years whose file exists): delete it and rerun if 2016 is wanted. Only 4 columns, so no stage durations or real response times in this download (a richer local extract exists, see below) |
| **Weather** (Open-Meteo, hourly, one point for all NYC) | Done | 2016-01-01 to 2022-12-31, 61,368 hours, no gaps; 122,736 rows on the 30-minute grid; 13 weather features | Citywide only: varies over time, not between zones |
| **Calendar** | Done | US federal holidays (72 in 2016-22) plus day before / after; 88 heat days (daily max >= 32 C), 239 heavy-rain days (>= 12.7 mm) | Heat days per year: 2016 17, 2017 7, 2018 9, 2019 7, 2020 11, 2021 13, 2022 24 (small samples) |
| **Zones (MODZCTA)** + ZIP crosswalk | Done | 178 zones, 219 ZIPs mapped | A few rare ZIPs (10121, 10000, 10281, ...) are in no zone: about 11% of panel rows but only about 0.1% of incidents that have a ZIP |
| **Hospitals, firehouses** (candidate staging posts) | Done | 60 hospitals, 219 firehouses, assigned to zones (2 firehouses unassigned) | True ambulance stations are not public |
| **FDNY monthly response times** | Done | Monthly by borough and incident class (fiscal-year summary rows dropped) | "Medical Emergencies" covers all first responders, not just ambulances: a rough benchmark only |
| **Asthma rates** (2017-19, 188 neighbourhoods) | Done, borough level | Borough asthma rate per 10,000 (as given in the source file; we have not confirmed whether it counts ED visits or hospitalisations): Bronx 190.7, Brooklyn 115.2, Staten Island 96.3, Manhattan 89.5, Queens 64.1 | No neighbourhood -> zone crosswalk, so borough only; constant within a borough |
| **Community health survey** | Done | 11 yearly rows, citywide only | No geography, so context only |
| **Zone context** (`zone_context.csv`: borough, location, population, asthma rate, distance to nearest hospital, hospitals / firehouses in zone) | Done | 178 zones | Borough comes from the community district containing the zone centroid |
| **Census ACS** (2022 5-year: population, median income, no-vehicle households, poverty) | Done | 177 of 178 zones; in `zone_context.csv` and so in `full` | Zone populations sum to about 8.7M vs roughly 8.3-8.5M for the city (ZIP areas extend past the boundary), so approximate. By borough, households without a vehicle: Manhattan 78%, Bronx 60%, Brooklyn 55%, Queens 34%, Staten Island 15% |
| **Road network (OpenStreetMap)** | Used on demand | Routes from a public OSRM server, cached; no full download | Decisions still use haversine × 1.35 at 30 km/h; OSMnx cannot be installed on the VM |
| Emergency call text | Not available | - | No public source; would need clearly labelled synthetic data |

**A richer local extract (git-ignored, provenance not recorded in the repo):** `notebooks/data/processed/incidents.csv.gz` has 3,255,184 incidents from 2019-01-01 to 2022-08-31 with call types, severity, zone, and stage durations (`dispatch_s`, `travel_s`, `response_s`, `onscene_s`, `to_hosp_s`, `job_s`); `service_times.csv` summarises them. Critical calls (severity 1-3): mean on scene 24.0 min, mean to hospital 11.8 min, mean whole job 68 min (others about 60 min). The demo export reads this file for call positions and the 2021 layout demand; **no notebook in the repo regenerates it**, so record how it was made before relying on it (open item). It also shows the replay's service times are shorter than reality (section 14).

### Pipeline and notebooks (from the data notes)
Two notebooks, each with one job. Everything is open data. Outputs live in `notebooks/data/`.
```
notebooks/DarraghC.ipynb        all NON-EMS data: fetch + clean  -->  notebooks/data/weather_calendar_30min.csv.gz
                                                                       notebooks/data/interim/*.csv   (small, tracked in git)
notebooks/HarryRowlands.ipynb   EMS incidents (API -> notebooks/ems_raw/), panel + lag features,
                                links the tables above into one big table `full`; then the model, the routing replay, the hand-off export
```
`notebooks/data/raw/` (downloads) and `notebooks/data/processed/` are git-ignored. Download code lives in `scripts/fetch_data.py` (used by `DarraghC.ipynb`). Run `DarraghC.ipynb` first (about 10 seconds once the raw files are cached), then `HarryRowlands.ipynb`.

| Dataset | Source | Where it is cleaned | Output | Role |
|---|---|---|---|---|
| **EMS incidents** (time, ZIP, severity, valid-response flag) | NYC Open Data `76xm-jjuj` via the Socrata API, 4 columns, 2016-2022 | `HarryRowlands.ipynb` | `notebooks/ems_raw/<year>.csv` | Demand history for the forecaster |
| **Weather** (hourly temperature, feels-like, humidity, rain, wind, cloud) | Open-Meteo archive API, requested in **New York local time** | `DarraghC.ipynb` 1-5 | `weather_calendar_30min.csv.gz` | Forecaster features; heat / rain days for surge case studies |
| **Calendar** (holidays, day before / after) | pandas US federal holidays | `DarraghC.ipynb` 4 | same file | Forecaster features |
| **Zones** (MODZCTA) + ZIP crosswalk | NYC Open Data `pri4-ifjk` | `DarraghC.ipynb` 7 | `interim/zones.csv`, `zip_to_zone.csv`; boundaries in `raw/modzcta_boundaries.geojson` | Spatial unit, centroids, demo polygons |
| **Hospitals, firehouses** | NY State Health `vn5v-hh5r`, FDNY `hc8x-tcnd` | `DarraghC.ipynb` 8 | `interim/posts.csv` | Hospital destinations in the replay and demo; candidate posts |
| **FDNY monthly response times** | NYC Open Data `j34j-vqvt` | `DarraghC.ipynb` 9 | `interim/fdny_monthly.csv` | Benchmark for the baseline |
| **Asthma rates** (NTA, 2017-19), **health survey** (citywide) | repo CSVs in `notebooks/data/` | `DarraghC.ipynb` 10 | `interim/asthma_*.csv`, `health_survey_citywide.csv` | Health context, equity narrative |
| **Census ACS 5-year** | Census API with a free key in the git-ignored `.env` (`CENSUS_API_KEY`, see `.env.example`) | `DarraghC.ipynb` 10 | `interim/census_zone.csv` | Population weighting, vulnerability. Skipped without a key |
| **Zone context** | built from the rows above | `DarraghC.ipynb` 10 | `interim/zone_context.csv` | One join gives borough, location, population, asthma rate, hospital access |
| Road network | OSRM public server (OSMnx optional) | `notebooks/street_routing.py` | `raw/route_cache.json` | Routes shown for the policy's moves |

**The big table (`full`, built in `HarryRowlands.ipynb` after the lag features):** `panel` (zipcode × 30-minute `bucket`, `y` = incidents, lags, time features) plus weather and calendar on `bucket` (`temp_c, feels_like_c, humidity_pct, precip_mm, is_raining, rain_3h_mm, rain_24h_mm, wind_kmh, cloud_pct, temp_max_24h_c, is_hot, is_heat_stress, is_cold`, holiday flags), zone context through `zipcode → zone` (`zone, borough, lon, lat, pop_est, asthma_rate_per_10k, dist_nearest_hospital_km, n_hospitals, n_firehouses`, Census fields), and `eval_is_heat_day` / `eval_is_heavy_rain_day` (**slice results only, never model inputs**). The model uses an **explicit feature list** (`FEATURE_GROUPS`, `src/grease/forecast/features.py`), so adding columns never changes the model by accident. For a model that predicts `h` buckets ahead, weather must use only what is known at forecast time.

**Memory.** The notebook streams the EMS files one year at a time, attaches only the columns the chosen features need, and trains all 8 models from one shared 32-bit dataset. Measured peak for one year of data: 1.9 GB (it was 4.1 GB before this was optimised); two years build the panel in 6 s and 1.6 GB on the VM. Extrapolated, not measured: about 6 GB for 4 years and 8-11 GB for all of 2016-2022. Set `EMS_YEARS` and the matching `TRAIN_START` / `VAL_START` / `TEST_START` in the settings cell to use fewer years. A kernel that dies mid-run is almost certainly out of memory.

**Data assumptions:** zone = MODZCTA (99.9% of EMS incidents with a ZIP map to a zone); weather in local time (the CSV originally in the repo was UTC and blank for Oct 16 and Oct 20-25 2022); borough of a zone is the community district containing its centroid; no ambulance location or fleet data is public, so fleet size and home posts are assumptions; locations are area-level only; FDNY "Medical Emergencies" covers all first responders, so it is only a rough benchmark; asthma rates are borough-level; the health survey is citywide only.

**Open data items:** the 4-column EMS download lacks stage durations and call type (the richer local extract above has them); confirm the severity-code meaning (assumed 1 = most critical); re-fetch 2016 if more years are wanted.


---

## 7. The forecast model

### Current (rewritten 2026-10-17, implements D8-D10)
**Output:** expected incidents per ZIP for the next 30 min / 1 h / 2 h / 3 h, separately for critical and other calls (8 LightGBM Poisson models), plus a derived 0-100 risk index per zone.
**Code:** `src/grease/forecast/` (`features.py`, `lgbm.py`, `baseline.py`, `evaluate.py`, `zones.py`), `src/grease/risk.py`, driven from `HarryRowlands.ipynb`; 10 unit tests in `tests/test_forecast.py`.
**Default features (29):** ZIP id; current and recent counts (`y`, `y_critical`, lags 30 min to 1 week, rolling sums 2 h / 6 h / 24 h, recent critical sum); time (`hour`, `dow`, `month`, `is_weekend`); weather (temperature, feels-like, rain now and 3 h, wind); holiday flags; zone location, population estimate and borough. Change `FEATURE_GROUPS` in the notebook to run the ablations in section 8.
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
- **Hand-off table written:** 3,101,040 rows, 177 zones, 24 columns (Appendix A).

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

Limitations of the previous model (all addressed by the rewrite, see section 9):

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

## 8. Feature decisions

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

## 9. Changes the model notebook needed (delivered)

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
| M11 | **Hand-off export**: zone x half-hour table with forecast, scheduled and oracle demand and the risk index; format in Appendix A | New cell in the notebook, `src/grease/forecast/export.py` | Must | Done (tested; written when the notebook runs) |
| M12 | **Long-run level features** (F16) | `features.py` | Should | Done; effect small, see F16 |

All ten are implemented. We skipped the intermediate "unchanged model on real data" baseline run that D10 suggested, because the original model cannot produce D8's output; the honest before / after is now the new model against the three baselines.

**Outcome:** the real runs are recorded in section 7; the F5-F7 ablations, F10 (COVID) and F1 (horizons) are still open.

---

## 10. Positioning, dispatch, the replay and its results

Code: `notebooks/routing_policy.py` (policy), `notebooks/ems_replay.py` (replay), `notebooks/street_routing.py` (real routes), `notebooks/demo_policy.py` (synthetic sanity check), tests in `tests/test_routing.py` and `tests/test_street_routing.py`. The skeleton packages `src/grease/sim`, `optimize`, `travel` and `dashboard` were the original plan's placeholders; the working implementation is the notebook-side code above.

### The policy (`RoutingPolicy`)
Every 30 simulated minutes, using the forecast issued at the end of the previous half-hour:
1. **Demand weights per zone** = critical forecast × 3 + other forecast × 1, scaled by (1 + 0.5 × vulnerability) so vulnerable zones count more (both weights are stated assumptions, D9).
2. **Objective: expected weighted demand covered (MEXCLP-style).** A zone covered by `n` units counts `w × (1 − q^n)`, where `q` is the live busy fraction of the fleet, so extra units on one zone have diminishing value and units spread out when the fleet is stretched. "Covered" means a unit could arrive within 8 minutes (critical) or 15 minutes (other), using the current travel-time matrix times the traffic multiplier.
3. **Greedy selection of staging zones** by marginal gain, seeding an **equity floor** first (at least one unit per borough), then **Hungarian assignment** (`scipy.optimize.linear_sum_assignment`) of idle units to those zones by drive time.
4. **Only worthwhile moves are made:** a discretionary move must add at least 0.3% of total weighted demand (stops thrashing), no unit is moved further than 20 minutes, at most 3 discretionary moves per cycle and 2 per unit per hour. Units that have just finished a job ("freed") have no position to protect, so they are sent to the best open staging zone without those caps.
5. **Failsafes and live data:** if the forecast is older than 45 minutes the policy holds all positions; an optional nowcast (Gamma-Poisson update) scales the forecast by what actually arrived in the last window, damped; every move carries a plain-language reason string (used by the demo log and, optionally, the Granite briefing).
6. **Dispatch:** a critical call gets the fastest idle unit with no trade-offs; other calls may take a unit up to 1 minute slower if its absence hurts coverage least.

**Dispatcher briefing (generative AI).** A notebook cell turns one cycle's moves into a short plain-language briefing with IBM Granite (`granite-3.3-2b-instruct`), with a guard: if the model mentions a ZIP that is not in the moves, the briefing falls back to the plain reason strings. It needs about 5 GB for the model plus a ~5 GB download, so the cell skips itself below 8 GB free (the VM had 5.6 GB); the plan was a quantised build through llama.cpp. **Implemented but not yet run on the VM, so not evidenced.**

### The replay (`run_replay`)
Steps through one day of **real incidents** minute by minute. Each call is dispatched to an idle unit (a repositioning unit stays dispatchable while it drives, and is dispatched from where it still is, which is conservative), then the unit spends 15 minutes on scene, drives to the nearest real hospital zone, spends 10-25 minutes there (drawn once per call so every policy faces identical service times), and becomes available at the hospital. The traffic multiplier is 1.4 in the 7-10 and 16-19 hour rush periods and 1.0 otherwise (an assumption). Calls with no idle unit queue, critical first.
- **Static** units drive back to their fixed home posts and stay dispatchable; **dynamic** units are repositioned by the policy.
- **Four policies on identical calls, fleet and travel times:** static (one layout optimised on the 2021 training-period average, never changed), scheduled (historical mean demand for that zone and time of week, no ML), forecast (our model), oracle (the demand that actually happened in the next hour: impossible, the ceiling).
- A trace option (`trace=True`) records every unit's drives and stops and every call's life so the demo can animate the run; a test checks it changes no numbers and that a unit is never in two places at once.

### Fleet size is an assumption, so it is calibrated and swept
No public data says how many units NYC fields or where they stand. About 4,600 calls a day with about an hour per job means about 190 busy units on average. At 200-300 units the replay is overloaded (calls queue for hours, about 20% unserved); at 400+ the static layout already beats reality. **330 units** was chosen so the static layout lands in a plausible range, and the sensitivity table in the notebook replays 300 and 370 units too: results swing a lot near the tipping point between about 300 and 400 units, so the fleet-size curve (section 11) is part of the result, not a footnote. The calibration should be re-checked against real travel times (`incident_travel_tm_seconds_qy` in the full EMS API).

### Street routing (`street_routing.py`)
OSMnx cannot be installed on the LinuxONE VM (it needs shapely, which has no s390x build, and the GEOS library is missing), so routes come from a public OSRM server: about 0.2 s per route, only `requests` needed, cached on disk (`notebooks/data/raw/route_cache.json`), a flagged straight-line fallback, OSMnx used only if present, and a bike profile for light units (bike times about 2-3× the car times on the trips tested; the speed is an assumption to check). It is used to show the real route and time of each move the policy recommends; **the policy still decides using the placeholder matrix (haversine × 1.35 at 30 km/h)**.

How far off is the placeholder? Five test trips, OSRM car time against our placeholder: short Bronx hop 3.9 vs 5.6 min, Bronx to East Harlem 13.4 vs 14.0, Queens to Midtown 10.7 vs 13.4, Brooklyn to East Harlem 30.9 vs 39.3, Staten Island to Brooklyn 30.4 vs 43.0. So the placeholder is roughly 4-30% too slow against OSRM; against OSMnx free-flow speed limits (no signals) it was about 40-60% too slow, which overstates it. The truth for an ambulance with a siren in traffic lies somewhere in this range. The earlier proof of concept (`notebooks/Mapping.ipynb`) computed one Manhattan route (Times Square to Washington Square Park: 5.2 minutes free-flow) with OSMnx; it was superseded by the OSRM router.

### Results (all modelled, none measured)
Notebook run, two real 2022 days (2022-07-19 heat wave and 2022-10-12), 330 units, ZIP-level, averages of both days:

| Policy | Critical calls reached in 8 min | 90th-percentile response (min) |
|---|---|---|
| Static layout | 76.4% | 17.1 |
| Scheduled (historical demand) | 84.4% | 12.7 |
| Forecast (our model) | 84.9% | 12.5 |
| Oracle (perfect foresight) | 86.2% | 11.6 |

Demo export (`scripts/export_demo.py`), 2022-07-19 only, zone level (178 zones, real hospital sites), 330 units, full day of 4,694 incidents:

| Policy | Critical within 8 min | Critical mean (min) | Critical 90th pct (min) | All calls mean (min) | Moves |
|---|---|---|---|---|---|
| Static | 82.3% | 5.67 | 10.47 | 5.87 | 0 |
| Model-guided (forecast) | 91.2% | 4.41 | 7.76 | 4.50 | 2,167 |
| Oracle | 93.4% | 4.26 | 7.39 | 4.53 | 2,112 |

Critical calls within 8 minutes by borough, same run: Bronx 80.8 → 89.6 (oracle 92.5); Brooklyn 81.8 → 93.8 (94.4); Manhattan 80.4 → 91.4 (95.3); Queens 83.6 → 89.2 (92.1); **Staten Island 94.6 → 89.2 (87.1)**. The two tables differ because of the day, the spatial level (a ZIP-level check of the static layout on 2022-07-19 gave 80.8% against 82.3% at zone level) and the averaging; compare policies within one table, not across.

**Reading it honestly**
- Most of the gain comes from positioning by expected demand at all (static → scheduled is +8 points on average); our ML forecast adds only about half a point over the historical schedule, consistent with the forecast scoring in section 7 (it is mostly last week's level times the usual daily shape). The gain held at 300 and 370 units.
- **Equity flag:** the dynamic policies did worse than static in Staten Island on every day tested (97.7% → 89.5% on one day, 58.8% → 57.5% on the other; 94.6 → 89.2 in the demo day). Units are drawn toward dense boroughs. This needs a fix (a larger floor for isolated boroughs) or an explicit statement; the equity floor of one unit per borough is not enough.
- Moves in the demo window are mostly freed units being sent to a staging zone ("back into service") rather than idle units being relocated (2 true relocations in the 6-hour window), because of the minimum-gain threshold and caps.
- Response times are modelled from an assumed fleet, scene and hospital times, traffic, and the placeholder travel matrix. **Only the difference between policies is meaningful.**

---

## 11. Product framing: lightweight first-response units (D13, D14)

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

**Replay results, routing and street-network work** are in section 10.


---

## 12. The demo

A React 19 + TypeScript + Vite + Tailwind + React-Leaflet app in `pulsenyc-source/pulsenyc` (built by Frederic as an in-browser synthetic simulation, then rewired by us to replay the real run). It reads `public/data/demo.json` and `zones.geojson`, produced by `scripts/export_demo.py`; nothing is simulated in the browser.

**What it shows**
- **Zones shaded by the model:** ZIP-zone polygons coloured by the model's risk index, the forecast calls for the next hour, or what actually happened; a new forecast lands every 30 minutes and the shading changes with it.
- **Live units**, coloured by what they are doing (available, moving to a new post, responding, on scene, to hospital, at hospital).
- **Where units go after a decision:** dashed cyan street routes (real OSRM routes) with a ring at the destination, and a log of each move with its reason.
- **Calls coming in:** a dot at the moment of the call (dark red until a unit is assigned, orange while one is en route), removed when a unit arrives; a live feed and counts.
- **Compare view:** static layout and model-guided fleet side by side on the same calls, with running critical-call response metrics, a "units in the highest-risk zones" table (static vs model) and the full-day result including the oracle.

**Data flow:** `forecast_zone_test.csv.gz` (hand-off table, Appendix A) + `incidents.csv.gz` → `ems_replay.run_replay(trace=True)` for static, forecast and oracle → `export_demo.py` (unit legs, call lifecycles, zone polygons simplified with shapely, per-cycle forecast and risk, OSRM routes for moves, summary table) → `demo.json` (about 2 MB). Default window: 2022-07-19, 14:00-20:00, 330 units; at 4× speed it plays in about 90 seconds, which fits the 3-minute video. `?t=150` opens the replay at a given minute.

**Checked:** headless Chrome plays the compare view at about 59 frames per second with no console errors; the app was run and viewed on the LinuxONE VM through an SSH tunnel (`npm run dev` on the VM, port 5173). Screenshots: `screenshot_examples.md`.

**Limits:** repositioning moves follow real streets, but responding and hospital legs are drawn as straight lines; units shown are the generic fleet, not the light units; `src/engine.ts` (the earlier synthetic engine) is unused. The VM cannot run the export script (shapely has no s390x wheel); the data is committed so the demo runs without it.

---
## 13. Platform notes (IBM LinuxONE)

What we know, and where it comes from:
- **Guidebook:** gives no hardware specification. Judging criterion 3 rewards AI/ML running on LinuxONE and "platform capabilities such as containerisation, security, scalability and compute resources, using screenshots as evidence". The two provided AI/ML labs are mandatory.
- **VM image** ([jupyter-lab-ml README](https://github.com/linuxone-community-cloud/jupyter-lab-ml)): Ubuntu 24.04 container with TensorFlow, PyTorch, XGBoost, scikit-learn, Keras and Llama.cpp. It says to use the `SSF2026` event code "to get an instance with sufficient resources for ML notebooks", and states no vCPU or memory figures. It mentions z17's Telum II on-chip AI inference accelerator. Snap ML and LightGBM are not in the list it describes (LightGBM does import on our VM).
- **What the accelerator does** ([IBM Z Accelerated for Snap ML](https://github.com/IBM/ibmz-accelerated-for-snapml), [Snap ML model import docs](https://snapml.readthedocs.io/en/latest/model_import.html)): on z16 and later, Snap ML can run **inference** of tree-ensemble models on the Integrated Accelerator for AI through zDNN. On older machines it runs on the CPU with no changes. It does not accelerate training or data preparation.
- **Could our LightGBM models use it?** The Snap ML import table lists LightGBM regressors (formats PMML, ONNX, text), but the same page says elsewhere that only PMML is supported, and it does not say whether the Poisson objective is supported. **Verified on the VM (below):** the class-level importer reads LightGBM text files. Pickled models are not portable between x86 and s390x (endianness); LightGBM text files are.
- **Installing Snap ML:** its GitHub page distributes it as a container image from `icr.io` that needs credentials; it does not mention a pip package. **Verified:** `pip install snapml` installs a PyPI wheel (1.17.2, cp312, manylinux_2_28 s390x) on the VM.

**Measured on the VM (`scripts/vm_check.py`, 2026-10-17; raw output in `screenshot_examples.md`):** s390x big-endian, IBM Z type 9175 (z17 generation) model 702 ME1, Linux 6.8 (Ubuntu 24.04), Python 3.12, **3 logical CPUs, 8.4 GB RAM (5.6 GB available at the time), no swap**; CPU flags `vx`, `vxe`, `vxe2`, **`nnpa` (the on-chip AI accelerator)**, `sort`, `dflt`; `libzdnn.so.0` installed. Libraries present: numpy 2.0.2, pandas 3.0.6, LightGBM 4.7.0, scikit-learn 1.9.0, XGBoost 3.4.1, PyTorch 2.11.0 (CPU); Snap ML and ONNX Runtime were not installed until we added Snap ML. The first run showed the libraries missing because a different Python (3.12.3) was active from the notebook environment's (3.12.15), which has them.

**Accelerator test results (run on the VM, 3 CPUs, `scripts/accel_check.py` and `scripts/accel_bench.py --quick`).**
- Exposed and usable: `nnpa` flag present, `libzdnn` installed, Snap ML 1.17.2 installs from a PyPI wheel (cp312, s390x) and selects the accelerator format (`zdnn_tensors`). Snap ML's class-level `BoostingMachineRegressor().import_model(file, "lightgbm")` reads LightGBM text files directly (the generic `snapml.import_model` is PMML-only; the docs table is misleading). The importer enforces PMML rules named in its own error messages.
- Correct for small numeric models: output equals LightGBM's raw score to about 1e-8 (apply `exp` for the Poisson mean).
- **Not faster.** Thousand rows per second, all engines at 3 threads (Snap ML defaults to 1 thread, which made the first comparison unfair):

| Model | Batch | LightGBM CPU | Snap ML CPU path | Snap ML accelerator |
|---|---|---|---|---|
| 15 leaves x 100 trees | 256 | 773 | 966 | 417 |
| 15 leaves x 100 trees | 100,000 | 908 | 1,020 | 239 |
| 63 leaves x 100 trees | 256 | 599 | 728 | 343 |
| 63 leaves x 100 trees | 100,000 | 655 | 783 | 240 |

  The accelerator is 0.26-0.57x the best LightGBM speed in every cell (also slower at 1 thread); Snap ML's own CPU path is 12-25% faster than LightGBM. Per-call latency for a live cycle of 256 rows is 0.3-0.7 ms on every engine, so inference speed is not a bottleneck for us.
- **Accuracy caveat:** for the 63-leaf model the accelerator's output differed from LightGBM by up to 0.049 in the raw score (about 5%), against 1e-8 for the 15-leaf model. My hypothesis (untested) is that the accelerator computes in 16-bit precision so rows near a split threshold can take the wrong branch.
- **Our real forecaster cannot be moved to Snap ML as it is:** it uses categorical features (zipcode, borough); the import succeeds without an error but the predictions are wrong (max difference 4.6 in the raw score). Using Snap ML would need a numeric-only retrain, and the gain (at most about 20% on a sub-second job) does not justify it.
- **Conclusion:** keep CPU LightGBM for production scoring. In the pitch: the accelerator is present and drivable on LinuxONE, we benchmarked it fairly against the CPU, and chose the faster and more accurate CPU path. Not claimed: any speed-up from the accelerator. The neural-network route (IBM Z Deep Learning Compiler) is the accelerator's real target and would only apply if we built a neural forecaster.

**What the VM is used for.** Training, scoring, the replay and the demo all run on the LinuxONE instance (the saved models are portable text files; pickles are not portable between x86 and s390x). The kernel died twice early on, caused by memory (the default then loaded seven years of data); the memory rewrite and the 2021-2022 default fixed it (section 6). Not run on the VM: the Granite briefing (needs about 8 GB free; it skips itself), the demo export (shapely has no s390x wheel), OSMnx.

---

## 14. Assumptions, limitations and open items

### Assumptions we state (every number rests on these)
| Assumption | Value | Where | How we handle it |
|---|---|---|---|
| Fleet size | 330 units | replay | Calibrated; 300 and 370 also replayed; sweep needed for the light-unit curve |
| Home posts for the static layout | optimised on 2021 average demand | replay | Fair strong baseline, never sees 2022 |
| Scene time / hospital time | 15 min / 10-25 min (random, same for every policy) | replay | Placeholders; only differences between policies matter |
| **Modelled job length vs the data** | about 50 min modelled (arrival + 15 scene + hospital leg + 10-25 at hospital) against a **mean of 68 min for critical calls (about 60 for others)** in the local incident extract (section 6) | replay | **Known gap: the replay frees units about 20 minutes too soon.** The 330-unit calibration was done with the shorter times, so re-draw service times from the data and re-calibrate the fleet (open item 3) |
| Traffic | ×1.4 in 7-10 h and 16-19 h | replay | Placeholder |
| Travel times in decisions | haversine × 1.35 at 30 km/h | policy | Placeholder, 4-30% slow against OSRM |
| Critical = severity 1-3 | F9 | forecast, replay | Needs the NYC data dictionary to confirm |
| Critical call worth 3 other calls; vulnerability boost up to +50% | D9 | risk index, policy | Explicit and adjustable without retraining |
| 8-minute coverage target (critical), 15 (other) | policy config | policy | Standard cardiac-arrest benchmark |
| Locations are zone centroids | D7 | everything | Public data is area-level only |

### What we can claim
- A forecaster that beats every baseline on Poisson deviance at every horizon on a held-out year (5-7% lower deviance), and is calibrated to about 4% low.
- Positioning idle units by expected demand, with an equity floor, improves modelled critical-call response against a fixed layout across fleet sizes and days (about 82% → 91% in 8 minutes on the demo day, 76% → 85% averaged over two days).
- The model's forecast is a small improvement over a schedule built from history alone; most of the value is in positioning.
- The IBM Z accelerator is present, drivable, and was benchmarked fairly (section 13).

### What we cannot claim
- Real response-time improvements or lives saved: times are modelled, there are no outcome data, and any survival figure is an illustration (D14).
- A weather-driven forecast: weather contributes about 1% of model gain in total.
- That AI "predicts the weather" or does "AI-driven path optimization": the policy is an optimiser fed by a forecast; routing is OSRM.
- Any speed-up from the IBM Z accelerator.
- That the light units work as modelled: the light-unit model is not built.
- Equity improvements: Staten Island does worse under the dynamic policy (section 10).

### Pitch and README claims to reconcile before submission
- The deck notebook (`notebooks/Grease_DISPATCH_Pitch_Deck (1).ipynb`, `docs/Grease DISPATCH Pitch Deck.pptx`) lists "weather related incident prediction", "AI-driven path optimization", theme "Wildcard" and a "mobile clinic fleet"; these conflict with the evidence above and with D13.
- The older pitch notebooks (`V1 Pitch.ipynb`, `Project Grease Pitch Deck.ipynb`) describe mobile clinic vans and "healthcare deserts"; superseded by D1/D13.
- README open-source table lists Streamlit; the dashboard package is a skeleton and the demo is React.

### Open items (priority order for the remaining time)
1. Build the light-unit replay (speed, no hospital leg, stays until the ambulance arrives, shorter target) and the fleet-size sweep, then re-derive the headline.
2. Fix or explain the Staten Island equity gap.
3. Re-draw service times from the incident extract (real jobs average 68 min for critical calls) and re-check the fleet calibration with real travel times; pass a street-route travel matrix into the policy (`tt_base`).
4. Capture VM screenshots (list in `screenshot_examples.md`); confirm the two mandatory labs are completed; run the Granite briefing on the VM (it skips itself under 8 GB free).
5. Record the 3-minute video and reconcile the deck.
6. Possible: re-fetch EMS with call type and stage durations for service times and cardiac-arrest share; offset model (F17) if the 4% level gap matters; F5-F7 ablations; F10 COVID year.

### Repository hygiene notes
`notebooks/ems_raw/` (about 673 MB of EMS CSVs) and the front end's `node_modules` have been committed by teammates; untracking them would delete teammates' local copies on pull, so it was left alone by request. A fresh `npm install` is needed after a clone because the committed `node_modules` is broken. Secrets: the Census key lives in the git-ignored `.env`; a Socrata app token is in a committed `notebooks/config.py` (historical).

---
## 15. Changelog

| Date | Change |
|---|---|
| 2026-10-17 | Log created. Data linked into `full`; model deliberately left unchanged (D3, D4) |
| 2026-10-17 | Census fetched and joined into the zone context (affects F7: population, income, vehicle access and poverty are now candidate features) |
| 2026-10-17 | Added D8-D10 (model output = expected demand; risk index is a derived layer; D3 ends after the baseline run), revised F8, and listed the model changes needed (section 5) |
| 2026-10-17 | Model rewritten per D8-D10 and M1-M10 (code in `src/grease/forecast/`, notebook updated, 10 tests pass). Smoke-tested on 2017 only; real run pending the complete EMS download |
| 2026-10-17 | Memory rewrite after the VM kernel died (out of memory at the linking cell): EMS streamed one year at a time, only the needed columns are linked (built from small lookups), targets computed one at a time, 8 models share one float32 dataset. Measured peak for one year of data fell from 4.1 GB to 1.9 GB; 15 tests pass |
| 2026-10-17 | Added D11 and section 6 (platform notes) and `scripts/vm_check.py` after the VM kernel died again at the panel cell |
| 2026-10-17 | VM measured: 8.4 GB RAM (5.6 GB free), 3 CPUs, IBM z17-generation LinuxONE with the `nnpa` AI-accelerator flag exposed; two years of data build the panel in 6 s and 1.6 GB. The kernel died because the notebook still defaulted to all seven years, so the repo default is now 2021-2022 (train Jan-Sep 2021, validate Q4 2021, test 2022). The full 2016-2022 history needs a bigger machine, so F11 / F10 experiments over many years are limited on the VM |
| 2026-10-17 | First real run recorded (section 7). Added 7-day / 28-day level features (F16), the zone-level hand-off export and the hand-off format (M11, Appendix A), the four-policy comparison with an oracle (D12), and F15-F17 |
| 2026-10-17 | Run 2 (with level features) recorded, including calibration, feature importance (roll_7d 71%, weather about 1% in total) and the risk snapshot. Corrected an earlier claim: the surge baseline is worse on MAE but not on deviance |
| 2026-10-17 | Reframed the product (D13, D14): lightweight first-response units as a second fleet, headline question = how many units to buy. Added section 7 (assumptions, required model changes, replay results, routing review). D1 revised |
| 2026-10-17 | Accelerator tested on the VM and benchmarked fairly: works but slower than CPU LightGBM; D11 decided, results in section 6 |
| 2026-10-17 | Demo wired to the real model run (`scripts/export_demo.py`, `ems_replay.run_replay(trace=True)`, React `replay.ts` / `DemoMap.tsx`): the browser replays real 2022-07-19 calls, the forecaster's per-cycle zone output (risk index shading) and the policy's moves on real street routes; the synthetic in-browser engine is no longer used. Zone-level replay (178 zones, hospitals at real sites) gives, for 14:00-20:00 and the full day at 330 units: critical calls within 8 min static 82.3% vs forecast-guided 91.2% vs oracle 93.4% (modelled times; notebook ZIP-level numbers can differ slightly) |
| 2026-10-17 | Consolidated every note (plan, data, research, hand-off, non-technical timeline, original plan) into this file; README rewritten; screenshots and VM output collected in `screenshot_examples.md` |

---

## Appendix A: forecast hand-off format

The forecaster answers one question: **for each zone, how many incidents do we expect in the next 30 min / 1 h / 2 h / 3 h, split into critical and other?** It does not place units or route them.

**The table** (`notebooks/data/processed/forecast_zone_test.csv.gz`, git-ignored, written by the last cells of `notebooks/HarryRowlands.ipynb`; rerun the notebook to regenerate). One row per **zone × 30-minute `bucket`** of the **test year (2022)**, about 3.1 million rows for 177 zones.

**Time convention.** `bucket` is the *start* of a half-hour that has just finished. The row is what a live system would know at the end of that half-hour (`bucket + 30 min`), and every value describes the window that **starts at `bucket + 30 min`**: `_30m` = the next half-hour, `_1h` = the next hour, `_2h`, `_3h` likewise. Windows that run past the end of the data are NaN.

| Column | Meaning | Policy that uses it |
|---|---|---|
| `zone` | MODZCTA zone id (string), joins to `notebooks/data/interim/zone_context.csv` and `posts.csv` | all |
| `bucket` | see above | all |
| `pred_critical_<h>`, `pred_other_<h>` | **the model's** expected incidents in the window (`<h>` = `30m`, `1h`, `2h`, `3h`) | **Forecast** policy |
| `hist_all_<h>` | historical mean for that zone, weekday and time of day, all calls (no ML) | **Scheduled** policy |
| `actual_critical_<h>`, `actual_other_<h>` | what really happened in the window | **Oracle** policy (the ceiling) and scoring |
| `risk_index_1h`, `risk_index_3h` | 0-100 risk index (`src/grease/risk.py`): need-weighted expected demand × vulnerability, ranked across zones within that half-hour | demo map shading, equity floor |

Critical = initial severity code 1-3 (assumption F9). The risk index weights (critical ×3, vulnerability boost up to 50%) are stated assumptions (D9).

**Reading the four-policy result.** If Forecast is close to Scheduled, forecast accuracy barely matters for dispatch and we should say so. If Forecast is close to Oracle, a better forecaster could not add much either. A large Forecast-to-Oracle gap means improving the model is worth the time. The same demand columns feed the light units; only the replay changes (speed, no hospital leg, stays until the ambulance arrives).

**Using it**
```python
import pandas as pd
t = pd.read_csv("notebooks/data/processed/forecast_zone_test.csv.gz", parse_dates=["bucket"], dtype={"zone": str})

now = pd.Timestamp("2022-07-20 16:00")                           # the half-hour that just finished
snap = t[t.bucket == now].set_index("zone")
demand_1h = snap.pred_critical_1h * 3 + snap.pred_other_1h       # whatever weighting the optimizer wants
top = snap.sort_values("risk_index_1h", ascending=False).head(10)
```
For a live or replay loop that must not read the answer table, call the model itself (`features_at` builds exactly the features the model was trained on; a test checks this):
```python
from grease.forecast.lgbm import ForecastModel
from grease.forecast import features as F, zones as Z
model = ForecastModel.load("notebooks/data/processed/forecast_models")
rows = F.features_at(history, now, weather=wx, zone_ctx=zone_ctx, zip_to_zone=zip_to_zone)   # `history`: zipcode, bucket, y, y_critical for the last 1345 buckets
pred = model.predict(rows)                                                                   # ZIP level
zone_pred = Z.aggregate_to_zones(pred.assign(zipcode=rows.zipcode, bucket=now), zip_to_zone, ["pred_critical_1h", "pred_other_1h"])
```

**Know before you rely on it**
- **The model is about 4% low on 2022** (predicted / actual = 0.96 in every slice and horizon; 0.94 before the 7-day / 28-day level features). It trained on Jan-Sep 2021 only and 2022 had about 6% more incidents. The shortfall is nearly uniform, so compare policies on *relative* demand and treat absolute counts as roughly 4% low.
- **Accuracy over the historical average is small** (0.0-1.6% lower MAE, 5-7% lower deviance at the ZIP-half-hour level, section 7). Zone totals over an hour or more are far less noisy than the single cells the score table measures.
- **`hist_all_*` and `pred_*` come from a model fitted before 2022; `actual_*` is 2022.** Do not tune policy parameters on the same year you report results for.
- **ZIP-level values are summed into zones**, exact for expected counts. About 0.1% of incidents are in ZIPs that belong to no zone and are dropped.
- Resolution is the 30-minute bucket. Locations are zone-level, so travel times use zone centroids.

---

## Appendix B: the original build plan (`plan_1.0`)

The original design we started from (replay engine → short-horizon forecaster → relocation optimizer → simulator/dashboard). Kept because it defines the evaluation we followed.

**System overview.** Replay engine (streams past incidents) → short-horizon forecaster (demand per zone, next 1-3 hours) → relocation optimizer (where should each unit stage now?) → simulator and dashboard (response times against baselines). Because NYC's dataset is historical, build a replay engine that feeds incidents in timestamp order as if they were arriving live, and say this openly in the pitch; designing the pipeline so a real feed could be swapped in is a plus.

**Unit type first.** Ambulances (or community-paramedic units) answering calls fit the data best, since actual response times can be simulated. Mobile clinics do not respond to calls, so "response time" would not be the metric (coverage or access would be). For a real-time critical-decision story the first is much stronger (this became D1, later revised by D13).

**Components**
1. *Short-horizon demand forecaster:* expected incidents per zone for the next 30-180 minutes. Baseline: historical average by zone, weekday and hour, scaled by a recent-surge multiplier. Model: LightGBM with a Poisson objective on lagged counts, time features, weather and holidays. Always compare against the baseline; if boosting barely wins, say so, since that is an honest finding (it did, see section 7).
2. *Relocation optimizer:* every 15-30 simulated minutes, solve for where idle units should stage (dynamic MEXCLP or demand-weighted travel time) with a maximum relocation distance, a maximum number of moves per unit per shift (fatigue and realism), a minimum coverage in every borough (equity floor), over candidate posts (hospital lots, fire stations or a grid). It should run in seconds. (Implemented as a greedy MEXCLP + Hungarian assignment in `routing_policy.py`, median 65 ms per cycle in the replay, rather than OR-Tools or PuLP.)
3. *Travel-time matrix:* zone-to-post drive times from the road network (OSMnx or OSRM) with time-of-day multipliers. (Partly done: OSRM on demand; the policy still uses a placeholder matrix.)
4. *Discrete-event simulator* (SimPy suggested): replay incidents, dispatch the nearest available unit, model service time (travel, on scene, to hospital, back), handling the busy-unit effect, which is where dynamic deployment actually helps. (Implemented as a minute-by-minute loop in `ems_replay.py`; SimPy is not used.)
5. *Dashboard:* units, predicted-demand heatmap, recommended moves, live metrics. (The React demo, section 12.)

**How to prove it works.** Run the same replayed days under static, scheduled and live-forecast policies (we added an oracle) and compare mean and 90th-percentile response time, percent of calls reached within a target (8 minutes), borough-level equity gaps and the number of relocations, on held-out days including surge days such as heat waves.

**Pitfalls to plan for.** Geography is coarse (ZIP, not exact coordinates); no unit-location data exists, so fleet size and baseline posts are assumptions to state and sweep; estimate service times from the data's timestamp fields rather than guessing (open item: the local incident extract has them, section 14); do not over-claim; reinforcement learning is a stretch goal, an optimizer fed by forecasts is easier to explain, validate and trust.

**Suggested scope.** Clean and aggregate; baseline then LightGBM; simulator with static policy; optimizer and dynamic policy; dashboard and surge-day case study; polish the story. Steps 3 and 4 are the heart of the project, so protect time for them over fancy forecasting.

---

## Appendix C: earlier requirements and pitch material

**Original requirements notes (mobile-clinic version, `Requirements.ipynb`).** Data acquisition from the NYC EMS Incident Dispatch Data (HIPAA means locations are aggregated to ZIP or similar, which suits routing to neighbourhood centroids) plus hospital locations and borough boundaries. Features: incident time, call type (overdose, cardiac arrest, respiratory distress), the aggregated location, and the call-to-on-scene delta as the target for slow-response areas. A three-step pipeline: (1) spatial analysis of "healthcare deserts" (0.5-1 mile buffers around hospitals, high-volume severe-call areas outside them); (2) time-series forecasting of severe incidents per area per 4-hour shift (XGBoost or random forest); (3) dynamic routing as a facility-location or vehicle-routing problem. **"Dream features":** a fleet of mobile health professionals on bicycles or motorcycles; a text-classification model that spots a severe emergency from the call; dispatch the nearest responder; the AHA 7-10% per minute rule of thumb as the reason a fast responder matters. This section is where the light-unit idea (D13) began. What changed: units are first-response units rather than clinic vans; forecasting is per ZIP zone per 30 minutes rather than per 4-hour shift; the target is demand, not the response delta; the call-text classifier is not built (no public call text).

**Simulation change list (`simulation-updates.ipynb`), all since done:** change the AI to the system we actually use; use actual 2022 data; put a real NYC street map behind the simulation (it showed a grey "API key required" screen); rename to Project Dispatch; include hospital names; use road routes with simulated traffic (now OSRM routes; traffic is a flat multiplier).

**Earlier pitch notebooks (`V1 Pitch.ipynb`, `Project Grease Pitch Deck.ipynb`, `Reponse_Times.ipynb`).** "Healthcare deserts" pitch: stationed mode (anchor in low-coverage hotspots) and patrolling mode (route through high-risk corridors); a coverage-radius map in which a hospital's radius shrinks where FDNY monthly response times are slow; a ZIP-code choropleth of response times (folium, hard-coded demo hospitals); "Tier Zero" stabilisation items: naloxone for overdoses, EpiPens and nebulizers for anaphylaxis and asthma, AEDs for cardiac arrest, de-escalation for behavioural crises (a useful starting list for the light-unit kit, to be confirmed with clinicians); feasibility and scalability claims (OpenStreetMap-based logic reusable in other cities); an ask for a pilot partnership to ingest live 911 data.

**Pitch deck notebook (`Grease_DISPATCH_Pitch_Deck (1).ipynb`).** Sections: the problem (13-minute response times, a paramedic lieutenant's quote), target demographic (NYC, 490,000 vehicles, 8.8 million people), the solution (priority locations from EMS data, predictive routing, weather-related incident prediction), technology (LinuxONE), demo, competitive advantage, value, the team (Jacob Barton, Dominic Easton, Harry Rowlands, Darragh Coleman, Frederic Tatham, Napat Chollabud) and the future. Several claims need reconciling with the evidence (section 14).
