# Data: where it comes from and how it is combined

Two notebooks, each with one job. Everything is open data. Outputs live in `notebooks/data/`.

```
notebooks/DarraghC.ipynb   all NON-EMS data: fetch + clean  -->  notebooks/data/weather_calendar_30min.csv.gz
                                                                 notebooks/data/interim/*.csv   (small, tracked in git)
notebooks/HarryRowlands.ipynb   EMS incidents (API -> notebooks/ems_raw/), panel + lag features,
                                then links the tables above into one big table `full`; then the model
```
`notebooks/data/raw/` (downloads) and `notebooks/data/processed/` are git-ignored. Download code lives in `scripts/fetch_data.py` (used by `DarraghC.ipynb`).
Run `DarraghC.ipynb` first (about 10 seconds once the raw files are cached), then `HarryRowlands.ipynb`.

## Datasets

| Dataset | Source | Where it is cleaned | Output | Role |
|---|---|---|---|---|
| **EMS incidents** (time, ZIP, severity, valid-response flag) | NYC Open Data `76xm-jjuj` via API, 4 columns, 2016-2022 | `HarryRowlands.ipynb` | `notebooks/ems_raw/<year>.csv` | Demand history for the forecaster |
| **Weather** (hourly temperature, feels-like, humidity, rain, wind, cloud) | Open-Meteo archive API, requested in **New York local time** | `DarraghC.ipynb` 1-5 | `weather_calendar_30min.csv.gz` (tracked) | Forecaster features; heat / rain days for surge case studies |
| **Calendar** (holidays, day before / after) | pandas US federal holidays | `DarraghC.ipynb` 4 | same file | Forecaster features |
| **Zones** (MODZCTA) + ZIP crosswalk | NYC Open Data `pri4-ifjk` | `DarraghC.ipynb` 7 | `interim/zones.csv`, `zip_to_zone.csv` | Spatial unit, centroids for travel times |
| **Hospitals, firehouses** | NY State Health `vn5v-hh5r`, FDNY `hc8x-tcnd` | `DarraghC.ipynb` 8 | `interim/posts.csv` | Transport destinations; candidate staging posts |
| **FDNY monthly response times** | NYC Open Data `j34j-vqvt` | `DarraghC.ipynb` 9 | `interim/fdny_monthly.csv` | Benchmark for the simulator's baseline |
| **Asthma rates** (NTA, 2017-19), **health survey** (citywide) | repo CSVs in `notebooks/data/` | `DarraghC.ipynb` 10 | `interim/asthma_*.csv`, `health_survey_citywide.csv` | Health context (borough asthma rate), equity narrative |
| **Census ACS 5-year** (population, income, no-vehicle, poverty) | Census API, **needs a free key** https://api.census.gov/data/key_signup.html (activate it from the email), put it in the git-ignored `.env` as `CENSUS_API_KEY=...` (see `.env.example`) | `DarraghC.ipynb` 10 | `interim/census_zone.csv` | Population weighting, fairness audit. Skipped without a key |
| **Zone context** (all of the zone-level sources above in one table) | built from the rows above | `DarraghC.ipynb` 10 | `interim/zone_context.csv` | One join gives borough, location, population, asthma rate, hospital access |
| Road network (OpenStreetMap) | OSMnx / OSRM | - | not fetched yet | Travel-time matrix (or haversine x detour factor as the fast option) |

## The big table (`full`, built in `HarryRowlands.ipynb` after the lag features)
`panel` (zipcode x 30-minute `bucket`, `y` = incidents, lags, time features) plus:
- weather + calendar on `bucket`: `temp_c, feels_like_c, humidity_pct, precip_mm, is_raining, rain_3h_mm, rain_24h_mm, wind_kmh, cloud_pct, temp_max_24h_c, is_hot, is_heat_stress, is_cold, is_holiday, is_day_before_holiday, is_day_after_holiday`
- zone context through `zipcode -> zone`: `zone, borough, lon, lat, pop_est, asthma_rate_per_10k, dist_nearest_hospital_km, n_hospitals, n_firehouses` (+ Census fields with a key)
- `eval_is_heat_day`, `eval_is_heavy_rain_day`: **slice results on heat days only, never model inputs** (they use the whole day's weather)

`panel` and the model cells are **not** modified: the model takes every column of `panel` as a feature, so merging into it would silently change the model.
Memory: about 4 GB for two years of data, so roughly 14 GB for all of 2016-2022; use a large machine or trim years.
For a model that predicts `h` buckets ahead, weather must be shifted back `h` buckets so it only uses what is known at forecast time (`src/grease/data/features.py` does this).

## Decisions and assumptions
- **Zone = MODZCTA.** 99.9% of EMS incidents with a ZIP map to a zone (a few rare ZIPs such as 10121 and 10000 do not).
- **Weather** comes from the API in local time. The CSV in the repo is UTC and blank for Oct 16 and Oct 20-25 2022, so it is used only for the original exploration cells.
- **Borough** of a zone is the community district containing its centroid, so zone context does not depend on EMS data.
- **No ambulance location or fleet data is public.** Fleet size and home posts are assumptions; we sweep a few sizes. Firehouses and hospitals stand in for staging posts.
- **Locations are area-level only.** Travel uses zone centroids, so response times are approximate.
- **FDNY "Medical Emergencies" covers all first responders**, so it is only a rough benchmark for ambulance response.
- Asthma rates are borough-level (no neighbourhood-to-zone crosswalk yet); the health survey is citywide only.
- The simulator shows what would happen **under these assumptions**, not proven real-world gains.

## Open items
1. **EMS download is 4 columns only** (time, ZIP, initial severity, response-valid flag). The simulator also needs stage durations (on scene, transport, job) and real response times, so the fetch will need extra columns (`first_on_scene_datetime`, `first_to_hosp_datetime`, `first_hosp_arrival_datetime`, `incident_close_datetime`, `incident_response_seconds_qy`, ...) before we build it.
2. Census key: added to `.env` but **Census currently rejects it** (probably not activated yet); once it works, rerun `DarraghC.ipynb` (section 6 fetches, section 10 joins it into `zone_context.csv`).
3. Road network for travel times, or accept haversine x detour factor.
4. Confirm the severity-code meaning and the target acuity band (assumed 1 = most critical).
