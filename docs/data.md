# Data: one notebook, one model table

Everything is open data, fetched and cleaned in **one notebook: `notebooks/DarraghC.ipynb`**. Run it top to bottom. It writes everything below
into `notebooks/data/`. (The old per-topic cleaning notebooks 01-07 were merged into it and removed.)

```
APIs / repo CSVs --> notebooks/DarraghC.ipynb --> notebooks/data/processed/model_table.csv.gz   (ONE table the model reads)
                                              --> notebooks/data/processed/incidents.csv.gz     (simulator: one row per incident)
                                              --> notebooks/data/interim/*.csv                  (small reference tables, tracked in git)
```
`notebooks/data/raw/` (downloads) and `notebooks/data/processed/` (large tables) are git-ignored and rebuilt by the notebook.
`scripts/fetch_data.py` holds the download code; the notebook calls it (do not duplicate it).

## Datasets

| Dataset | Source | Notebook section | Output | Role |
|---|---|---|---|---|
| **EMS incidents** (call time, severity, assignment / on-scene / hospital / close timestamps, borough, ZIP) | NYC Open Data `76xm-jjuj` (API, monthly files in `raw/ems_parts/`) | 11 | `incidents.csv.gz`, `service_times.csv` | Demand history, the replay stream, real response times and service-time distributions |
| **Weather** (hourly temperature, feels-like, humidity, rain, wind, cloud) | Open-Meteo archive API, requested in **New York local time** | 1-5 | `notebooks/data/weather_calendar_30min.csv.gz` (tracked) | Forecaster features; heat / rain days for surge case studies |
| **Calendar** (holidays, day before / after) | pandas US federal holidays | 4 | same file | Forecaster features |
| **Zones** (MODZCTA) + ZIP crosswalk | NYC Open Data `pri4-ifjk` | 7 | `interim/zones.csv`, `zip_to_zone.csv` | Spatial unit, centroids for travel times |
| **Hospitals, firehouses** | NY State Health `vn5v-hh5r`, FDNY `hc8x-tcnd` | 8 | `interim/posts.csv` | Transport destinations; candidate staging posts |
| **FDNY monthly response times** | NYC Open Data `j34j-vqvt` | 9 | `interim/fdny_monthly.csv` | Benchmark for the simulator's baseline |
| **Asthma rates** (NTA, 2017-19), **health survey** (citywide) | repo CSVs in `notebooks/data/` | 10 | `interim/asthma_borough.csv`, `asthma_nta.csv`, `health_survey_citywide.csv` | **Health context combined into the model table** (borough asthma rate), equity narrative |
| **Census ACS 5-year** (population, income, no-vehicle, poverty) | Census API, **needs a free key**: https://api.census.gov/data/key_signup.html then `CENSUS_API_KEY=<key>` | 10 | `interim/census_zone.csv` | Population weighting, fairness audit. Skipped without a key |
| Road network (OpenStreetMap) | OSMnx / OSRM | - | not fetched yet | Travel-time matrix (or haversine x detour factor as the fast option) |

## The model table (`processed/model_table.csv.gz`)
One row per **zone x 30-minute `bucket`** for every month loaded (zero-filled so quiet periods count). Columns:
- target: `n_incidents`, `n_low_mid` (mid / low acuity)
- weather + calendar: `temp_c, feels_like_c, humidity_pct, precip_mm, is_raining, rain_3h_mm, rain_24h_mm, wind_kmh, cloud_pct, temp_max_24h_c, is_hot, is_heat_stress, is_cold, is_holiday, is_day_before_holiday, is_day_after_holiday`
- zone / health context: `borough`, `asthma_rate_per_10k`, Census columns when available; time: `hour, dow, month, year`
- `eval_is_heat_day`, `eval_is_heavy_rain_day`: **slice results on heat days only, never model inputs** (they use the whole day's weather)

Current build: 177 zones x 37,872 buckets = 6.7M rows (26 months: 2019 Jan-Sep, 2021 Jan-Sep, 2022 Jan-Aug; see below), 109 MB.

### Using it in the forecaster (`HarryRowlands.ipynb`)
See section 13 of the notebook. In short: `pd.read_csv("data/processed/model_table.csv.gz", parse_dates=["bucket"], dtype={"zone": str})`,
rename `zone -> zipcode`, `n_incidents -> y`, then add lags. **Compute lags only inside contiguous runs of buckets**: the table skips months we do not have.
For a model predicting h buckets ahead use `lag_weather(table, h)` so it only sees weather available at forecast time. `src/grease/data/features.py`
does the same weather / calendar join for any other table.

## Decisions and assumptions
- **Zone = MODZCTA.** 98.9% of incidents map to a zone, so we use it (community district is the fallback).
- **EMS months present:** 2019 Jan-Sep, 2021 Jan-Sep, 2022 Jan-Aug (26 months). Set `FETCH_MISSING_EMS = True` in section 11 to download the rest (about 40 s per month).
  2020 is excluded on purpose (COVID distorts demand). The table only contains months we have, so there are no fake zero gaps.
- **Weather** comes from the API in local time; the CSV in the repo (UTC, with Oct 16 and Oct 20-25 2022 empty) is only used for the original exploration cells.
- **Acuity bands** (critical 1-3, mid 4-6, low 7-8) assume severity code 1 is most critical. **Verify against the NYC data dictionary.**
- **No ambulance location or fleet data is public.** Fleet size and home posts are assumptions; we sweep a few sizes. Firehouses and hospitals stand in for staging posts.
- **Locations are area-level only.** Travel uses zone centroids, so response times are approximate.
- **FDNY "Medical Emergencies" covers all first responders**, so it is only a rough benchmark for ambulance response.
- Standby and transfer incidents are dropped (not new demand). Response time uses only rows flagged valid and between 1 s and 3 h.
- Asthma rates are borough-level (no NTA to zone crosswalk yet); the health survey is citywide only.
- The simulator shows what would happen **under these assumptions**, not proven real-world gains.

## Open items
1. Free Census API key, then rerun section 10 (adds zone-level population / income / vehicle access).
2. Fetch the missing EMS months (`FETCH_MISSING_EMS = True`); pre-2019 years if the forecaster should train on more history.
3. Road network for travel times, or accept haversine x detour factor.
4. Confirm the severity-code meaning and the target acuity band.
