# Data inventory and how we use it

Everything is open data. Raw files are **not** in git (`data/raw/` is ignored); regenerate with
`python scripts/fetch_data.py`. Cleaning notebooks are in `notebooks/cleaning/` and run in numeric order.

## Pipeline

```
scripts/fetch_data.py  ->  data/raw/          (APIs, untouched)
notebooks/cleaning/01-06 -> data/interim/     (typed, validated, one topic per file)
notebooks/cleaning/07  ->  data/processed/    (modelling tables)
```

Run order: `fetch_data.py` -> 02, 03, 04, 05, 06 (any order) -> 01 -> 07. (07 needs 01, 02, 03.)

## Datasets

| # | Dataset | Source | How we get it | Raw file | Cleaned by | Role in the project |
|---|---------|--------|---------------|----------|-----------|---------------------|
| 1 | **EMS incident dispatch** (call time, severity, assignment/on-scene/hospital/close timestamps, borough, ZIP, community district) | NYC Open Data `76xm-jjuj` (Socrata API) | `fetch_data.py ems`; years 2019, 2021, 2022 (~1.5M rows/year) | `ems_incidents_<year>.csv.gz` | NB 01 | **Core.** Demand history for the forecaster; the stream the replay engine feeds; empirical service times and real response times for the simulator and its validation |
| 2 | **Weather** (hourly temp, precip, wind, cloud) | Open-Meteo archive (already in repo) | `notebooks/data/NYC_Weather_2016_2022.csv` | same | NB 02 | Forecaster feature; heat/rain days define the surge-day case studies. **Ends 2022-10-25** |
| 3 | **Calendar** (federal holidays, weekday, heat day >=32C, heavy rain >=12.7mm) | derived from #2 + pandas holiday calendar | built in NB 02 | - | NB 02 | Forecaster features |
| 4 | **Zone boundaries** (MODZCTA, community districts) | NYC Open Data `pri4-ifjk`, `5crt-au7u` (Socrata GeoJSON) | `fetch_data.py small` | `modzcta_boundaries.geojson`, `community_district_boundaries.geojson` | NB 03 | Spatial unit, centroids for travel times, ZIP crosswalk |
| 5 | **Hospitals** | NY State Health `vn5v-hh5r` (Socrata, filtered to NYC counties, `description = Hospital`) | `fetch_data.py small` | `nys_hospitals_nyc.csv` | NB 04 | Transport destinations in the simulator; candidate staging posts |
| 6 | **FDNY firehouses** | NYC Open Data `hc8x-tcnd` | `fetch_data.py small` | `fdny_firehouses.csv` | NB 04 | Candidate staging posts |
| 7 | **FDNY monthly response times** | NYC Open Data `j34j-vqvt` | `fetch_data.py small` | `fdny_monthly_response_times.csv` | NB 05 | Benchmark: sanity-check simulated baseline response times |
| 8 | **Census ACS 5-year** (population, income, no-vehicle households, poverty) by ZCTA | Census API | `CENSUS_API_KEY=<key> fetch_data.py census` (**free key needed**: https://api.census.gov/data/key_signup.html) | `census_acs5_2022_zcta.csv` | NB 06 | Population-weighted demand; equity audit (is slow response concentrated in low-income / low-vehicle areas?) |
| 9 | **Asthma rates by NTA** (2017-19) | already in repo | `notebooks/data/NYC_asthma_nta_rates_2017_2019.csv` | same | NB 06 | Public-health context, borough-level equity narrative. NTA-level; no ZIP crosswalk yet |
| 10 | **Community health survey** | already in repo | `notebooks/data/NYC_community_health_survey.csv` | same | NB 06 | Citywide context only (no geography) |
| 11 | **Road network** (OpenStreetMap) | OSMnx / OSRM | **not fetched yet** - done in the travel-matrix step (`pip install osmnx`) | - | - | Zone-centroid <-> post drive times |
| 12 | **Emergency call text** | - | **not available** (no public data) | - | - | Optional stretch; would need clearly-labelled synthetic data |

## Cleaned outputs

`data/interim/` - `ems_incidents_clean.csv.gz`, `weather_hourly.csv`, `calendar_daily.csv`, `zones_modzcta.csv`,
`zip_to_modzcta.csv`, `zones_cd.csv`, `hospitals.csv`, `firehouses.csv`, `candidate_posts.csv`,
`fdny_monthly_clean.csv`, `census_zone.csv` (needs key), `asthma_nta.csv`, `asthma_borough.csv`, `health_survey_citywide.csv`.

`data/processed/` - `demand_zone_30min.csv.gz` (zone x 30-min counts + weather + calendar, zero-filled),
`incidents_sim.csv.gz` (one row per incident with zone, acuity, durations), `service_times.csv`.

## How the data feeds the plan (see `claudeplan.md`, `docs/plan.md`)

| Stage | Uses |
|-------|------|
| Replay engine | `incidents_sim.csv.gz` sorted by timestamp, played as if live |
| Forecaster (baseline -> LightGBM Poisson) | `demand_zone_30min.csv.gz`: lags, hour/weekday/month, weather, holidays, heat/rain flags |
| Travel-time matrix | `zones_modzcta.csv` centroids + `candidate_posts.csv` + OSM road network |
| Simulator | incidents, `service_times.csv` (on-scene, transport, job duration by acuity), hospitals for transport legs |
| Optimizer | forecast demand per zone, candidate posts, travel matrix, borough equity floor |
| Evaluation | simulated vs. real `response_s` and FDNY monthly averages; borough gaps; surge days from `calendar_daily.csv` |
| Fairness audit | Census + asthma + survey joined by zone / borough |

## Decisions and assumptions (state these in the pitch)

- **Zone = MODZCTA** (ZIP-based). Notebook 07 reports how many incidents map to a zone; if < ~95%, switch to community district.
- **Years 2019, 2021, 2022.** 2020 is skipped (COVID distorts demand). Weather ends 2022-10-25, so 2022 modelling should stop there. Suggested split: train 2019 + 2021, test on 2022 (including heat days).
- **Acuity bands** (critical 1-3, mid 4-6, low 7-8) assume severity code 1 = most critical. **Verify against the NYC data dictionary.** The plan targets low/mid-acuity calls.
- **No ambulance location or fleet data is public.** Fleet size and home posts are assumptions; we sweep several sizes. Firehouses/hospitals stand in for staging posts.
- **Locations are area-level only.** Travel uses zone centroids, so response times are approximate.
- **FDNY monthly "Medical Emergencies" includes all first responders**, so it is only a rough benchmark for ambulance response.
- **Standby and transfer incidents are dropped** (not new demand). Response times only use rows flagged valid and between 1 s and 3 h.
- The simulator shows what would happen **under these assumptions**, not proven real-world gains.

## Open items

1. Get a free Census API key and run the census fetch (someone on the team).
2. Download the OSM road network for the travel-time matrix (heavy; do once, cache in `data/interim/`).
3. Check the severity-code meaning and decide the target acuity band.
4. Decide ambulances vs mobile clinics (README still says clinics; the plan recommends ambulances).
5. Optional: NTA -> zone crosswalk so asthma rates can join at zone level.
