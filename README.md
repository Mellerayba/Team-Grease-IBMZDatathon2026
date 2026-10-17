# Team Grease_IBM Z Datathon 2026 Project: Project Dispatch

IBM Z Datathon 2026 · event code `SSF2026` · Team Grease (Jacob Barton, Dominic Easton, Harry Rowlands, Darragh Coleman, Frederic Tatham, Napat Chollabud)

**Project Dispatch decides where emergency-response units should wait in New York City before calls arrive, and which unit to send, using a demand forecast trained and run on an IBM LinuxONE VM.** Every number below is *modelled* from real 2022 incidents under stated assumptions, not measured in the field.

Everything we decided, measured and learned is in one file: [`docs/decisions.md`](docs/decisions.md). Evidence screenshots and VM output: [`docs/screenshot_examples.md`](docs/screenshot_examples.md).

## The problem
In New York City the first minutes of a critical emergency decide whether someone survives, but ambulance response times are at record highs. Ambulances are large, expensive, staffed by a shrinking workforce and stuck in the same traffic as everyone else, so the city cannot simply add more of them quickly.

**Response times are getting worse.**
- NYC response times for life-threatening medical emergencies rose from 9:34 in 2021 to 13:09 in the latest fiscal year reported ([EMS1](https://www.ems1.com/response-times/nyc-ambulance-response-times-hit-record-high-amid-ems-staffing-shortage)).
- FDNY's EMS chief told the City Council the department loses about 10 in-service ambulances a day because of staffing shortages (same source).

**The burden is uneven across boroughs.**
- The NYC Independent Budget Office found the share of advanced life support calls answered within 10 minutes fell in every borough between 2014 and 2024. In 2024 it was about 82% in Staten Island and about 75% in the Bronx. For basic life support, the Bronx was about 48% against about 68% in Staten Island ([IBO analysis summary](https://isoclivecivic.substack.com/p/exploring-an-independent-perspective)).

**Minutes decide survival in cardiac arrest.**
- The American Heart Association rule of thumb is that survival chances fall by 7-10% for each minute without CPR or defibrillation ([CARES registry report](https://mycares.net/sitepages/uploads/2024/2023_flipbook/inc/html/34.html)). Newer studies refine this, showing much better survival with defibrillation within about four minutes ([FireRescue1](https://www.firerescue1.com/fire-products/aeds/articles/ny-countys-cardiac-emergency-response-time-subpar-ISf1uNeBLRnlC7jS/)).

**Recognising the emergency on the phone is part of the problem.**
- Dispatchers fail to identify roughly 25% of out-of-hospital cardiac arrests ([study abstract](https://swov.nl/en/publicatie/machine-learning-supportive-tool-recognize-cardiac-arrest-emergency-calls)). In a Copenhagen study of 108,607 calls, a machine learning model recognised cardiac arrest more often and faster than dispatchers, but with more false alarms (same source).

**Why we use decision support, not automation.**
- A randomised trial across 169,049 calls found that alerting dispatchers did not significantly improve their recognition of cardiac arrest, even though the model outperformed humans on its own ([trial summary](https://busqueda.bvsalud.org/portal/resource/es/mdl-33404620)). We keep a human dispatcher in the loop and test our system by simulation.

**Data limitation.**
- NYC's public EMS dispatch data aggregates incident locations to protect patient privacy, so our analysis works at area level, not street level ([data.gov](https://catalog.data.gov/dataset/ems-incident-dispatch-data)).

More sources and the news reporting behind the problem statement: `docs/decisions.md` section 3.

## The solution
**Project Dispatch** adds a second, much cheaper layer to the emergency response: a fleet of **lightweight electric first-response units** (for example e-bikes) that can move through traffic and carry enough equipment to stabilise a critically ill person until an ambulance arrives. Ambulances still transport patients; the light units buy the first minutes. AI decides where those units wait and which one goes. The question we answer for the city: **how many units should NYC buy, and where should they stand?**

## What the project actually does
1. **Collects and cleans open data** from the NYC Open Data API (EMS incidents 2016-2022, zones, hospitals and firehouses, FDNY response-time reports), Open-Meteo (hourly weather in New York time), the Census API (population, income, vehicle access, poverty), health data (asthma rates, community health survey) and US holidays; combines them into one modelling table keyed by ZIP zone and 30-minute bucket. (`notebooks/DarraghC.ipynb`, `scripts/fetch_data.py`, `notebooks/HarryRowlands.ipynb`)
2. **Forecasts demand.** Eight LightGBM Poisson models give the expected number of critical and other calls in each ZIP over the next 30 minutes, 1, 2 and 3 hours, from recent counts, time, weather, holidays and zone context. They are trained on Jan-Sep 2021, validated on Q4 2021 and tested on all of 2022, scored against three baselines (historical mean, surge-scaled mean, same window last week) overall and on heat days, heavy-rain days, the busiest ZIPs and each borough, and checked for calibration. (`src/grease/forecast/`)
3. **Scores neighbourhood risk.** A 0-100 risk index = need-weighted expected demand (critical calls count 3×) × a vulnerability factor (poverty, households without a vehicle, asthma rate), with the weights stated as assumptions. (`src/grease/risk.py`)
4. **Positions units.** Every 30 minutes a policy picks where idle units should wait to cover the most expected, vulnerable demand within 8 minutes (an expected-coverage objective with diminishing returns per extra unit), assigns units to those spots by drive time, keeps at least one unit in every borough, caps relocations to avoid thrashing, holds still if the forecast is stale, can correct the forecast with what actually arrived, and explains every move in plain language. (`notebooks/routing_policy.py`)
5. **Dispatches.** A critical call gets the fastest available unit; other calls may take a slightly slower unit if that protects coverage.
6. **Replays real incidents** minute by minute (call, dispatch, scene time, hospital trip, return) under four policies on identical calls, fleet and travel times: static layout, scheduled (historical demand, no ML), forecast (our model) and oracle (perfect foresight, the ceiling), plus a fleet-size sensitivity run. (`notebooks/ems_replay.py`)
7. **Routes on real streets.** Moves the policy recommends get real street routes and drive times from the OSRM routing service (with a bike profile for the light units), cached on disk, with a flagged fallback. (`notebooks/street_routing.py`)
8. **Shows it in a demo.** A React and Leaflet app replays the real run: live units, where they head after each decision, incoming calls, and the ZIP zones shaded by the model's output, with static and model-guided fleets side by side and running response metrics. It runs on the LinuxONE VM. (`pulsenyc-source/pulsenyc`, data from `scripts/export_demo.py`)
9. **Tests the IBM Z AI accelerator honestly.** On the VM we verified the accelerator is exposed, drove it through Snap ML, benchmarked it fairly against CPU LightGBM and found it slower and, for larger trees, slightly less accurate, so scoring stays on the CPU. (`scripts/vm_check.py`, `scripts/accel_check.py`, `scripts/accel_bench.py`)
10. **Briefs the dispatcher (generative AI, optional).** A notebook cell asks IBM Granite to summarise a cycle's moves in plain language, guarded so it cannot mention a ZIP that was not in the moves; implemented, not yet run on the VM (it needs about 8 GB free).

## Results so far (modelled)
Real 2022-07-19 heat-wave day, 4,694 incidents, 330 units, demo export:

| Policy | Critical calls within 8 min | Critical 90th-percentile response |
|---|---|---|
| Static layout | 82.3% | 10.5 min |
| **Model-guided (our forecast)** | **91.2%** | **7.8 min** |
| Oracle (perfect foresight) | 93.4% | 7.4 min |

Averaged over two real days (heat wave and an October day) in the model notebook: static 76.4%, scheduled 84.4%, forecast 84.9%, oracle 86.2%. Be clear about what this means: most of the gain comes from positioning by expected demand at all; our ML forecast adds only a little over a schedule built from history, and its accuracy gain over the historical average is small (5-7% lower Poisson deviance, 0-1.6% lower MAE). Weather adds almost nothing to the forecast. The dynamic policy does worse than the static layout in Staten Island, which we have flagged and not fixed. Full numbers, caveats and the equity analysis: `docs/decisions.md` sections 7 and 10.

## Status (honest)
**Built and tested (33 automated tests pass):** the data pipeline, forecaster, risk index, positioning and dispatch policy, replay with four policies, street routing, the demo, the accelerator benchmark. **Not built yet:** the light-unit replay model (speed, no hospital trip, stays until the ambulance arrives) and the fleet-size sweep for it, so today's replay uses a generic ambulance-like fleet; travel times inside the policy's decisions are a placeholder (the real street routes are only displayed); the Granite briefing on the VM. All response times are modelled from an assumed fleet size, scene and hospital times, traffic and travel times; only the difference between policies is meaningful. No outcome data exists, so any "lives saved" figure would be an illustration. See `docs/decisions.md` section 14.

## How it uses IBM Z
- The notebooks (data build, training, replay) and the demo run on an IBM LinuxONE VM (s390x, z17 generation, 3 CPUs, 8.4 GB RAM), with memory-lean code (about 1.9 GB per year of data) because of that limit.
- The on-chip AI accelerator (`nnpa`, zDNN) was tested through Snap ML and benchmarked against CPU LightGBM; the result and the reasoning are documented rather than assumed.
- Models are saved as portable LightGBM text files (pickles are not portable between x86 and s390x).
- VM output and the screenshots still to capture: `docs/screenshot_examples.md`.

## Run it
```bash
# Tests
PYTHONPATH=src python3 -m pytest tests

# Data and model (on the VM or locally): run in order
#   notebooks/DarraghC.ipynb      fetches and cleans the non-EMS data (needs a free Census key in .env, see .env.example)
#   notebooks/HarryRowlands.ipynb fetches EMS data, trains and scores the forecaster, runs the replay, writes the hand-off table

# The demo (works straight after a clone: the replay data is committed)
cd pulsenyc-source/pulsenyc
rm -rf node_modules && npm install      # the committed node_modules is broken
npm run dev                             # open the URL Vite prints, press RUN DEMO

# Regenerate the demo data (not on the VM: shapely has no s390x wheel)
python scripts/export_demo.py --street-routes

# On the VM: platform and accelerator checks
python3 scripts/vm_check.py --years 2
python3 scripts/accel_check.py
python3 scripts/accel_bench.py --quick
```
To view the demo from the VM, run `npm run dev` there and tunnel the port: `ssh -L 5173:localhost:5173 <user>@<vm-host>`, then open `http://localhost:5173`.

## Repository map
| Path | What it is |
|---|---|
| `docs/decisions.md` | The single record: problem evidence, decisions, data, model results, policy and replay, demo, platform notes, assumptions, open items |
| `docs/screenshot_examples.md`, `docs/screenshots/` | Evidence: demo screenshots, test and export runs, verbatim VM output, and the list of VM screenshots still to take |
| `notebooks/DarraghC.ipynb`, `HarryRowlands.ipynb` | Data pipeline, forecaster, replay, hand-off export (the two notebooks that matter) |
| `notebooks/routing_policy.py`, `ems_replay.py`, `street_routing.py`, `demo_policy.py` | Positioning and dispatch policy, replay, OSRM routing, synthetic sanity check |
| `src/grease/forecast/`, `risk.py`, `data/` | Forecaster (features, LightGBM, baselines, evaluation, zone aggregation, export), risk index, data helpers. `sim/`, `optimize/`, `travel/`, `dashboard/` are early skeletons superseded by the notebook-side code |
| `scripts/` | `fetch_data.py`, `export_demo.py`, `vm_check.py`, `accel_check.py`, `accel_bench.py` |
| `pulsenyc-source/pulsenyc/` | The React demo (`src/replay.ts`, `DemoMap.tsx`, `main.tsx`, `StreetMap.tsx`; data in `public/data/`) |
| `notebooks/data/interim/` | Small cleaned reference tables (zones, hospitals, Census, health, FDNY) |
| `tests/` | 33 tests: features, forecast, routing policy and replay, street routing |
| `notebooks/*Pitch*.ipynb`, `Reponse_Times.ipynb`, `Mapping.ipynb`, `docs/Grease DISPATCH Pitch Deck.pptx` | Earlier pitch and exploration material (some claims superseded, see `docs/decisions.md` section 14) |

## Target audience
1. **City emergency-management and budget decision-makers:** Use the fleet-size curve to decide how many units to buy and where to station them.
2. **Emergency medical dispatchers and operations leaders:** Use the recommendations to decide which unit to send and where to position idle units.
3. **Public health and policy planners:** Use the fairness analysis to find neighbourhoods with the slowest response and target staffing and funding.
4. **Hospital and community health managers:** Plan outreach and support around predicted demand spikes.
5. **Residents of underserved areas:** The end beneficiaries, who gain faster care and fairer allocation of resources.
6. **Researchers and civic technologists:** Reuse the open-source approach with other cities' open data.

## Datasets
All data is open, properly licensed and contains no personal information.
- [x] **NYC EMS incident dispatch data** (NYC Open Data `76xm-jjuj`): incident time, ZIP, severity; 2017-2022 downloaded complete (2016 truncated, not used).
- [x] **FDNY monthly response times** (`j34j-vqvt`): benchmark.
- [x] **Hospital and firehouse locations** (NY State Health `vn5v-hh5r`, FDNY `hc8x-tcnd`): hospital destinations and candidate posts.
- [x] **Zone boundaries and ZIP crosswalk** (MODZCTA, `pri4-ifjk`): the spatial unit and the demo's polygons.
- [x] **Census ACS 5-year** (population, income, vehicle access, poverty): vulnerability and population.
- [x] **Weather** (Open-Meteo archive API, New York local time) and **calendar** (US federal holidays): forecaster features.
- [x] **Health data** (asthma rates by neighbourhood, community health survey): vulnerability and equity context.
- [x] **OpenStreetMap** (via the OSRM routing service and map tiles): street routes and the map.
- [ ] **Emergency call text:** no public source; not used.

## Open-source libraries and tools

| Component | Open-source library | License | Usage in project |
| :--- | :--- | :--- | :--- |
| **Data processing** | [pandas](https://github.com/pandas-dev/pandas) | BSD 3-Clause | Loading, cleaning and aggregating incident, weather and zone data |
| **Numerical computing** | [NumPy](https://github.com/numpy/numpy) | BSD 3-Clause | Array calculations for features, coverage maths and travel times |
| **Forecasting** | [LightGBM](https://github.com/microsoft/LightGBM) | MIT | The 8 Poisson demand models |
| **Routing assignment** | [SciPy](https://github.com/scipy/scipy) | BSD 3-Clause | Hungarian assignment of units to staging zones |
| **Street routing** | [OSRM](https://github.com/Project-OSRM/osrm-backend) (public demo server) and [OpenStreetMap](https://www.openstreetmap.org) data | BSD 2-Clause; ODbL data | Real street routes and drive times for the policy's moves |
| **HTTP and data APIs** | [requests](https://github.com/psf/requests), [sodapy](https://github.com/xmunoz/sodapy), [python-dotenv](https://github.com/theskumar/python-dotenv) | Apache 2.0; MIT; BSD 3-Clause | Calling Open-Meteo, Census, OSRM and the NYC Open Data (Socrata) API; keeping keys out of git |
| **Geometry** | [Shapely](https://github.com/shapely/shapely) | BSD 3-Clause | Simplifying zone polygons for the demo (not available on the s390x VM, so used off-VM) |
| **Accelerator test** | [Snap ML](https://pypi.org/project/snapml/) (IBM), [Nyoka](https://github.com/SoftwareAG/nyoka) | IBM package licence; Apache 2.0 | Testing inference on the IBM Z AI accelerator; PMML export attempt |
| **Testing** | [pytest](https://github.com/pytest-dev/pytest) | MIT | 33 tests |
| **Web interface** | [React](https://github.com/facebook/react) and React DOM | MIT | The demo |
| **Web mapping** | [Leaflet](https://github.com/Leaflet/Leaflet) and [React-Leaflet](https://github.com/PaulLeCam/react-leaflet) | BSD 2-Clause; Hippocratic-2.1 | Interactive map, zone polygons, units, routes |
| **Frontend language and build** | [TypeScript](https://github.com/microsoft/TypeScript), [Vite](https://github.com/vitejs/vite), [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react) | Apache 2.0; MIT | Type-checked build and dev server |
| **Frontend styling** | [Tailwind CSS](https://github.com/tailwindlabs/tailwindcss), [PostCSS](https://github.com/postcss/postcss), [Autoprefixer](https://github.com/postcss/autoprefixer) | MIT | Styling the demo |
| **Icons** | [Lucide React](https://github.com/lucide-icons/lucide) | ISC | Icon set (dependency of the demo) |
| **Exploration notebooks** | [folium](https://github.com/python-visualization/folium), [OSMnx](https://github.com/gboeing/osmnx) | MIT | Earlier response-time maps and the first routing proof of concept (OSMnx cannot be installed on the VM) |
| **Generative AI (optional)** | [IBM Granite](https://huggingface.co/ibm-granite) models, [llama.cpp](https://github.com/ggml-org/llama.cpp) | Apache 2.0; MIT | Plain-language dispatcher briefing (implemented, not yet run on the VM) |

`requirements.txt` also lists OR-Tools, SimPy, Streamlit, pydeck and GeoPandas from the original plan; the final code does not use them (the policy is a greedy expected-coverage optimiser with SciPy assignment, the replay is a minute-by-minute loop, the dashboard is the React demo).
