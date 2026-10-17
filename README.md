# Team-Grease-IBMZDatathon2026
SSF2026

## The problem
In New York City the first minutes of a critical emergency decide whether someone survives, but ambulance response times are at record highs. Ambulances are large, expensive, staffed by a shrinking workforce and stuck in the same traffic as everyone else, so the city cannot simply add more of them quickly.

## Why this problem matters

**Response times are getting worse.**
- NYC response times for life-threatening medical emergencies rose from 9:34 in 2021 to 13:09 in the latest fiscal year reported ([EMS1](https://www.ems1.com/response-times/nyc-ambulance-response-times-hit-record-high-amid-ems-staffing-shortage)).
- FDNY's EMS chief told the City Council the department loses about 10 in-service ambulances a day because of staffing shortages (same source).

**The burden is uneven across boroughs.**
- The NYC Independent Budget Office found the share of advanced life support calls answered within 10 minutes fell in every borough between 2014 and 2024. In 2024 it was about 82% in Staten Island and about 75% in the Bronx. For basic life support, the Bronx was about 48% against about 68% in Staten Island ([IBO analysis summary](https://isoclivecivic.substack.com/p/exploring-an-independent-perspective)).

**Minutes decide survival in cardiac arrest.**
- The American Heart Association rule of thumb is that survival chances fall by 7-10% for each minute without CPR or defibrillation ([CARES registry report](https://mycares.net/sitepages/uploads/2024/2023_flipbook/inc/html/34.html)). Newer studies refine this, showing much better survival with defibrillation within about four minutes ([FireRescue1](https://www.firerescue1.com/fire-products/aeds/articles/ny-countys-cardiac-emergency-response-time-subpar-ISf1uNeBLRnlC7jS/)).

**Recognising the emergency on the phone is part of the problem.**
- Dispatchers fail to identify roughly 25% of out-of-hospital cardiac arrests ([study abstract](https://swov.nl/en/publicatie/machine-learning-supportive-tool-recognize-cardiac-arrest-emergency-calls)).
- In a Copenhagen study of 108,607 calls, a machine learning model recognised cardiac arrest more often and faster than dispatchers, but with more false alarms (same source).

**Why we use decision support, not automation.**
- A randomised trial across 169,049 calls found that alerting dispatchers did not significantly improve their recognition of cardiac arrest, even though the model outperformed humans on its own ([trial summary](https://busqueda.bvsalud.org/portal/resource/es/mdl-33404620)). We keep a human dispatcher in the loop and test our system by simulation.

**Data limitation.**
- NYC's public EMS dispatch data aggregates incident locations to protect patient privacy, so our analysis works at area level, not street level ([data.gov](https://catalog.data.gov/dataset/ems-incident-dispatch-data)).

## Proposed Solution
**Project Dispatch** adds a second, much cheaper layer to the emergency response: a fleet of **lightweight electric first-response units** (for example e-bikes) that can move through traffic and carry enough equipment to stabilise a critically ill person until an ambulance arrives. Ambulances still transport patients; the light units buy the first minutes.

AI decides where those units wait and which one goes:
1. **Forecast** the expected number of critical and other calls in each zone for the next 30 minutes to 3 hours, from historical EMS data, weather and the calendar.
2. **Position** idle units so that as much expected critical demand as possible is within a few minutes' ride, weighting vulnerable neighbourhoods more and keeping a minimum presence in every borough.
3. **Dispatch** the unit that can arrive fastest, with a human dispatcher in the loop and a plain-language briefing of each recommended move.
4. **Replay** real 2022 incidents under each policy so every policy faces identical calls.

**The question we answer for the city: how many units should NYC buy, and where should they stand?** Because the same incidents can be replayed with 0, 50, 100, 300 or more units, we can show how the share of critical calls reached quickly rises with fleet size, where the returns start to diminish, and which boroughs gain most.

**Status (honest).** Built and tested: the demand forecaster, the risk index, the positioning and dispatch policy, and a minute-by-minute replay of real 2022 incidents, which today models a single generic fleet. Being added: the light-unit model (speed, no hospital transport, stays until the ambulance arrives), street-network travel times and the fleet-size sweep. All response times are *modelled*, not measured, and rest on stated assumptions (see `docs/decisions.md`).

## Target Audience
1. **City emergency-management and budget decision-makers:** Use the fleet-size curve to decide how many units to buy and where to station them.
2. **Emergency medical dispatchers and operations leaders:** Use the recommendations to decide which unit to send and where to position idle units. 
3. **Public health and policy planners:** Use the fairness analysis to find neighbourhoods with the slowest response and target staffing and funding. 
4. **Hospital and community health managers:** Plan outreach and support around predicted demand spikes. 
5. **Residents of underserved areas:** The end beneficiaries, who gain faster care and fairer allocation of resources. 
6. **Researchers and civic technologists:** Reuse the open-source approach with other cities' open data.

## Datasets
We will use open, publicly available datasets: 
- [ ] **Public EMS dispatch data** (e.g. NYC Open Data EMS Incident Dispatch Data): Incident time, severity level, and dispatch and travel times, aggregated by area to protect privacy. 
- [ ] **Public response-time reports** (e.g. FDNY monthly response times): Benchmarks to validate our findings. 
- [ ] **Hospital and station locations** from public health and city sources. 
- [x] **Census data:** Population, income and vehicle access by area, to weight demand and audit fairness. 
- [x] **Weather data** (e.g. NOAA): To improve demand forecasts. 
- [ ] **Open road network data** (OpenStreetMap): For travel-time estimates. 
- [ ] **Emergency call text:** Public or synthetic, clearly labelled, with no personal data. 

All data is open, properly licensed and contains no personal information.


## Open-Source Libraries and Tools Usage

The project uses the following libraries and tools for its Python workflows and browser-based PulseNYC simulation:

| Component | Open-source library | License | Usage in project |
| :--- | :--- | :--- | :--- |
| **Data processing** | [pandas](https://github.com/pandas-dev/pandas) | BSD 3-Clause | Loading, cleaning, and aggregating incident and forecast data |
| **Numerical computing** | [NumPy](https://github.com/numpy/numpy) | BSD 3-Clause | Array calculations for forecasting, risk scoring, and travel-time calculations |
| **Forecasting** | [LightGBM](https://github.com/microsoft/LightGBM) | MIT | Training and running the incident-demand forecasting model |
| **Routing assignment** | [SciPy](https://github.com/scipy/scipy) | BSD 3-Clause | Hungarian assignment in the Python routing policy |
| **Python dashboard** | [Streamlit](https://github.com/streamlit/streamlit) | Apache 2.0 | Python dashboard framework; the current dashboard is a skeleton |
| **Web interface** | [React](https://github.com/facebook/react) | MIT | Building the PulseNYC simulation interface |
| **Browser rendering** | [React DOM](https://github.com/facebook/react) | MIT | Rendering the React interface in the browser |
| **Web mapping** | [Leaflet](https://github.com/Leaflet/Leaflet) | BSD 2-Clause | Rendering the interactive map |
| **Frontend language** | [TypeScript](https://github.com/microsoft/TypeScript) | Apache 2.0 | Implementing the browser simulation engine and interface |
| **Frontend build tooling** | [Vite](https://github.com/vitejs/vite) | MIT | Running the development server and building the frontend |
| **Frontend styling** | [Tailwind CSS](https://github.com/tailwindlabs/tailwindcss) | MIT | Styling the PulseNYC interface |
| **CSS processing** | [PostCSS](https://github.com/postcss/postcss) | MIT | Processing frontend CSS |
| **CSS compatibility** | [Autoprefixer](https://github.com/postcss/autoprefixer) | MIT | Adding browser compatibility prefixes to CSS |
| **Icons** | [Lucide React](https://github.com/lucide-icons/lucide) | ISC | Providing icons for the frontend |
| **React build integration** | [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react) | MIT | Enabling React support and fast refresh in Vite |
