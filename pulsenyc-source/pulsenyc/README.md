# Project Dispatch: demo

React + TypeScript + Leaflet front end that **replays the real model run**: real 2022 NYC EMS calls, the LightGBM forecast for each half hour, and the routing policy's decisions, side by side with a fixed (static) layout. Nothing is simulated in the browser.

## Run

```bash
npm install
npm run dev          # then open the localhost URL Vite prints, and press RUN DEMO
npm run build        # production build
```

The page needs internet for the map tiles. It reads `public/data/demo.json` and `public/data/zones.geojson`, which are committed so the demo runs straight after a clone. `?t=150` in the URL opens the replay at minute 150.

## Regenerate the data (from the repo root)

```bash
python scripts/export_demo.py --street-routes     # needs notebooks/data/processed/forecast_zone_test.csv.gz and incidents.csv.gz
```

This replays one day (default 2022-07-19, the heat wave; window 14:00-20:00, 330 units) under the static layout and the model-guided policy (and the oracle, for the full-day table), and writes the unit timelines, call lifecycles, zone polygons and the forecast and risk index for every 30-minute cycle. `--street-routes` fetches real street routes (public OSRM, cached in `notebooks/data/raw/route_cache.json`) for every move the policy makes, so repositioning units drive along streets; about 400 routes take 7 minutes the first time. Options: `--day`, `--from-hour`, `--hours`, `--units`.

## What you see

- **Zones** shaded by the model: risk index, forecast calls for the next hour, or what actually happened. A new forecast lands every 30 minutes and the shading changes with it.
- **Live units** (flat squares, in the pitch deck's navy, red and gold) coloured by what they are doing (available, moving to a new post, responding, on scene, to hospital, at hospital).
- **Where they are going after a change:** dashed navy street routes with a gold square at the destination for every move the model makes; the right-hand log gives each move's reason.
- **Calls coming in:** a diamond appears the minute a call arrives (red until a unit is assigned, amber while one is en route) and disappears when a unit reaches it.
- **Compare view:** static layout and model-guided fleet on the same calls, with running critical-call response metrics and the full-day result.

## Limits

Response times are **modelled**, not measured: fleet size, scene and hospital times, traffic and (for responses) travel times are assumptions; routes for repositioning moves are real streets but responding and transport legs are drawn as straight lines. Only the difference between layouts is meaningful. `src/engine.ts` is the earlier in-browser synthetic simulation and is no longer used.
