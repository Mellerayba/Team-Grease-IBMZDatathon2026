# PulseNYC

Interactive NYC ambulance dispatch and demand-aware relocation simulation built with React, TypeScript, Tailwind CSS and React Leaflet.

## Run

```bash
npm install
npm run dev
```

Open the localhost URL shown by Vite. To make a production build, run `npm run build`.

## Features

- Reproducible synthetic incident stream, with a Brooklyn demand surge
- Browser-based dispatch and service lifecycle simulation
- Idle unit relocation every 15 simulated minutes using a rule-based demand heuristic
- Shared incident arrivals and initial fleet for static and AI policy comparison
- Interactive dark NYC map using CARTO/OpenStreetMap tiles
- Playback, speed, fleet size, policy and visualization controls
- Response time and 90th percentile metrics derived from simulated outcomes

## Important limitations

This is a demonstrator, not an operational EMS model. Forecasts are rule-based, not machine learning. Travel follows straight-line interpolation and approximated urban speed, not a routed road network. Fleet deployment, service durations, incidents and hospitals are illustrative. The app needs internet access for map tiles. No historical NYC data is bundled.
