# Project Dispatch

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
- NYC street map using OpenStreetMap tiles with a CARTO fallback, without an API key
- Playback, speed, fleet size, policy and visualization controls
- Response time and 90th percentile metrics derived from simulated outcomes

## Important limitations

This is a demonstrator, not an operational EMS model. Forecasts are rule-based, not machine learning. Travel follows straight-line interpolation and approximated urban speed, not a routed road network. Fleet deployment, service durations and incidents are illustrative. Hospital markers use real campus reference coordinates; they are not ambulance entrance locations. The app needs internet access for map tiles. No historical NYC data is bundled.

## Hospital locations

Five named example facilities are shown, one per borough; this is not a complete hospital directory. Coordinates come from [NYC Health + Hospitals facilities](https://data.cityofnewyork.us/resource/q6fj-vxf8.json) and [NYC Facilities Database](https://data.cityofnewyork.us/resource/67g2-p84d.json). Staten Island uses the South campus at 375 Seguine Avenue. The simulation selects the nearest of these five facilities using straight-line distance. It does not model hospital capacity, clinical suitability, or road routing.
