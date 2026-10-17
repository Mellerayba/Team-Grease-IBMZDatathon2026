Here’s how I’d structure it so it’s achievable and demonstrably “real-time AI for critical decisions.”

System overview
Replay engine → short-horizon forecaster → relocation optimizer → simulator/dashboard
(streams past      (demand per zone,        (where should each        (shows response times
 incidents)         next 1-3 hours)          unit stage now?)          vs. baselines)

Because NYC’s dataset is historical, build a replay engine that feeds incidents in timestamp order as if they were arriving live. Say this openly in your pitch. Designing the pipeline so a real feed could be swapped in is a plus.

Decide your unit type first

“Mobile units or ambulances” changes the problem. I’d pick one:

Ambulances (or community-paramedic units) answering low and mid-acuity calls. This fits the data best, since you can simulate actual response times.
Mobile clinics, which don’t respond to calls, so “response time” isn’t your metric. You’d measure access or coverage instead.

For a real-time critical-decision story, the first is much stronger. You can still show the same system generalizes to clinics.

Components

1. Short-horizon demand forecaster

Predict expected incidents per zone for the next 30 to 180 minutes.
Baseline: historical average by zone, weekday, and hour, scaled by a recent-surge multiplier (last hour vs. normal).
Model: LightGBM with a Poisson objective, using lagged counts, time features, weather, and holidays.
Always compare against the baseline. If boosting barely wins, say so, since that’s an honest finding.

2. Relocation optimizer (the decision engine)

Every 15 to 30 simulated minutes, solve for where idle units should stage.
Formulation: a dynamic Maximal Expected Covering Location problem, or simpler, minimize expected demand-weighted travel time, with constraints on:
Max relocation distance per move
Max moves per unit per shift (crew fatigue and realism)
Minimum coverage in every borough (equity floor)
Candidate staging posts: hospital lots, fire stations, or a grid of hubs.
Solve with OR-Tools or PuLP. It should run in seconds, which is what makes it “real-time.”

3. Travel-time matrix

Precompute zone-to-post drive times from the road network (OSMnx or OSRM), with time-of-day multipliers for traffic.

4. Discrete-event simulator (SimPy works well)

Replay incidents, dispatch the nearest available unit, and model service time (travel, on scene, to hospital, back).
Handles the busy-unit effect, which is where dynamic deployment actually helps.

5. Dashboard

A map showing units, predicted demand heatmap, recommended moves, and live metrics.
How you prove it works

Run the same replayed days under three policies and compare:

Static: units stay at fixed home posts.
Scheduled: relocate by historical hour-of-day averages.
Your system: relocate using live forecasts.

Metrics: mean and 90th percentile response time, percent of calls reached within a target (e.g. 8 minutes), borough-level equity gaps, and number of relocations. Test on held-out days, ideally including surge days like heat waves or large events, where dynamic deployment should shine.

Pitfalls to plan for
Geography may be coarse. As far as I recall, this dataset gives location by ZIP code, precinct, or community district rather than exact coordinates. Check the columns, then use zones as your spatial unit and zone centroids for travel times.
No unit location data. The dataset doesn’t tell you where ambulances actually were, so fleet size and baseline posts are assumptions. State them clearly and test a few fleet sizes.
Service times: Estimate them from the data’s timestamp fields rather than guessing.
Don’t over-claim. The simulator shows what would happen under your assumptions, not proven real-world gains.
RL is a stretch goal. An optimizer fed by forecasts is easier to explain, validate, and trust. Add RL only if you finish early.
Suggested scope for a hackathon
Clean and aggregate data to zones and 30-minute buckets.
Build the baseline forecaster, then LightGBM.
Build the simulator with static policy and measure baseline response times.
Add the optimizer and the dynamic policy.
Add the dashboard and surge-day case study.
Polish the story: problem, method, results, limitations.

Steps 3 and 4 are the heart of the project, so protect time for them over fancy forecasting.
