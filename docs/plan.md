# Build order (from claudeplan.md)
1. Clean/aggregate EMS data to zones x 30-min buckets (`data/`)
2. Baseline forecaster, then LightGBM (`forecast/`)
3. Travel-time matrix + simulator with static policy -> baseline response times (`travel/`, `sim/`)  **protect time here**
4. Optimizer + dynamic policy (`optimize/`, `sim/policies.py`)
5. Dashboard + surge-day case study (`dashboard/`)
6. Story: problem, method, results, limitations

# Submission checklist (IBM Z Datathon, Oct 17-18 2026)
- Project title must contain team name: `Team Grease_IBM Z Datathon 2026 Project`
- <=3 min video: problem, data, solution, and how it uses IBM Z (show LinuxONE instance + model)
- Code repo link; credit open-source libraries; event code SSF2026
- Both provided AI/ML labs (hybrid AI; predictive + generative) must be completed first
- Screenshots as evidence of LinuxONE capabilities (containers, security, scalability, compute)
- Must be built within the hack window; optional flowchart + prototype

# Open decisions
- Unit type: **decided** - ambulances stay as the transporting fleet; the lever we study is a second fleet of lightweight electric first-response units (see `docs/decisions.md` D13 and section 7)
- Zone unit (ZIP / precinct / community district) - check dataset columns
