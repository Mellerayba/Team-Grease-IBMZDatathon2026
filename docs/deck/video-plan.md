# Video plan: 3 minutes, with the stack and IBM Z up front

The brief: no more than 3 minutes covering the problem statement, data supporting the problem, the solution, and, most importantly, how the solution leverages IBM Z (the tech stack, the LinuxONE instance and the model). 3:00 is a hard limit, so the video cannot be longer. Instead, **replace the Implementation and Technology slides with one new slide, `stack-slide.png`**, and give the live LinuxONE section about a minute. The script below is 2:29 of speech; with the live recording running its full minute the finished video is about 2:50, which is under the limit with 10 seconds spare. If it runs long, trim the silent demo footage first.

## Slide order (8 slides)

| # | Slide | Change needed |
|---|---|---|
| 1 | Title | none (3 s) |
| 2 | The Problem | add the 9:34 (2021) to 13:09 trend and the 7-10% per minute line |
| 3 | Target demographic (map) | add the Bronx 75% vs Staten Island 82% gap as one stat |
| 4 | The Solution | fine; drop or reword the "incident prediction: weather-triggered" card (weather adds almost nothing to the forecast) |
| 5 | **The Stack on IBM Z (new)** | `docs/deck/stack-slide.png` replaces both Implementation and Technology. The old Technology slide says "Granite LLM for driver communication", which has not been run on the VM |
| 6 | Live: LinuxONE instance, model, demo | screen recording, see the shot list |
| 7 | The Competitive Advantage | replace 9.65 min vs 13 min (a modelled number against a reported one) with 82% to 91% of critical calls within 8 minutes, labelled modelled |
| 8 | Team and close | merge the Team slide and the final slide |

## Time budget and script

| Segment | On screen | Words | Seconds |
|---|---|---|---|
| Title | title slide | 0 | 3 |
| 1. Problem | slide: The Problem (add the 9:34 to 13:09 trend) | 63 | 24 |
| 2. Data behind the problem | slide: Target demographic with the map (add the borough gap as one stat) | 45 | 17 |
| 3. Solution | slide: The Solution | 43 | 17 |
| 4. The stack on IBM Z | slide: The Stack on IBM Z (new) | 89 | 34 |
| 5. Live: the instance, the model, the demo | screen recording of the VM (see shot list) | 96 | 37 |
| 6. Result | slide: The Competitive Advantage (corrected) | 40 | 15 |
| 7. Close | slide: Team and close | 5 | 2 |
| **Total** | | **381** | **about 149 s (2:29)** |

About 2.6 words a second. If you run over, shorten segment 2 or 6, never segment 4 or 5.

### 1. Problem  (24 s)
*slide: The Problem (add the 9:34 to 13:09 trend)*

> In New York, an ambulance now takes about thirteen minutes to reach a life-threatening emergency, up from nine and a half minutes in 2021. In cardiac arrest, every minute without care cuts survival by seven to ten percent, and the standard is eight minutes. The city loses around ten ambulances a day to staffing shortages, so it can't simply buy its way out.

### 2. Data behind the problem  (17 s)
*slide: Target demographic with the map (add the borough gap as one stat)*

> The burden is also uneven. In 2024, about 75 percent of advanced life support calls in the Bronx were answered within ten minutes, against 82 percent in Staten Island. We use the city's open EMS data, which is area-level, so we work by ZIP zone.

### 3. Solution  (17 s)
*slide: The Solution*

> Project Dispatch adds a cheap second layer: lightweight first-response units, like e-bikes, that stabilise a patient until the ambulance arrives. Our AI forecasts where calls will come from, and decides where those units wait, so help is already close when the call comes.

### 4. The stack on IBM Z  (34 s)
*slide: The Stack on IBM Z (new)*

> Everything runs on an IBM LinuxONE instance: s390x under z/VM, with eight gigabytes of memory. On top sit Python with pandas and LightGBM for the forecast, our own positioning policy and replay, and a React map for the demo. We built it to fit the instance, and saved the models as portable files, because s390x is big-endian. We also tested the on-chip AI accelerator through Snap ML. It worked, but ran slower than the CPU, so we kept the faster path, and all thirty-three tests pass on the instance.

### 5. Live: the instance, the model, the demo  (37 s)
*screen recording of the VM (see shot list)*

> Here is the instance. The machine is s390x, and the accelerator flag, nnpa, is exposed. These are the eight forecast models, trained on this instance and tested on all of 2022. They beat every baseline on error, though the gain is small, and we say so. Here we replay real calls: the model-guided layout reaches more critical patients in eight minutes than a fixed one. And this is the demo, served from the instance. Calls come in, units move toward the zones the model expects to be busy, and the shading changes with each new forecast.

### 6. Result  (15 s)
*slide: The Competitive Advantage (corrected)*

> On a heat-wave day, the share of critical calls reached within eight minutes rose from 82 to 91 percent. These times are modelled, not measured. Next, we build the e-bike model, and answer how many units the city should buy.

### 7. Close  (2 s)
*slide: Team and close*

> Project Dispatch, by Team Grease.

## Shot list for the live LinuxONE section (60 seconds of footage)
Record one take with the VM's JupyterLab (IBM logo) in frame. The notebook cells are already run; you only scroll and click. Practise once.

| Time | Show | What it proves |
|---|---|---|
| 0-6 s | A terminal or cell: `uname -m` and `grep -m1 -o nnpa /proc/cpuinfo` | the instance is s390x and the AI accelerator flag is exposed |
| 6-12 s | `!python3 ../scripts/vm_check.py --years 2` output | the machine, its memory and the library stack |
| 12-22 s | Harry notebook: the eight training lines, then the MAE and deviance tables | the model was trained and scored on LinuxONE |
| 22-28 s | The replay table (static vs forecast vs oracle), then the accelerator verdict and benchmark lines | the replay result, and the accelerator tested honestly |
| 28-60 s | The demo in the browser (served from the VM through the tunnel): press Run demo, speed 4x, let units and calls move, point at a red zone and a dashed route | the finished product running from the instance |

The screenshots in `docs/screenshots/vm-01` to `vm-12` are your backup if a live run fails: cut to them. The narration for this segment is 96 words (about 37 seconds); the recording itself runs a full 60 seconds, so let the demo play for the last 20-25 seconds with no talking, or over music.

## Say and do not say
- **Say:** "modelled, not measured"; "slower than the CPU, so we kept the faster path"; "served from the instance".
- **Do not say:** that weather drives the forecast; that the forecast beats a schedule built from history (on coverage it is level); that the Granite briefing runs on the VM; that the accelerator made anything faster.
