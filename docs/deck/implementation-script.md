# The Implementation: slide and 30-second script

Slide: `implementation-slide.png` (960x540 at 2x, matches the deck's style; source `implementation-slide.html`).

## Script (about 27 seconds, 70 words, room to breathe in a 30-second slot)
*On the slide, point to each card in turn: Data, Forecast, Position, Replay, then the navy strip.*

> This is how it works, and all of it runs on our IBM LinuxONE VM. A LightGBM model trained on real 2021 New York 911 calls forecasts demand in every ZIP zone for the next three hours. Our policy moves idle units toward that demand every thirty minutes. Replaying real 2022 calls, critical calls reached within eight minutes rose from 82 to 91 percent. Those times are modelled, not measured.

| Beat | Words on screen | Time |
|---|---|---|
| Opening, "runs on LinuxONE" (navy strip) | "This is how it works, and all of it runs on our IBM LinuxONE VM." | 0-4 s |
| Data and Forecast cards | "A LightGBM model trained on real 2021 ... next three hours." | 4-13 s |
| Position card | "Our policy moves idle units ... every thirty minutes." | 13-18 s |
| Replay card | "Replaying real 2022 calls ... 82 to 91 percent." | 18-25 s |
| Honest caveat | "Those times are modelled, not measured." | 25-27 s |

## If you need 20 seconds (47 words)
> Real 2021 New York 911 calls train a LightGBM forecast on our IBM LinuxONE VM. Every thirty minutes, idle units move toward the zones where demand is expected. Replaying real 2022 calls, critical calls reached within eight minutes rose from 82 to 91 percent. Modelled, not measured.

## Facts behind every line (so nothing is overstated)
- **2021 calls:** the model trained on January to September 2021 and validated on Q4 2021; 1,491,454 calls in 2021 (decisions.md, section 6 and 7).
- **Forecast horizon:** next 30 minutes, 1, 2 and 3 hours, per ZIP zone; "three hours" is the longest.
- **Every thirty minutes:** the policy cycle (decisions.md, section 10).
- **82% to 91%:** critical calls within 8 minutes on 2022-07-19, static layout vs model-guided, 4,694 calls, 330 assumed units (demo export; the VM notebook run gives 81.85% and 91.67%). The two-day average is lower (76% to 85%), so say "on our heat-wave test day" if asked.
- **Modelled:** fleet size, scene and hospital times, traffic and travel times are assumptions.
- Do not say "weather-based" or "predicts the weather": weather adds almost nothing to the forecast. Do not say the forecast beats a schedule built from history: on coverage it is level with it.
