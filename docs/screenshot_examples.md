# Screenshot examples: evidence of how we used the technology

What this file is: the evidence the guidebook asks for ("screenshots as evidence of platform capabilities"), organised by the technology we used. Context and conclusions are in [`decisions.md`](decisions.md).

**How each piece of evidence was produced (so nothing is overstated):**
- **Captured** = a screenshot taken for this file, on 2026-10-17, from the running demo (headless Chrome) or from real command output rendered in a terminal-style frame. Files are in [`screenshots/`](screenshots/).
- **VM output (verbatim)** = text the team copied from the LinuxONE VM terminal and pasted into our working session; reproduced exactly below. These are not screenshots; take the matching screenshots on the VM (section D).
- **To capture** = evidence that does not exist yet. Nothing in that list is claimed.

## Technology map

| Technology | What we used it for | Evidence | Status |
|---|---|---|---|
| **IBM LinuxONE VM (s390x, z17-generation)** | Runs the notebooks (data build, model training, replay) and serves the demo | A1 platform check, A4 demo served from the VM | Verbatim text; screenshots to capture (D1, D2, D7) |
| **IBM Z AI accelerator (`nnpa`, zDNN) via Snap ML** | Tested for forecaster inference; benchmarked fairly; production stays on CPU | A2, A3 | Verbatim text; screenshot to capture (D3, D4) |
| **LightGBM (Poisson objective)** | Demand forecaster, 8 models | C1 tests; B4 zones shaded by its output | Captured |
| **Python replay and routing policy** | Four-policy comparison on real incidents | C2 export run; B6 results | Captured |
| **OSRM (open street routing)** | Real street routes for the policy's moves | B3 (dashed cyan routes) | Captured |
| **React + Leaflet + OpenStreetMap tiles** | The demo | B1-B6 | Captured |
| **IBM Granite via llama.cpp** | Plain-language briefing of each cycle's moves | none | Implemented, not run on the VM |
| **Mandatory IBM labs (hybrid AI; predictive and generative)** | Required by the guidebook | none | Completion to be confirmed and captured |
| **Containers / security / scalability** | Guidebook criterion 3 | none | To capture (D) |

---

## A. LinuxONE VM evidence (verbatim output from the VM)

### A1. Platform check (`python3 scripts/vm_check.py --years 2`, on `linux1@grease`)
Shows the machine is an IBM Z (s390x, big-endian), the memory we have to work within, and that the on-chip AI accelerator flag (`nnpa`) is exposed.

```
== Platform
machine        : s390x (big-endian)
os             : Linux-6.8.0-147-generic-s390x-with-glibc2.39
python         : 3.12.15
logical cpus   : 3
IBM Z machine  : manufacturer IBM, type 9175, model 702              ME1

== Memory (this is what decides whether a notebook kernel survives)
total          : 8.4 GB
available now  : 5.6 GB   <- what is free right now, after other processes and old kernels
swap           : 0.0 GB

== CPU features and AI accelerator
vx     : YES  vector facility (SIMD)
vxe    : YES  vector enhancements 1
vxe2   : YES  vector enhancements 2
nnpa   : YES  neural network processing assist = the on-chip AI accelerator (IBM z16 / Telum and later)
sort   : YES  sort instruction
dflt   : YES  deflate (compression) instruction

== Libraries
numpy       : 2.0.2
pandas      : 3.0.6
lightgbm    : 4.7.0
sklearn     : 1.9.0
snapml      : not available (ModuleNotFoundError)
onnxruntime : not available (ModuleNotFoundError)
xgboost     : 3.4.1
torch       : 2.11.0+cpu
```
(Excerpt: the tip and explanatory lines of the script's output are omitted. In the very first run, from a different Python, the libraries were missing and the pipeline test failed with `No module named 'numpy'`; the notebook environment has them.)

### A2. Accelerator test (`python3 scripts/accel_check.py`)
Snap ML 1.17.2 installed from a PyPI s390x wheel and picked the accelerator format. For a small numeric model the accelerator matches LightGBM; for our real model (categorical features) the predictions do not match, so it cannot be used as is.

```
== 1. Hardware and zDNN library
machine: s390x | python 3.12.15
AI accelerator flag (nnpa): YES
zDNN library: libzdnn.so.0

== 2. Snap ML
snapml  imported

== 3A. Synthetic numeric Poisson model: does the accelerator path work at all?
LightGBM CPU: 500,000 rows in 1.43s (0.35M rows/s)
[direct lightgbm import] tree_format=auto           chosen=zdnn_tensors     0.08M rows/s (  0.2x LightGBM) | output equals: raw score (apply exp for the Poisson mean) | max diff 2.5e-08
[direct lightgbm import] tree_format=zdnn_tensors   chosen=zdnn_tensors     0.08M rows/s (  0.2x LightGBM) | output equals: raw score (apply exp for the Poisson mean) | max diff 2.5e-08
[direct lightgbm import] tree_format=compress_trees chosen=compress_trees   0.12M rows/s (  0.4x LightGBM) | output equals: raw score (apply exp for the Poisson mean) | max diff 2.5e-08
VERDICT [direct lightgbm import]: accelerator (zdnn_tensors) selected: YES | predictions match LightGBM: YES

== 3B. Our real model (critical_1h; has categorical features zipcode and borough)
31 features, 305 trees, categorical splits: YES
test inputs: realistic ranges; categorical columns [0, 30] get valid category codes
LightGBM CPU: 500,000 rows in 10.27s (0.05M rows/s)
[our model] tree_format=auto           chosen=zdnn_tensors     0.05M rows/s (  1.0x LightGBM) | output equals: NEITHER (mismatch) | max diff 4.6e+00
[our model] tree_format=zdnn_tensors   chosen=zdnn_tensors     0.05M rows/s (  1.0x LightGBM) | output equals: NEITHER (mismatch) | max diff 4.6e+00
[our model] tree_format=compress_trees chosen=compress_trees   0.05M rows/s (  1.1x LightGBM) | output equals: NEITHER (mismatch) | max diff 4.6e+00
VERDICT [our model]: accelerator (zdnn_tensors) selected: YES | predictions match LightGBM: NO
```
Earlier attempts on the same VM, kept because they are part of the story: the generic importer rejected LightGBM files (`Generic model import currently supports only 'pmml' as the input type`), and the PMML route through Nyoka failed with `undefined feature used at line 86` and `non-supported value defined for algorithmName attribute`. The working route is the class-level `BoostingMachineRegressor().import_model(file, "lightgbm")`.

### A3. Fair benchmark (`python3 scripts/accel_bench.py --quick`, 3 CPUs, thousand rows per second | milliseconds per call)
All engines at 3 threads in the lower rows (Snap ML defaults to 1 thread, which made the first comparison unfair).

```
=== leaves=15 trees=100 | accelerator output max diff vs LightGBM: 8.1e-09
engine                                 batch 256          batch 4096        batch 100000
LightGBM CPU        1t               299 |    0.9         310 |   12.9         314 |  318.9
Snap ML CPU path     1t              351 |    0.7         352 |   11.4         356 |  281.2
Snap ML ACCELERATOR 1t               188 |    1.4         119 |   33.5         185 |  541.8
LightGBM CPU        3t               773 |    0.3         891 |    4.5         908 |  110.1
Snap ML CPU path     3t              966 |    0.3        1026 |    3.9        1020 |   98.0
Snap ML ACCELERATOR 3t               417 |    0.6         379 |   10.6         239 |  417.6

=== leaves=63 trees=100 | accelerator output max diff vs LightGBM: 4.9e-02
LightGBM CPU        1t               222 |    1.2         226 |   17.7         230 |  434.0
Snap ML CPU path     1t              222 |    1.2         223 |   18.0         223 |  448.2
Snap ML ACCELERATOR 1t               165 |    1.6         167 |   23.9         168 |  596.4
LightGBM CPU        3t               599 |    0.4         652 |    6.1         655 |  152.6
Snap ML CPU path     3t              728 |    0.4         759 |    5.3         783 |  127.7
Snap ML ACCELERATOR 3t               343 |    0.7         345 |   11.6         240 |  417.4

== SUMMARY: accelerator throughput divided by the best LightGBM throughput (above 1.0 = the accelerator wins)
 leaves  trees     batch 256    batch 4096  batch 100000
     15    100         0.54x         0.43x         0.26x
     63    100         0.57x         0.53x         0.37x
Best case: leaves=63, trees=100, batch=256: accelerator is 0.57x LightGBM (still slower).
```
**Conclusion we report:** the accelerator is present, drivable and benchmarked; it was slower than CPU LightGBM (0.26-0.57×) and, for larger trees, slightly less accurate, so production scoring stays on the CPU. We do not claim a speed-up. A live 256-row cycle takes 0.3-0.7 ms on every engine, so inference speed is not a bottleneck.

### A4. The demo running on the VM
The team ran the demo on the LinuxONE VM (`npm run dev`, port 5173) and viewed it through an SSH tunnel at `http://localhost:5173`; it worked. **No screenshot of that session has been saved yet** (see D). It looks the same as section B below.

---

## B. The demo (captured from the running app)

Every figure is the real replay of 2022-07-19: real incidents, the LightGBM forecast for each half-hour, and the policy's moves. Response times are modelled.

### B1. Compare view at the start of the replay
Static layout (left) and model-guided fleet (right) on the same calls and the same 330 units; zones shaded by the model's risk index.

![Compare view at the start](screenshots/demo-01-compare-start.jpg)

### B2. Mid-replay (16:30): live units, incoming calls, new forecast
Unit colours show what each is doing; red and orange dots are calls (red = no unit assigned yet); the right-hand panels show waiting and en-route calls, the highest-risk zones with units parked there (static vs model), and the model's recent decisions. The banner reads "Forecast issued 16:30 · new forecast just landed".

![Mid-replay compare view](screenshots/demo-02-compare-mid-replay.jpg)

### B3. Where units are going after a decision
Model-guided view, zoomed. Dashed cyan lines are the real street routes (OSRM) from a unit to the post the model chose; the ring marks the destination. Red shading is the model's highest-risk zones.

![Destination routes on streets](screenshots/demo-03-model-destination-routes.jpg)

### B4. Zones changing with the model output (same moment, 17:30)
The same map shaded three ways: the model's risk index, the forecast calls for the next hour, and what actually happened. Comparing the second and third shows where the forecast was right and where it missed.

![Zones by risk index](screenshots/demo-04-zones-risk.jpg)
![Zones by forecast calls](screenshots/demo-05-zones-forecast.jpg)
![Zones by actual calls](screenshots/demo-06-zones-actual.jpg)

### B5. Calls, highest-risk zones and decision log
Close-up of the right-hand column at 17:20: the call feed (812 calls reached so far), the highest-risk zones with the number of units in or heading to each (static vs model), and each move with its reason and drive time.

![Calls, zones and decisions](screenshots/demo-08-calls-zones-decisions.png)

### B6. End of the replay: result tables
Running metrics for the 6-hour window, and the full-day result for static, model-guided and the oracle (82.3% → 91.2% → 93.4% of critical calls within 8 minutes).

![End of replay with results](screenshots/demo-07-end-of-replay-results.jpg)

---

## C. Command output captured locally (real runs, rendered in a terminal-style frame)

### C1. Automated tests: 33 passed
Forecast features and model, risk index, routing policy, replay (including the trace used by the demo), street routing.

![Tests](screenshots/terminal-01-tests.png)

### C2. Demo export run
The replay of the heat-wave day for the three policies (4,694 incidents, 330 units), the borough breakdown, and the export of 1,508 window calls, 743 moves and 400 street routes. This is the run that produced the committed `demo.json` (output file path shortened).

![Export run](screenshots/terminal-02-export-demo.png)

---

## D. To capture on the VM (not done yet, do not claim until captured)

Take each as a screenshot with the VM prompt (`linux1@grease`) visible, and add it to `docs/screenshots/` and this file.

| # | Evidence | Command (run on the VM) | Shows |
|---|---|---|---|
| D1 | Platform | `uname -m && lscpu \| head -20` | s390x, IBM Z |
| D2 | AI accelerator flag | `grep -m1 -o nnpa /proc/cpuinfo; python3 scripts/vm_check.py --years 2` | `nnpa: YES`, memory, libraries |
| D3 | Accelerator verdict | `python3 scripts/accel_check.py` | the VERDICT lines (section A2) |
| D4 | Benchmark | `python3 scripts/accel_bench.py --quick` | the summary table (section A3) |
| D5 | Compute | `python3 -m pytest tests -q` and `htop` (or `top`) during model training in the notebook | tests pass on s390x; CPU use |
| D6 | Model on the VM | the Harry notebook's score table and calibration cells (rendered output) | forecaster trained and scored on LinuxONE |
| D7 | Demo on the VM | the browser at `localhost:5173` through the tunnel, with the `npm run dev` terminal and `uname -m` visible in the same frame or recording | the demo is served from LinuxONE |
| D8 | Containers | `docker ps` or `podman ps`, `cat /proc/1/cgroup` (if the VM is a container), or the instance page showing the image | containerisation (guidebook criterion) |
| D9 | Security | the instance's access method (SSH key login, user separation); only show what is real | security (guidebook criterion) |
| D10 | Scalability | `free -h` and `nproc`, and a sentence of what we did about the memory limit (the 2021-2022 default; 1.9 GB per year measured) | resource handling |
| D11 | Mandatory labs | the completion screens for both IBM labs (hybrid AI; predictive and generative) | required for eligibility |
| D12 | Granite briefing (optional) | run the briefing cell with enough free memory | the generative-AI step, currently not evidenced |

Guidebook criterion 3 ("platform capabilities such as containerisation, security, scalability and compute resources, using screenshots as evidence") is the reason for D8-D10; if a capability is not real for our instance, say so rather than screenshotting around it.
