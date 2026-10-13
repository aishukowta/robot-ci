# Robot CI

**Continuous regression testing for robot policies — manipulation in simulation, and a handwriting robot from simulation to real recordings.**

Robot CI treats robot policies like software versions. A *baseline* and a *candidate* policy run under identical conditions (same tasks, same seeds); their behaviour is measured with several metrics; configurable thresholds flag regressions; results are stored as JSON and shown in a Streamlit dashboard. For the handwriting robot, the same plan the simulator executes is exported to the real robot, and a phone recording of the result is compared with the simulation.

---

## What is verified (honest status)

| Capability | Implemented | Unit tested | End-to-end run | Real data | Live robot |
|---|---|---|---|---|---|
| Manipulation regression (PyBullet + mock, 5 scenarios, 6 metrics) | ✅ | ✅ | ✅ | – | – |
| Trial variability (bootstrap 95% CI) for `--trials N` | ✅ | ✅ | ✅ | – | – |
| Handwriting: font (A–Z, 0–9), 12 targets, 7 writing policies | ✅ | ✅ | ✅ | – | – |
| Handwriting simulation: kinematic plotter + PyBullet arm (GUI + MP4) | ✅ | ✅ | ✅ | – | – |
| Handwriting metrics + paired regression detection | ✅ | ✅ | ✅ | – | – |
| Export for real robot (SVG, CSV, plan JSON; G-code once profile is filled) | ✅ | ✅ | ✅ | – | n/a (files only) |
| Upload recording → real-vs-sim comparison (photo / video) | ✅ | ✅ (synthetic) | ✅ (synthetic self-test) | **pending lab recording** | n/a |
| Evidence-based diagnosis (5 categories) | ✅ | ✅ | ✅ (synthetic) | pending | – |
| Video marker tracking / shape-only video-vs-sim (manipulation) | ✅ | ✅ | ✅ (synthetic) | not yet | – |
| Hardware readiness checker | ✅ | ✅ | ✅ (reports BLOCKED — correct) | – | – |
| ROS 2 adapter | scaffold only (no publishers/subscribers) | mappings only | – | – | – |

"Real data" for the handwriting robot needs exactly one more step: recording the robot drawing an exported plan and uploading it (see **Lab day** below).

---

## Install (Windows PowerShell)

```powershell
cd "C:\Users\Aishwarya Kowta\Desktop\robot-ci"
python -m venv .venv                 # skip if .venv already exists
.venv\Scripts\python.exe -m pip install -e ".[dev,mcap]"
.venv\Scripts\python.exe -m pytest -q          # expect: 220 passed
```

`requirements-tested.txt` lists the exact versions the suite was verified with (use only to rebuild a broken environment).

Run every command from the repository root (targets, configs and results use relative paths).

---

## 5-minute demo

```powershell
# 1. Manipulation regression in PyBullet (prints REGRESSION_DETECTED: the candidate is deliberately degraded)
.venv\Scripts\python.exe -m robot_ci run --simulator pybullet --trials 3

# 2. Watch the handwriting robot write in PyBullet (opens a window)
.venv\Scripts\python.exe -m robot_ci writing-sim --target word_robot --policy writing_baseline --gui

# 3. Handwriting regression: baseline vs candidate, all 12 targets, 3 trials
.venv\Scripts\python.exe -m robot_ci writing-run --backend kinematic --trials 3

# 4. Prove the upload pipeline works (synthetic phone photo + video with a known 1.2 mm offset)
.venv\Scripts\python.exe -m robot_ci writing-selftest

# 5. Dashboard (Sections A-D)
.venv\Scripts\python.exe -m robot_ci dashboard
```

---

## Part 1 — Manipulation regression (original framework)

* **Policies** (`robot_ci/policies/`): `baseline` is a rule-based waypoint (proportional) controller. It contains an untrained, hand-initialised network hook whose effect on actions is negligible (≤ 0.00004 m per step), so it is not a learned policy. `candidate` is a deliberately degraded copy (lower gain, noise, bias, no lift) used as an **injected regression**.
* **Scenarios** (`scenarios/`): `reach_simple`, `reach_diagonal`, `pick_and_place_easy|medium|hard`. Initial positions are fixed; seeds change noise.
* **Simulators**: PyBullet Franka Panda (grasp = kinematic attachment), and a fast mock.
* **Metrics**: success rate, completion time (steps), trajectory deviation, collision count (steps in contact with obstacle zones), action difference, reward difference.
* **Detection**: `config/regression_thresholds.yaml` (validated on load). With `--trials N` the run also stores per-metric spread and a bootstrap 95% CI of the change.
* **Video comparison** (`extract-trajectory`, `compare-video-sim`): colour-marker tracking; shape-only mode normalises both paths with one uniform scale, flips image *v* to point up (`--image-axes`), aligns with rotation-only Procrustes and reports a separate mirror check. `--source` is required and fixtures are always labelled `synthetic_fixture`; same-policy claims need `--same-policy-on-robot` and are recorded as user-declared.

---

## Part 2 — Handwriting robot

### Pipeline

```
target (YAML) ─► writing policy ─► PenPlan (waypoints, pen up/down, feed, SHA-256 hash)
                                     │
             ┌───────────────────────┼─────────────────────────────┐
             ▼                       ▼                             ▼
   simulate (kinematic /     regression: baseline vs        export for robot
   PyBullet arm, GUI/MP4)    candidate, paired stats        (SVG / G-code / CSV + plan.json)
             │                       │                             │ robot draws it, phone records
             ▼                       ▼                             ▼
     metrics per run   ──►   findings + CIs + images     upload photo/video ─► page detection ─►
                                     │                   homography (mm) ─► ink skeleton ─►
                                     │                   compare with intended + simulated ink
                                     └──────────────► diagnosis (5 categories) ◄───┘
```

### Targets (`writing_targets/*.yaml`)
Words `HELLO`, `ROBOT`; digits `0–9`; alphabet `A–M`, `N–Z`; patterns square, circle, triangle, star, spiral, zigzag, sine wave. Page frame: **mm, origin at the paper's top-left, x right, y down** (the same frame as a straightened photo). Add a target by copying a YAML file.

### Writing policies (`robot_ci/writing/policies.py`)
Rule-based stroke planners (not learned models). `writing_baseline` follows the font faithfully (1 mm waypoints, 30 mm/s drawing, 60 mm/s travel). Candidates contain **injected defects**: `writing_candidate` (slower, corner cutting, 0.25 mm tremor, small offset, skipped pen lifts), plus single-defect variants `_offset`, `_jitter`, `_slow`, `_cornercut`, `_drag`. The manipulation policies are untouched.

### Simulators (`robot_ci/writing/simulate.py`)
* **kinematic** — plotter model: first-order servo, speed/acceleration limits, latency, noise. ~0.02 s per run.
* **pybullet** — Franka Panda **stand-in arm** holding a 10 cm pen over an A4 sheet: IK every 1/60 s, velocity-limited joint servo, pen-down by tip height. Joint-limit entries and robot–table contact are counted. `--gui` draws ink live; `--record file.mp4` saves a video (off-screen). The real robot's model is unknown, so arm-specific numbers are simulation properties.

### Metrics (`robot_ci/writing/metrics.py`)
| Metric | Definition | Better |
|---|---|---|
| Trajectory error (mean / max, mm) | distance from each ink point to the nearest point of the intended path | lower |
| Path coverage | share of the intended path with ink within 1 mm | higher |
| Writing accuracy | share of characters / pattern recognised by a template recogniser over the robot's own font (not general OCR) | higher |
| Execution time (s) | first to last motion sample | lower |
| Path efficiency | measured pen-down length / intended length; regression uses \|ratio − 1\| | closer to 1 |
| Repeatability (mm) | mean pairwise RMS distance between repeated runs of the same policy + target | lower |
| Safety events | joint-limit entries + unexpected contacts + stall-like tracking failures (where detectable) | lower |

### Regression rule (`config/writing_thresholds.yaml`)
Paired over (target, trial) with matched seeds. A metric regresses only if the mean change in the bad direction ≥ threshold **and**, with ≥ 3 trials, the paired bootstrap 95% CI excludes zero — so one noisy run cannot trigger a regression.

### Real-to-sim analysis (`robot_ci/writing/analysis.py`)
1. Page image = photo, or the sharpest video frame after motion stops.
2. Paper corners detected automatically (or given manually) → perspective transform to page mm. Valid because writing is planar, so one uncalibrated phone gives millimetre measurements on the paper.
3. Ink → adaptive threshold → skeleton → points in mm. Page orientation is chosen by best match to the target.
4. Rigid registration separates **paper placement offset** from **stroke-shape error**; raw and aligned numbers are both reported.
5. Compared against the intended path **and** the simulation of the **same plan**; video gives execution time (motion energy); an optional colour marker on the pen holder gives an approximate motion track.
6. Not measurable from video (stated in every report): pen force, joint angles, motor stalls.

The synthetic self-test injects a 1.2 / −0.8 mm offset into a rendered, perspective-distorted phone photo and video; the pipeline recovers it within ~0.2–0.3 mm and reads all characters.

### Diagnosis (`robot_ci/writing/diagnosis.py`)
`POLICY_REGRESSION_LIKELY`, `SIM_TO_REAL_MISMATCH_LIKELY`, `BOTH_ISSUES_LIKELY`, `NO_REGRESSION_DETECTED`, `INCONCLUSIVE_INSUFFICIENT_EVIDENCE`. Only a declared robot recording counts as physical evidence; policy-level sim-to-real conclusions need `--executed-plan` (the robot drew the exported file, not its own font); poor page detection or low ink coverage → inconclusive.

---

## Lab day (recording the real robot)

See **[docs/LAB_DAY.md](docs/LAB_DAY.md)** for the full checklist. In short:

```powershell
# before the lab: export the plans you will draw
.venv\Scripts\python.exe -m robot_ci writing-export --target word_hello --policy writing_baseline
.venv\Scripts\python.exe -m robot_ci writing-export --target word_hello --policy writing_candidate
# at the lab: fill in config/writing_robot_profile.yaml, draw each drawing.svg (or .gcode), record + photograph
# after: upload in Dashboard > Section D > Upload recording, or
.venv\Scripts\python.exe -m robot_ci writing-analyze --plan results/writing/exports/<folder>/plan.json --video run.mp4 --photo page.jpg --source robot_phone_recording --executed-plan --run results/writing/run_<id>.json
```

---

## CLI reference

| Command | Purpose |
|---|---|
| `run` | manipulation baseline-vs-candidate regression (`--simulator pybullet|mock`, `--trials`) |
| `demo` | PyBullet GUI rollout of a manipulation policy |
| `dashboard` | Streamlit dashboard (Sections A–D) |
| `analyze-telemetry` | offline JSON/JSONL/MCAP(JSON-encoded) telemetry analysis |
| `extract-trajectory` | marker trajectory from a video (`--source` required, `--roi`, `--max-jump`) |
| `compare-video-sim` | video trajectory vs simulation (`--image-axes`, `--same-policy-on-robot`) |
| `check-hardware` | hardware readiness audit (never moves anything) |
| `writing-targets` | list handwriting targets |
| `writing-sim` | simulate one writing policy (`--gui`, `--record`, `--backend`) |
| `writing-run` | handwriting regression (`--baseline`, `--candidate`, `--targets`, `--trials`, `--backend`) |
| `writing-export` | export plan files for the real robot |
| `writing-analyze` | analyse a robot recording against simulation (+ diagnosis with `--run`) |
| `writing-diagnose` | diagnosis from saved run / analysis JSON |
| `writing-selftest` | synthetic end-to-end check of the upload pipeline |

---

## Project structure

```
robot_ci/
  pipeline/ policies/ scenarios/ simulation/ replay/ evaluation/ regression/ results/   manipulation framework
  telemetry/ transport/ hardware/ vision/ diagnosis/                                    telemetry, video, readiness, diagnosis
  stats.py                                                                              bootstrap CIs
  writing/  font targets plan policies simulate metrics regression export analysis diagnosis selftest render cli
dashboard/app.py, dashboard/writing_section.py
config/   regression_thresholds.yaml  writing_thresholds.yaml  writing_robot_profile.yaml  example_franka_panda_config.json
scenarios/  writing_targets/  tests/  docs/
```

---

## Limitations

* All manipulation results are simulation-only; scenarios have fixed initial positions; the candidate is an injected regression.
* Handwriting simulation uses a stand-in Franka Panda; real-robot kinematics are unknown until the lab.
* Real-to-sim from a phone measures geometry on the paper well; it cannot measure pen force, joint angles or stalls. If the robot imports SVG through a vendor app, its own speed planning makes execution time non-comparable (reported, not compared).
* Writing accuracy is template recognition over the robot's own font, not general OCR.
* Diagnosis is rule-based evidence interpretation, not causal inference.
* ROS 2 integration is a scaffold; Robot CI never drives hardware directly.

## Future work (priority order)

1. Record and analyse baseline and candidate on the real robot (several repeats each) → real repeatability and sim-to-real numbers.
2. Robot-specific kinematic model in PyBullet once the model is known.
3. Scenario randomisation for manipulation; per-scenario repeated trials.
4. Live telemetry (vendor SDK or ROS 2) only if the robot exposes it.
