# Robot CI

**A Continuous Integration Framework for Regression Testing of Robotic Manipulation Policies with Sim-to-Real Reliability Assessment**

---

## Project Purpose

Modern robots increasingly rely on policies learned from data. When these policies are retrained or fine-tuned, their behaviour can change — a new checkpoint may unintentionally break manipulation behaviour that a previous version handled successfully.

**Robot CI** brings software-style regression discipline to robotic manipulation:

```
Policy Update → Automated Replay → Regression Detection → Confidence Assessment → Deployment Decision
```

The central question Robot CI answers is:

> *"Did the policy actually get worse, and how much should we trust this failure?"*

---

## Current Milestone: Milestone 1

**Core Software Regression Pipeline** — the first complete vertical slice of the Robot CI architecture.

```
Policy → Scenario Corpus → Replay → Behavioural Evaluation → Regression Detection → Results → Dashboard
```

### Implemented Now ✅

| Module | Status | Description |
|--------|--------|-------------|
| Policy Manager | ✅ | PyTorch policy interface with baseline and candidate policies |
| Scenario / Test Manager | ✅ | YAML-based reusable test corpus (5 scenarios) |
| Mock Simulator | ✅ | Lightweight 3D kinematics simulator for development |
| Replay Engine | ✅ | Automated rollout execution with deterministic seeds |
| Evaluation Engine | ✅ | 6 behavioural metrics computed from actual rollout data |
| Regression Detector | ✅ | Configurable threshold-based regression detection |
| Results Data Model | ✅ | Extensible JSON storage with future-proof fields |
| Pipeline Runner + CLI | ✅ | End-to-end pipeline orchestration |
| Streamlit Dashboard | ✅ | Visual results dashboard reading stored data |

### Future Milestones ⏳

| Milestone | Module | Status |
|-----------|--------|--------|
| 2 | Robust metrics + experiment framework | ⏳ |
| 3 | Sim-to-Real Analyzer | ⏳ |
| 4 | Confidence / Reliability Engine | ⏳ |
| 5 | Deployment Decision + Dashboard polish | ⏳ |
| 6 | Isaac Sim integration | ⏳ |
| 7 | GitHub Actions + Docker CI | ⏳ |
| 8 | ROS 2 / MoveIt 2 integration | ⏳ |
| 9 | 6-DOF physical arm validation | ⏳ |

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Robot CI Pipeline                  │
├──────────────┬───────────────┬───────────────────────┤
│ Policy       │ Scenario      │ Mock Simulator         │
│ Manager      │ Manager       │ (future: Isaac Sim)    │
├──────────────┴───────────────┴───────────────────────┤
│                  Replay Engine                        │
├──────────────────────────────────────────────────────┤
│               Evaluation Engine                       │
│  ┌────────┬────────┬──────────┬─────────┬──────────┐ │
│  │Success │Complet.│Trajectory│Collision│ Action   │ │
│  │Rate    │Time    │Deviation │Count    │ Diff     │ │
│  └────────┴────────┴──────────┴─────────┴──────────┘ │
│  ┌──────────┐                                        │
│  │ Reward   │                                        │
│  │ Diff     │                                        │
│  └──────────┘                                        │
├──────────────────────────────────────────────────────┤
│             Regression Detector                       │
├──────────────────────────────────────────────────────┤
│    [Future: Sim-to-Real → Confidence → Decision]     │
├──────────────────────────────────────────────────────┤
│            Results Store + Dashboard                  │
└──────────────────────────────────────────────────────┘
```

---

## Module Responsibilities

### Policy Manager (`robot_ci/policies/`)

- **PolicyInterface** — Abstract base class defining the standard policy API
- **BaselinePolicy** — PyTorch MLP with hand-initialized weights for proportional control
- **CandidatePolicy** — Same architecture with intentionally degraded weights (noise injection)
- **Registry** — Name-based policy loading (`get_policy("baseline")`)

### Scenario / Test Manager (`robot_ci/scenarios/`)

- **Scenario** — Dataclass defining test case: initial state, target, environment, criteria
- **ScenarioManager** — Loads and validates YAML scenario files

### Mock Simulator (`robot_ci/simulation/`)

- **SimulatorBackend** — Abstract interface (future Isaac Sim will implement this)
- **MockSimulator** — Lightweight 3D kinematics with collision detection, grasping, reward

> **Note**: The mock simulator is a development tool. It does NOT simulate realistic robot physics. It validates the Robot CI software architecture before later Isaac Sim integration.

### Replay Engine (`robot_ci/replay/`)

- **Rollout** — Complete record of a single episode (observations, actions, states, rewards)
- **ReplayEngine** — Runs a policy against scenarios and records full rollout data

### Evaluation Engine (`robot_ci/evaluation/`)

Six behavioural metrics computed from actual rollout data:

| Metric | What it measures |
|--------|-----------------|
| **Success Rate** | Percentage of scenarios completed successfully |
| **Completion Time** | Mean steps to completion (or max if failed) |
| **Trajectory Deviation** | RMS Euclidean deviation between baseline/candidate trajectories (with resampling) |
| **Collision Count** | Total collision/safety violations |
| **Action Difference** | Mean L2 difference between action sequences |
| **Reward Difference** | Change in cumulative reward |

#### Trajectory Deviation Design

Because baseline and candidate episodes may have different lengths, trajectories are **resampled to a common number of points** (default: 100) using linear interpolation before computing the RMS Euclidean deviation. This approach is modular — Dynamic Time Warping (DTW) or other methods can be added as alternative metrics later.

### Regression Detector (`robot_ci/regression/`)

- Compares baseline vs candidate metrics against configurable thresholds
- Assigns severity: `none`, `minor`, `major`, `critical`
- Thresholds stored in `config/regression_thresholds.yaml`

### Results (`robot_ci/results/`)

- **TestRunResult** — Extensible data model with future fields for sim-to-real, confidence, deployment recommendation
- **ResultStore** — JSON file persistence

---

## Installation

```bash
# Clone and navigate to the project
cd robot-ci

# Install in development mode
pip install -e ".[dev]"
```

### Dependencies

- Python ≥ 3.9
- PyTorch ≥ 2.0
- NumPy ≥ 1.24
- Pandas ≥ 2.0
- PyYAML ≥ 6.0
- Streamlit ≥ 1.24
- Matplotlib ≥ 3.7
- pytest ≥ 7.0 (dev)

---

## How to Run the Pipeline

```bash
# Run the full regression testing pipeline
python -m robot_ci run --baseline baseline --candidate candidate --scenarios scenarios/

# With custom options
python -m robot_ci run \
    --baseline baseline \
    --candidate candidate \
    --scenarios scenarios/ \
    --seed 42 \
    --results-dir results/ \
    --config-dir config/
```

The pipeline will:
1. Load baseline and candidate policies
2. Load all test scenarios from the YAML corpus
3. Run baseline rollouts (deterministic, seeded)
4. Run candidate rollouts (same seeds)
5. Calculate 6 behavioural metrics from actual rollout data
6. Detect regressions against configurable thresholds
7. Save structured results as JSON
8. Print a formatted summary

### Seed Strategy

Each scenario has a fixed seed. The **same seed** is used when running both baseline and candidate policies on that scenario, ensuring deterministic matched comparison. Seed values are stored in rollout metadata. The code is structured to support multiple seeds per scenario in future milestones.

---

## How to Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=robot_ci

# Run a specific test
pytest tests/test_pipeline_e2e.py -v
```

### End-to-End Test

The `test_pipeline_e2e.py` test runs the complete pipeline with an intentionally degraded candidate policy and verifies that regression is **actually detected from measured behaviour** — not from hard-coded values.

---

## How to Launch Dashboard

```bash
# Via CLI
python -m robot_ci dashboard

# Or directly
streamlit run dashboard/app.py
```

The dashboard reads actual stored JSON results and displays:
- Metric comparison bar charts
- Regression findings table with severity
- Per-scenario rollout results
- Historical run selector

---

## Example Experiment

Running the pipeline with the built-in baseline and candidate policies:

```bash
python -m robot_ci run --baseline baseline --candidate candidate --scenarios scenarios/
```

**Expected behaviour**: The candidate policy has intentionally degraded weights (noise injection), which produces:
- Lower success rate
- Longer completion times
- Greater trajectory deviation
- More collisions
- Different action distributions
- Lower cumulative reward

The regression detector identifies these drops and reports them with severity levels.

---

## Project Structure

```
robot-ci/
├── README.md                          # This file
├── pyproject.toml                     # Project config and dependencies
│
├── config/
│   ├── regression_thresholds.yaml     # Configurable regression thresholds
│   └── simulator.yaml                 # Mock simulator defaults
│
├── robot_ci/
│   ├── __init__.py
│   ├── __main__.py                    # CLI entry point
│   │
│   ├── policies/                      # Policy Manager
│   │   ├── base.py                    # PolicyInterface ABC
│   │   ├── baseline.py               # BaselinePolicy (PyTorch)
│   │   ├── candidate.py              # CandidatePolicy (degraded)
│   │   └── registry.py               # Name-based loading
│   │
│   ├── scenarios/                     # Scenario / Test Manager
│   │   ├── models.py                  # Scenario dataclass
│   │   └── manager.py                # YAML loader + validator
│   │
│   ├── simulation/                    # Simulation Backend
│   │   ├── backend.py                # SimulatorBackend ABC
│   │   └── mock.py                   # MockSimulator
│   │
│   ├── replay/                        # Replay Engine
│   │   ├── rollout.py                # Rollout data model
│   │   └── engine.py                 # ReplayEngine
│   │
│   ├── evaluation/                    # Evaluation Engine
│   │   ├── evaluator.py              # EvaluationEngine orchestrator
│   │   └── metrics/
│   │       ├── base.py               # Metric ABC + MetricResult
│   │       ├── success_rate.py
│   │       ├── completion_time.py
│   │       ├── trajectory_deviation.py
│   │       ├── collision.py
│   │       ├── action_difference.py
│   │       └── reward_difference.py
│   │
│   ├── regression/                    # Regression Detection
│   │   └── detector.py               # RegressionDetector
│   │
│   ├── results/                       # Results Management
│   │   ├── models.py                 # TestRunResult
│   │   └── store.py                  # JSON persistence
│   │
│   └── pipeline/                      # Pipeline Orchestration
│       └── runner.py                 # PipelineRunner
│
├── scenarios/                         # Test scenario corpus
│   ├── reach_simple.yaml
│   ├── reach_diagonal.yaml
│   ├── pick_and_place_easy.yaml
│   ├── pick_and_place_medium.yaml
│   └── pick_and_place_hard.yaml
│
├── results/                           # Pipeline output (gitignored)
│
├── dashboard/
│   └── app.py                        # Streamlit dashboard
│
└── tests/
    ├── conftest.py
    ├── test_policies.py
    ├── test_scenarios.py
    ├── test_simulator.py
    ├── test_replay.py
    ├── test_evaluation.py
    ├── test_regression.py
    ├── test_results.py
    └── test_pipeline_e2e.py
```

---

## Current Limitations

1. **Mock simulator only** — No realistic physics. The simulator validates the software architecture; Isaac Sim integration is planned for Milestone 6.
2. **No sim-to-real analysis** — The sim-to-real gap measurement (Wasserstein, KL, JS divergence) is deferred to Milestone 3.
3. **No confidence scoring** — The reliability/confidence engine is deferred to Milestone 4.
4. **No deployment decision** — The Promote/Investigate/Reject logic is deferred to Milestone 5.
5. **No physical hardware** — 6-DOF arm validation is deferred to Milestone 9.
6. **Single seed per scenario** — One deterministic seed per scenario. Multiple seeds for statistical robustness planned for Milestone 2.
7. **Simple policies** — The baseline and candidate are simple PyTorch MLPs, not state-of-the-art manipulation policies.
8. **No CI/CD** — GitHub Actions and Docker deployment are deferred to Milestone 7.
9. **No ROS 2 / MoveIt 2** — Robot communication and motion planning integration deferred to Milestone 8.

---

## Future Milestones

The complete Robot CI vision extends this foundation with:

- **Sim-to-Real Analyzer**: Compare simulation and real-world distributions using Wasserstein distance, KL divergence, or Jensen-Shannon divergence
- **Confidence / Decision Engine**: Assess failure reliability by combining regression evidence with sim-to-real gap
- **Deployment Decision**: Automated Promote / Investigate / Reject recommendations
- **Isaac Sim**: High-fidelity physics simulation for policy testing
- **GitHub Actions + Docker**: Automated CI execution on policy push
- **ROS 2 + MoveIt 2**: Real robot communication and motion planning
- **6-DOF Arm Validation**: Physical hardware testing of selected scenarios

---

## License

University Project I — Fall Semester 2026–27
