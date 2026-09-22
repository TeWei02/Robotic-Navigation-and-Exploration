# Robotic Navigation and Exploration

[![Live demo](https://img.shields.io/badge/live%20demo-GitHub%20Pages-2ea44f)](https://tewei02.github.io/Robotic-Navigation-and-Exploration/)
[![CI](https://github.com/TeWei02/Robotic-Navigation-and-Exploration/actions/workflows/ci.yml/badge.svg)](https://github.com/TeWei02/Robotic-Navigation-and-Exploration/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-%233776AB?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-%23EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Deep reinforcement learning for path tracking and autonomous navigation, submitted for the
**TAICA AI Satellite Course — Robotic Navigation and Exploration**. The repository contains a
PPO path-tracking stack (HW3-1), a curriculum-learning navigation agent built on the external
Proly simulator (HW3-2), and the archived assignment report.

**Online version:** <https://tewei02.github.io/Robotic-Navigation-and-Exploration/>

The published page is a browser replay of the `HW3-1` path-tracking environment. It re-implements
the course environment and the trained-agent interface in JavaScript, so the environment dynamics
can be inspected without a Python toolchain, and it verifies itself against the repository by
re-running the same reference episodes in the browser.

---

## Table of contents

- [Online demo](#online-demo)
- [Repository contents](#repository-contents)
- [Environment and interfaces](#environment-and-interfaces-hw3-1)
- [Quick start](#quick-start)
- [Reproducibility and verification](#reproducibility-and-verification)
- [Data and model availability](#data-and-model-availability)
- [Project structure](#project-structure)
- [Tech stack](#tech-stack)
- [License](#license)

---

## Online demo

| Item | Value |
|------|-------|
| URL | <https://tewei02.github.io/Robotic-Navigation-and-Exploration/> |
| Source | `docs/` on the `main` branch, served by GitHub Pages |
| Offline | installable PWA (`docs/manifest.json`, `docs/sw.js`, 8 cached assets) |
| Runtime | vanilla JavaScript + Canvas, no build step, no third-party libraries |

The page offers six pre-computed episodes: `corridor-turn`, `wide-sweep`, `tight-arc`,
`offset-entry`, `biased-driver` and `recovery`. Each episode can be played in two ways:

- **Replay** — plays back the recorded command sequence, which is exactly what the Python
  generator produced.
- **Closed loop** — the JavaScript controller drives the environment step by step, so the
  trajectory reflects the browser implementation rather than the recording.

A self-check runs on load: the browser replays the reference commands and compares the resulting
trajectory, reward and observation stream against the committed data set. The result of that
comparison is shown on the page, and it is the same comparison that `tools/build_web_data.py --check`
performs in Python.

Honest scope of the demo:

- The control sequences come from the **deterministic pure-pursuit baseline** in
  `HW3-1/PathTracking/pure_pursuit.py`, not from the trained PPO policy. Training writes
  checkpoints to a local `save/` directory that is not distributed with this repository, so a
  recorded PPO rollout cannot be published here.
- The environments are the `basic` (unicycle) kinematic model with the reward, termination
  and observation layout of `HW3-1/wrapper.py`.
- The page reports environment behaviour. It does not claim to reproduce the training curves or
  the scores quoted in the archived report.

---

## Repository contents

| Path | Content | Status |
|------|---------|--------|
| `HW3-1/` | PPO path tracking: `model.py` (PolicyNet / ValueNet), `env_runner.py` (GAE), `multi_env.py`, `agent.py`, `train.py`, `wrapper.py`, `cubic_spline.py`, `eval_score_check.py` | code complete and runnable |
| `HW3-1/PathTracking/` | `Controller` base class, `utils.search_nearest`, and the deterministic `PurePursuitController` baseline | runnable |
| `HW3-1/Simulation/` | Kinematic models (basic, bicycle, differential drive) and their simulators | runnable |
| `HW3-2/` | Curriculum navigation agent: `train_curriculum_random.sh`, `champion_play.py`, `auto_optimize.sh`, `rl_play.py`, `dummy_env.py`, `PROLY_SETUP.md` | requires the external Proly runtime, see below |
| `HW3/` | Archived original submission, including `report_4b4g0077.md` and the report PDF | read-only record |
| `docs/` | Published static site: `index.html`, `app.js`, `engine.js`, `data/reference.json`, PWA assets | generated from repository code |
| `tools/` | `build_web_data.py` — regenerates and verifies the published data set | runnable |
| `tests/` | `pytest` suite for the data set, the baseline and the published assets | runnable |

### Assignment parts

**HW3-1 — path tracking with PPO.** The agent receives a 14-dimensional observation and emits a
normalised yaw-rate command for the `basic` kinematic model. The implementation includes the
clipped surrogate objective with an entropy bonus, generalised advantage estimation in
`EnvRunner`, and a multi-environment rollout collector.

**HW3-2 — navigation with curriculum learning.** A progressive-difficulty training script and a
best-model evaluation script that play against stronger opponents through the Proly simulator.
`PROLY_SETUP.md` documents how the simulator is obtained and installed; `get_proly.sh` fetches it.
This part cannot be executed from a clean clone of this repository, because the Proly application
and the checkpoints produced by training are not distributed here (see
[Data and model availability](#data-and-model-availability)).

---

## Environment and interfaces (HW3-1)

Framework: a Gym-style environment (`HW3-1/wrapper.py`) over the kinematic simulators in
`HW3-1/Simulation/`, with paths generated procedurally by `cubic_spline.cubic_spline_2d` from four
randomised control points.

| Item | Value |
|------|-------|
| State | `(x, y, yaw)`, plus a recorded pose history |
| Action | One normalised value in `[-1, 1]`, mapped to the yaw rate of the `basic` model; forward speed is fixed at `0.6 × v_range` |
| Observation (14) | Two previous poses `(x, y, yaw)` — positions divided by 600, yaw in radians — followed by four future waypoints (8 values) spaced 16 path indices apart, starting at the nearest path sample (also divided by 600) |
| Reward | `0.8·exp(−0.1·d²) + 0.2·exp(−0.1·Δψ²) + progress`, where `d²` is the squared distance to the nearest path sample, `Δψ` the wrapped heading error in degrees against that sample's heading, and `progress ∈ {+0.1, 0, −1.0}` for advancing, stalling and regressing along the path |
| Termination | Reaching the final path sample, coming within 10 units of the goal, or 400 steps |
| Limits | `v_range = 20`, `w_range = 60` (deg/s), `dt = 0.1` s, start pose sampled around `(200, 50)` within ±20 units |
| Determinism | Path generation and start pose depend on `numpy.random`; a fixed seed reproduces an episode exactly |

---

## Quick start

Python 3.10+ is required. No Node.js toolchain is needed: the demo is plain static files.

```bash
git clone https://github.com/TeWei02/Robotic-Navigation-and-Exploration
cd Robotic-Navigation-and-Exploration
python3 -m pip install -r requirements.txt

# 1. Rebuild and verify the published data set (compares against a live Python run)
python3 tools/build_web_data.py
python3 tools/build_web_data.py --check

# 2. Run the deterministic baseline against the course environment
python3 - <<'PY'
import numpy as np, sys
sys.path.insert(0, "HW3-1")
import wrapper
from PathTracking.pure_pursuit import PurePursuitController

np.random.seed(1)
env = wrapper.PathTrackingEnv()
ctrl = PurePursuitController(lookahead=30.0, v=12.0, w_range=60.0)
ctrl.set_path(env.path)

total = 0.0
for _ in range(400):
    _, w = ctrl.command(env.simulator.state.pose())
    _, reward, done, _ = env.step(np.array([w / 60.0]))
    total += reward
    if done:
        break
print("return =", round(total, 3))
PY

# 3. Run the test suite
python3 -m pytest -q

# 4. Serve the published site locally
python3 -m http.server 8000 --directory docs   # http://localhost:8000

# 5. Train the PPO agent (PyTorch; writes checkpoints to HW3-1/save/, not distributed here)
cd HW3-1 && python3 train.py
```

The baseline snippet prints `return = 232.06` after 297 steps on this machine, which is exactly
scenario 1 of the published data set — the Python generator and the snippet above agree.

### Reference runs in the published data set

Produced by `tools/build_web_data.py` with the deterministic pure-pursuit baseline above. They
document environment behaviour under six path/start conditions; they are not PPO results and not
a benchmark.

| # | Scenario | Control mode | Steps | Return | Termination |
|---|----------|--------------|-------|--------|-------------|
| 1 | `corridor-turn` | pure pursuit | 297 | 232.06 | goal |
| 2 | `wide-sweep` | pure pursuit | 289 | 261.55 | goal |
| 3 | `tight-arc` | pure pursuit | 331 | 290.81 | goal |
| 4 | `offset-entry` | pure pursuit | 306 | 224.98 | goal |
| 5 | `biased-driver` | steering bias injected | 340 | 42.41 | goal |
| 6 | `recovery` | start pose pushed off-path | 400 | 247.76 | step limit |

Trajectories, observations, rewards and per-step counters for all six scenarios are stored in
[`docs/data/reference.json`](docs/data/reference.json) and printed by
`python3 tools/build_web_data.py --check`. The file, not this table, is the source of truth.

---

## Reproducibility and verification

Every figure shown on the published page is generated by code in this repository. Three
independent checks guard that claim:

| Check | Command | Result |
|-------|---------|--------|
| Data set parity (Python) | `python3 tools/build_web_data.py --check` | all six scenarios rebuild with `max|Δ| = 0` against the committed file |
| Data set parity (browser) | open the published page | the in-page self-check replays each episode and reports the same comparison |
| Tests and lint | `python3 -m pytest -q`, `python3 -m ruff check .` | 9 tests pass, lint clean |

`tests/test_reference_data.py` covers the published data set structure, the honesty of the
provenance note, determinism of the baseline, the three control modes, and the integrity of the
published assets (web-app manifest, service-worker cache list, in-page scope notice, absence of
placeholder text). Continuous integration runs the same commands on every push
(`.github/workflows/ci.yml`).

`ruff` is configured in `pyproject.toml` and deliberately excludes the archived `HW3/` and
`HW3-1/` course submissions, which are kept in their original form.

---

## Data and model availability

The following are **not** distributed with this repository. Nothing on the published page or in
the tests is derived from them, and no number in this README is quoted from an unreproducible run.

| Missing artefact | Why it is absent | What this repository provides instead |
|------------------|------------------|----------------------------------------|
| PPO checkpoints (`HW3-1/save/model.pt`, value and optimiser states) | Written locally by `train.py`; weight files are excluded from the repository | The full training code, the environment, and a deterministic baseline that runs end to end |
| PPO training curves and evaluation scores | Produced during the original course run; the raw logs are not part of this repository | Instructions to retrain; a baseline episode that can be compared under identical conditions |
| HW3-2 checkpoints and opponent statistics | The agent trains inside the external Proly application, which is not redistributable here | `PROLY_SETUP.md` and `get_proly.sh` to obtain the simulator, plus the training and evaluation scripts |
| Report figures and tables | Fixed in the archived PDF report (`HW3/`), which was produced from the original training run | The report is kept as a read-only record and clearly separated from the regenerated material |
| External datasets | Not used: reference paths and start poses are generated procedurally inside the environment | `wrapper.gen_path()` is the complete generator; consequences are reproducible from a seed |

Only one external dependency is required to run the environment and regenerate the published data
set: `opencv-python-headless`, imported at module level by `HW3-1/Simulation/` even when nothing is
rendered. `torch`, `gymnasium` and `matplotlib` are needed only for training and evaluation, and
are intentionally not pinned in `requirements.txt`.

---

## Project structure

```
Robotic-Navigation-and-Exploration/
├── HW3-1/                       # PPO path tracking (active code)
│   ├── PathTracking/            # Controller base, nearest-point search, pure-pursuit baseline
│   ├── Simulation/              # basic / bicycle / differential-drive kinematics
│   ├── agent.py                 # PPO agent loop
│   ├── model.py                 # PolicyNet / ValueNet
│   ├── env_runner.py            # Rollout collection and GAE
│   ├── multi_env.py             # Environment batching
│   ├── wrapper.py               # Gym-style environment: reward, observation, termination
│   ├── cubic_spline.py          # Reference path generation
│   ├── eval_score_check.py      # Evaluation entry point
│   └── train.py                 # Training entry point (writes save/)
├── HW3-2/                       # Curriculum navigation with the external Proly simulator
│   ├── train_curriculum_random.sh
│   ├── champion_play.py
│   ├── auto_optimize.sh
│   ├── dummy_env.py
│   └── PROLY_SETUP.md
├── HW3/                         # Archived original submission and report (read-only)
├── docs/                        # Published static site (GitHub Pages, PWA)
│   ├── index.html
│   ├── app.js                   # Playback, closed-loop control, in-page parity self-check
│   ├── engine.js                # Environment, reward, controller, parity harness
│   ├── data/reference.json      # Committed reference episodes
│   └── manifest.json, sw.js, icons/
├── tools/build_web_data.py      # Regenerates and verifies the published data set
├── tests/test_reference_data.py # Data set, determinism and asset tests
├── pyproject.toml               # Ruff and pytest configuration
├── requirements.txt
└── LICENSE
```

---

## Tech stack

| Layer | Technology |
|-------|------------|
| Reinforcement learning | PyTorch (PPO with clipped surrogate and GAE), NumPy |
| Environment | Custom Gym-style wrapper over kinematic models, cubic-spline reference paths |
| Simulator integration (HW3-2) | Proly application, shell-driven curriculum and hyper-parameter search |
| Verification | pytest, ruff, Python/JavaScript parity checking |
| Published site | HTML, CSS, vanilla JavaScript, Canvas, web-app manifest and service worker |
| Hosting | GitHub Pages (`docs/`, `main` branch), GitHub Actions CI |

---

## License

Released under the MIT License — see [LICENSE](LICENSE).

## Author

Te-Wei Ko
