# Robotic Navigation and Exploration

[![Python](https://img.shields.io/badge/Python-3.10+-%233776AB?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2-%23EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1-%23008080)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Deep Reinforcement Learning assignments for the **NTHU CS Robotic Navigation and Exploration** course. Implements PPO-based path tracking and RL-based autonomous navigation agents.

## Contents

| Part | Topic | Description |
|------|-------|-------------|
| HW3-1 | Path Tracking with PPO | Proximal Policy Optimization for trajectory following |
| HW3-2 | Proly RL Navigation | Curriculum learning for autonomous navigation |
| Report | Analysis & Results | HW3-1 + HW3-2 performance analysis (PDF) |

## Key Implementations

### HW3-1 — Path Tracking

- **PolicyNet / ValueNet** — Neural architectures for PPO agent
- **EnvRunner** — Gym-style multi-environment wrapper
- **PPO Clipped Loss** — Surrogate objective with entropy bonus
- **Reward Engineering** — Custom reward functions for navigation
- **Kinematic Simulators** — Basic, bicycle, and differential drive models

### HW3-2 — Proly Navigation

- **Curriculum Random Training** — Progressive difficulty scheduling
- **Champion Play** — Best-model evaluation vs strong opponents
- **Auto Optimization** — Shell-based hyperparameter search

## Quick Start

```bash
# HW3-1: Path tracking
cd HW3-1
python train.py

# HW3-2: Navigation agent
cd HW3-2
python rl_play.py
```

## Project Structure

```
Robotic-Navigation-and-Exploration/
├── HW3-1/                  # PPO Path Tracking
│   ├── PathTracking/       # Pure pursuit controller
│   ├── Simulation/         # Kinematic simulators
│   ├── agent.py            # PPO agent loop
│   ├── model.py            # Policy/Value networks
│   ├── multi_env.py        # Parallel env runner
│   ├── train.py            # Training entry point
│   └── cubic_spline.py     # Spline path generation
├── HW3-2/                  # RL Navigation
│   ├── rl_play.py          # Test trained model
│   ├── train_curriculum_random.sh
│   └── champion_play.py    # Best-model evaluation
├── HW3/                    # Original submission archive
│   ├── report_4b4g0077.md
│   └── report_4b4g0077_HW3-*.pdf
└── README.md
```

## Dependencies

- Python 3.10+
- PyTorch 2.x
- Gymnasium
- NumPy, Matplotlib

## License

MIT
