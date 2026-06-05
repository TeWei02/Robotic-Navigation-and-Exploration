# Robotic Navigation and Exploration — HW3

Deep Reinforcement Learning assignments for NTHU CS Robotic Navigation and Exploration course.

## Contents

| Part | Topic | Weight |
|------|-------|--------|
| HW3-1 | Path Tracking with PPO | 40% |
| HW3-2 | Proly — RL-based Navigation | 50% |
| Report | Analysis & Results | 10% |

## Key Implementations

- **PolicyNet / ValueNet** — Neural network architectures for PPO agent
- **EnvRunner** — Gym-style environment wrapper for training loop
- **PPO Clipped Loss** — Proximal Policy Optimization with clipping
- **Reward Engineering** — Custom reward functions for navigation tasks

## Quick Start

```bash
cd HW3-1
python train.py
```

```bash
cd HW3-2
python rl_play.py
```

## Dependencies

- Python 3.10+
- PyTorch
- Gym / Gymnasium
- NumPy, Matplotlib

## License

MIT