# HW3 Report — 4b4g0077

---

## HW3-1: Deep Reinforcement Learning on Path Tracking

### Implementation

**PolicyNet (TODO 1)**
Built as a two-layer MLP with 64 hidden units and Tanh activations, followed by a `DiagGaussian` head that outputs a `FixedNormal` distribution over the action space.

```
State (14) → Linear(64) → Tanh → Linear(64) → Tanh → DiagGaussian → FixedNormal(μ, σ)
```

Tanh is chosen over ReLU because the bounded output stabilizes training for continuous control.  
All linear layers use orthogonal initialization to ensure well-conditioned gradient flow from the start.

**ValueNet (TODO 2)**
Same backbone as PolicyNet, with an additional `Linear(64 → 1)` output layer.  
`forward()` returns `self.main(state)[:, 0]` — squeezing to a scalar per sample.

**EnvRunner.run() (TODO 3)**
At each of the `n_step` steps:
1. Record current `states` and `dones` into the mini-batch buffers.
2. Use `policy_net` to sample `actions` and `log_probs` from the current policy, without gradient.
3. Use `value_net` to estimate `V(s)`.
4. Step the environment to obtain `rewards` and next `states`.

After the loop, `compute_gae()` computes GAE returns using `γ = 0.99`, `λ = 0.95`.

**PPO Clipped Loss (TODO 4)**
Already implemented in `agent.py`. The probability ratio is computed in log-space for numerical stability:

$$r_t(\theta) = \exp(\log\pi_\theta(a_t|s_t) - \log\pi_{\theta_\text{old}}(a_t|s_t))$$

The clipped surrogate objective:

$$L^{\text{CLIP}} = -\mathbb{E}\left[\min\left(r_t \hat{A}_t,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

with `ε = 0.2`. The value loss uses clipped MSE to prevent critic overfitting.

### Training Result

| Metric | Value |
|--------|-------|
| Total iterations | 30,000 |
| Initial mean return | −1155 |
| Final mean return | ~152 |
| **Evaluation Score** | **129.22** |

The reward increases monotonically over training, confirming the PPO implementation is correct.

---

## HW3-2: Deep Reinforcement Learning on Proly

### Reward Design (TODO 6)

**`calculate_flag_capture_reward()`**  
Compares `last_checkpoint_index` between consecutive frames. When the index increases (a new flag is captured), return `+50.0`. Otherwise return `0.0`.  
The large positive value ensures the agent treats flag capture as the dominant objective.

**`calculate_distance_reward()`**  
Computes the Euclidean distance to the target (`np.linalg.norm(target_position)`) in the current and previous frame.
- Getting closer → `+1.0`
- Getting farther → `−0.5`
- No change → `0.0`

The asymmetry (penalty smaller than reward) encourages exploration: the agent is not overly afraid of temporarily moving away when navigating around obstacles.

**`calculate_survival_reward()`**  
If `agent_health ≤ 0` (fell off the cliff or died), return `−30.0`.  
This teaches the agent that survival is necessary to accumulate flag rewards.

**`calculate_reward()` — composition**  
```python
total = flag_reward + distance_reward + survival_reward
```

| Component | Value | Purpose |
|-----------|-------|---------|
| Flag capture | +50 | Primary sparse signal: task progress |
| Distance (closer) | +1 | Dense guidance toward next flag |
| Distance (farther) | −0.5 | Soft penalty to discourage aimless wandering |
| Survival (death) | −30 | Prevent cliff-jumping as an escape strategy |

The flag reward dominates all other per-step signals, so the agent consistently prioritizes reaching the next checkpoint over any local incentive.
