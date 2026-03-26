# HW3: Deep Reinforcement Learning — 實作說明

> **課程**：Robotic Navigation and Exploration（CS, NTHU）  
> **助教**：Min-Chun Hu  
> **截止日期**：2026/4/12 23:59（NTU COOL，硬截止）

---

## 目錄

- [作業概覽](#作業概覽)
- [HW3-1：Path Tracking（40%）](#hw3-1path-tracking40)
  - [TODO 1 — PolicyNet 架構](#todo-1--policynet-架構-modelpy)
  - [TODO 2 — ValueNet 架構](#todo-2--valuenet-架構-modelpy)
  - [TODO 3 — EnvRunner.run()](#todo-3--envrunnerrun-env_runnerpy)
  - [TODO 4 — PPO Clipped Loss](#todo-4--ppo-clipped-loss-agentpy)
  - [TODO 5 — 訓練超參數](#todo-5--訓練超參數-trainpy)
  - [如何訓練與評估](#如何訓練與評估)
- [HW3-2：Proly（50%）](#hw3-2proly50)
  - [TODO 6 — Reward Functions](#todo-6--reward-functions-rl_playpy)
  - [如何訓練 Proly](#如何訓練-proly)
- [Report（10%）](#report10)
- [提交格式](#提交格式)

---

## 作業概覽

| 部分 | 檔案 | 配分 | 說明 |
|------|------|------|------|
| HW3-1 | `model.py`, `env_runner.py`, `agent.py`, `train.py` | 40% | PPO 實作於 Path Tracking 環境 |
| HW3-2 | `rl_play.py` | 50% | 設計 Reward 讓 PPO agent 玩 Proly |
| Report | `report_<student_id>.pdf` | 10% | 各 5%，簡短說明實作 |

---

## HW3-1：Path Tracking（40%）

### 環境規格

| 項目 | 說明 |
|------|------|
| **Observation** | `o ∈ ℝ¹⁴`：位置（過去2步 + 當前）+ 最近路徑點 + 3個 waypoint |
| **Action** | 角速度 `a ∈ [-1, 1]`（1D 連續） |
| **Reward** | `r = 0.8·r_d + 0.2·r_y + r_p` |
| **終止條件** | 進度 100% 或到達 400 timesteps |

---

### TODO 1 — PolicyNet 架構 (`model.py`)

**目標**：建立 `self.main`（backbone）與 `self.dist`（輸出動作分佈）。

```python
# model.py — PolicyNet.__init__() 中
self.main = nn.Sequential(
    init_(nn.Linear(s_dim, 64)),  # 14 -> 64
    nn.Tanh(),
    init_(nn.Linear(64, 64)),     # 64 -> 64
    nn.Tanh(),
)
self.dist = DiagGaussian(64, a_dim, std)  # 輸出 Gaussian 分佈
```

**設計說明**：
- 兩層 64-unit MLP，使用 **Tanh** 激活（比 ReLU 更適合連續控制）
- `DiagGaussian` 輸出一個平均值 `μ` 和固定標準差 `σ`，形成 `FixedNormal` 分佈
- 訓練時隨機採樣（exploration），評估時取 `mode()`（deterministic）

---

### TODO 2 — ValueNet 架構 (`model.py`)

**目標**：建立 `self.main`，輸出單一 state value。

```python
# model.py — ValueNet.__init__() 中
self.main = nn.Sequential(
    init_(nn.Linear(s_dim, 64)),  # 14 -> 64
    nn.Tanh(),
    init_(nn.Linear(64, 64)),     # 64 -> 64
    nn.Tanh(),
    init_(nn.Linear(64, 1)),      # 64 -> 1（scalar value）
)
```

**設計說明**：
- 結構與 PolicyNet backbone 相同，但最後多一層輸出 1 個 scalar
- `forward()` 已寫好：`return self.main(state)[:, 0]`，會把 `(B,1)` squeeze 成 `(B,)`

---

### TODO 3 — EnvRunner.run() (`env_runner.py`)

**目標**：跑 `n_step` 步收集 trajectory 資料存入 mini-batch buffer。

```python
# env_runner.py — EnvRunner.run() 的 for loop 中
self.mb_states[step, :] = self.states          # 記錄當前 state
self.mb_dones[step, :]  = self.dones           # 記錄 done flag

states_tensor = torch.from_numpy(self.states).float().to(self.device)
with torch.no_grad():
    actions, a_logps = policy_net(states_tensor)   # 採樣動作
    values = value_net(states_tensor)               # 估計 state value

actions_np = actions.cpu().numpy()
self.mb_actions[step, :] = actions_np
self.mb_a_logps[step, :] = a_logps.cpu().numpy()
self.mb_values[step, :]  = values.cpu().numpy()

self.states, rewards, self.dones, info = self.env.step(actions_np)
self.mb_rewards[step, :] = rewards              # 記錄收到的 reward
```

**各 buffer 的 shape**：

| Buffer | Shape |
|--------|-------|
| `mb_states` | `(n_step, n_env, s_dim)` |
| `mb_actions` | `(n_step, n_env, a_dim)` |
| `mb_dones` | `(n_step, n_env)` |
| `mb_a_logps` | `(n_step, n_env)` |
| `mb_values` | `(n_step, n_env)` |
| `mb_rewards` | `(n_step, n_env)` |

---

### TODO 4 — PPO Clipped Loss (`agent.py`)

**目標**：實作 PPO 的 clipped surrogate policy gradient loss。

**數學公式**：

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

```python
# agent.py — PPO.train() 中
# 計算 probability ratio（用 log space 做再 exp，數值穩定）
ratio = torch.exp(sample_a_logps - sample_old_a_logps)

# 兩個 surrogate objective
surr1 = ratio * sample_advs
surr2 = torch.clamp(ratio, 1.0 - self.clip_val, 1.0 + self.clip_val) * sample_advs

# 最小化負的 clipped objective（= 最大化 objective）
pg_loss = -torch.min(surr1, surr2).mean()
```

**為什麼要 clip**：防止 policy 更新幅度過大，穩定訓練。

---

### TODO 5 — 訓練超參數 (`train.py`)

預設值（符合助教示例）：

```python
n_env         = 8       # 平行環境數（記憶體不足時減少）
n_step        = 128     # 每次 rollout 的步數
sample_mb_size = 64     # mini-batch 大小
sample_n_epoch = 4      # 每次 rollout 訓練幾個 epoch
a_std         = 0.5     # 動作分佈的標準差
lamb          = 0.95    # GAE λ
gamma         = 0.99    # discount factor
clip_val      = 0.2     # PPO clip epsilon ε
lr            = 1e-4    # learning rate
n_iter        = 30000   # 訓練總 iteration 數
device        = "cpu"
```

> **如果遇到 OSError: The paging file is too small**，把 `n_env` 降到 4 或 2。

---

### 如何訓練與評估

```bash
# 進入 HW3-1 目錄
cd HW3-1/

# 安裝相依套件
pip install torch numpy>=1.20,<2.0 matplotlib opencv-python cloudpickle

# 開始訓練（會自動存檔到 save/）
python train.py

# 推論（視覺化看車子跑）
python play.py

# 評估分數
python eval.py

# 畫 training curve
python plot.py
```

訓練完成後 `save/` 資料夾應有：
- `model.pt`：最新模型權重
- `return.txt`：訓練曲線數據
- `model-XXXXX.pt`：各 checkpoint

---

## HW3-2：Proly（50%）

### 環境規格

| 項目 | 說明 |
|------|------|
| **Observation** | `target_position`（到下個 checkpoint 的相對位置）、`agent_health`、`last_checkpoint_index`、`terrain_grid`（5×5 地形） |
| **Action** | 加速度 `(A_x, A_y) ∈ [-1, 1]²`（2D 連續） |
| **目標** | 依序捕獲所有 flag（checkpoint） |

---

### TODO 6 — Reward Functions (`rl_play.py`)

**核心原則**：好的 reward 設計 = 引導 agent 完成任務的關鍵。

#### calculate_flag_capture_reward()

```python
def calculate_flag_capture_reward(self):
    if self.prev_observation is None:
        return 0.0
    prev_checkpoint = self.prev_observation["last_checkpoint_index"]
    curr_checkpoint = self.observation["last_checkpoint_index"]
    if curr_checkpoint > prev_checkpoint:
        return 50.0   # 成功捕獲 flag → 大獎勵
    return 0.0
```

#### calculate_distance_reward()

```python
def calculate_distance_reward(self):
    if self.prev_observation is None:
        return 0.0
    prev_distance = np.linalg.norm(self.prev_observation["target_position"])
    curr_distance = np.linalg.norm(self.observation["target_position"])
    if curr_distance < prev_distance:
        return 1.0    # 靠近目標 → 小獎勵（dense reward）
    elif curr_distance > prev_distance:
        return -0.5   # 遠離目標 → 小懲罰
    return 0.0
```

#### calculate_survival_reward()

```python
def calculate_survival_reward(self):
    if self.observation["agent_health"] <= 0:
        return -20.0  # 掉下懸崖 → 大懲罰
    return 0.0
```

#### calculate_reward()（主函數）

```python
def calculate_reward(self):
    flag_reward     = self.calculate_flag_capture_reward()
    distance_reward = self.calculate_distance_reward()
    survival_reward = self.calculate_survival_reward()
    return flag_reward + distance_reward + survival_reward
```

**Reward 設計邏輯**：

| 事件 | Reward | 理由 |
|------|--------|------|
| 捕獲 flag | +50 | 明確的任務進度信號，量大讓 agent 優先追求 |
| 靠近目標 | +1 | Dense reward，避免 sparse 問題，加快學習 |
| 遠離目標 | -0.5 | 非對稱懲罰（比獎勵小），不讓 agent 因懼怕懲罰而靜止不動 |
| 死亡（health=0） | -20 | 教 agent 避開懸崖，量介於 flag 獎勵和 step reward 之間 |

---

### 如何訓練 Proly

```bash
# 安裝套件
pip install gymnasium==1.2.3 numpy<2.0.0 mlgame3d==0.8.0 \
            stable-baselines3==2.7.1 tensorboard==2.20.0 \
            setuptools<81.0 torch==2.10.0

# 下載 Proly 執行檔（選對應 OS）
# https://github.com/PAIA-Playful-AI-Arena/Proly/releases/tag/1.4.0-beta.1
# macOS → Proly.app, Linux → Proly.x86_64, Windows → Proly.exe

# 訓練（map 1/2/3 輪流跑效果更好）
python -m mlgame3d -i rl_play.py -i hidden -i hidden -i hidden \
       -e 10000 -gp items 0 -gp audio false \
       -gp map 1 -gp checkpoint 10 -gp max_time 120 Proly.app

# 推論（用訓練好的 model.zip）
python -m mlgame3d -i model_play.py -i hidden -i hidden -i hidden \
       -e 10000 -gp items 0 -gp audio false \
       -gp map 3 -gp checkpoint 10 -gp max_time 120 Proly.app

# 加速訓練（無渲染 + 加速時間）
# 加入 -ng -ts 10 讓訓練快 10 倍
```

訓練完會在 `HW3-2/` 下產生：
- `model.zip`：最新模型（提交用）
- `models/<timestamp>/`：各 checkpoint
- `tensorboard/`：訓練指標（用 `tensorboard --logdir tensorboard/` 開啟）

---

## Report（10%）

每份 report 各 5%，**寫太多不會加分**，簡短說明即可。

**建議格式**（每份 1 頁以內）：

```
HW3-1 Report
- PolicyNet：兩層 64-unit MLP + Tanh，輸出 DiagGaussian 分佈
- ValueNet：相同結構輸出 scalar value
- PPO Loss：使用 clipped surrogate，ε = 0.2
- 訓練結果：Evaluation Score = ？（目標 > 120 拿滿分）

HW3-2 Report
- Reward 設計：flag capture (+50) + distance (+1/-0.5) + survival (-20)
- 設計理由：flag reward 主導方向，distance reward 提供 dense signal，
            survival reward 防止 agent 亂跳
```

---

## 提交格式

提交 `HW3_<學號>.zip`，結構如下：

```
HW3_<your_student_id>.zip
├── HW3-1/
│   ├── save/
│   │   ├── model.pt        ← 訓練好的模型權重
│   │   └── return.txt      ← 訓練曲線數據
│   ├── agent.py
│   ├── env_runner.py
│   ├── model.py
│   ├── train.py
│   └── report_<student_id>.pdf
└── HW3-2/
    ├── model.zip           ← 訓練好的 Proly 模型
    ├── model_play.py
    ├── rl_play.py
    └── report_<student_id>.pdf
```

> **注意**：不要提交 `plot.py`、`eval.py`，助教會用原版評分。  
> 多餘檔案每個扣 5 分。

---

## 評分標準

### HW3-1（40%）

| 項目 | 配分 | 說明 |
|------|------|------|
| `plot.py` (return.txt) | 10% | reward 曲線必須上升 |
| `eval.py` 評估分數 | 30% | `30 × ES / 120`，ES > 120 拿滿 |
| Report | 5% | 簡短說明實作 |

### HW3-2（50%）

| 項目 | 配分 | 說明 |
|------|------|------|
| 5 張地圖各 10 flag | 50% | 1% per flag，最高分的 run 計算 |
| Report | 5% | 簡短說明 reward 設計 |

---

*實作完成於 2026/03/25*
