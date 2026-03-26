# pyright: reportMissingImports=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportOptionalSubscript=false, reportOptionalMemberAccess=false

import os
import time

import numpy as np
import torch
from dummy_env import DummyEnv
from stable_baselines3 import PPO
from stable_baselines3.common.utils import safe_mean


class RewardManager:
    def __init__(self):
        self.prev_observation = None
        self.observation = None

    def update(self, observation):
        self.prev_observation = self.observation
        self.observation = observation

    def reset(self):
        self.prev_observation = None
        self.observation = None

    def calculate_flag_capture_reward(self):
        """
        [Flag Capture Reward]
        Goal: When a new flag is capture, give a large reward to encourage the agent to move along the correct path.

        Hints:
        1. Compare 'last frame's checkpoint index' (self.prev_observation["last_checkpoint_index"])
           with 'current frame's checkpoint index' (self.observation["last_checkpoint_index"]).
        2. If the current frame's index > the previous frame's index, it means progress was made. Return a positive reward
        3. If there is no change, return 0.0.
        """
        if self.prev_observation is None:
            return 0.0
        prev_checkpoint = self.prev_observation["last_checkpoint_index"]
        curr_checkpoint = self.observation["last_checkpoint_index"]
        if curr_checkpoint > prev_checkpoint:
            return 120.0
        if curr_checkpoint < prev_checkpoint:
            return -80.0
        return 0.0

    def calculate_distance_reward(self):
        """
        [Distance Reward]
        Goal: Guide the agent to constantly move closer to the target point.

        Hints:
        1. Calculate 'distance to target in the previous frame' (prev_distance) and 'distance to target in the current frame' (current_distance).
           (Hint: use numpy.linalg.norm to calculate the vector length of target_position)
        2. Compare the two:
           - If current_distance < prev_distance (getting closer) -> reward
           - If current_distance > prev_distance (getting farther) -> penalize
        3. If the distance hasn't changed, return 0.0.
        """
        if self.prev_observation is None:
            return 0.0
        prev_target = self.prev_observation["target_position"]
        curr_target = self.observation["target_position"]
        prev_distance = np.linalg.norm(prev_target)
        curr_distance = np.linalg.norm(curr_target)
        delta = prev_distance - curr_distance
        if delta > 0:
            return min(3.0, float(delta * 12.0))
        elif delta < 0:
            return max(-6.0, float(delta * 20.0))
        return 0.0

    def calculate_survival_reward(self):
        """
        [Survival Reward]
        Goal: Teach the agent the importance of survival - avoid jumpping off the cliff

        Hints:
        Check if agent's health(agent_health) reaches 0
        """
        if self.observation["agent_health"] <= 0:
            return -200.0
        return 0.0

    def calculate_health_change_reward(self):
        if self.prev_observation is None:
            return 0.0

        prev_h = float(self.prev_observation.get("agent_health", 100.0))
        curr_h = float(self.observation.get("agent_health", 100.0))
        delta = curr_h - prev_h
        if delta < 0.0:
            return float(delta * 3.0)
        if curr_h < 25.0:
            return -1.2
        return 0.0

    def calculate_obstacle_reward(self):
        """
        [Obstacle Penalty]
        Goal: Penalize risky positions when nearby terrain indicates obstacles.

        This uses terrain_grid as a local map around the agent. Non-zero values
        are treated as risky/obstacle-like cells.
        """
        grid = self.observation.get("terrain_grid")
        if grid is None:
            return 0.0

        grid = np.asarray(grid, dtype=object)
        if grid.ndim < 2 or grid.shape[0] == 0 or grid.shape[1] == 0:
            return 0.0

        def cell_to_float(cell):
            if isinstance(cell, dict):
                if "terrain_type" in cell:
                    return float(cell["terrain_type"])
                if "value" in cell:
                    return float(cell["value"])
                return 0.0
            try:
                return float(cell)
            except (TypeError, ValueError):
                return 0.0

        grid_num = np.vectorize(cell_to_float)(grid)

        h, w = grid.shape[0], grid.shape[1]
        cy, cx = h // 2, w // 2
        y0, y1 = max(0, cy - 1), min(h, cy + 2)
        x0, x1 = max(0, cx - 1), min(w, cx + 2)
        local = grid_num[y0:y1, x0:x1]

        obstacle_mask = np.abs(local) > 0.5
        if not np.any(obstacle_mask):
            return 0.0

        # Harsh center-cell penalty + neighborhood danger penalty.
        center_is_obstacle = bool(np.abs(grid_num[cy, cx]) > 0.5)
        nearby_count = int(np.count_nonzero(obstacle_mask))
        penalty = -3.0 * nearby_count
        if center_is_obstacle:
            penalty -= 40.0
        return penalty

    def calculate_reward(self):
        """
        [Main Update Loop] (executed every frame)
        Goal: Calculate the total score for this instant

        Hints:
        1. Call the reward each reward components:
           Use self.calculate_...() to get scores for each component.

        2. Sum up rewards:
           total_reward = checkpoint_score + distance_score + survival_score + ...

        Return values:
        - total_reward (float): The total score for this frame.
        """
        # TODO 6: Complete the reward function
        flag_reward = self.calculate_flag_capture_reward()
        distance_reward = self.calculate_distance_reward()
        survival_reward = self.calculate_survival_reward()
        health_reward = self.calculate_health_change_reward()
        obstacle_reward = self.calculate_obstacle_reward()
        # Penalize every frame to discourage stalling and force faster progress.
        step_penalty = -0.1
        total_reward = (
            flag_reward
            + distance_reward
            + survival_reward
            + health_reward
            + obstacle_reward
            + step_penalty
        )
        return total_reward


class MLPlay:
    def __init__(self, observation_structure, action_space_info, *args, **kwargs):
        self.reward_manager = RewardManager()

        self.config = {
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "clip_range": 0.2,
            "gamma": 0.99,
            "ent_coef": 0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "tensorboard_log": os.path.join(os.path.dirname(__file__), "tensorboard"),
            "policy_kwargs": {"net_arch": [64, 64], "activation_fn": torch.nn.Tanh},
        }
        self.dummy_env = DummyEnv(observation_structure, action_space_info)
        self.prev_observation = None
        self.prev_action = None
        self.prev_log_prob = None
        self.prev_value = None
        self.episode_rewards = []
        self.total_steps = 0
        self.episode_steps = 0
        self.episode_count = 1
        self.update_count = 0
        self.no_progress_steps = 0
        self.last_checkpoint_index = 0
        self.prev_target_distance = None
        self.idle_steps = 0
        self.start_time = time.strftime("%Y%m%d_%H%M%S")
        self.model_save_dir = os.path.join(os.path.dirname(__file__), "models", self.start_time)
        self.model_path = os.path.join(os.path.dirname(__file__), "model" + ".zip")
        self.shield_trigger_count = 0
        self.violation_count = 0
        self.violation_rate_hist = []
        self.debug_logged = False

        os.makedirs(self.model_save_dir, exist_ok=True)

        self._initialize_model()
        print("PPO initialized in training mode")

    def reset(self):
        if self.episode_rewards:
            total_reward = sum(self.episode_rewards)
            print(
                f"Episode {self.episode_count}: Total Reward = {total_reward:.2f}, Steps = {len(self.episode_rewards)}"
            )
            print(f"Episode {self.episode_count}: Safety Shield Triggered = {self.shield_trigger_count}")
            violation_rate = self.violation_count / max(1, self.episode_steps)
            print(f"Episode {self.episode_count}: Collision Violation Rate = {violation_rate:.4f}")
            self.violation_rate_hist.append(violation_rate)
            self.episode_rewards = []
            self.shield_trigger_count = 0
            self.violation_count = 0
            self.episode_steps = 0

        self._update_policy()

        self.prev_observation = None
        self.prev_action = None
        self.prev_log_prob = None
        self.prev_value = None
        self.episode_count += 1
        self.no_progress_steps = 0
        self.last_checkpoint_index = 0
        self.prev_target_distance = None
        self.idle_steps = 0

        self.reward_manager.reset()

    def update(self, raw_observation, done, *args, **kwargs):
        if not self.debug_logged:
            keys = sorted(list(raw_observation.keys()))
            print(f"Observation keys: {keys}")
            tg = raw_observation.get("terrain_grid")
            print(f"terrain_grid type: {type(tg).__name__}")
            self.debug_logged = True

        self.reward_manager.update(raw_observation)
        observation = raw_observation["flattened"]

        checkpoint_idx = int(raw_observation.get("last_checkpoint_index", 0))
        curr_target_dist = float(np.linalg.norm(np.asarray(raw_observation.get("target_position", [0.0, 0.0]))))
        made_progress = checkpoint_idx > self.last_checkpoint_index
        if self.prev_target_distance is not None and curr_target_dist < self.prev_target_distance - 0.03:
            made_progress = True

        if made_progress:
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1
        self.last_checkpoint_index = checkpoint_idx
        self.prev_target_distance = curr_target_dist

        vel = np.asarray(raw_observation.get("agent_velocity", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
        speed = float(np.linalg.norm(vel[:2])) if vel.size >= 2 else float(np.linalg.norm(vel))
        if speed < 0.03:
            self.idle_steps += 1
        else:
            self.idle_steps = 0

        reward = self.reward_manager.calculate_reward()
        if self.no_progress_steps > 120:
            reward -= min(3.0, 0.01 * float(self.no_progress_steps - 120))
        if self.idle_steps > 20:
            reward -= min(3.0, 0.04 * float(self.idle_steps - 20))

        action, log_prob, value = self._predict_action(observation)
        nav_action = np.asarray(action, dtype=np.float32)
        safe_action = self._apply_safety_shield(raw_observation, nav_action)
        action = self._fuse_navigation_and_safety(raw_observation, nav_action, safe_action)
        action = self._fallback_target_controller(raw_observation, action)

        if self._is_collision_violation(raw_observation, action):
            self.violation_count += 1

        if self.prev_observation is not None:
            self.episode_rewards.append(reward)

            if not self.model.rollout_buffer.full:
                self._add_to_rollout_buffer(
                    obs=self.prev_observation,
                    action=self.prev_action,
                    reward=reward,
                    done=done,
                    value=self.prev_value,
                    log_prob=self.prev_log_prob,
                )
                if self.model.rollout_buffer.full:
                    done_tensor = np.array([done])
                    value_tensor = torch.as_tensor(value).unsqueeze(0) if value.ndim == 0 else torch.as_tensor(value)
                    self.model.rollout_buffer.compute_returns_and_advantage(last_values=value_tensor, dones=done_tensor)

        self.prev_observation = observation
        self.prev_action = action
        self.prev_log_prob = log_prob
        self.prev_value = value
        self.total_steps += 1
        self.episode_steps += 1

        # NOTE: DO NOT MODIFY.
        # Sending additional dummy discrete actions that would not be needed for this assignment
        return action, (0, 0)

    def _action_risk_score(self, grid, action):
        h, w = grid.shape
        cy, cx = h // 2, w // 2

        step_x = int(np.sign(action[0]))
        step_y = -int(np.sign(action[1]))

        risk = 0.0
        for k, weight in ((1, 1.0), (2, 0.7)):
            y = int(np.clip(cy + step_y * k, 0, h - 1))
            x = int(np.clip(cx + step_x * k, 0, w - 1))
            cell = abs(float(grid[y, x]))
            if cell > 0.2:
                risk += 100.0 * weight
            risk += cell * 10.0 * weight

        return risk

    def _is_collision_violation(self, raw_observation, action):
        grid = self._extract_grid(raw_observation)
        if grid is None or grid.ndim != 2:
            return False

        h, w = grid.shape
        cy, cx = h // 2, w // 2
        if abs(float(grid[cy, cx])) > 0.5:
            return True

        risk = self._action_risk_score(grid, np.asarray(action, dtype=np.float32))
        return risk >= 90.0

    def _fuse_navigation_and_safety(self, raw_observation, nav_action, safe_action):
        nav = np.asarray(nav_action, dtype=np.float32)
        safe = np.asarray(safe_action, dtype=np.float32)
        grid = self._extract_grid(raw_observation)

        risk = 0.0
        if grid is not None and grid.ndim == 2:
            risk = self._action_risk_score(grid, nav)

        # Adaptive fusion: higher risk/stuck => rely more on safety branch.
        alpha = 0.2
        if risk > 40.0:
            alpha = min(0.95, 0.2 + (risk - 40.0) / 120.0)
        if self.no_progress_steps > 100:
            alpha = min(0.95, alpha + 0.25)

        fused = (1.0 - alpha) * nav + alpha * safe
        return np.clip(fused, -1.0, 1.0)

    def _to_float_array(self, obj):
        if obj is None:
            return np.array([], dtype=np.float32)

        if isinstance(obj, (int, float, np.integer, np.floating, bool)):
            return np.array([float(obj)], dtype=np.float32)

        if isinstance(obj, np.ndarray):
            try:
                return obj.astype(np.float32).reshape(-1)
            except (TypeError, ValueError):
                return np.array([], dtype=np.float32)

        if isinstance(obj, dict):
            preferred_keys = ["flattened", "data", "values", "value", "grid", "array"]
            for key in preferred_keys:
                if key in obj:
                    arr = self._to_float_array(obj[key])
                    if arr.size > 0:
                        return arr

            chunks = [self._to_float_array(v) for v in obj.values()]
            chunks = [c for c in chunks if c.size > 0]
            if chunks:
                return np.concatenate(chunks, axis=0)
            return np.array([], dtype=np.float32)

        if isinstance(obj, (list, tuple)):
            chunks = [self._to_float_array(v) for v in obj]
            chunks = [c for c in chunks if c.size > 0]
            if chunks:
                return np.concatenate(chunks, axis=0)
            return np.array([], dtype=np.float32)

        return np.array([], dtype=np.float32)

    def _extract_grid(self, raw_observation):
        grid_obj = raw_observation.get("terrain_grid")
        if grid_obj is None:
            return None

        def cell_to_float(cell):
            if isinstance(cell, dict):
                if "terrain_type" in cell:
                    return float(cell["terrain_type"])
                if "value" in cell:
                    return float(cell["value"])
                return 0.0
            try:
                return float(cell)
            except (TypeError, ValueError):
                return 0.0

        # Preferred path: keep native 2D layout to preserve spatial meaning.
        if isinstance(grid_obj, np.ndarray) and grid_obj.ndim == 2:
            if grid_obj.dtype == object:
                return np.vectorize(cell_to_float)(grid_obj).astype(np.float32)
            return grid_obj.astype(np.float32)

        if isinstance(grid_obj, (list, tuple)) and len(grid_obj) > 0 and isinstance(grid_obj[0], (list, tuple, np.ndarray)):
            rows = [[cell_to_float(c) for c in row] for row in grid_obj]
            return np.asarray(rows, dtype=np.float32)

        # Fallback path for flattened/unknown encodings.
        arr = self._to_float_array(grid_obj)
        if arr.size == 0:
            return None

        side = int(np.sqrt(arr.size))
        if side * side == arr.size and side >= 3:
            return arr.reshape(side, side)

        if arr.size >= 25:
            return arr[:25].reshape(5, 5)

        return None

    def _extract_target_dir(self, raw_observation):
        target = raw_observation.get("target_position", [0.0, 0.0])

        if isinstance(target, dict):
            if "x" in target and "z" in target:
                vec = np.array([float(target["x"]), -float(target["z"])], dtype=np.float32)
            elif "x" in target and "y" in target:
                vec = np.array([float(target["x"]), -float(target["y"])], dtype=np.float32)
            else:
                arr = self._to_float_array(target)
                if arr.size >= 2:
                    vec = np.array([arr[0], -arr[1]], dtype=np.float32)
                else:
                    vec = np.array([0.0, 0.0], dtype=np.float32)
        else:
            arr = self._to_float_array(target)
            if arr.size >= 2:
                vec = np.array([arr[0], -arr[1]], dtype=np.float32)
            else:
                vec = np.array([0.0, 0.0], dtype=np.float32)

        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            vec /= norm
        return vec

    def _apply_safety_shield(self, raw_observation, action):
        base_action = np.asarray(action, dtype=np.float32).copy()
        base_action = np.clip(base_action, -1.0, 1.0)
        target_dir = self._extract_target_dir(raw_observation)

        grid = self._extract_grid(raw_observation)
        if grid is None or grid.ndim != 2 or grid.shape[0] < 3 or grid.shape[1] < 3:
            return self._stabilize_action(base_action, target_dir)

        h, w = grid.shape
        cy, cx = h // 2, w // 2
        center_danger = abs(float(grid[cy, cx])) > 0.5

        # Anti-spin smoothing: limit abrupt direction flips.
        if self.prev_action is not None:
            prev = np.asarray(self.prev_action, dtype=np.float32)
            delta = base_action - prev
            delta_norm = float(np.linalg.norm(delta))
            max_delta = 0.35
            if delta_norm > max_delta and delta_norm > 1e-6:
                base_action = prev + delta * (max_delta / delta_norm)

        base_risk = self._action_risk_score(grid, base_action)

        # Only intervene when agent is in danger or moving into clear danger.
        if not center_danger and base_risk < 70.0:
            return self._stabilize_action(base_action, target_dir=target_dir)

        candidates = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([-1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.0, -1.0], dtype=np.float32),
            np.array([0.7, 0.7], dtype=np.float32),
            np.array([0.7, -0.7], dtype=np.float32),
            np.array([-0.7, 0.7], dtype=np.float32),
            np.array([-0.7, -0.7], dtype=np.float32),
            base_action,
        ]

        best = base_action
        best_score = float("inf")
        for cand in candidates:
            risk = self._action_risk_score(grid, cand)
            align = float(np.dot(cand, target_dir))
            score = risk - 7.0 * align
            if score < best_score:
                best_score = score
                best = cand

        if np.linalg.norm(best - base_action) > 1e-5:
            self.shield_trigger_count += 1

        return self._stabilize_action(best, target_dir)

    def _stabilize_action(self, action, target_dir):
        raw = np.asarray(action, dtype=np.float32).copy()
        act = raw.copy()

        # Never accelerate opposite to target direction.
        target_norm = float(np.linalg.norm(target_dir))
        if target_norm > 1e-6:
            align = float(np.dot(act, target_dir))
            if align < 0.0:
                act = act - align * target_dir

            if self.no_progress_steps > 80:
                # If stuck, bias toward checkpoint direction to break loops.
                act = 0.6 * act + 0.4 * target_dir

        # Final smoothing against previous action to reduce spinning.
        if self.prev_action is not None:
            prev = np.asarray(self.prev_action, dtype=np.float32)
            prev_w = 0.45 if self.no_progress_steps > 80 or self.idle_steps > 25 else 0.7
            act = prev_w * prev + (1.0 - prev_w) * act

        # Cap action magnitude to reduce aggressive spins/dives.
        norm = float(np.linalg.norm(act))
        max_norm = 0.8
        if norm > max_norm and norm > 1e-6:
            act = act * (max_norm / norm)

        # Anti-stall: if output is too small while idle/no-progress, force forward intent.
        if float(np.linalg.norm(target_dir)) > 1e-6 and (self.idle_steps > 10 or self.no_progress_steps > 60):
            act = 0.2 * act + 0.8 * target_dir

        min_norm = 0.0
        if self.idle_steps > 12:
            min_norm = 0.35
        elif self.no_progress_steps > 70:
            min_norm = 0.25

        norm = float(np.linalg.norm(act))
        if norm < min_norm and float(np.linalg.norm(target_dir)) > 1e-6:
            act = target_dir * min_norm

        act = np.clip(act, -1.0, 1.0)
        return act

    def _fallback_target_controller(self, raw_observation, action):
        # If stuck for too long, switch to deterministic target-seeking action
        # to break circular behavior and recover progress.
        if self.no_progress_steps < 90 and self.idle_steps < 30:
            return action

        target_dir = self._extract_target_dir(raw_observation)
        if float(np.linalg.norm(target_dir)) <= 1e-6:
            return action

        guided = 0.95 * target_dir
        if self.prev_action is not None:
            prev = np.asarray(self.prev_action, dtype=np.float32)
            mix = 0.25 if self.no_progress_steps > 140 or self.idle_steps > 50 else 0.5
            guided = mix * prev + (1.0 - mix) * guided

        self.shield_trigger_count += 1
        return np.clip(guided, -1.0, 1.0)

    def _initialize_model(self):
        print("Initializing PPO model...")
        if os.path.exists(self.model_path):
            try:
                self.model = PPO.load(self.model_path, env=self.dummy_env, **self.config, verbose=1)
                print(f"Model loaded from {self.model_path}")
            except Exception as e:
                print(f"Error loading model from {self.model_path}: {e}")
                print("Creating new model...")
                self.model = PPO("MlpPolicy", env=self.dummy_env, **self.config, verbose=1)
        else:
            print(f"No pre-trained model found at {self.model_path}. Creating new model...")
            self.model = PPO("MlpPolicy", env=self.dummy_env, **self.config, verbose=1)

        # NOTE: SB3 is not used in the standard way here. Normally model.learn() drives the
        # entire training loop; here, total_timesteps=0 is used only to initialize the
        # TensorBoard logger and internal SB3 state. The actual rollout collection and policy
        # updates are driven manually by mlgame3d's game loop via _add_to_rollout_buffer()
        # and _update_policy(), because mlgame3d controls the environment stepping externally.
        self.model.learn(total_timesteps=0, tb_log_name=f"PPO_{self.start_time}")

    def _save_model(self):
        if self.model is not None:
            self.model.save(self.model_path)
            print(f"Model saved to {self.model_path}")

            update_path = f"{self.model_save_dir}/ppo_model_{self.update_count}.zip"
            self.model.save(update_path)
            print(f"Model saved to {update_path}")

    def _predict_action(self, obs):
        obs_tensor = torch.as_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, value, log_prob = self.model.policy(obs_tensor)
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten(), value.cpu().numpy().flatten()

    def _add_to_rollout_buffer(self, obs, action, reward, done, value, log_prob):
        if not self.model.rollout_buffer.full:
            self.model.rollout_buffer.add(
                obs=torch.as_tensor(obs).unsqueeze(0),
                action=torch.as_tensor(action).unsqueeze(0),
                reward=torch.as_tensor([reward]),
                episode_start=torch.as_tensor([done]),
                value=torch.as_tensor(value).unsqueeze(0) if value.ndim == 0 else torch.as_tensor(value),
                log_prob=torch.as_tensor(log_prob).unsqueeze(0) if log_prob.ndim == 0 else torch.as_tensor(log_prob),
            )

    def _update_policy(self):
        if self.model.rollout_buffer.size() == 0 or not self.model.rollout_buffer.full:
            return

        print(f"Updating PPO policy with {self.model.rollout_buffer.size()} experiences...")

        self.model.num_timesteps += self.model.rollout_buffer.size()
        self.model.train()
        self.update_count += 1

        self.model.logger.record("train/mean_reward", safe_mean(self.model.rollout_buffer.rewards))
        self.model.logger.record("param/n_steps", self.model.n_steps)
        self.model.logger.record("param/batch_size", self.model.batch_size)
        self.model.logger.record("param/n_epochs", self.model.n_epochs)
        self.model.logger.record("param/gamma", self.model.gamma)
        self.model.logger.record("param/gae_lambda", self.model.gae_lambda)
        self.model.logger.record("param/ent_coef", self.model.ent_coef)
        self.model.logger.record("param/vf_coef", self.model.vf_coef)
        self.model.logger.record("param/max_grad_norm", self.model.max_grad_norm)
        if self.violation_rate_hist:
            self.model.logger.record("safety/violation_rate", float(np.mean(self.violation_rate_hist[-20:])))
        self.model.logger.record("safety/shield_trigger_count", float(self.shield_trigger_count))
        self.model.logger.dump(self.update_count)

        self.model.rollout_buffer.reset()
        print("PPO policy updated successfully")

        self._save_model()
