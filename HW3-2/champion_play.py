# pyright: reportMissingImports=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportOptionalSubscript=false, reportOptionalMemberAccess=false

import os
from typing import Any

import numpy as np
from stable_baselines3 import PPO

try:
    from rl_play import MLPlay as FastRLController
except Exception:
    FastRLController = None


class MLPlay:
    def __init__(self, observation_structure: Any, action_space_info: Any, *args: Any, **kwargs: Any):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_candidates = [
            os.path.join(current_dir, "model.zip"),
            os.path.join(current_dir, "best_model.zip"),
        ]

        self.model = None
        self.use_model_action = False
        self.agent_tag = hex(id(self))[-4:]
        for model_path in model_candidates:
            if os.path.exists(model_path):
                try:
                    self.model = PPO.load(model_path)
                    print(f"[champion_play] Loaded model: {model_path}")
                    break
                except Exception as exc:
                    print(f"[champion_play] Failed to load {model_path}: {exc}")

        if self.model is None:
            print("[champion_play] No usable model found, fallback heuristic enabled")
        else:
            print("[champion_play] Safety-first mode: PPO used as weak prior")

        if hasattr(action_space_info, "continuous_size"):
            self.action_dim = int(action_space_info.continuous_size)
        elif isinstance(action_space_info, dict) and "continuous_size" in action_space_info:
            self.action_dim = int(action_space_info["continuous_size"])
        else:
            self.action_dim = 2

        self.prev_action = np.zeros((self.action_dim,), dtype=np.float32)
        # Per-agent personality to avoid all bots moving identically.
        self.side_pref = float(np.random.uniform(-0.35, 0.35))
        self.use_item_cooldown = int(np.random.randint(10, 28))
        self.race_mode = True
        self.invert_forward_axis = False
        self.use_fast_rl_controller = FastRLController is not None
        self.fast_controller = None
        if self.use_fast_rl_controller:
            try:
                self.fast_controller = FastRLController(observation_structure, action_space_info, *args, **kwargs)
                print("[champion_play] Fast RL controller enabled (from rl_play)")
            except Exception as exc:
                self.fast_controller = None
                self.use_fast_rl_controller = False
                print(f"[champion_play] Fast RL controller disabled: {exc}")
        self.tick = 0
        self.prev_checkpoint = 0
        self.prev_target_distance = None
        self.orbit_steps = 0
        self.prev_hp = 100.0
        self.prev_respawning = False
        self.prev_dead = False
        self.no_progress_steps = 0
        self.fall_like_count = 0
        self.estimated_death_count = 0
        self.hearts = 5
        self.permadeath = False
        self.last_life_loss_tick = -99999
        self.life_loss_armed = True
        self.life_count_enabled = False
        self.stats_path = os.path.join(current_dir, "champion_stats.csv")
        self.debug_budget = 24
        if not os.path.exists(self.stats_path):
            with open(self.stats_path, "w", encoding="utf-8") as f:
                f.write("tick,hp,checkpoint,no_progress,fall_like,deaths_est,hearts,permadeath,use_item_cd,side_pref\n")

    def reset(self) -> None:
        self.prev_action = np.zeros((self.action_dim,), dtype=np.float32)
        self.tick = 0
        self.prev_checkpoint = 0
        self.prev_target_distance = None
        self.orbit_steps = 0
        self.prev_hp = 100.0
        self.prev_respawning = False
        self.prev_dead = False
        self.no_progress_steps = 0
        self.hearts = 5
        self.permadeath = False
        self.last_life_loss_tick = -99999
        self.life_loss_armed = True
        self.life_count_enabled = False
        self.debug_budget = 0

    def _to_float_array(self, obj: Any) -> np.ndarray:
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
            # Prefer terrain cell values if present.
            if "terrain_type" in obj:
                return self._to_float_array(obj.get("terrain_type"))

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

    def _extract_grid(self, raw_observation: dict[str, Any]) -> np.ndarray | None:
        grid_obj = raw_observation.get("terrain_grid")

        # Already a numeric 2D grid.
        if isinstance(grid_obj, np.ndarray) and grid_obj.ndim == 2:
            try:
                return grid_obj.astype(np.float32)
            except (TypeError, ValueError):
                pass

        # Structured list of cells (each may be dict with terrain_type).
        if isinstance(grid_obj, list):
            terrain_vals: list[float] = []
            for cell in grid_obj:
                if isinstance(cell, dict) and "terrain_type" in cell:
                    terrain_vals.extend(self._to_float_array(cell["terrain_type"]).tolist())
                else:
                    terrain_vals.extend(self._to_float_array(cell).tolist())

            arr = np.asarray(terrain_vals, dtype=np.float32).reshape(-1)
        else:
            arr = self._to_float_array(grid_obj)

        if arr.size == 0:
            return None

        side = int(np.sqrt(arr.size))
        if side * side == arr.size and side >= 3:
            return arr.reshape(side, side)

        if arr.size >= 25:
            return arr[:25].reshape(5, 5)

        return None

    def _update_runtime_stats(self, raw_observation: dict[str, Any]) -> None:
        hp = float(raw_observation.get("agent_health", 100.0))
        checkpoint = int(raw_observation.get("last_checkpoint_index", 0))
        respawning = bool(raw_observation.get("is_respawning", False))
        dead_now = hp <= 0.5

        if checkpoint > self.prev_checkpoint:
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1

        hp_drop = self.prev_hp - hp
        if hp_drop >= 0.95:
            self.fall_like_count += 1
        # Approximate death-like event: very low hp with respawning state.
        if respawning and hp <= 0.5:
            self.estimated_death_count += 1

        # Rearm once respawn phase ends, so each death cycle costs exactly one heart.
        if (not respawning) and self.prev_respawning:
            self.life_loss_armed = True

        # Enable life counting only after leaving initial spawn/respawn phase.
        if (not respawning) and hp > 1.0:
            self.life_count_enabled = True

        respawn_event = respawning and (not self.prev_respawning)
        death_event = respawn_event

        min_gap = 30
        if (
            self.life_count_enabled
            and death_event
            and self.life_loss_armed
            and self.tick > 5
            and (self.tick - self.last_life_loss_tick >= min_gap)
        ):
            prev_hearts = self.hearts
            self.hearts = max(0, self.hearts - 1)
            self.last_life_loss_tick = self.tick
            self.life_loss_armed = False
            print(
                f"[champion_play:{self.agent_tag}] life lost, hearts {prev_hearts}->{self.hearts}, "
                f"hp={hp:.2f}, respawning={int(respawning)}, tick={self.tick}"
            )
            if self.hearts <= 0:
                self.permadeath = True
                print(f"[champion_play:{self.agent_tag}] permadeath engaged (hearts depleted)")

        if self.hearts <= 0:
            self.permadeath = True

        self.prev_checkpoint = checkpoint
        self.prev_hp = hp
        self.prev_respawning = respawning
        self.prev_dead = dead_now

        if self.tick % 30 == 0:
            with open(self.stats_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{self.tick},{hp:.3f},{checkpoint},{self.no_progress_steps},"
                    f"{self.fall_like_count},{self.estimated_death_count},"
                    f"{self.hearts},{int(self.permadeath)},"
                    f"{self.use_item_cooldown},{self.side_pref:.4f}\n"
                )

        # Self-analysis & adaptation: if stuck too long, become more aggressive toward checkpoints.
        if self.no_progress_steps > 120:
            self.side_pref *= 0.8
            self.use_item_cooldown = max(8, self.use_item_cooldown - 1)

    def _extract_target_vec(self, raw_observation: dict[str, Any]) -> np.ndarray:
        target = raw_observation.get("target_position", [0.0, 0.0])
        if isinstance(target, dict):
            if "x" in target and "z" in target:
                vec = np.array([float(target["x"]), -float(target["z"])], dtype=np.float32)
            elif "x" in target and "y" in target:
                vec = np.array([float(target["x"]), -float(target["y"])], dtype=np.float32)
            else:
                arr = self._to_float_array(target)
                if arr.size >= 3:
                    vec = np.array([arr[0], -arr[2]], dtype=np.float32)
                elif arr.size >= 2:
                    vec = np.array([arr[0], -arr[1]], dtype=np.float32)
                else:
                    vec = np.zeros((2,), dtype=np.float32)
        else:
            arr = self._to_float_array(target)
            if arr.size >= 3:
                vec = np.array([arr[0], -arr[2]], dtype=np.float32)
            elif arr.size >= 2:
                vec = np.array([arr[0], -arr[1]], dtype=np.float32)
            else:
                vec = np.zeros((2,), dtype=np.float32)

        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            vec = vec / norm
        else:
            vec = np.array([0.0, 1.0], dtype=np.float32)
        return vec

    def _extract_target_distance(self, raw_observation: dict[str, Any]) -> float:
        target = raw_observation.get("target_position", [0.0, 0.0])
        arr = self._to_float_array(target)
        if arr.size >= 3:
            return float(np.linalg.norm(np.array([arr[0], arr[2]], dtype=np.float32)))
        if arr.size >= 2:
            return float(np.linalg.norm(np.array([arr[0], arr[1]], dtype=np.float32)))
        return 0.0

    def _anti_orbit_correction(self, raw_observation: dict[str, Any], action: np.ndarray) -> np.ndarray:
        target = self._extract_target_vec(raw_observation)
        dist = self._extract_target_distance(raw_observation)

        if self.prev_target_distance is None:
            self.prev_target_distance = dist
            return action

        cp = int(raw_observation.get("last_checkpoint_index", 0))
        getting_farther = dist > self.prev_target_distance + 0.02
        near_checkpoint = dist < 2.2

        if getting_farther and self.no_progress_steps > 8:
            self.orbit_steps += 1
        else:
            self.orbit_steps = max(0, self.orbit_steps - 1)

        corrected = np.asarray(action, dtype=np.float32).copy()

        # If close to checkpoint but no progress, suppress wide circling and pull hard to target.
        if near_checkpoint and self.no_progress_steps > 6:
            corrected = 0.35 * corrected + 0.65 * target
            corrected[0] = float(np.clip(corrected[0], -0.35, 0.35))

        # If persistent orbiting is detected, switch to aggressive target lock.
        if self.orbit_steps > 12:
            corrected = 0.15 * corrected + 0.85 * target
            corrected[0] = float(np.clip(corrected[0], -0.28, 0.28))

        self.prev_target_distance = dist
        corrected = np.clip(corrected, -1.0, 1.0).astype(np.float32)
        return corrected

    def _ensure_min_move(self, steer: np.ndarray, target: np.ndarray, min_norm: float = 0.35) -> np.ndarray:
        vec = np.asarray(steer, dtype=np.float32)
        n = float(np.linalg.norm(vec))
        if n < 1e-6:
            vec = target.copy()
            n = float(np.linalg.norm(vec))
        if n > 1e-6 and n < min_norm:
            vec = vec * (min_norm / n)
        return vec

    def _forward_priority(self, vec: np.ndarray, target: np.ndarray, min_forward: float = 0.28) -> np.ndarray:
        out = np.asarray(vec, dtype=np.float32).copy()
        t = np.asarray(target, dtype=np.float32)

        # Limit excessive lateral steering that often causes circular motion.
        out[0] = float(np.clip(out[0], -0.45, 0.45))

        # Keep alignment with target direction from collapsing.
        # This avoids forcing a fixed global axis that can drive into hazards.
        t_norm = float(np.linalg.norm(t))
        if t_norm > 1e-6:
            t = t / t_norm
            align = float(np.dot(out, t))
            if align < min_forward:
                out = out + (min_forward - align) * t

        # Prevent sudden reverse direction wrt previous action.
        prev = self.prev_action[:2]
        if float(np.dot(prev, out)) < -0.05:
            out = 0.75 * prev + 0.25 * out

        # If still too sideways, blend back toward target direction.
        if abs(float(out[0])) > 0.35:
            out = 0.6 * out + 0.4 * t

        return out

    def _repel_from_players(self, raw_observation: dict[str, Any]) -> np.ndarray:
        repel = np.zeros((2,), dtype=np.float32)
        others = raw_observation.get("other_players", [])
        if not isinstance(others, list):
            return repel

        for p in others:
            try:
                rel = np.asarray(p.get("relative_position", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
                if rel.size < 3:
                    continue
                # relative_position is (x, y, z), movement plane is x-z.
                vec = np.array([rel[0], -rel[2]], dtype=np.float32)
                dist = float(np.linalg.norm(vec))
                if dist < 1e-4:
                    continue
                if dist < 4.0:
                    # Stronger separation when agents are very close.
                    repel -= (vec / dist) * float((4.0 - dist) / 4.0)
            except Exception:
                continue
        return repel

    def _safety_adjust(self, raw_observation: dict[str, Any], steer: np.ndarray) -> np.ndarray:
        grid = self._extract_grid(raw_observation)
        if grid is None:
            return steer

        if grid.ndim != 2 or grid.shape[0] < 3 or grid.shape[1] < 3:
            return steer

        h, w = grid.shape
        cy, cx = h // 2, w // 2

        def action_risk(act: np.ndarray) -> float:
            ax = int(np.sign(float(act[0])))
            ay = -int(np.sign(float(act[1])))
            risk = 0.0
            for k, wt in ((1, 1.0), (2, 0.7)):
                y = int(np.clip(cy + ay * k, 0, h - 1))
                x = int(np.clip(cx + ax * k, 0, w - 1))
                cell = abs(float(grid[y, x]))
                if cell > 0.45:
                    risk += 100.0 * wt
                risk += 8.0 * cell * wt
            return risk

        center_danger = abs(float(grid[cy, cx])) > 0.5
        base = np.asarray(steer, dtype=np.float32)
        base_risk = action_risk(base)

        if (not center_danger) and base_risk < 70.0:
            return base

        target = self._extract_target_vec(raw_observation)
        candidates = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([-1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.0, -1.0], dtype=np.float32),
            np.array([0.7, 0.7], dtype=np.float32),
            np.array([0.7, -0.7], dtype=np.float32),
            np.array([-0.7, 0.7], dtype=np.float32),
            np.array([-0.7, -0.7], dtype=np.float32),
            base,
        ]

        best = base
        best_score = float("inf")
        for cand in candidates:
            score = action_risk(cand) - 5.0 * float(np.dot(cand, target))
            if score < best_score:
                best_score = score
                best = cand
        return best

    def _heuristic_action(self, raw_observation: dict[str, Any]) -> np.ndarray:
        target = self._extract_target_vec(raw_observation)
        repel = self._repel_from_players(raw_observation)
        # Side bias creates lane-like separation between competitors.
        side = np.array([self.side_pref, 0.0], dtype=np.float32)
        steer = 1.30 * target + 0.20 * repel + 0.05 * side

        hp = float(raw_observation.get("agent_health", 100.0))
        if hp <= 1.0:
            # One-heart survival style: conservative motion when low HP.
            steer *= 0.75

        # If stuck, force stronger checkpoint-seeking behavior.
        if self.no_progress_steps > 30:
            steer = 1.80 * target + 0.12 * repel
        elif self.no_progress_steps > 12:
            steer = 1.55 * target + 0.15 * repel

        steer = self._safety_adjust(raw_observation, steer)
        steer = self._forward_priority(steer, target, min_forward=0.45)
        steer = self._ensure_min_move(steer, target, min_norm=0.70)
        norm = float(np.linalg.norm(steer))
        if norm > 1e-6:
            steer = steer / norm

        action = np.zeros((self.action_dim,), dtype=np.float32)
        n = min(2, self.action_dim)
        action[:n] = (1.0 * steer[:n]).astype(np.float32)
        if n >= 2 and self.invert_forward_axis:
            action[1] = -action[1]

        # Lighter smoothing in race mode for faster turn-in and acceleration.
        prev_w = 0.40
        if self.no_progress_steps > 30:
            prev_w = 0.15
        elif self.no_progress_steps > 12:
            prev_w = 0.28
        action = prev_w * self.prev_action + (1.0 - prev_w) * action
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        if self.debug_budget > 0:
            cp = int(raw_observation.get("last_checkpoint_index", 0))
            hp = float(raw_observation.get("agent_health", 100.0))
            print(
                f"[champion_play][dbg] t={self.tick} cp={cp} hp={hp:.1f} "
                f"target=({target[0]:.3f},{target[1]:.3f}) action=({action[0]:.3f},{action[1]:.3f})"
            )
            self.debug_budget -= 1
        self.prev_action = action.copy()
        return action

    def _postprocess_model_action(self, raw_observation: dict[str, Any], base_action: np.ndarray) -> np.ndarray:
        act = np.asarray(base_action, dtype=np.float32).reshape(-1)
        if act.size < self.action_dim:
            padded = np.zeros((self.action_dim,), dtype=np.float32)
            padded[: act.size] = act
            act = padded

        steer = np.zeros((2,), dtype=np.float32)
        steer[: min(2, self.action_dim)] = act[: min(2, self.action_dim)]

        target = self._extract_target_vec(raw_observation)
        repel = self._repel_from_players(raw_observation)
        side = np.array([self.side_pref, 0.0], dtype=np.float32)

        # Safety-first blend: model is a weak prior, heuristic drives survival/progress.
        steer = 0.35 * steer + 0.55 * target + 0.32 * repel + 0.10 * side
        steer = self._safety_adjust(raw_observation, steer)
        steer = self._forward_priority(steer, target, min_forward=0.26)

        norm = float(np.linalg.norm(steer))
        if norm > 1e-6:
            steer = steer / norm

        out = np.zeros((self.action_dim,), dtype=np.float32)
        out[: min(2, self.action_dim)] = steer[: min(2, self.action_dim)]
        if self.action_dim >= 2 and self.invert_forward_axis:
            out[1] = -out[1]
        out = 0.72 * self.prev_action + 0.28 * out

        if self.no_progress_steps > 100:
            out[: min(2, self.action_dim)] = 0.55 * target[: min(2, self.action_dim)] + 0.45 * out[
                : min(2, self.action_dim)
            ]

        out[: min(2, self.action_dim)] = self._forward_priority(
            out[: min(2, self.action_dim)],
            target[: min(2, self.action_dim)],
            min_forward=0.24,
        )[: min(2, self.action_dim)]

        out[: min(2, self.action_dim)] = self._ensure_min_move(
            out[: min(2, self.action_dim)],
            target[: min(2, self.action_dim)],
            min_norm=0.40,
        )[: min(2, self.action_dim)]

        out = np.clip(out, -1.0, 1.0).astype(np.float32)
        self.prev_action = out.copy()
        return out

    def _decide_discrete_action(self, raw_observation: dict[str, Any]) -> tuple[int, int]:
        if self.permadeath:
            return (0, 0)
        # Use item periodically; prefer defensive usage when low HP.
        self.tick += 1
        hp = float(raw_observation.get("agent_health", 100.0))
        if hp <= 1.0:
            return (0, 1)
        if self.tick % self.use_item_cooldown == 0:
            return (0, 1)
        return (0, 0)

    def update(self, raw_observation: dict[str, Any], done: bool, *args: Any, **kwargs: Any):
        self._update_runtime_stats(raw_observation)

        if self.permadeath:
            return np.zeros((self.action_dim,), dtype=np.float32), (0, 0)

        if self.fast_controller is not None:
            try:
                fast_action, _ = self.fast_controller.update(raw_observation, done, *args, **kwargs)
                fast_action = np.asarray(fast_action, dtype=np.float32).reshape(-1)
                if fast_action.size < self.action_dim:
                    padded = np.zeros((self.action_dim,), dtype=np.float32)
                    padded[: fast_action.size] = fast_action
                    fast_action = padded
                fast_action = self._anti_orbit_correction(raw_observation, fast_action)
                self.prev_action = np.clip(fast_action, -1.0, 1.0).astype(np.float32)
                return self.prev_action, self._decide_discrete_action(raw_observation)
            except Exception as exc:
                print(f"[champion_play] fast controller fallback: {exc}")

        if self.model is not None and self.use_model_action:
            obs = raw_observation.get("flattened")
            if obs is not None:
                action, _ = self.model.predict(obs, deterministic=True)
                action = self._postprocess_model_action(raw_observation, action)
                action = self._anti_orbit_correction(raw_observation, action)
                return action, self._decide_discrete_action(raw_observation)

        action = self._heuristic_action(raw_observation)
        action = self._anti_orbit_correction(raw_observation, action)
        return action, self._decide_discrete_action(raw_observation)
