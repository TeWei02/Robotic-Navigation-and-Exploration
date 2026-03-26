import os
import numpy as np
from stable_baselines3 import PPO


class MLPlay:
    def __init__(self, observation_structure, action_space_info, *args, **kwargs):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "model.zip")

        self.model = None
        if os.path.exists(model_path):
            self.model = PPO.load(model_path)
            print(f"[eval_play] Loaded model: {model_path}")
        else:
            print(f"[eval_play] model.zip not found: {model_path}")

        if hasattr(action_space_info, "continuous_size"):
            self.action_dim = int(action_space_info.continuous_size)
        elif isinstance(action_space_info, dict) and "continuous_size" in action_space_info:
            self.action_dim = int(action_space_info["continuous_size"])
        else:
            self.action_dim = 2

    def reset(self):
        pass

    def update(self, raw_observation, done, *args, **kwargs):
        if self.model is None:
            return np.zeros((self.action_dim,), dtype=np.float32), (0, 0)

        obs = raw_observation.get("flattened")
        if obs is None:
            return np.zeros((self.action_dim,), dtype=np.float32), (0, 0)

        action, _ = self.model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != self.action_dim:
            padded = np.zeros((self.action_dim,), dtype=np.float32)
            n = min(self.action_dim, action.size)
            padded[:n] = action[:n]
            action = padded

        return action, (0, 0)
