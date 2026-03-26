import os
import numpy as np
import torch

import wrapper
from model import PolicyNet


def evaluate(n_iter=50):
    save_dir = "./save"
    device = "cpu"

    policy_net = PolicyNet(14, 1).to(device)
    ckpt_path = os.path.join(save_dir, "model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"model not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path)
    policy_net.load_state_dict(checkpoint["PolicyNet"])
    policy_net.eval()

    env = wrapper.PathTrackingEnv()
    total = 0.0

    for _ in range(n_iter):
        ob, _ = env.reset()
        ep_reward = 0.0
        while True:
            state = torch.tensor(np.expand_dims(ob, axis=0), dtype=torch.float32, device=device)
            action = policy_net.action_step(state, deterministic=True).cpu().detach().numpy()
            ob, reward, done, _ = env.step(action[0])
            ep_reward += reward
            if done:
                total += ep_reward
                break

    score = total / n_iter
    print(f"Evaluation Score: {score:.4f}")
    return score


if __name__ == "__main__":
    score = evaluate(50)
    with open("save/eval_score.txt", "w", encoding="utf-8") as f:
        f.write(f"{score:.4f}\n")
    print("Saved to save/eval_score.txt")
