import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repo_paths import LUNAR_LANDER_RESULTS, ensure_dir
from utils.recording import continuous_gif_kwargs, record_episode_gif

from PPO import MODEL_PATH, LunarLanderAgent

LUNAR_GIF_KWARGS = continuous_gif_kwargs(expected_steps=500, target_duration_sec=5.0)


def load_trained_agent():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained agent found at {MODEL_PATH}. Run PPO.py first."
        )

    env = gym.make('LunarLander-v3')
    agent = LunarLanderAgent(
        env=env,
        policy_lr=3e-4,
        value_lr=1e-3,
        gamma=0.99,
        lam=0.95,
        n_hidden=128,
        n_steps=1024,
        entropy_coef_start=0.02,
        entropy_coef_end=0.001,
        max_grad_norm=0.5,
    )
    agent.load(MODEL_PATH)
    agent.policy_net.eval()
    agent.value_net.eval()
    return agent


def _run_lunar_episode(env, agent):
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        with torch.no_grad():
            dist = agent.policy_net(
                torch.tensor(obs, dtype=torch.float32, device=agent.device)
            )
            action = dist.probs.argmax().item()

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

    return total_reward, steps


def test_agent(agent, n_tests=3, render=True, record_video=False):
    if render and record_video:
        raise ValueError("Use either render=True or record_video=True, not both.")

    print("\n--- Testing Trained Agent (PPO) ---")

    if render:
        render_mode = "human"
    elif record_video:
        render_mode = "rgb_array"
    else:
        render_mode = None

    test_env = gym.make('LunarLander-v3', render_mode=render_mode)
    gif_path = ensure_dir(LUNAR_LANDER_RESULTS) / "ppo_lunar_lander.gif"

    rewards = []
    for i in range(n_tests):
        if record_video and i == 0:
            total_reward, steps = record_episode_gif(
                test_env,
                lambda capture_env: _run_lunar_episode(capture_env, agent),
                gif_path,
                **LUNAR_GIF_KWARGS,
            )
            print(f"Saved demo GIF to {gif_path}")
        else:
            total_reward, steps = _run_lunar_episode(test_env, agent)

        print(f"Test Episode {i+1} | Steps: {steps} | Reward: {total_reward:.2f}")
        rewards.append(total_reward)
    test_env.close()
    if rewards:
        print(f"\nEval summary ({len(rewards)} episodes): "
              f"mean reward = {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")


if __name__ == "__main__":
    test_agent(load_trained_agent(), n_tests=10, render=False, record_video=False)
