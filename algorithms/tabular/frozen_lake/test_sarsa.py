import pickle
import sys
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repo_paths import FROZEN_LAKE_RESULTS, ensure_dir
from utils.recording import DISCRETE_GIF_KWARGS, record_episode_gif

from SARSA import FrozenLakeAgent, Q_VALUES_PATH


def load_trained_agent():
    if not Q_VALUES_PATH.exists():
        raise FileNotFoundError(
            f"No trained agent found at {Q_VALUES_PATH}. Run SARSA.py first."
        )

    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    agent = FrozenLakeAgent(env, epsilon=0, epsilon_decay=1.0, learning_rate=0.1)
    with Q_VALUES_PATH.open("rb") as f:
        q_values = pickle.load(f)
    agent.q_values = defaultdict(lambda: np.zeros(env.action_space.n), q_values)
    return agent


def test_agent(agent, n_tests: int = 3, render=True, record_video=False):
    if render and record_video:
        raise ValueError("Use either render=True or record_video=True, not both.")

    print("\n--- Testing Trained Agent (SARSA) ---")

    if render:
        render_mode = "human"
    elif record_video:
        render_mode = "rgb_array"
    else:
        render_mode = None

    test_env = gym.make(
        'FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode=render_mode
    )
    evaluator = FrozenLakeAgent(test_env, epsilon=0, epsilon_decay=1.0, learning_rate=0.1)
    evaluator.q_values = agent.q_values

    gif_path = ensure_dir(FROZEN_LAKE_RESULTS) / "sarsa.gif"
    rewards = []
    for i in range(n_tests):
        if record_video and i == 0:
            def run_episode(capture_env):
                old_env = evaluator.env
                evaluator.env = capture_env
                try:
                    return evaluator.run_episode(training=False)
                finally:
                    evaluator.env = old_env

            reward, steps = record_episode_gif(
                test_env, run_episode, gif_path, **DISCRETE_GIF_KWARGS,
            )
            print(f"Saved demo GIF to {gif_path}")
        else:
            reward, steps = evaluator.run_episode(training=False)
        print(f"Test Episode {i+1} | Steps: {steps} | Reward: {reward}")
        rewards.append(reward)
    test_env.close()
    if rewards:
        print(f"\nEval summary ({len(rewards)} episodes): "
              f"mean reward = {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")


if __name__ == "__main__":
    test_agent(load_trained_agent(), n_tests=10, render=False, record_video=False)
