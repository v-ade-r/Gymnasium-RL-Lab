from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repo_paths import HALFCHEETAH_PPO_FINAL, HALFCHEETAH_PPO_RESULTS, ensure_dir
from utils.recording import CONTINUOUS_GIF_KWARGS, record_episode_gif

import gymnasium as gym
import numpy as np
import torch

from PPO import HalfCheetahAgent, make_env

MODEL_PATH = HALFCHEETAH_PPO_FINAL
ENV_ID = "HalfCheetah-v5"
NUM_ENVS = 1


def load_trained_agent(model_path=MODEL_PATH):
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"No trained agent found at {model_path}. Run PPO.py first."
        )

    envs = gym.vector.AsyncVectorEnv([make_env(ENV_ID) for _ in range(NUM_ENVS)])
    agent = HalfCheetahAgent(
        envs=envs,
        num_envs=NUM_ENVS,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        n_steps=256,
        entropy_coef=0.0,
        vf_coef=0.5,
        clip_epsilon=0.2,
        max_grad_norm=0.5,
        ppo_epochs=10,
        num_minibatches=32,
        anneal_lr=True,
        clip_vloss=True,
        target_kl=0.015,
        norm_reward=True,
        total_timesteps=10_000_000,
    )
    agent.load(model_path)
    agent.actor.eval()
    agent.critic.eval()
    envs.close()
    return agent


def _run_ppo_episode(env, agent):
    obs, _ = env.reset()
    obs = np.array(obs, dtype=np.float32)
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        norm_obs = agent.obs_rms.normalize(obs.reshape(1, -1)).astype(np.float32)
        with torch.no_grad():
            obs_tensor = torch.from_numpy(norm_obs).to(agent.device)
            dist = agent.actor(obs_tensor)
            action = dist.mean.cpu().numpy()[0]

        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, terminated, truncated, _ = env.step(action)
        obs = np.array(obs, dtype=np.float32)
        done = terminated or truncated
        total_reward += reward
        steps += 1

    return total_reward, steps


def test_agent(agent, n_tests=3, render=True, record_video=False):
    """Test trained agent with deterministic policy (Gaussian mean, no sampling)."""
    if render and record_video:
        raise ValueError("Use either render=True or record_video=True, not both.")

    print("\n--- Testing Trained HalfCheetah Agent ---")

    if render:
        render_mode = "human"
    elif record_video:
        render_mode = "rgb_array"
    else:
        render_mode = None

    test_env = gym.make(ENV_ID, render_mode=render_mode)
    gif_path = ensure_dir(HALFCHEETAH_PPO_RESULTS) / "ppo.gif"

    rewards = []
    for i in range(n_tests):
        if record_video and i == 0:
            total_reward, steps = record_episode_gif(
                test_env,
                lambda capture_env: _run_ppo_episode(capture_env, agent),
                gif_path,
                **CONTINUOUS_GIF_KWARGS,
            )
            print(f"Saved demo GIF to {gif_path}")
        else:
            total_reward, steps = _run_ppo_episode(test_env, agent)

        print(f"Test Episode {i+1} | Steps: {steps} | Reward: {total_reward:.1f}")
        rewards.append(total_reward)
    test_env.close()
    if rewards:
        print(f"\nEval summary ({len(rewards)} episodes): "
              f"mean reward = {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")


if __name__ == "__main__":
    test_agent(load_trained_agent(), n_tests=10, render=False, record_video=False)
