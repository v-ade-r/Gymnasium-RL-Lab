import gymnasium as gym
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repo_paths import HALFCHEETAH_SAC_FINAL, HALFCHEETAH_SAC_RESULTS, ensure_dir
from utils.recording import CONTINUOUS_GIF_KWARGS, record_episode_gif
from models import Actor
from sac import Args


def _run_sac_episode(env, actor, device):
    obs, _ = env.reset()
    terminated = False
    truncated = False
    ep_reward = 0.0

    while not (terminated or truncated):
        with torch.no_grad():
            mu, _ = actor(torch.FloatTensor(obs).to(device))
            action = torch.tanh(mu).cpu().numpy()

        obs, reward, terminated, truncated, _ = env.step(action)
        ep_reward += reward

    return ep_reward


def evaluate(model_path, num_episodes=5, render=True, record_video=False):
    """Test the agent with a deterministic policy (tanh(mean), no sampling)."""
    if render and record_video:
        raise ValueError("Use either render=True or record_video=True, not both.")

    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if render:
        render_mode = "human"
    elif record_video:
        render_mode = "rgb_array"
    else:
        render_mode = None

    env = gym.make(args.env_id, render_mode=render_mode)
    gif_path = ensure_dir(HALFCHEETAH_SAC_RESULTS) / "sac.gif"

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    actor = Actor(obs_dim, action_dim).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint['actor'])
    actor.eval()

    print(f"Loaded model from {model_path}. Starting evaluation...")

    rewards = []
    for ep in range(num_episodes):
        if record_video and ep == 0:
            ep_reward = record_episode_gif(
                env,
                lambda capture_env: _run_sac_episode(capture_env, actor, device),
                gif_path,
                **CONTINUOUS_GIF_KWARGS,
            )
            print(f"Saved demo GIF to {gif_path}")
        else:
            ep_reward = _run_sac_episode(env, actor, device)

        print(f"Episode {ep+1}: Reward = {ep_reward:.2f}")
        rewards.append(ep_reward)

    env.close()
    if rewards:
        print(f"\nEval summary ({len(rewards)} episodes): "
              f"mean reward = {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")


if __name__ == "__main__":
    evaluate(HALFCHEETAH_SAC_FINAL, num_episodes=10, render=False, record_video=False)
