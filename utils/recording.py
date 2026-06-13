"""GIF episode recording for evaluation demos."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v3 as iio
import numpy as np

# Short discrete episodes (e.g. FrozenLake): brief pause per grid step.
DISCRETE_GIF_KWARGS = {"hold_frames": 3, "fps": 10, "frame_skip": 1}

def continuous_gif_kwargs(
    *,
    expected_steps: int = 1000,
    target_duration_sec: float = 5.0,
    fps: int = 20,
    hold_frames: int = 1,
) -> dict:
    """Pick ``frame_skip`` so a full rollout GIF is about ``target_duration_sec`` long."""
    target_frames = max(2, int(target_duration_sec * fps))
    frame_skip = max(1, (expected_steps + 1) // target_frames)
    return {
        "hold_frames": hold_frames,
        "fps": fps,
        "frame_skip": frame_skip,
    }


# Long continuous rollouts (HalfCheetah etc.): ~5 s for a 1000-step episode.
CONTINUOUS_GIF_KWARGS = continuous_gif_kwargs()


class FrameCaptureWrapper(gym.Wrapper):
    """Capture a copied rgb_array frame after every reset and step."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.frames: list[np.ndarray] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._capture()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._capture()
        return obs, reward, terminated, truncated, info

    def _capture(self):
        frame = self.env.render()
        if frame is not None:
            self.frames.append(np.asarray(frame, dtype=np.uint8).copy())


def save_episode_gif(
    frames: list[np.ndarray],
    output_path: Path,
    *,
    hold_frames: int = 3,
    fps: int = 10,
    frame_skip: int = 1,
) -> None:
    if not frames:
        raise ValueError("No frames to save.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected = frames[::frame_skip]
    expanded = [frame for frame in selected for _ in range(hold_frames)]
    iio.imwrite(output_path, expanded, extension=".gif", fps=fps, loop=0)


def record_episode_gif(
    env: gym.Env,
    run_episode_fn: Callable[[gym.Env], Any],
    output_path: Path,
    *,
    hold_frames: int = 3,
    fps: int = 10,
    frame_skip: int = 1,
) -> Any:
    """Run ``run_episode_fn`` on a frame-capturing env wrapper and save a GIF."""
    wrapped = FrameCaptureWrapper(env)
    result = run_episode_fn(wrapped)
    save_episode_gif(
        wrapped.frames,
        output_path,
        hold_frames=hold_frames,
        fps=fps,
        frame_skip=frame_skip,
    )
    return result
