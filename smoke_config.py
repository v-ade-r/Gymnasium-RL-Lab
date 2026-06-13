"""Smoke-test mode: short training budgets for CI and sanity checks.

Enable via ``--smoke`` on the command line or ``SMOKE=1`` in the environment.
"""

from __future__ import annotations

import os
import sys
from typing import Any

_smoke: bool | None = None

TABULAR_EPISODES = 50
LUNAR_LANDER_TIMESTEPS = 4_096
HALFCHEETAH_PPO_TIMESTEPS = 4_096
HALFCHEETAH_PPO_NUM_ENVS = 2
HALFCHEETAH_SAC_TIMESTEPS = 10_000
HALFCHEETAH_SAC_LEARNING_STARTS = 500
HALFCHEETAH_SAC_BUFFER_SIZE = 10_000
EVOLUTION_TIMESTEPS = 8_000
EVOLUTION_N_WORKERS = 2
EVOLUTION_POP_SIZE = 8
MAP_ELITES_EMITTERS = 2
MAP_ELITES_EMITTER_POP = 4
MAP_ELITES_RESOLUTION = (10, 10)


def smoke_mode() -> bool:
    """Return True once if smoke mode is active; strip ``--smoke`` from argv."""
    global _smoke
    if _smoke is None:
        _smoke = (
            os.environ.get("SMOKE", "").strip().lower() in ("1", "true", "yes")
            or "--smoke" in sys.argv
        )
        while "--smoke" in sys.argv:
            sys.argv.remove("--smoke")
        if _smoke:
            os.environ.setdefault("WANDB_MODE", "disabled")
    return _smoke


def apply_sac_smoke(args: Any) -> Any:
    if not smoke_mode():
        return args
    args.total_timesteps = HALFCHEETAH_SAC_TIMESTEPS
    args.learning_starts = HALFCHEETAH_SAC_LEARNING_STARTS
    args.buffer_size = HALFCHEETAH_SAC_BUFFER_SIZE
    args.save_every = 999_999_999
    args.exp_name = f"{args.exp_name}_smoke"
    return args


def apply_cma_es_smoke(args: Any) -> Any:
    if not smoke_mode():
        return args
    args.total_timesteps = EVOLUTION_TIMESTEPS
    args.n_workers = min(args.n_workers, EVOLUTION_N_WORKERS)
    args.pop_size = EVOLUTION_POP_SIZE
    args.save_every_gens = 10_000
    args.exp_name = f"{args.exp_name}_smoke"
    return args


def apply_nes_smoke(args: Any) -> Any:
    return apply_cma_es_smoke(args)


def apply_map_elites_smoke(args: Any) -> Any:
    if not smoke_mode():
        return args
    args.total_timesteps = EVOLUTION_TIMESTEPS
    args.n_workers = min(args.n_workers, EVOLUTION_N_WORKERS)
    args.n_emitters = MAP_ELITES_EMITTERS
    args.emitter_pop_size = MAP_ELITES_EMITTER_POP
    args.archive_resolution = MAP_ELITES_RESOLUTION
    if hasattr(args, "n_episodes_per_candidate"):
        args.n_episodes_per_candidate = 1
    args.save_every_gens = 10_000
    args.heatmap_every_gens = 10_000
    args.exp_name = f"{args.exp_name}_smoke"
    return args


def apply_halfcheetah_ppo_config(config: dict) -> dict:
    if not smoke_mode():
        return config
    updated = dict(config)
    updated["total_timesteps"] = HALFCHEETAH_PPO_TIMESTEPS
    updated["num_envs"] = HALFCHEETAH_PPO_NUM_ENVS
    updated["exp_name"] = f"{updated['exp_name']}_smoke"
    return updated
