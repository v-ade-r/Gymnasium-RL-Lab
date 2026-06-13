"""Shared Weights & Biases setup for HalfCheetah trainers.

Metric keys follow CleanRL-style prefixes so SAC, PPO, and evolution runs
plot on the same axes in the W&B dashboard. W&B does not auto-capture rewards;
every metric must be passed explicitly to wandb.log().
"""

from dataclasses import asdict, is_dataclass
from typing import Optional

import wandb

WANDB_PROJECT = "gymnasium-rl-lab"

# Shared across RL trainers (SAC, PPO)
METRIC_EPISODIC_RETURN = "charts/episodic_return"
METRIC_EPISODIC_LENGTH = "charts/episodic_length"
METRIC_AVG_RETURN_100 = "charts/avg_return_100"
METRIC_AVG_STEPS_100 = "charts/avg_steps_100"
METRIC_LEARNING_RATE = "charts/learning_rate"

# Shared across evolution trainers (CMA-ES, NES, MAP-Elites)
METRIC_BEST_RETURN = "charts/best_return"
METRIC_MEAN_RETURN = "charts/mean_return"
METRIC_WORST_RETURN = "charts/worst_return"
METRIC_BEST_OVERALL_RETURN = "charts/best_overall_return"
METRIC_GENERATION = "charts/generation"
METRIC_TOTAL_ENV_STEPS = "charts/total_env_steps"
METRIC_GEN_SECONDS = "charts/gen_seconds"
METRIC_OBS_NORM_COUNT = "charts/obs_norm_count"


def _to_config(config) -> dict:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return config
    return dict(vars(config))


def init_wandb(exp_name: str, config, tags: Optional[list] = None, group: Optional[str] = None):
    return wandb.init(
        project=WANDB_PROJECT,
        name=exp_name,
        config=_to_config(config),
        tags=tags or [],
        group=group,
    )


def finish_wandb():
    if wandb.run is not None:
        wandb.finish()
