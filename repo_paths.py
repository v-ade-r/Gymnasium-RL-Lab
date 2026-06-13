"""Repository output paths for results, trained models, and other artifacts."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

RESULTS = REPO_ROOT / "results"
MODELS = REPO_ROOT / "models"

# Mirror algorithms/ layout under results/
FROZEN_LAKE_RESULTS = RESULTS / "tabular" / "frozen_lake"
LUNAR_LANDER_RESULTS = RESULTS / "discrete_control" / "lunar_lander"
HALFCHEETAH_SAC_RESULTS = RESULTS / "continuous_control" / "halfcheetah" / "sac"
HALFCHEETAH_PPO_RESULTS = RESULTS / "continuous_control" / "halfcheetah" / "ppo"
HALFCHEETAH_EVOLUTION_RESULTS = RESULTS / "continuous_control" / "halfcheetah" / "evolution"

# Backward-compatible aliases used by evolution trainers
HALFCHEETAH_EVOLUTION = HALFCHEETAH_EVOLUTION_RESULTS
HALFCHEETAH_RESULTS = RESULTS / "continuous_control" / "halfcheetah"

# Mirror algorithms/ layout under models/
HALFCHEETAH_PPO_MODELS = MODELS / "continuous_control" / "halfcheetah" / "ppo"
HALFCHEETAH_SAC_MODELS = MODELS / "continuous_control" / "halfcheetah" / "sac"
HALFCHEETAH_EVOLUTION_MODELS = MODELS / "continuous_control" / "halfcheetah" / "evolution"
LUNAR_LANDER_MODELS = MODELS / "discrete_control" / "lunar_lander"
FROZEN_LAKE_MODELS = MODELS / "tabular" / "frozen_lake"

# Final checkpoints (used by trainers and test scripts)
HALFCHEETAH_PPO_FINAL = HALFCHEETAH_PPO_MODELS / "halfcheetah_ppo_final.pth"
HALFCHEETAH_SAC_FINAL = HALFCHEETAH_SAC_MODELS / "sac_final_model.pth"
CMAES_FINAL = HALFCHEETAH_EVOLUTION_MODELS / "cmaes_final_model.npz"
# NES family: one final model per variant so comparative runs don't overwrite.
SNES_FINAL = HALFCHEETAH_EVOLUTION_MODELS / "snes_final_model.npz"
XNES_FINAL = HALFCHEETAH_EVOLUTION_MODELS / "xnes_final_model.npz"
MAP_ELITES_FINAL = HALFCHEETAH_EVOLUTION_MODELS / "cma_me_final_archive.npz"
LUNAR_LANDER_FINAL = LUNAR_LANDER_MODELS / "ppo_lunar_lander_final_model.pth"
FROZEN_LAKE_Q_SARSA = FROZEN_LAKE_MODELS / "q_values_SARSA.pkl"
FROZEN_LAKE_Q_LEARNING = FROZEN_LAKE_MODELS / "q_values_Q-learning.pkl"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_path(model_dir: Path, filename: str | Path) -> Path:
    """Build a path under the algorithm-specific models directory."""
    path = Path(filename)
    if path.is_absolute():
        return path
    return model_dir / path
