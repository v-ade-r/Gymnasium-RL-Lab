# HalfCheetah-v5 — SAC vs PPO vs Evolution

Three paradigms on the same MuJoCo locomotion task:

| Approach | Folder | Policy type |
|----------|--------|---------------|
| SAC | [`sac/`](sac/) | Stochastic policy, entropy-regularized off-policy updates |
| PPO | [`ppo/`](ppo/) | Gaussian policy, clipped on-policy updates |
| Evolution | [`evolution/`](evolution/) | Deterministic MLP policy, gradient-free optimization |

HalfCheetah trainers log to W&B project **`gymnasium-rl-lab`**. How to train, evaluate, and
record demos is documented in each folder's README (linked below).

**SAC vs PPO in the code.** The [`sac/`](sac/) sources (`sac.py`, `models.py`, `buffer.py`) contain
inline comments tagged **`[SAC vs PPO]`** — they contrast this off-policy implementation with
[`ppo/PPO.py`](ppo/PPO.py) on the same environment (replay buffer vs rollout buffer, Q(s,a) vs V(s),
tanh squashing with Jacobian correction, update loop, and related design choices). Read SAC and PPO
side by side when learning how continuous RL variants differ.

## Results

The following numbers come from single educational training runs (one seed each) and should
not be interpreted as statistically robust benchmarks.

**Training curves in `results/`.** Several trainers can save matplotlib plots locally
(e.g. PPO writes `learning_curve_HalfCheetah_PPO.png` after a full run). The PNGs
committed in this repo under `results/continuous_control/halfcheetah/` were **exported
manually from W&B** (not from those matplotlib outputs) — they match the charts logged
during training and are kept here as static portfolio artifacts.

### At a glance

| Algorithm | Paradigm | Training steps | Best reported return | Policy | Learning Curves |
|-----------|----------|---------------:|---------------------:|--------|-----------------|
| SAC | Off-policy RL | 3M | 15,199 ± 63 (eval) | MLP 256×256 | <img src="../../../results/continuous_control/halfcheetah/sac/sac_avg_return_100.png" width="120" alt="SAC learning curve"> |
| PPO | On-policy RL | 10M | 8,669 ± 86 (eval) | MLP 256×256 | <img src="../../../results/continuous_control/halfcheetah/ppo/ppo_avg_return.png" width="120" alt="PPO learning curve"> |
| CMA-ES | Evolution | 20M | 2,141 ± 880 (eval) | MLP 32×32 | <img src="../../../results/continuous_control/halfcheetah/evolution/CMA-ES_best_overall_return.png" width="120" alt="CMA-ES learning curve"> |
| sNES | Evolution | 20M | 2,990 ± 66 (eval) | MLP 32×32 | <img src="../../../results/continuous_control/halfcheetah/evolution/sNES_best_overall_return.png" width="120" alt="sNES learning curve"> |
| MAP-Elites | Evolution / Quality-Diversity | 50M | 1,644 ± 751 (eval) | MLP 32×32 | <img src="../../../results/continuous_control/halfcheetah/evolution/MAP-Elites_best_fitness.png" width="120" alt="MAP-Elites learning curve"> |

Deep RL (SAC, PPO) reaches the highest returns with larger networks and gradient-based
updates. Evolution methods use a compact MLP (32×32, d ≈ 1.8k) rather than the 256×256
policies above — a deliberate choice to keep the search space small under local hardware
constraints (population-based trainers evaluate many candidates per generation and are far
more memory- and compute-intensive per step than a single gradient update). Lower returns
are therefore expected; MAP-Elites in particular trades peak performance for a diverse
behavior archive. None of these runs are multi-seed benchmarks; they document working
implementations on the same environment.

### SAC

| Metric | Value |
|---|---:|
| Environment | HalfCheetah-v5 |
| Algorithm | SAC |
| Training steps | 3M |
| Seeds | 1 |
| Policy architecture | MLP 256×256 (actor + twin Q-networks) |
| Training return, last 100 episodes | 14,819 |
| Deterministic evaluation episodes | 10 |
| Deterministic evaluation return | 15,199 ± 63 |

More detail: [`sac/README.md`](sac/README.md).

| Artifact | Path |
|----------|------|
| Training curve | [`sac_avg_return_100.png`](../../../results/continuous_control/halfcheetah/sac/sac_avg_return_100.png) |
| Demo GIF | [`sac.gif`](../../../results/continuous_control/halfcheetah/sac/sac.gif) |

### PPO

| Metric | Value |
|---|---:|
| Environment | HalfCheetah-v5 |
| Algorithm | PPO |
| Training steps | 10M |
| Seeds | 1 |
| Policy architecture | MLP 256×256 (separate actor + critic) |
| Training return, last 100 episodes | 6,972 |
| Deterministic evaluation episodes | 10 |
| Deterministic evaluation return | 8,669 ± 86 |

More detail: [`ppo/README.md`](ppo/README.md).

| Artifact | Path |
|----------|------|
| Training curve | [`ppo_avg_return.png`](../../../results/continuous_control/halfcheetah/ppo/ppo_avg_return.png) |
| Demo GIF | [`ppo.gif`](../../../results/continuous_control/halfcheetah/ppo/ppo.gif) |

### CMA-ES

| Metric | Value |
|---|---:|
| Environment | HalfCheetah-v5 |
| Algorithm | CMA-ES (IPOP restarts) |
| Training steps | 20M |
| Seeds | 1 |
| Policy architecture | MLP 32×32 (d = 1,830) |
| Population λ | 32 → 128 (2 IPOP restarts) |
| Training return, honest re-eval (30 episodes) | 2,344 ± 503 |
| Independent evaluation episodes | 10 |
| Independent evaluation return | 2,141 ± 880 |

The **training return** is from the final re-evaluation inside `CMA-ES.py`: top 10
single-episode finalists plus the distribution mean, each averaged over 30 episodes;
the distribution mean won (2,343.5 ± 502.8). The **evaluation return** is a separate
`test_all.py cmaes` run on the saved checkpoint.

| Artifact | Path |
|----------|------|
| Training curve (`charts/best_overall_return` vs env steps) | [`CMA-ES_best_overall_return.png`](../../../results/continuous_control/halfcheetah/evolution/CMA-ES_best_overall_return.png) |

### sNES

| Metric | Value |
|---|---:|
| Environment | HalfCheetah-v5 |
| Algorithm | sNES (separable NES, default in `NES.py`) |
| Training steps | 20M |
| Seeds | 1 |
| Policy architecture | MLP 32×32 (d = 1,830) |
| Population λ | 32 → 128 (2 restarts) |
| Training return, honest re-eval (30 episodes) | 2,950 ± 238 |
| Independent evaluation episodes | 10 |
| Independent evaluation return | 2,990 ± 66 |

The **training return** is from the final re-evaluation inside `NES.py`: top 10
single-episode finalists plus the distribution mean, each averaged over 30 episodes;
`finalist_6` won (2,950.2 ± 237.5). The **evaluation return** is a separate
`test_all.py nes` run on the saved checkpoint (2,990.0 ± 66.3; min 2,877.2, max 3,103.4).

| Artifact | Path |
|----------|------|
| Training curve (`charts/best_overall_return` vs env steps) | [`sNES_best_overall_return.png`](../../../results/continuous_control/halfcheetah/evolution/sNES_best_overall_return.png) |

### MAP-Elites (CMA-MAE)

| Metric | Value |
|---|---:|
| Environment | HalfCheetah-v5 |
| Algorithm | CMA-MAE (soft-threshold MAP-Elites emitters) |
| Training steps | 50M |
| Seeds | 1 |
| Policy architecture | MLP 32×32 (d = 1,830) |
| Archive | 25×25 grid, BDs (vx, \|action\|) |
| Coverage (filled cells) | 8.8% (55 / 625) |
| Training best fitness, honest re-eval (10 episodes) | 1,914 |
| Independent evaluation episodes | 10 |
| Independent evaluation return | 1,644 ± 751 |

MAP-Elites optimizes a **diverse archive** of policies, not a single peak performer —
lower return than sNES/CMA-ES is expected. The product is the behavior map
(fast/slow × high/low control effort), not one best cheetah.

The **training best fitness** is from the final archive re-evaluation inside
`MAP-Elites.py` (every elite averaged over 10 episodes). The **evaluation return** is
a separate `test_all.py map-elites evaluate --best` run on the saved archive.

| Artifact | Path |
|----------|------|
| Training curve (`qd/best_fitness` vs env steps) | [`MAP-Elites_best_fitness.png`](../../../results/continuous_control/halfcheetah/evolution/MAP-Elites_best_fitness.png) |
| Archive heatmap | [`MAP_Elites_heatmap_final.png`](../../../results/continuous_control/halfcheetah/evolution/MAP_Elites_heatmap_final.png) |

More detail on evolution trainers (run commands, `test_all.py`, NES variants):
[`evolution/README.md`](evolution/README.md).
