# Evolution — HalfCheetah-v5

Gradient-free evolutionary methods for MuJoCo `HalfCheetah-v5`. All three trainers
optimize a small deterministic MLP (32×32, **d ≈ 1,830** flat parameters) with
Welford observation normalization and parallel rollout evaluation.

| Script | Algorithm | Output |
|--------|-----------|--------|
| `CMA-ES.py` | CMA-ES with IPOP restarts | `models/.../evolution/cmaes_final_model.npz` |
| `NES.py` | sNES (default) or xNES | `snes_final_model.npz` / `xnes_final_model.npz` |
| `MAP-Elites.py` | CMA-MAE (MAP-Elites emitters) | `models/.../evolution/cma_me_final_archive.npz` |

## Run

```bash
cd algorithms/continuous_control/halfcheetah/evolution
python CMA-ES.py
python NES.py              # sNES (default)
python MAP-Elites.py
```

To run full-covariance xNES instead of sNES, set `Args.nes_variant = "xnes"` in `NES.py`.

Training logs to W&B project **`gymnasium-rl-lab`** (requires `wandb login`; disable with
`WANDB_MODE=disabled` for smoke tests).

## Results

The following numbers come from single educational training runs (one seed each) and
should not be interpreted as statistically robust benchmarks.

**Training curves in `results/`.** Trainers log metrics to W&B during training. The PNGs
committed under `results/continuous_control/halfcheetah/evolution/` were **exported
manually from W&B** — they are static portfolio artifacts, not matplotlib outputs from
the training scripts.

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

Because these experiments use a single seed and are not hyperparameter-tuned for
leaderboard scores, the results demonstrate that the implementations work, not
state-of-the-art competition scores.

## Visualizing results

Evaluate saved checkpoints with the unified CLI:

```bash
python test_all.py cmaes
python test_all.py nes
python test_all.py nes --variant xnes
python test_all.py map-elites list-top --n 20
python test_all.py map-elites evaluate --best
python test_all.py map-elites evaluate --cell 45 5
python test_all.py map-elites heatmap --save-path archive_heatmap.png
python test_all.py map-elites heatmap --show
```

Run `python test_all.py -h` for all options.

Committed artifacts live in
[`results/continuous_control/halfcheetah/evolution/`](../../../../results/continuous_control/halfcheetah/evolution/):

| File | Description |
|------|-------------|
| `CMA-ES_best_overall_return.png` | CMA-ES training curve (`charts/best_overall_return`) |
| `sNES_best_overall_return.png` | sNES training curve (`charts/best_overall_return`) |
| `MAP-Elites_best_fitness.png` | MAP-Elites training curve (`qd/best_fitness`) |
| `MAP_Elites_heatmap_final.png` | Archive heatmap (behavior descriptors) |

## Why evolution here

- **No gradients:** optimizes policy weights directly — complements SAC/PPO on the same env
- **Small policy, large search:** 32×32 MLP keeps **d ≈ 1,830** tractable for CMA-ES / NES
- **Shared infrastructure:** Welford obs normalization, multiprocessing rollouts, honest final re-eval
- **MAP-Elites:** quality-diversity archive (velocity × control effort), not a single peak policy

## NES variant: sNES is the default (and why)

`NES.py` ships two variants, selected with `nes_variant` in `Args`. **Default is `"snes"`**
(separable / diagonal NES, Schaul et al. 2011).

Why sNES rather than full-covariance xNES at this budget? HalfCheetah episodes are always
1000 steps, so 20M steps ≈ a few hundred generations:

- **Learning rate.** xNES covariance rate `(9 + 3 ln d) / (5 d √d) ≈ 8e-5` for `d = 1830` —
  the `1/d` factor freezes the shape matrix (~0.025 movement over the whole run). sNES uses
  `(3 + ln d) / (5 √d) ≈ 0.049`, about **600× larger**, so per-coordinate scales adapt.
- **Memory.** xNES stores a `d × d` matrix (~27 MB) and does `O(d²)` work per generation;
  sNES keeps a length-`d` std vector (`O(d)`).

Both variants share rank utilities, mirrored sampling, Welford normalization, L2 weight decay,
restart-on-stagnation, and honest final re-evaluation. Checkpoints are separate so runs do not
overwrite each other.

### Which NES variant to reach for

| Variant | Adapts | Best when | Cost |
|---------|--------|-----------|------|
| **sNES** (default) | per-coordinate scale (diagonal) | **high d / few generations / limited RAM** — this setup | `O(d)` |
| **xNES** (full covariance) | full scale + rotations/correlations | **low d (≤ few hundred)** with many affordable generations | `O(d²)`–`O(d³)` |

Rule of thumb: pick **xNES** (or CMA-ES) when a full covariance fits the budget and parameter
coupling matters; pick **sNES** when `d` is large and generations are scarce.

## Requirements

See [`requirements/mujoco.txt`](../../../../requirements/mujoco.txt).
