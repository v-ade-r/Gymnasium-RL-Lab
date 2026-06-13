# SAC — HalfCheetah-v5

Soft Actor-Critic for MuJoCo `HalfCheetah-v5`.

## Run

```bash
cd algorithms/continuous_control/halfcheetah/sac && python sac.py
python test.py
```

## Results

The following result comes from a single educational training run and should not be interpreted as a statistically robust benchmark.

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

Because this experiment uses a single seed and is not hyperparameter-tuned for leaderboard scores, the result is meant to demonstrate that the implementation works, not to establish a state-of-the-art comparison.

## Visualizing results

After training, evaluate and record a demo:

```bash
python sac.py          # train → models/continuous_control/halfcheetah/sac/sac_final_model.pth
python test.py         # record GIF → results/continuous_control/halfcheetah/sac/sac.gif
```

Headless eval without GIF:

```python
evaluate(..., num_episodes=10, render=False, record_video=False)
```

Committed artifacts live in [`results/continuous_control/halfcheetah/sac/`](../../../../results/continuous_control/halfcheetah/sac/).

## Why SAC here

- **Sample efficiency:** off-policy learning with 1M replay buffer
- **Maximum entropy:** reward + exploration (H(π))
- **Twin Q-networks:** reduces overestimation in continuous actions

## SAC vs PPO (comments in source)

`sac.py`, `models.py`, and `buffer.py` annotate design choices with **`[SAC vs PPO]`**, comparing
this trainer to [`../ppo/PPO.py`](../ppo/PPO.py) on the same task — e.g. replay vs on-policy
rollouts, critic architecture, entropy and tanh-Jacobian in the actor, target networks, and the
step-by-step training loop. The module docstring in `sac.py` points to the same tag throughout
the folder.

## Jacobian correction

Tanh-squashed Gaussian policy with corrected log-prob:

```
log_prob -= (2 * (log(2) - z - softplus(-2 * z))).sum(dim=-1)
```

## Requirements

See [`requirements/mujoco.txt`](../../../../requirements/mujoco.txt).
