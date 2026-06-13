# PPO — HalfCheetah-v5

Proximal Policy Optimization for MuJoCo `HalfCheetah-v5`.

## Run

```bash
cd algorithms/continuous_control/halfcheetah/ppo && python PPO.py
python test.py
```

## Results

The following result comes from a single educational training run and should not be interpreted as a statistically robust benchmark.

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

Because this experiment uses a single seed and is not hyperparameter-tuned for leaderboard scores, the result is meant to demonstrate that the implementation works, not to establish a state-of-the-art comparison. The **evaluation return** is from `test.py` on `halfcheetah_ppo_final.pth` (deterministic Gaussian mean, 10 episodes).

**Demo**

<img src="../../../../results/continuous_control/halfcheetah/ppo/ppo.gif" width="480" alt="PPO demo">

**Learning curve**

<img src="../../../../results/continuous_control/halfcheetah/ppo/ppo_avg_return.png" width="560" alt="PPO learning curve">

## Visualizing results

After training, evaluate and record a demo:

```bash
python PPO.py          # train → models/continuous_control/halfcheetah/ppo/halfcheetah_ppo_final.pth
python test.py         # record GIF → results/continuous_control/halfcheetah/ppo/ppo.gif
```

Headless eval without GIF:

```bash
python test.py   # default: 10 episodes, render=False
```

Committed artifacts live in [`results/continuous_control/halfcheetah/ppo/`](../../../../results/continuous_control/halfcheetah/ppo/).

## Why PPO here

- **On-policy stability:** clipped surrogate objective limits destructive policy updates
- **Parallel rollouts:** 8 `AsyncVectorEnv` workers for fast data collection
- **MuJoCo best practices:** observation normalization, truncation bootstrap, SB3-style KL early stopping, `log_std` clamp

## Requirements

See [`requirements/mujoco.txt`](../../../../requirements/mujoco.txt).
