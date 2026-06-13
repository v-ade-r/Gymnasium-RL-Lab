# LunarLander — PPO mini-batch

Actor-Critic PPO with GAE(λ=0.95) on `LunarLander-v3`.

## Run

```bash
cd algorithms/discrete_control/lunar_lander
python PPO.py
python test.py
```

Default training: **1 000 000 timesteps**, `n_steps=1024`, `policy_lr=3e-4`, `value_lr=1e-3`, `gamma=0.99`, `lam=0.95`, `n_hidden=128`, entropy coefficient linearly annealed from `0.02` to `0.001`. PPO update: 4 epochs, mini-batch size 64, `clip_epsilon=0.2`.

Run training first, then `test.py` loads the saved checkpoint for evaluation or demo recording.

## Results

The following result comes from a single educational training run and should not be interpreted as a statistically robust benchmark.

| Metric | Value |
|---|---:|
| Environment | LunarLander-v3 |
| Algorithm | PPO (mini-batch + GAE) |
| Training steps | 1M |
| Seeds | 1 |
| Policy architecture | MLP 128×128 (actor + value) |
| Eval policy | greedy (argmax over action logits) |
| Evaluation episodes | 10 |
| Mean evaluation return | 271.12 ± 37 |

Because this experiment uses a single seed and is not hyperparameter-tuned for leaderboard scores, the result is meant to demonstrate that the implementation works, not to establish a state-of-the-art comparison.

**Demo**

<img src="../../../results/discrete_control/lunar_lander/ppo_lunar_lander.gif" width="480" alt="LunarLander PPO demo">

**Learning curve**

<img src="../../../results/discrete_control/lunar_lander/ppo_learning_curve.png" width="560" alt="PPO learning curve">

## Outputs

Learning curves saved to [`results/discrete_control/lunar_lander/`](../../../results/discrete_control/lunar_lander/):

| File | Description |
|------|-------------|
| `ppo_learning_curve.png` | Rolling average reward and steps |
| `ppo_lunar_lander.gif` | Demo rollout (`test.py`, `record_video=True`) |

Headless eval (10 episodes, no GIF):

```python
test_agent(load_trained_agent(), n_tests=10, render=False, record_video=False)
```

Model checkpoint saved to [`models/discrete_control/lunar_lander/`](../../../models/discrete_control/lunar_lander/):

| File | Description |
|------|-------------|
| `ppo_lunar_lander_final_model.pth` | Policy + value network checkpoint (used by `test.py`) |

## Key idea

Collects `n_steps` transitions per rollout, computes GAE advantages once, then runs 4 PPO epochs over shuffled mini-batches of size 64. Truncated episodes are bootstrapped with `gamma * V(s_final)`.
