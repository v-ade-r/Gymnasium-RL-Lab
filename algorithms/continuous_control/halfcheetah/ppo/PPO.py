import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import wandb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from smoke_config import apply_halfcheetah_ppo_config, smoke_mode
from repo_paths import (
    HALFCHEETAH_PPO_FINAL,
    HALFCHEETAH_PPO_MODELS,
    HALFCHEETAH_PPO_RESULTS,
    ensure_dir,
    model_path,
)
from wandb_utils import (
    finish_wandb,
    init_wandb,
    METRIC_AVG_RETURN_100,
    METRIC_AVG_STEPS_100,
    METRIC_EPISODIC_LENGTH,
    METRIC_EPISODIC_RETURN,
    METRIC_LEARNING_RATE,
)

"""
State-of-the-art PPO for MuJoCo HalfCheetah-v5

Key differences vs discrete_control PPO:

- Continuous actions → Gaussian policy N(mu(s), diag(exp(log_std))) instead of Categorical.
  log_std is a learnable parameter INDEPENDENT of state — state-dependent std oscillates
  in PPO and breaks the trust region. Constant log_std gives more stable training.

- MLP [256, 256] with tanh — discrete_control PPO also uses an MLP backbone but with ReLU.
  HalfCheetah observation is a 17D vector (joint positions + velocities), same low-dimensional
  state-vector setting as discrete_control (not pixels). Tanh instead of ReLU — fits continuous control better,
  because it bounds activations and stabilizes gradients in deeper layers.

- Separate Actor/Critic networks — in continuous control a shared backbone is worse.
  Critic optimizes MSE to returns (scale ~thousands), actor optimizes policy
  gradient (scale ~0.01). These very different gradients interfere in a shared backbone.
  "Implementation Matters in Deep RL" (Engstrom et al.) confirms this observation.

- Observation normalization (RunningMeanStd) — CRITICAL for MuJoCo. Different observation
  dimensions have drastically different scales (qpos ~0.1 vs qvel ~10). We normalize
  to ~N(0,1) using Welford's algorithm.

- No tanh squashing of actions — that is a technique from SAC, where the entropy term requires exact
  log_prob with Jacobian correction. In PPO the clipped objective itself limits policy changes,
  so squashing is not needed. Actions are simply clipped to env bounds.

- entropy_coef = 0 — Gaussian with learnable std explores on its own (std decreases
  naturally during training). Entropy bonus is useful in discrete (Categorical),
  in continuous it is unnecessary.

- clip_epsilon = 0.2 (discrete_control PPO also uses 0.2) — standard for continuous control.
  Continuous actions have smaller ratio shifts than discrete, so we tolerate a larger clip.

- Learning rate annealing — linear decay to 0, prevents oscillations at the end of
  training when policy is already close to optimum.

- Value function clipping — limits jumps in value predictions, analogous to
  policy clipping. Stabilizes critic training.

- Advantage normalization per MINIBATCH — each minibatch has zero-mean advantages,
  which prevents bias in the gradient direction within the minibatch.

- Reward scaling (running return variance) — divides rewards by sqrt(var) of current
  discounted return sums. We do not subtract the mean (that would introduce bias in the policy
  gradient — the agent starts treating neutral rewards as penalties), we only scale
  by std. Stabilizes the scale of returns and advantage estimation.

- Truncation vs Termination — correct handling of time-limited episodes (HalfCheetah
  truncates after 1000 steps). When an episode ends due to time limit (truncation),
  V(s_next) is NOT 0 — the agent could have kept running. We bootstrap V(s_final) from the critic
  and add gamma * V(s_final) to the reward at the last step. Without this the agent systematically
  undervalues states near the time limit (because it "thinks" that V(s_1000) = 0).

- KL Early Stopping — monitors approximated KL divergence between old and new
  policy in each minibatch. If KL > target_kl (default 0.015), it stops
  the ppo_epochs loop for that update. Prevents catastrophic policy changes
  when an unlucky minibatch "pulls" the policy too far beyond the trust region.

Expected results: ~5000-8000+ reward in 2M timesteps (~15-30 min on RTX 3070).
Based on: CleanRL, "Implementation Matters" (Engstrom et al., 2020), SpinningUp.
"""


class RunningMeanStd:
    """Welford's online algorithm — maintains running mean and variance.

    Normalizes observations to ~N(0,1). Critical for MuJoCo where:
    - qpos (joint positions):    values ~[-0.1, 0.1]
    - qvel (joint velocities):   values ~[-10, 10]
    - other features:            may have yet other scales

    Without normalization, neurons in the first MLP layer must simultaneously
    process signals at scales differing by 100x — gradients dominated
    by high-scale dimensions, low-scale dimensions do not learn.

    Welford's algorithm instead of naive (sum/count):
    Naive method: mean = sum(x) / N, var = sum(x²)/N - mean²
    Problem: with large N, sum(x²) and mean² are close → catastrophic cancellation
    (floating point precision loss). Welford accumulates variance incrementally,
    avoiding this problem.

    Batch update (Chan et al., 1979) — combines statistics of two groups:
    combined_var = (n_a * var_a + n_b * var_b + delta² * n_a * n_b / n_total) / n_total
    The third term corrects for the difference in means between groups.
    """
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, batch):
        batch = np.asarray(batch, dtype=np.float64)
        # reshape(-1, *self.mean.shape) works universally:
        # - shape=(17,): obs (8,17) → (8,17), single obs (17,) → (1,17)
        # - shape=():    returns (8,) → (8,), scalar → (1,)
        batch = batch.reshape(-1, *self.mean.shape)
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x):
        return (x - self.mean.astype(np.float32)) / (np.sqrt(self.var.astype(np.float32)) + 1e-8)


# Floor (and ceiling) on log_std — safeguard against DISTRIBUTION DEGENERATION, NOT a remedy
# for KL explosion (that is what KL early stopping in update() is for).
# Consciously set LOW: empirically this agent's best result came at
# std≈0.055 (Avg R 5434), so the floor MUST be clearly below that regime so as not to
# cut it off. Training crash occurred at fixed std≈0.056 and KL=7.69 — it was KL,
# not low std, that destroyed the policy. Floor only protects against std → ~0, where
# log_prob/ratio become numerically unstable. Clamping log_std is standard from
# SAC (here sac/models.py uses -5) and many continuous PPO impls.
LOG_STD_MIN = -4.0   # std ≳ 0.018 — below the tested regime (0.055–0.135), does not interfere
LOG_STD_MAX = 1.0    # std ≲ 2.72 (hygiene, does not let std escape upward)


class Actor(nn.Module):
    """Gaussian policy: obs → N(mean(obs), diag(exp(log_std))).

    Architecture:
    - 2 hidden layers [256, 256] with tanh activation
    - mean_head: 256 → action_dim, with gain=0.01 (small weights → mean close to 0 at start
      → actions initially small and symmetric, which is natural for locomotion)
    - log_std: learnable parameter, initialized to log_std_init=-2.0 → std≈0.135
      at start (SB3-zoo recipe for HalfCheetah). Low initial std + low LR
      prevents premature collapse of exploration in heavily trained PPO.

    Why is log_std independent of state?
    State-dependent std (mean and std from the network) can oscillate —
    in one update std drops, in the next it rises, causing large jumps
    in log_prob ratio and breaking the PPO trust region. With state-independent std,
    std changes slowly and monotonically (usually decreases).

    Why WITHOUT tanh squashing?
    PPO does not need exact log_prob for the entropy term (unlike SAC).
    PPO clips ratio = exp(new_log_prob - old_log_prob) — both sides of the ratio
    have the same bias from the lack of squashing, so the bias cancels out.
    Tanh squashing compresses gradients near ±1 (tanh gradient → 0),
    which makes it harder to learn actions near the bounds. Better to simply
    clip actions to [-1, 1] at env.step().
    """
    def __init__(self, obs_dim, action_dim, log_std_init=-2.0):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

    def forward(self, obs):
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        mean = self.mean_head(x)
        # Floor/ceiling on log_std: clamp gradient is 0 outside the range, so the parameter
        # naturally stabilizes at the boundary instead of crossing it (as in SAC).
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp().expand_as(mean)
        return Normal(mean, std) # the 'mean_head' layer only actually learns 'mean' because it is wrapped
        # in the Normal() function together with std; if it were wrapped in Categorical, it would learn
        # the entire probability distribution by itself because it would be the only 'component'.

    def get_action(self, obs):
        """Action sampling during rollout collection.
        Returns the action and its log_prob (scalar per env, summed over action dimensions).

        log_prob = sum_i log N(a_i | mu_i, sigma_i) — sum over action dimensions,
        because dimensions are independent (diagonal covariance).
        """
        dist = self.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        # dist.log_prob(action) → (batch, action_dim) — log-prob per dimension
        # .sum(dim=-1) → (batch,) — combined log-prob
        # We sum because P(a1,a2,...,a6) = P(a1)*P(a2)*...*P(a6) (independent dimensions)
        # → log P = log P(a1) + log P(a2) + ... + log P(a6)
        return action, log_prob

    def evaluate(self, obs, actions):
        """Re-evaluation of log_prob and entropy for actions from the rollout (PPO update).
        We must recompute log_prob from CURRENT weights (not old ones),
        to compute ratio = exp(new_log_prob - old_log_prob).
        """
        dist = self.forward(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class Critic(nn.Module):
    """Value function V(s) — separate network from the actor.

    Same architecture as actor (256, 256, tanh) but with a single output (scalar).
    Orthogonal init with gain=1.0 on value_head — standard scale for value predictions
    (in HalfCheetah V(s) ~thousands after convergence).
    """
    def __init__(self, obs_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.value_head = nn.Linear(256, 1)
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs):
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        return self.value_head(x).squeeze(-1)


def make_env(env_id="HalfCheetah-v5"):
    """Returns a thunk that creates the MuJoCo environment.
    No preprocessing wrappers — MuJoCo returns a raw state vector,
    we do observation normalization ourselves in the agent (RunningMeanStd).
    """
    def thunk():
        env = gym.make(env_id)
        return env
    return thunk


class HalfCheetahAgent:
    def __init__(
        self,
        envs,
        num_envs=8,
        lr=1e-4,
        gamma=0.98,
        lam=0.92,
        n_steps=256,
        entropy_coef=4e-4,
        vf_coef=0.58,
        clip_epsilon=0.1,
        max_grad_norm=0.8,
        ppo_epochs=20,
        num_minibatches=32,
        anneal_lr=True,
        clip_vloss=False,
        target_kl=0.05,
        norm_reward=False,
        total_timesteps=2_000_000,
    ):
        self.envs = envs
        self.num_envs = num_envs
        self.gamma = gamma
        self.lam = lam
        self.n_steps = n_steps
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.clip_epsilon = clip_epsilon
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.num_minibatches = num_minibatches
        self.anneal_lr = anneal_lr
        self.clip_vloss = clip_vloss
        self.target_kl = target_kl
        self.norm_reward = norm_reward
        self.total_timesteps = total_timesteps
        self.lr = lr

        self.obs_dim = envs.single_observation_space.shape[0]   # 17 for HalfCheetah
        self.action_dim = envs.single_action_space.shape[0]     # 6 for HalfCheetah
        self.action_low = envs.single_action_space.low          # [-1, -1, ..., -1]
        self.action_high = envs.single_action_space.high        # [1, 1, ..., 1]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)

        # One optimizer for both networks. Even though the networks are separate (not shared backbone),
        # one optimizer with one lr schedule is simpler and works equally well.
        # Gradients do not interfere anyway — backward() for policy_loss generates gradients
        # ONLY in the actor (critic is not in the policy loss computational graph), and vice versa.
        # Combined loss = policy_loss + vf_coef * value_loss automatically routes
        # gradients to the appropriate networks.
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr, eps=1e-5,
        )

        self.obs_rms = RunningMeanStd(shape=(self.obs_dim,))

        # Reward scaling: we divide rewards by running std of discounted returns.
        # ret_rms with shape=() → tracks scalar variance of returns.
        # self.ret is a per-env accumulator: ret = ret * gamma + reward.
        # We update ret_rms on EVERY step (not only at episode end),
        # because even partial returns give information about reward scale.
        self.ret_rms = RunningMeanStd(shape=())
        self.ret = np.zeros(num_envs, dtype=np.float64)

        self.obs, _ = self.envs.reset()
        self.obs = np.array(self.obs, dtype=np.float32)   # (num_envs, 17)

        self.ep_rewards = np.zeros(num_envs, dtype=np.float64)
        self.ep_steps = np.zeros(num_envs, dtype=np.int64)

        self.reward_history = []
        self.steps_history = []
        self.start_update = 0

        self.batch_size = self.n_steps * self.num_envs
        self.mini_batch_size = self.batch_size // self.num_minibatches
        self.last_approx_kl = 0.0
        self.last_policy_loss = 0.0
        self.last_value_loss = 0.0
        self.last_entropy = 0.0
        self.global_step = 0

    def normalize_obs(self, obs, update=True):
        """Normalizes observations using running statistics.

        update=True during rollout collection (update statistics + normalize).
        update=False during testing and when computing last_values (only normalize,
        do not change statistics because these are not new training data).
        """
        if update:
            self.obs_rms.update(obs)
        # Clip to [-10, 10] (CleanRL/SB3 standard) — protects against extreme
        # outliers in normalized observations that destabilize the MLP.
        return np.clip(self.obs_rms.normalize(obs), -10.0, 10.0).astype(np.float32)

    def collect_rollout(self):
        """
        Collects n_steps transitions from num_envs environments simultaneously.

        Buffer shapes (continuous vs discrete):
        - states:    (n_steps, num_envs, 17)    ← vector, not image
        - actions:   (n_steps, num_envs, 6)     ← 6D continuous, not int
        - rewards, dones, values, log_probs: (n_steps, num_envs)

        Observations are normalized BEFORE saving to the buffer.
        Rewards go through a 3-stage pipeline:
        1. Reward scaling: reward / sqrt(running_var_of_returns)
        2. Truncation bootstrap: reward += gamma * V(s_final) for truncated envs
        3. Save to buffer (buffer contains normalized rewards with bootstrap)
        """
        states = np.zeros((self.n_steps, self.num_envs, self.obs_dim), dtype=np.float32)
        actions = np.zeros((self.n_steps, self.num_envs, self.action_dim), dtype=np.float32)
        rewards = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)
        dones = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)
        values = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)
        log_probs = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)

        for t in range(self.n_steps):
            norm_obs = self.normalize_obs(self.obs)
            states[t] = norm_obs

            obs_tensor = torch.from_numpy(norm_obs).to(self.device)

            with torch.no_grad():
                action, log_prob = self.actor.get_action(obs_tensor)
                value = self.critic(obs_tensor)

            action_np = action.cpu().numpy()
            # We clip actions to env bounds ONLY for step() — in the buffer we keep
            # the original (unclipped) actions, because log_prob was computed for them.
            # If we saved clipped actions, then in update() actor.evaluate()
            # would compute log_prob for clipped values, but old_log_prob
            # would be for unclipped ones → inconsistency in ratio.
            clipped_action = np.clip(action_np, self.action_low, self.action_high)

            actions[t] = action_np
            values[t] = value.cpu().numpy()
            log_probs[t] = log_prob.cpu().numpy()

            next_obs, reward, terminated, truncated, infos = self.envs.step(clipped_action)
            next_obs = np.array(next_obs, dtype=np.float32)
            done = np.logical_or(terminated, truncated)

            # --- Episode tracking (raw rewards, before normalization) ---
            self.ep_rewards += reward
            self.ep_steps += 1

            for i in range(self.num_envs):
                if done[i]:
                    self.reward_history.append(self.ep_rewards[i])
                    self.steps_history.append(int(self.ep_steps[i]))
                    self.ep_rewards[i] = 0.0
                    self.ep_steps[i] = 0

            # --- Reward scaling ---
            # 1. Accumulate discounted return per env: ret = ret * gamma + reward
            # 2. Update running variance of returns (ret_rms)
            # 3. Divide reward by sqrt(var) — scale, but do NOT subtract the mean
            # 4. Reset accumulator for finished episodes
            #
            # Why do we not subtract the mean?
            # Subtracting the mean from rewards would introduce bias in the policy gradient:
            # rewards close to the mean would become ~0, which "tells" the agent that
            # those actions are neutral — even if they are objectively good.
            # Dividing by std only scales, preserving relative differences.
            if self.norm_reward:
                self.ret = self.ret * self.gamma + reward
                self.ret_rms.update(self.ret)
                reward = reward / (np.sqrt(self.ret_rms.var) + 1e-8)
                # Clip normalized rewards to [-10, 10] (CleanRL/SB3 standard).
                reward = np.clip(reward, -10.0, 10.0)
                self.ret *= (1.0 - done)

            # --- Truncation bootstrap ---
            # When an episode ends due to time limit (truncated), not due to
            # actual "death" (terminated), the agent could have continued.
            # V(s_next) for truncated is NOT 0 — we must bootstrap.
            #
            # Problem: VectorEnv auto-resets, so next_obs is already from a NEW
            # episode. The old final_obs is available in infos["final_observation"].
            #
            # Solution: we add gamma * V(s_final) to the reward. Then in GAE,
            # dones[t]=1 zeros V(s_next) from the new episode (good, because it is a different
            # episode), but the reward already contains bootstrap from V(s_final).
            # Effect: delta = (r + gamma*V(s_final)) + 0 - V(s) = r + gamma*V(s_final) - V(s) ✓
            if "final_observation" in infos:
                for i in range(self.num_envs):
                    if truncated[i]:
                        final_obs = np.array(infos["final_observation"][i], dtype=np.float32)
                        norm_final = self.normalize_obs(final_obs.reshape(1, -1), update=False)
                        with torch.no_grad():
                            obs_t = torch.from_numpy(norm_final).to(self.device)
                            terminal_value = self.critic(obs_t).item()
                        reward[i] += self.gamma * terminal_value

            rewards[t] = reward
            dones[t] = done.astype(np.float32)

            self.obs = next_obs

        norm_last_obs = self.normalize_obs(self.obs, update=False)
        with torch.no_grad():
            obs_tensor = torch.from_numpy(norm_last_obs).to(self.device)
            last_values = self.critic(obs_tensor).cpu().numpy()
        last_values = last_values * (1.0 - dones[-1])

        return states, actions, rewards, dones, values, log_probs, last_values

    def compute_gae(self, rewards, values, dones, last_values):
        """GAE(gamma, lambda) — identical algorithm as in discrete_control PPO."""
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(self.num_envs, device=self.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = torch.tensor(last_values, dtype=torch.float32, device=self.device)
            else:
                next_val = values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self, states, actions, rewards, dones, values, log_probs, last_values):
        """
        PPO update with separate actor/critic networks.

        Key techniques:
        - Advantage normalization per MINIBATCH (not per batch) — each minibatch
          has zero-mean advantages, which eliminates bias in the gradient direction.
        - Value function clipping — limits jump of V(s) to ±clip_epsilon.
        - Log ratio trick: ratio = exp(log_ratio) — numerically stable.
        - KL Early Stopping — if approx_kl > target_kl, stops the update.
          Prevents catastrophic policy changes in a single update.
        """
        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device)
        rewards_t = torch.tensor(rewards, device=self.device)
        dones_t = torch.tensor(dones, device=self.device)
        values_t = torch.tensor(values, device=self.device)
        old_log_probs_t = torch.tensor(log_probs, device=self.device)

        with torch.no_grad():
            advantages, returns = self.compute_gae(rewards_t, values_t, dones_t, last_values)
            advantages = advantages.reshape(-1)
            returns = returns.reshape(-1)

        states_t = states_t.reshape(-1, self.obs_dim)
        actions_t = actions_t.reshape(-1, self.action_dim)
        old_log_probs_t = old_log_probs_t.reshape(-1)
        old_values_t = values_t.reshape(-1)

        indices = np.arange(self.batch_size)

        # KL guardrail (Stable-Baselines3 style): we check approx_kl PER MINIBATCH,
        # BEFORE the optimizer step. If a single minibatch would push
        # the policy beyond the threshold (1.5 * target_kl), we discard that and all subsequent updates
        # of this rollout. This way a CATASTROPHIC update (e.g. KL=7.69 at std≈0.05,
        # where a small change in mean gives a huge ratio) is NEVER applied —
        # instead of destroying the policy, we simply stop early and collect a fresh rollout.
        kl_stop_threshold = 1.5 * self.target_kl if self.target_kl is not None else None
        continue_training = True

        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, self.batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_idx = indices[start:end]

                mb_states = states_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_old_log_probs = old_log_probs_t[mb_idx]
                mb_old_values = old_values_t[mb_idx]

                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                new_log_probs, entropy = self.actor.evaluate(mb_states, mb_actions)
                new_values = self.critic(mb_states)

                # --- PPO Clipped Policy Loss ---
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = log_ratio.exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    self.last_approx_kl = approx_kl

                # --- KL early stopping (SB3) ---
                # Checked BEFORE optimizer.step(): if KL already exceeded the threshold,
                # this minibatch is NOT applied (break before the gradient step),
                # and the flag also breaks the epoch loop. This is the "safety fuse"
                # that prevents one unlucky update from destroying training.
                """In simple terms: KL is a measure of how far the "new" policy has drifted from
                the "old" one (from the rollout). Formula $KL ≈ (ratio - 1) - log(ratio)$ — cheap
                estimator. Threshold 1.5 * target_kl is a heuristic from SB3."""
                if kl_stop_threshold is not None and approx_kl > kl_stop_threshold:
                    continue_training = False
                    break

                # --- Value Loss (with optional clipping) ---
                if self.clip_vloss:
                    v_loss_unclipped = (new_values - mb_returns) ** 2
                    v_clipped = mb_old_values + torch.clamp(
                        new_values - mb_old_values, -self.clip_epsilon, self.clip_epsilon
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    value_loss = 0.5 * nn.functional.mse_loss(new_values, mb_returns)

                entropy_loss = entropy.mean()

                self.last_policy_loss = policy_loss.item()
                self.last_value_loss = value_loss.item()
                self.last_entropy = entropy_loss.item()

                loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

            if not continue_training:
                break

    def train(self, total_timesteps=None, save_every=500_000):
        if total_timesteps is None:
            total_timesteps = self.total_timesteps

        steps_per_update = self.n_steps * self.num_envs
        n_updates = total_timesteps // steps_per_update
        print_every = max(1, n_updates // 20)
        save_every_updates = max(1, save_every // steps_per_update)

        start_update = self.start_update
        self.global_step = start_update * steps_per_update

        print(f"Training HalfCheetah PPO | Device: {self.device}")
        print(f"  {total_timesteps/1e6:.1f}M timesteps, {n_updates} updates")
        print(f"  batch_size={self.batch_size}, mini_batch_size={self.mini_batch_size}")
        print(f"  ppo_epochs={self.ppo_epochs}, clip_epsilon={self.clip_epsilon}")
        print(f"  Actor params: {sum(p.numel() for p in self.actor.parameters()):,}")
        print(f"  Critic params: {sum(p.numel() for p in self.critic.parameters()):,}")

        for update_idx in tqdm(range(start_update, n_updates), initial=start_update, total=n_updates):
            # Linear learning rate annealing
            if self.anneal_lr:
                frac = 1.0 - update_idx / n_updates
                lr_now = self.lr * frac
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr_now

            states, actions, rewards, dones, values, log_probs, last_values = self.collect_rollout()
            eps_before = len(self.reward_history)
            self.update(states, actions, rewards, dones, values, log_probs, last_values)

            self.start_update = update_idx + 1
            self.global_step += steps_per_update

            if wandb.run is not None:
                for ep_reward, ep_steps in zip(
                    self.reward_history[eps_before:],
                    self.steps_history[eps_before:],
                ):
                    wandb.log({
                        METRIC_EPISODIC_RETURN: ep_reward,
                        METRIC_EPISODIC_LENGTH: ep_steps,
                    }, step=self.global_step)
                wandb.log({
                    "losses/policy_loss": self.last_policy_loss,
                    "losses/value_loss": self.last_value_loss,
                    "losses/entropy": self.last_entropy,
                    "losses/approx_kl": self.last_approx_kl,
                }, step=self.global_step)

            if (update_idx + 1) % save_every_updates == 0:
                steps_done = (update_idx + 1) * steps_per_update
                self.save(
                    model_path(
                        HALFCHEETAH_PPO_MODELS,
                        f"halfcheetah_ppo_checkpoint_{steps_done // 1000}k.pth",
                    )
                )

            if (update_idx + 1) % print_every == 0 and len(self.reward_history) > 0:
                avg_reward = np.mean(self.reward_history[-100:])
                avg_steps = np.mean(self.steps_history[-100:])
                n_eps = len(self.reward_history)
                lr_current = self.optimizer.param_groups[0]['lr']
                avg_std = self.actor.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().mean().item()
                wandb.log({
                    METRIC_AVG_RETURN_100: avg_reward,
                    METRIC_AVG_STEPS_100: avg_steps,
                    METRIC_LEARNING_RATE: lr_current,
                    "charts/policy_std": avg_std,
                }, step=self.global_step)
                print(f"Update {update_idx+1}/{n_updates} | "
                      f"Eps: {n_eps} | "
                      f"Avg R: {avg_reward:.1f} | "
                      f"Avg Steps: {avg_steps:.0f} | "
                      f"LR: {lr_current:.2e} | "
                      f"Std: {avg_std:.3f} | "
                      f"KL: {self.last_approx_kl:.4f}")

    def save(self, filename=HALFCHEETAH_PPO_FINAL):
        filename = model_path(HALFCHEETAH_PPO_MODELS, filename)
        ensure_dir(filename.parent)
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'obs_rms_mean': self.obs_rms.mean,
            'obs_rms_var': self.obs_rms.var,
            'obs_rms_count': self.obs_rms.count,
            'ret_rms_mean': self.ret_rms.mean,
            'ret_rms_var': self.ret_rms.var,
            'ret_rms_count': self.ret_rms.count,
            'reward_history': self.reward_history,
            'steps_history': self.steps_history,
            'start_update': self.start_update,
        }
        torch.save(checkpoint, filename)
        print(f"Model saved as {filename}")

    def load(self, filename=HALFCHEETAH_PPO_FINAL):
        filename = model_path(HALFCHEETAH_PPO_MODELS, filename)
        checkpoint = torch.load(filename, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.obs_rms.mean = checkpoint['obs_rms_mean']
        self.obs_rms.var = checkpoint['obs_rms_var']
        self.obs_rms.count = checkpoint['obs_rms_count']
        self.ret_rms.mean = checkpoint.get('ret_rms_mean', self.ret_rms.mean)
        self.ret_rms.var = checkpoint.get('ret_rms_var', self.ret_rms.var)
        self.ret_rms.count = checkpoint.get('ret_rms_count', self.ret_rms.count)
        self.reward_history = checkpoint.get('reward_history', [])
        self.steps_history = checkpoint.get('steps_history', [])
        self.start_update = checkpoint.get('start_update', 0)
        print(f"Model loaded from {filename} (resuming from update {self.start_update})")


def plot_results(rewards, steps):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    rolling_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
    ax1.plot(rolling_avg)
    ax1.set_title("HalfCheetah PPO - Learning Progress (Rolling Average Reward)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Average Reward (last 100)")
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.grid(True)

    rolling_steps = np.convolve(steps, np.ones(100)/100, mode='valid')
    ax2.plot(rolling_steps, color='orange')
    ax2.set_title("HalfCheetah PPO - Steps per Episode (Rolling Average)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Average Steps (last 100)")
    ax2.grid(True)

    plt.tight_layout()
    ensure_dir(HALFCHEETAH_PPO_RESULTS)
    path = HALFCHEETAH_PPO_RESULTS / "learning_curve_HalfCheetah_PPO.png"
    plt.savefig(path)
    print(f"Plot saved as {path}")
    plt.close()


if __name__ == "__main__":
    RESUME_FROM = None  
    EXP_NAME = "PPO_HalfCheetah_v5"
    ENV_ID = "HalfCheetah-v5"

    TRAIN_CONFIG = apply_halfcheetah_ppo_config({
        "exp_name": EXP_NAME,
        "env_id": ENV_ID,
        "num_envs": 8,
        "lr": 1e-4,
        "gamma": 0.98,
        "lam": 0.92,
        "n_steps": 256,
        "entropy_coef": 4e-4,
        "vf_coef": 0.58,
        "clip_epsilon": 0.1,
        "max_grad_norm": 0.8,
        "ppo_epochs": 20,
        "num_minibatches": 32,
        "anneal_lr": True,
        "clip_vloss": False,
        "target_kl": 0.05,
        "norm_reward": False,
        "total_timesteps": 10_000_000,
    })
    NUM_ENVS = TRAIN_CONFIG["num_envs"]
    if smoke_mode():
        print(
            f"Smoke mode: {TRAIN_CONFIG['total_timesteps']} timesteps, "
            f"{NUM_ENVS} envs"
        )

    init_wandb(TRAIN_CONFIG["exp_name"], TRAIN_CONFIG, tags=["ppo", "halfcheetah"])

    envs = gym.vector.AsyncVectorEnv([make_env(ENV_ID) for _ in range(NUM_ENVS)])

    agent = HalfCheetahAgent(
        envs=envs,
        num_envs=NUM_ENVS,
        lr=TRAIN_CONFIG["lr"],
        gamma=TRAIN_CONFIG["gamma"],
        lam=TRAIN_CONFIG["lam"],
        n_steps=TRAIN_CONFIG["n_steps"],
        entropy_coef=TRAIN_CONFIG["entropy_coef"],
        vf_coef=TRAIN_CONFIG["vf_coef"],
        clip_epsilon=TRAIN_CONFIG["clip_epsilon"],
        max_grad_norm=TRAIN_CONFIG["max_grad_norm"],
        ppo_epochs=TRAIN_CONFIG["ppo_epochs"],
        num_minibatches=TRAIN_CONFIG["num_minibatches"],
        anneal_lr=TRAIN_CONFIG["anneal_lr"],
        clip_vloss=TRAIN_CONFIG["clip_vloss"],
        target_kl=TRAIN_CONFIG["target_kl"],
        norm_reward=TRAIN_CONFIG["norm_reward"],
        total_timesteps=TRAIN_CONFIG["total_timesteps"],
    )

    if RESUME_FROM:
        agent.load(RESUME_FROM)

    save_every = 999_999_999 if smoke_mode() else 500_000
    agent.train(save_every=save_every)
    agent.save(HALFCHEETAH_PPO_FINAL)

    envs.close()

    if not smoke_mode():
        plot_results(agent.reward_history, agent.steps_history)
    finish_wandb()
