import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_STD_MIN = -5
LOG_STD_MAX = 2


class Actor(nn.Module):
    """[SAC vs PPO] Fundamentally different actor from PPO:

    1. State-dependent log_std — the network has TWO outputs: mean_head and log_std_head.
       In PPO, log_std was an nn.Parameter (fixed, independent of state).
       In SAC, state-dependent std enables ADAPTIVE exploration:
       large std in unfamiliar states, small std in well-known ones.
       Stable thanks to entropy regularization (α compensates automatically).

    2. Tanh squashing: a = tanh(z), z ~ N(μ(s), σ(s)).
       Guarantees a ∈ [-1, 1] MATHEMATICALLY (not via clipping as in PPO).
       Requires Jacobian correction in log_prob:
       log π(a|s) = log N(z|μ,σ) - Σ log(1 - tanh²(zᵢ))

       Why must SAC have squashing, and PPO need not?
       SAC EXPLICITLY uses log π(a|s) in the objective: loss = E[α·log π - Q].
       If log_prob were wrong (no Jacobian), the actor would optimize the
       wrong objective and α would not converge to the correct value.
       PPO uses RATIO = exp(new_log_prob - old_log_prob) — both sides share
       the same bias from omitting squashing, so the bias cancels.

    3. ReLU instead of tanh in hidden layers — the off-policy replay buffer yields
       more stable gradients (i.i.d. batches), so ReLU is safe and faster.
       PPO used tanh because on-policy gradients (temporally correlated) are less stable.

    4. Reparameterization trick: z = μ + σ·ε, ε ~ N(0,1).
       rsample() instead of sample() — gradients from the Q-network flow THROUGH z
       into the actor weights. In PPO the gradient flowed through ratio·advantage.
    """
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def get_action(self, obs):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        """Normal for continuous action spaces (e.g. joint torques); Categorical
        when actions are discrete (e.g. left/right).

        [SAC vs PPO] rsample() instead of sample():
        rsample() = reparameterized sample = μ + σ·ε, with gradients through μ and σ.
        sample() cuts the gradient (does not propagate to network parameters).
        In PPO we used sample() because the gradient went through the ratio, not the action.
        In SAC the gradient flows from Q(s,a) THROUGH a into the actor weights — rsample() is required."""
        z = dist.rsample()
        action = torch.tanh(z)

        """[SAC vs PPO] Jacobian correction — key difference!
        Tanh is a nonlinear transform, so it changes the probability density. The action has a different density than z because
        we applied tanh. Tanh warps the distribution differently and we must subtract a correction for that squashing to get correct log_prob:
        log π(a|s) = log N(z|μ,σ) - Σ log(1 - tanh²(zᵢ))
        Numerically stable form: log(1 - tanh²(z)) = 2·(log2 - z - softplus(-2z))
        Without this correction: the entropy estimate is wrong → α converges to a bad value
        → too little or too much exploration → worse results.
        In PPO there was no tanh squashing → no Jacobian correction."""
        log_prob = dist.log_prob(z).sum(dim=-1)
        log_prob -= (2 * (np.log(2) - z - F.softplus(-2 * z))).sum(dim=-1)
        return action, log_prob


class SoftQNetwork(nn.Module):
    """[SAC vs PPO] Q(s,a) instead of V(s):

    - Input: concat(obs, action) — the critic knows WHICH action the agent took.
      PPO's V(s) saw ONLY the state, not which action the agent would take.
      Q(s,a) gives finer feedback: "this specific action in this state
      is worth X", vs V(s): "this state is generally worth X".

    - Twin Q: two separate Q-networks; we take min(Q1, Q2) as the target.
      Reduces overestimation bias — Q-networks tend to overestimate
      value (optimistic bias from bootstrapping + function approximation).
      min() yields a pessimistic estimate, which is more stable.
      In PPO this was less of an issue — on-policy data does not produce the same overestimation
      (GAE bootstraps from V, not Q, and V is updated frequently).

    - ReLU instead of tanh (like the actor) — off-policy stability.
    """
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q_value = nn.Linear(256, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q_value(x).squeeze(-1)
