import numpy as np
import torch


class ReplayBuffer:
    """[SAC vs PPO] Replay buffer instead of rollout buffer.

    PPO rollout buffer:
    - Stores a full rollout 
    - Data used ONCE (at most a few PPO epochs), then discarded — on-policy
    - Contains: states, actions, rewards, dones, values, log_probs
    - Requires: GAE for advantages, advantage normalization

    SAC replay buffer:
    - Circular buffer for SINGLE transitions (s, a, r, s', terminated)
    - Capacity 1M — data reused many times (off-policy)
    - Random sampling — independent and identically distributed minibatches (breaks temporal correlations)
    - Does NOT store values/log_probs — computed on the fly from current networks
    - Stores `terminated` (not `done`) — truncation = bootstrap, termination = no bootstrap

    NumPy arrays instead of deque/list — O(1) random access.
    Pre-allocated memory — no dynamic allocation during training.
    """
    def __init__(self, obs_dim, action_dim, capacity, device):
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, terminated):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.terminated[self.ptr] = terminated
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.obs[idx]).to(self.device),
            torch.FloatTensor(self.actions[idx]).to(self.device),
            torch.FloatTensor(self.rewards[idx]).to(self.device),
            torch.FloatTensor(self.next_obs[idx]).to(self.device),
            torch.FloatTensor(self.terminated[idx]).to(self.device),
        )
