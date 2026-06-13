import sys
from pathlib import Path

import gymnasium as gym

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repo_paths import LUNAR_LANDER_FINAL, LUNAR_LANDER_RESULTS, ensure_dir
from smoke_config import LUNAR_LANDER_TIMESTEPS, smoke_mode
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm

RESULTS_DIR = LUNAR_LANDER_RESULTS
MODEL_PATH = LUNAR_LANDER_FINAL

"""
1. General techniques and assumptions
- Agent is based on three pillars:

- Actor-Critic Architecture: Two networks. Actor (PolicyNet) outputs actions, Critic (ValueNet) evaluates the state.
- On-Policy Learning: Learns only from recently collected data. No Replay Buffer.
- Uses TD(lambda=0.95) for GAE computation.
- Stability first: PPO ensures the new policy does not deviate drastically from the old one.
"""


class PolicyNet(nn.Module):
    def __init__(self, obs_dims, n_hidden, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dims, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_actions),
        )

    def forward(self, x):
        return Categorical(logits=self.net(x))


class ValueNet(nn.Module):
    def __init__(self, obs_dims, n_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dims, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LunarLanderAgent():
    def __init__(
        self,
        env,
        policy_lr=3e-4,
        value_lr=1e-3,
        gamma=0.99,
        lam=0.95,
        n_hidden=128,
        n_steps=2048,
        entropy_coef_start=0.01,
        entropy_coef_end=0.001,
        max_grad_norm=0.5,
    ):
        self.env = env
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.gamma = gamma
        self.lam = lam
        self.n_steps = n_steps
        self.entropy_coef_start = entropy_coef_start
        self.entropy_coef_end = entropy_coef_end
        self.max_grad_norm = max_grad_norm

        self.obs_dims = env.observation_space.shape[0]
        n_actions = env.action_space.n

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_net = PolicyNet(self.obs_dims, n_hidden, n_actions).to(self.device)
        self.value_net = ValueNet(self.obs_dims, n_hidden).to(self.device)

        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.value_lr)

    
        self.obs, _ = self.env.reset()  # obs as an attribute — when rollouts are chained across updates. (PPO, A2C)
        # because we do rollout over a fixed number of steps, the next rollout must start from the last state; if we did not
        # keep this as an attribute we would lose continuity

        self.reward_history = []
        self.steps_history = []

    def get_entropy_coef(self, progress):
        """Linear decay: explores at the beginning, exploits at the end. coef_start >> coef_end"""
        return self.entropy_coef_start + (self.entropy_coef_end - self.entropy_coef_start) * progress

    def collect_rollout(self):
        """
        Collects n_steps of transitions.
        Saves states, actions, rewards, dones, values, and log_probs.

        Episodes end and restart naturally in the middle of a rollout.
        """
        states = np.zeros((self.n_steps, self.obs_dims), dtype=np.float32)
        actions = np.zeros(self.n_steps, dtype=np.int64)
        rewards = np.zeros(self.n_steps, dtype=np.float32)
        dones = np.zeros(self.n_steps, dtype=np.float32)
        values = np.zeros(self.n_steps, dtype=np.float32)
        log_probs = np.zeros(self.n_steps, dtype=np.float32)

        ep_reward = 0.0
        ep_steps = 0

        for t in range(self.n_steps):
            states[t] = self.obs
            
            # Use from_numpy for efficiency, unsqueeze to add batch dimension
            obs_tensor = torch.from_numpy(self.obs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # Rollout data serves in PPO as fixed reference points (the so-called old policy) necessary to compute
                # the ratio coefficient. We use no_grad and .item() to "freeze" these values, which allows the update
                # phase to correctly optimize the new strategy relative to historical results without incorrectly
                # passing gradients through past data.
                dist = self.policy_net(obs_tensor)
                value = self.value_net(obs_tensor)
                
                action = dist.sample()
                log_prob = dist.log_prob(action)

            # Store numpy values
            actions[t] = action.item()
            values[t] = value.item()
            log_probs[t] = log_prob.item()

            next_obs, reward, terminated, truncated, _ = self.env.step(actions[t])
            # next_obs is the 'last' frame in which termination or truncation occurred (if they occurred). That is,
            # the agent transitioned to this state or crashed or the episode time ran out. next_obs is never
            # empty or 0.
            done = terminated or truncated

            rewards[t] = reward
            dones[t] = float(done)  # float because in update we use (1 - done) — bool would work poorly

            ep_reward += reward
            ep_steps += 1

            # Handle end of episode
            if done:
                if truncated and not terminated:
                    # When the episode is cut off by the time limit (truncated), the done=True flag forces GAE to zero
                    # out future state values. Then even though the agent e.g. was flying well but ran out of time, it
                    # suddenly "dies" as if terminal. Manually adding bootstrap_value to the reward lets the agent
                    # correctly estimate the value of the final state without mathematically connecting it to the start
                    # of the new episode. So the last state does not lead to "death" but to a potential next state whose
                    # value we estimate (bootstrap).
                    with torch.no_grad():
                        next_obs_tensor = torch.from_numpy(next_obs).unsqueeze(0).to(self.device)
                        bootstrap_value = self.value_net(next_obs_tensor).item()
                    rewards[t] += self.gamma * bootstrap_value

                self.reward_history.append(ep_reward)
                self.steps_history.append(ep_steps)
                ep_reward = 0.0
                ep_steps = 0
                self.obs, _ = self.env.reset()
            else:
                self.obs = next_obs

        # Bootstrap value for the last state in the rollout (if episode didn't end naturally)
        last_value = 0.0
        # If the rollout ends while the agent is still in the game (not done), last_value serves as a forecast of
        # future rewards (bootstrapping). This value is necessary for GAE to correctly compute advantages for the
        # last steps in the buffer, preventing the false assumption that the game ends after the data is cut off.
        # We use it as the starting point for backward computation (from the end of the buffer to the beginning).
        # In the function computing GAE/Advantages it typically looks like:
        #
        # next_value = last_value  # We start from our forecast
        # for t in reversed(range(n_steps)):
        #     # Compute the TD (Temporal Difference) error
        #     delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        if not done:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(self.obs).unsqueeze(0).to(self.device)
                last_value = self.value_net(obs_tensor).item()

        return states, actions, rewards, dones, values, log_probs, last_value

    def compute_gae(self, rewards, values, dones, last_value):
        """
        GAE(gamma, lambda)
        Calculated backwards to propagate future rewards.
        """
        T = len(rewards)
        advantages = torch.zeros(T, device=self.device)
        gae = 0.0

        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else values[t + 1]
            # delta is the difference between what happened and what we thought would happen.
            # If delta is positive, it means: "Oh, it was better than I thought!"
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            # GAE stabilizes learning by combining short- and long-term prediction errors.
            # We compute advantages backward to propagate successes from the end of the episode to earlier states,
            # and returns form the updated target for the Critic network.
            #
            # A(s, a) = Q(s, a) - V(s), where Q — return, V — value
            #
            # Return = Advantage + Value
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            # Above we compute delta for the given state and add the weighted sum of deltas (gae) from the previous state
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self, states, actions, rewards, dones, values, log_probs, last_value, entropy_coef):
        """
        PPO-style update with multiple epochs, clipping, and mini-batches.
        """
        # Convert to tensors once
        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device)
        rewards_t = torch.tensor(rewards, device=self.device)
        dones_t = torch.tensor(dones, device=self.device)
        values_t = torch.tensor(values, device=self.device)
        old_log_probs_t = torch.tensor(log_probs, device=self.device)

        # 1. Calculate advantages and returns ONCE before epochs
        with torch.no_grad():
            advantages, returns = self.compute_gae(rewards_t, values_t, dones_t, last_value)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # - with torch.no_grad(): we compute baseline values. We do not want to train the networks now, only extract
            #   their current "opinions" about states
            # - compute_gae: we run the mechanism discussed earlier, which looks at the rollout and says: "this action was
            #   X better/worse than we predicted"
            # - Advantage normalization: this is a key engineering trick. We bring advantages to mean 0 and std 1.
            #   Thanks to this gradients do not explode and the network learns stably regardless of whether rewards in
            #   the game are on the order of 1 or 1000.

        # PPO and Mini-batch Hyperparameters
        ppo_epochs = 4
        mini_batch_size = 64
        clip_epsilon = 0.2
        batch_size = self.n_steps
        indices = np.arange(batch_size)

        # 2. Optimize for multiple epochs
        for _ in range(ppo_epochs):
            # - ppo_epochs: we go through the same data 4 times. Thanks to PPO this is safe and lets us squeeze maximum
            #   knowledge from the collected experience.
            # - shuffle: we shuffle the order of steps. The network should not learn that "after step 5 comes step 6",
            #   because that leads to overfitting to a specific play session.
            # Shuffle indices to break temporal correlations
            np.random.shuffle(indices)
            
            for start in range(0, batch_size, mini_batch_size):
                # Instead of giving the network 1024 examples at once, we give it 16 batches of 64 examples. Smaller
                # batches = more frequent weight updates and better convergence.
                end = start + mini_batch_size
                mb_idx = indices[start:end]

                # Sample a mini-batch
                mb_states = states_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_old_log_probs = old_log_probs_t[mb_idx]

                # Forward pass on mini-batch
                dist = self.policy_net(mb_states)
                log_probs = dist.log_prob(mb_actions)
                entropies = dist.entropy()
                current_values = self.value_net(mb_states)

                # --- Actor Loss (PPO Clipped) ---
                # ratio: the ratio of the new strategy to the old. If ratio > 1, the action is now more probable than
                # before. If ratio < 1, less.
                ratio = torch.exp(log_probs - mb_old_log_probs)

                # - surr1: standard gain (like in REINFORCE)
                # - surr2 (The Clip): if ratio goes outside [0.8, 1.2], we clip it
                # - min(surr1, surr2): we take the pessimistic variant. This is the essence of PPO — if the new strategy
                #   wants to take too large a step toward the reward, PPO holds it back. The minus at the start is because
                #   PyTorch minimizes functions and we want to maximize gain.
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy: a bonus for being uncertain. If the network thinks all actions are similarly good, entropy is
                # high. This forces the agent to keep exploring and prevents premature lock-in to one strategy.
                entropy_loss = -entropy_coef * entropies.mean()
                actor_loss = policy_loss + entropy_loss

                # Optimize Actor
                self.policy_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()

                # --- Critic Loss ---
                value_loss = nn.functional.mse_loss(current_values, mb_returns)

                # Optimize Critic
                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.value_optimizer.step()

    def train(self, total_timesteps):
        n_updates = total_timesteps // self.n_steps
        print_every = max(1, n_updates // 20)

        for update_idx in tqdm(range(n_updates)):
            progress = update_idx / n_updates
            entropy_coef = self.get_entropy_coef(progress)

            # Unpack the new values from rollout
            states, actions, rewards, dones, values, log_probs, last_value = self.collect_rollout()
            self.update(states, actions, rewards, dones, values, log_probs, last_value, entropy_coef)

            if (update_idx + 1) % print_every == 0 and len(self.reward_history) > 0:
                avg_reward = np.mean(self.reward_history[-100:])
                avg_steps = np.mean(self.steps_history[-100:])
                n_eps = len(self.reward_history)
                print(f"Update {update_idx+1}/{n_updates} | "
                      f"Eps: {n_eps} | "
                      f"Avg R: {avg_reward:.2f} | "
                      f"Avg Steps: {avg_steps:.1f} | "
                      f"H: {entropy_coef:.4f}")

    def save(self, filename=MODEL_PATH):
        filename = Path(filename)
        ensure_dir(filename.parent)
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'reward_history': self.reward_history,
            'steps_history': self.steps_history
        }
        torch.save(checkpoint, filename)
        print(f"Model saved as {filename}")

    def load(self, filename=MODEL_PATH):
        checkpoint = torch.load(filename, map_location=self.device, weights_only=False) 
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        self.reward_history = checkpoint.get('reward_history', [])
        self.steps_history = checkpoint.get('steps_history', [])
        print(f"Model loaded from {filename}")


def plot_results(rewards, steps):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    rolling_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
    ax1.plot(rolling_avg)
    ax1.set_title("Learning Progress (Rolling Average Reward)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Average Reward (last 100)")
    ax1.grid(True)

    rolling_steps = np.convolve(steps, np.ones(100)/100, mode='valid')
    ax2.plot(rolling_steps, color='orange')
    ax2.set_title("Steps per Episode (Rolling Average)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Average Steps (last 100)")
    ax2.grid(True)

    plt.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "ppo_learning_curve.png"
    plt.savefig(path)
    print(f"Plot saved as {path}")
    plt.close()


# --- Execution ---

if __name__ == "__main__":
    total_timesteps = LUNAR_LANDER_TIMESTEPS if smoke_mode() else 1_000_000
    if smoke_mode():
        print(f"Smoke mode: training {total_timesteps} timesteps")

    env = gym.make('LunarLander-v3')

    agent = LunarLanderAgent(
        env=env,
        policy_lr=3e-4,
        value_lr=1e-3,
        gamma=0.99,
        lam=0.95,
        n_hidden=128,
        n_steps=1024,
        entropy_coef_start=0.02,
        entropy_coef_end=0.001,
        max_grad_norm=0.5,
    )

    agent.train(total_timesteps=total_timesteps)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    agent.save(MODEL_PATH)

    if not smoke_mode():
        plot_results(agent.reward_history, agent.steps_history)
