import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import wandb
import sys
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from smoke_config import apply_sac_smoke, smoke_mode
from repo_paths import (
    HALFCHEETAH_SAC_FINAL,
    HALFCHEETAH_SAC_MODELS,
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
)

from models import Actor, SoftQNetwork
from buffer import ReplayBuffer

"""
State-of-the-art SAC (Soft Actor-Critic) for MuJoCo HalfCheetah-v5

SAC is an off-policy, entropy-regularized algorithm, fundamentally different from PPO.
Key differences are marked [SAC vs PPO].
"""

@dataclass
class Args:
    exp_name: str = "SAC_HalfCheetah_v5"
    env_id: str = "HalfCheetah-v5"
    total_timesteps: int = 3_000_000  # [SAC: 3M is enough; PPO needed 10M]
    learning_rate: float = 3e-4  # same as PPO
    buffer_size: int = 1_000_000  # [SAC only] 1M transitions replay buffer
    gamma: float = 0.99  # same as PPO
    tau: float = 0.005  # [SAC only] Polyak averaging rate
    batch_size: int = 256  # [SAC: 256 random samples per batch]
    learning_starts: int = 5000  # [SAC only] random exploration phase
    policy_lr: float = 3e-4  # same as PPO
    q_lr: float = 1e-3  # same as PPO
    alpha_lr: float = 3e-4  # [SAC only] temperature learning rate
    save_every: int = 100_000
    seed: int = 1


def make_env(env_id="HalfCheetah-v5"):
    """[SAC vs PPO] Single env, not VectorEnv.
    SAC uses one environment — the off-policy replay buffer provides
    enough data diversity. No auto-reset — manual reset.
    PPO often uses AsyncVectorEnv with parallel envs for faster collection.
    """
    return gym.make(env_id)


class SACAgent:
    def __init__(self, args, env):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        obs_dim = env.observation_space.shape[0]     # 17
        action_dim = env.action_space.shape[0]       # 6

        # --- Actor ---
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.policy_lr)

        """
        --- Twin Q-networks + targets ---
        [SAC vs PPO] PPO has ONE critic V(s) with one optimizer.
        SAC has FOUR Q networks: Q1, Q2 (trained), Q1_target, Q2_target (Polyak).
        Two optimizers: one for the actor, one for both Q networks together."""
        self.qf1 = SoftQNetwork(obs_dim, action_dim).to(self.device)
        self.qf2 = SoftQNetwork(obs_dim, action_dim).to(self.device)

        """Target networks: copies of the Q-networks, updated slowly (Polyak averaging).
        They provide a stable TD target — if we used Q directly, the target
        would change at every gradient step → unstable feedback loop."""
        self.qf1_target = SoftQNetwork(obs_dim, action_dim).to(self.device)
        self.qf2_target = SoftQNetwork(obs_dim, action_dim).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=args.q_lr)

        """
        --- Automatic alpha (temperature) tuning ---
        [SAC vs PPO] No direct equivalent in PPO.
        α controls the trade-off between reward and entropy:
        J(π) = E[Σ r_t + α · H(π(·|s_t))]
        
        target_entropy = -dim(A) is the heuristic from the original SAC paper:
        "entropy should be approximately equal to the negative of the action dimensionality".
        For 6D actions: target = -6.
        
        If current entropy > target → decrease α (less incentive to explore)
        If current entropy < target → increase α (more incentive to explore)
        This automatically balances exploration vs exploitation WITHOUT manual tuning."""
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args.alpha_lr)
        self.alpha = self.log_alpha.exp().item()

        """--- Replay buffer ---
        [SAC vs PPO] PPO had no replay buffer — it collected data and discarded it."""
        self.replay_buffer = ReplayBuffer(obs_dim, action_dim, args.buffer_size, self.device)

    def save_checkpoint(self, path, global_step=0):
        path = model_path(HALFCHEETAH_SAC_MODELS, path)
        ensure_dir(path.parent)
        torch.save({
            'actor': self.actor.state_dict(),
            'qf1': self.qf1.state_dict(),
            'qf2': self.qf2.state_dict(),
            'qf1_target': self.qf1_target.state_dict(),
            'qf2_target': self.qf2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu().numpy(),
            'global_step': global_step,
        }, path)

    def load_checkpoint(self, path):
        path = model_path(HALFCHEETAH_SAC_MODELS, path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        self.qf1.load_state_dict(checkpoint['qf1'])
        self.qf2.load_state_dict(checkpoint['qf2'])
        self.qf1_target.load_state_dict(checkpoint['qf1_target'])
        self.qf2_target.load_state_dict(checkpoint['qf2_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        with torch.no_grad():
            self.log_alpha.copy_(torch.tensor(checkpoint['log_alpha'], device=self.device))
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.args.alpha_lr)
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        return checkpoint.get('global_step', 0)

    def update(self):
        """One gradient step for all networks.

        [SAC vs PPO] Update order:
        1. Critics (Q1, Q2) — MSE to TD target
        2. Actor — maximize Q - α·log_prob
        3. Alpha — adjust temperature toward target entropy
        4. Target networks — Polyak averaging
        """
        obs, actions, rewards, next_obs, terminated = self.replay_buffer.sample(self.args.batch_size)

        """===== 1. CRITIC UPDATE =====
        [SAC vs PPO] TD target with entropy bonus:
        target = r + γ·(1 - terminated)·(min(Q1_target, Q2_target) - α·log π(a'|s'))
        
        PPO target: returns = GAE advantages + values (multi-step, lambda-weighted)
        
        Key differences in the TD target:
        - min(Q1_target, Q2_target): pessimistic estimate, reduces overestimation
        - α·log_prob: soft Bellman target — penalizes overly confident policies, encourages exploration
        - 1-step bootstrap (not multi-step like GAE)
        - terminated (not done): truncation = bootstrap, termination = no bootstrap
        
        [SAC vs PPO] Truncation handling is AUTOMATIC:
        We store `terminated` (not `done`). For truncated episodes:
        terminated=False → (1-terminated)=1 → bootstrap Q(s_next). Correct!
        In PPO you often add gamma·V(s_final) to the return by hand."""
        with torch.no_grad():
            next_state_actions, next_state_log_pi = self.actor.get_action(next_obs)
            qf1_next_target = self.qf1_target(next_obs, next_state_actions)
            qf2_next_target = self.qf2_target(next_obs, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + (1 - terminated) * self.args.gamma * min_qf_next_target
            """td_target = rewards + gamma * (min_q_target - alpha * next_log_probs)
            next_log_probs is the log-probability of the action — always negative (range (-∞, 0]),
            typical values around -2, -5, -20. The more random the action (higher entropy), the lower the value.
            alpha is trained with a separate optimizer and acts as a scale bridge between
            next_log_probs and min_q_target — so both terms are on comparable magnitudes."""

        qf1_values = self.qf1(obs, actions)
        qf2_values = self.qf2(obs, actions)
        qf1_loss = torch.nn.functional.mse_loss(qf1_values, next_q_value)
        qf2_loss = torch.nn.functional.mse_loss(qf2_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        """Loss for both Q networks because both must learn; it does not matter that we often use
        only one of the two predictions in the target."""

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        """
        ===== 2. ACTOR UPDATE =====
        [SAC vs PPO] Fundamentally different policy gradient:
        SAC: loss = E[α·log π(a|s) - min(Q1(s,a), Q2(s,a))]
          We minimize: LOW log_prob (= high entropy) + HIGH Q (= good actions)
          Gradients from Q flow THROUGH the action (reparameterization trick) into actor weights.
        
        PPO: loss = -min(ratio·advantage, clip(ratio)·advantage)
          ratio = exp(new_log_prob - old_log_prob), clipping limits policy change.
          Gradient flows through the log-prob ratio, NOT through the action."""
        pi, log_pi = self.actor.get_action(obs)
        qf1_pi = self.qf1(obs, pi)
        qf2_pi = self.qf2(obs, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        """
        ===== 3. ALPHA (TEMPERATURE) UPDATE =====
        [SAC vs PPO] No direct equivalent in PPO.
        alpha_loss = -E[log(α) · (log π(a|s) + target_entropy)]
        
        If log_prob < -target_entropy (entropy too high) → loss > 0 → decrease α
        If log_prob > -target_entropy (entropy too low) → loss < 0 → increase α
        
        log_probs.detach() — gradient ONLY to log_alpha, not to the actor.
        Actor update (step 2) and alpha update have DIFFERENT objectives:
        the actor minimizes α·log_prob - Q (wants lower log_prob)
        alpha seeks α such that entropy ≈ target (wants log_prob ≈ -target_entropy)"""
        alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        """
        ===== 4. TARGET NETWORK SOFT UPDATE =====
        [SAC vs PPO] Polyak averaging: θ_target = τ·θ + (1-τ)·θ_target
        τ = 0.005 → target changes ~200× slower than the online network.
        This yields a stable TD target (slow-moving target instead of a jumping one).
        In PPO there were no target networks — on-policy data + clipping gave stability."""
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1.0 - self.args.tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1.0 - self.args.tau) * target_param.data)

        return qf_loss.item(), actor_loss.item(), alpha_loss.item()


def train(resume_from=None):
    """
    [SAC vs PPO] Different data collection and update loop:

    PPO is on-policy:
        collect a rollout with the current policy
        compute advantages/returns
        reuse this rollout for several epochs/minibatches
        then discard it

    SAC is off-policy:
        take env steps and store transitions in replay_buffer
        sample random minibatches from replay_buffer
        update critics/actor/alpha from old and recent transitions

    Key difference:
    - PPO uses a temporary rollout buffer.
    - SAC uses a replay buffer with many past transitions.
    """
    args = Args()
    args = apply_sac_smoke(args)
    if smoke_mode():
        print(
            f"Smoke mode: {args.total_timesteps} timesteps, "
            f"learning_starts={args.learning_starts}"
        )
    init_wandb(args.exp_name, args, tags=["sac", "halfcheetah"])

    env = make_env(args.env_id)
    agent = SACAgent(args, env)

    start_step = 0
    if resume_from:
        start_step = agent.load_checkpoint(resume_from)
        print(f"Resumed from {resume_from} at step {start_step}")

    obs, _ = env.reset(seed=args.seed)
    episode_reward = 0
    ep_steps = 0
    reward_history = []
    steps_history = []
    print_every = max(1, args.total_timesteps // 20)

    print(f"Training HalfCheetah SAC | Device: {agent.device}")
    print(f"  {args.total_timesteps / 1e6:.1f}M timesteps")
    print(f"  batch_size={args.batch_size}, learning_starts={args.learning_starts}")
    print(f"  tau={args.tau}, target_entropy={agent.target_entropy}")
    print(f"  buffer_size={args.buffer_size:,}")
    print(f"  Actor params: {sum(p.numel() for p in agent.actor.parameters()):,}")
    print(f"  Critic params (x2): {sum(p.numel() for p in agent.qf1.parameters()):,} each")

    for global_step in tqdm(range(start_step, args.total_timesteps), initial=start_step, total=args.total_timesteps):
        """[SAC vs PPO] Random exploration phase.
        The first learning_starts steps = RANDOM actions from env.action_space.
        Fills the buffer with DIVERSE experience before Q-networks start
        learning. Without this: Q learns from an almost-deterministic policy
        at the start, which biases Q-estimates and slows convergence.
        In PPO there is no such phase — the policy is stochastic from the start
        (Gaussian with large initial std), and on-policy data does not need an
        exploration "seed"."""
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            """Select an action. deterministic=True → mean (testing), False → sample (training).

            [SAC vs PPO] In PPO testing also used the mean, but WITHOUT tanh:
            action = dist.mean (Gaussian mean, clipped to [-1,1]).
            In SAC testing: action = tanh(mean) — squashing is an integral part of the policy.
            """
            with torch.no_grad():
                action, _ = agent.actor.get_action(torch.FloatTensor(obs).to(agent.device))
                action = action.cpu().numpy()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        """[SAC vs PPO] We store terminated (NOT done) in the buffer.
        Truncation handling is automatic: terminated=False → bootstrap."""
        agent.replay_buffer.add(obs, action, reward, next_obs, terminated)

        obs = next_obs
        episode_reward += reward
        ep_steps += 1

        """[SAC vs PPO] Manual reset (single env, not VectorEnv with auto-reset)."""
        if terminated or truncated:
            reward_history.append(episode_reward)
            steps_history.append(ep_steps)
            wandb.log({
                METRIC_EPISODIC_RETURN: episode_reward,
                METRIC_EPISODIC_LENGTH: ep_steps,
            }, step=global_step)
            obs, _ = env.reset()
            episode_reward = 0
            ep_steps = 0

        """[SAC vs PPO] Gradient update EVERY step (after the exploration phase).
        In PPO: update ONCE per full rollout.
        SAC has FRESHER data at every step."""
        # [SAC: 1 per env step, PPO: 10 epochs × 32 minibatches per rollout]
        if global_step > args.learning_starts:
            q_loss, a_loss, alpha_loss = agent.update()
            if global_step % 100 == 0:
                wandb.log({
                    "losses/qf_loss": q_loss,
                    "losses/actor_loss": a_loss,
                    "losses/alpha_loss": alpha_loss,
                    "charts/alpha": agent.alpha,
                }, step=global_step)

        if (global_step + 1) % args.save_every == 0:
            ckpt_path = model_path(
                HALFCHEETAH_SAC_MODELS,
                f"sac_checkpoint_{(global_step + 1) // 1000}k.pth",
            )
            agent.save_checkpoint(ckpt_path, global_step=global_step + 1)

        if (global_step + 1) % print_every == 0 and len(reward_history) > 0:
            avg_reward = np.mean(reward_history[-100:])
            avg_steps = np.mean(steps_history[-100:])
            n_eps = len(reward_history)
            wandb.log({
                METRIC_AVG_RETURN_100: avg_reward,
                METRIC_AVG_STEPS_100: avg_steps,
                "charts/alpha": agent.alpha,
                "charts/buffer_size": agent.replay_buffer.size,
            }, step=global_step)
            print(f"Step {global_step + 1}/{args.total_timesteps} | "
                  f"Eps: {n_eps} | "
                  f"Avg R: {avg_reward:.1f} | "
                  f"Avg Steps: {avg_steps:.0f} | "
                  f"Alpha: {agent.alpha:.4f} | "
                  f"Buffer: {agent.replay_buffer.size:,}")

    final_path = HALFCHEETAH_SAC_FINAL
    ensure_dir(final_path.parent)
    torch.save({
        'actor': agent.actor.state_dict(),
        'qf1': agent.qf1.state_dict(),
        'qf2': agent.qf2.state_dict(),
        'log_alpha': agent.log_alpha.detach().cpu().numpy(),
    }, final_path)
    print(f"Training finished. Model saved as {final_path}")
    env.close()
    finish_wandb()

if __name__ == "__main__":
    train(resume_from=None)
