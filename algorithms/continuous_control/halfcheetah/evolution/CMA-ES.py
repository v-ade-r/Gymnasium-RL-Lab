"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for MuJoCo HalfCheetah-v5.

This is a black-box, gradient-free optimizer. The whole policy network is treated
as a single parameter vector theta in R^d, and we maintain a multivariate
Gaussian distribution N(m, sigma^2 * C) over theta. Each generation we:

    1. ASK   - sample lambda candidates theta_i from the distribution
    2. EVAL  - run each candidate as a full episode in the env, return the
               cumulative reward as fitness
    3. TELL  - sort candidates by fitness and update m, sigma, C so that
               the distribution drifts toward better-performing regions

Notable features in this implementation (Hansen 2016 + community improvements):
    - Active CMA-ES (negative weights for the worst mu candidates)
    - Mirrored / antithetic sampling (variance reduction of the gradient estimate)
    - Lazy eigendecomposition of C (every ~1/(10*d*c1) generations)
    - Welford running observation normalizer (critical for MuJoCo)
    - Mean-based L2 weight decay in the fitness (regularization)
    - Multiprocessing pool for parallel candidate evaluation
    - IPOP-CMA-ES restart strategy (population doubles on stagnation)

References:
    Hansen, "The CMA Evolution Strategy: A Tutorial" (2016, arXiv:1604.00772)
    Auger & Hansen, "A Restart CMA Evolution Strategy With Increasing Population Size" (2005)
    Jastrebski & Arnold, "Improving Evolution Strategies through Active Covariance Matrix Adaptation" (2006)
    Salimans et al., "Evolution Strategies as a Scalable Alternative to RL" (2017)
"""
import os
import math
import time
import sys
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import wandb
from tqdm import tqdm

_HC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HC_ROOT))
sys.path.insert(0, str(_HC_ROOT.parents[2]))
from smoke_config import apply_cma_es_smoke, smoke_mode
from repo_paths import CMAES_FINAL, HALFCHEETAH_EVOLUTION_MODELS, ensure_dir, model_path
from wandb_utils import (
    finish_wandb,
    init_wandb,
    METRIC_BEST_OVERALL_RETURN,
    METRIC_BEST_RETURN,
    METRIC_GEN_SECONDS,
    METRIC_GENERATION,
    METRIC_MEAN_RETURN,
    METRIC_OBS_NORM_COUNT,
    METRIC_TOTAL_ENV_STEPS,
    METRIC_WORST_RETURN,
)


# ============================================================================
# 1. POLICY NETWORK
# ============================================================================

class PolicyNet(nn.Module):
    """Small deterministic MLP: obs (17) -> action (6) in [-1, 1].

    The network is intentionally small. Full CMA-ES stores a d x d covariance
    matrix and performs an O(d^3) eigendecomposition periodically, so we want
    d in the low thousands. With hidden = (32, 32) we get
    17*32 + 32 + 32*32 + 32 + 32*6 + 6 = 1830 parameters.

    There is no log_std head here: in CMA-ES exploration comes from sampling
    weights theta from N(m, sigma^2 * C). The forward pass is fully deterministic.
    The output tanh squashes actions into the valid action range.
    """

    def __init__(self, obs_dim, action_dim, hidden=(32, 32)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers += [nn.Linear(prev, action_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)


def flat_params(net):
    """Concatenate all network parameters into a single 1D float64 numpy array.

    float64 is used for CMA-ES numerics (covariance matrix is sensitive to
    rounding); we cast back to float32 only when writing into the network.
    """
    return torch.cat([p.data.view(-1) for p in net.parameters()]).cpu().numpy().astype(np.float64)


def set_flat_params(net, flat):
    """Inverse of flat_params: write a 1D array back into the network in-place."""
    flat_t = torch.from_numpy(flat.astype(np.float32))
    idx = 0
    for p in net.parameters():
        n = p.numel()
        p.data.copy_(flat_t[idx:idx + n].view_as(p))
        idx += n


def init_random_params(net, std=0.1, seed=0):
    """Reinitialize all weights with N(0, std^2) noise and return the flat vector.

    A small std means the initial policy is close to "do nothing" (network
    outputs ~0 -> tanh(0) = 0 actions). CMA-ES then expands sigma during the
    first generations and discovers useful behavior. Starting from a non-trivial
    pretrained policy would also work, but for a from-scratch benchmark we
    start as close to neutral as possible while keeping symmetry-breaking noise.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.data.normal_(mean=0.0, std=std, generator=g)
    return flat_params(net)


# ============================================================================
# 2. RUNNING OBSERVATION NORMALIZER (Welford / Chan online algorithm)
# ============================================================================

class RunningStats:
    """Numerically stable running mean and variance for observations.

    On MuJoCo, raw observation dimensions have wildly different scales
    (joint angles vs angular velocities vs body positions). Without
    normalization the fitness landscape is so distorted that the global
    step-size sigma cannot adapt to it, and CMA-ES fails to make progress.

    We use Welford's online algorithm (and Chan et al. 1979 formula for
    merging two partial sets of statistics from worker processes), which
    is stable even when count grows very large.
    """

    def __init__(self, dim):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)  # sum of squared deviations
        self.count = 0

    def update_batch(self, x):
        """Incorporate a batch of observations (n x dim)."""
        if x.size == 0:
            return
        n = x.shape[0]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        delta = batch_mean - self.mean
        new_count = self.count + n
        self.mean = self.mean + delta * (n / new_count)
        self.M2 = self.M2 + batch_var * n + delta ** 2 * (self.count * n / new_count)
        self.count = new_count

    def merge(self, other_mean, other_M2, other_count):
        """Combine partial Welford stats produced by a worker process."""
        if other_count == 0:
            return
        if self.count == 0:
            self.mean = other_mean.copy()
            self.M2 = other_M2.copy()
            self.count = other_count
            return
        delta = other_mean - self.mean
        total = self.count + other_count
        self.mean = self.mean + delta * (other_count / total)
        self.M2 = self.M2 + other_M2 + delta ** 2 * (self.count * other_count / total)
        self.count = total

    @property
    def std(self):
        if self.count < 2:
            return np.ones_like(self.mean)
        return np.sqrt(self.M2 / self.count)

    def state_dict(self):
        return {"mean": self.mean, "M2": self.M2, "count": self.count}

    def load_state_dict(self, sd):
        self.mean = sd["mean"]
        self.M2 = sd["M2"]
        self.count = sd["count"]


# ============================================================================
# 3. CMA-ES CORE
# ============================================================================

class CMAES:
    """Covariance Matrix Adaptation Evolution Strategy with active update.

    Distribution state:
        m       : mean vector  (R^d)
        sigma   : global step size (scalar)
        C       : covariance matrix (d x d), with eigendecomposition C = B D^2 B^T

    Evolution paths (low-pass filtered "momenta"):
        p_sigma : used by Cumulative Step-size Adaptation (CSA)
        p_c     : used by the rank-1 update of C

    Convention: TELL receives fitnesses where LOWER is BETTER.
    For RL, the caller passes -episode_return so that the maximization of
    return becomes minimization of fitness.

    Hyperparameter defaults follow Hansen 2016 (eqs. 48-53) and adapt to
    problem dimension d and effective parent number mu_eff.
    """

    def __init__(self, mean_init, sigma_init, pop_size=None, seed=0):
        self.d = len(mean_init)
        self.mean = np.array(mean_init, dtype=np.float64)
        self.sigma = float(sigma_init)
        self.rng = np.random.default_rng(seed)

        # Population size (forced even for clean antithetic / mirrored pairs).
        if pop_size is None:
            pop_size = 4 + int(3 * np.log(self.d))
        if pop_size % 2 == 1:
            pop_size += 1
        self.lam = pop_size
        self.mu = self.lam // 2  # number of selected parents

        # Raw recombination weights (Hansen 2016, eq. 49):
        #     w_i_raw = ln((lam+1)/2) - ln(i)   for i = 1 .. lam
        # First mu values are positive (parents), the rest are negative (active update).
        weights_raw = np.log((self.lam + 1) / 2) - np.log(np.arange(1, self.lam + 1))

        # Positive weights normalize to sum = 1.
        self.weights_pos = weights_raw[:self.mu] / weights_raw[:self.mu].sum()

        # Effective number of selected parents (variance-effective selection mass).
        # Higher mu_eff -> more averaging -> slower but more stable progress.
        self.mu_eff = 1.0 / (self.weights_pos ** 2).sum()

        # Negative weights for the active rank-mu update (worst mu candidates).
        weights_neg_raw = weights_raw[self.mu:]  # all negative numbers
        mu_eff_neg = (weights_neg_raw.sum() ** 2) / (weights_neg_raw ** 2).sum()

        # Strategy parameters from Hansen 2016, eqs. 55-58.
        # c_sigma : learning rate for the step-size evolution path
        # d_sigma : damping for sigma updates
        # c_c     : learning rate for the rank-1 evolution path
        # c_1     : learning rate for the rank-1 update of C
        # c_mu    : learning rate for the rank-mu update of C
        self.c_sigma = (self.mu_eff + 2) / (self.d + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0.0, np.sqrt((self.mu_eff - 1) / (self.d + 1)) - 1) + self.c_sigma
        self.c_c = (4 + self.mu_eff / self.d) / (self.d + 4 + 2 * self.mu_eff / self.d)
        self.c_1 = 2 / ((self.d + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1 - self.c_1,
            2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.d + 2) ** 2 + self.mu_eff),
        )

        # Rescale negative weights so that the active update keeps C positive
        # definite and stable (Hansen 2016, eq. 53, three constraints).
        alpha_mu_neg = 1 + self.c_1 / self.c_mu
        alpha_mueff_neg = 1 + (2 * mu_eff_neg) / (self.mu_eff + 2)
        alpha_posdef_neg = (1 - self.c_1 - self.c_mu) / (self.d * self.c_mu)
        scale_neg = min(alpha_mu_neg, alpha_mueff_neg, alpha_posdef_neg) / abs(weights_neg_raw.sum())
        self.weights_neg = weights_neg_raw * scale_neg
        self.weights = np.concatenate([self.weights_pos, self.weights_neg])  # length lam

        # Evolution paths start at zero.
        self.p_sigma = np.zeros(self.d)
        self.p_c = np.zeros(self.d)

        # Covariance matrix and its eigendecomposition C = B * D^2 * B^T.
        # B is orthogonal (eigenvectors as columns), D is the vector of sqrt(eigenvalues).
        self.C = np.eye(self.d)
        self.B = np.eye(self.d)
        self.D = np.ones(self.d)

        self.gen = 0
        self.eigen_eval = 0  # generation at which eigendecomp was last refreshed

        # E[||N(0, I_d)||] - expected norm of a d-dimensional standard normal,
        # used as the "neutral" reference length in CSA (Hansen 2016, eq. 36).
        self.chi_n = math.sqrt(self.d) * (1 - 1 / (4 * self.d) + 1 / (21 * self.d ** 2))

        # Lazy eigendecomposition interval (Hansen 2016, eq. 49).
        # Recomputing eigenvectors of a d x d matrix costs O(d^3); we postpone
        # it because C changes only slowly between generations.
        self.eigen_update_interval = max(1, int(1 / ((self.c_1 + self.c_mu) * self.d * 10)))

        # Buffers populated by ask() and consumed by tell().
        self._last_z = None
        self._last_y = None

    def ask(self):
        """Sample lam candidate parameter vectors x_i = m + sigma * B * D * z_i.

        Mirrored (antithetic) sampling: half of the noise vectors z_i are
        negated. This guarantees that for every direction explored, the
        opposite direction is also explored. It reduces the variance of the
        evolutionary gradient estimate and helps escape shallow local minima.
        """
        half = self.lam // 2
        z_half = self.rng.standard_normal((half, self.d))
        z = np.vstack([z_half, -z_half])  # shape (lam, d), antithetic pairs

        # y_i = B * D * z_i  (vectorized: BD has columns scaled by D, then z @ BD^T)
        BD = self.B * self.D  # broadcasting D over columns: shape (d, d)
        y = z @ BD.T  # shape (lam, d)
        x = self.mean + self.sigma * y

        self._last_z = z
        self._last_y = y
        return x

    def tell(self, fitnesses):
        """Update m, sigma, C using the fitnesses of the last ask()-batch.

        fitnesses[i] is the fitness of self._last_y[i]. LOWER is BETTER.
        """
        if self._last_y is None:
            raise RuntimeError("tell() called before ask()")
        self.gen += 1

        # 1. Sort by fitness (ascending). Best (lowest fitness) goes first.
        order = np.argsort(fitnesses)
        y_sorted = self._last_y[order]
        z_sorted = self._last_z[order]

        # 2. Recombination: weighted mean of the best mu y-vectors.
        #    The new mean drifts in the direction of better-performing samples.
        y_w = (self.weights_pos[:, None] * y_sorted[:self.mu]).sum(axis=0)
        z_w = (self.weights_pos[:, None] * z_sorted[:self.mu]).sum(axis=0)
        self.mean = self.mean + self.sigma * y_w  # m <- m + sigma * <y>_w

        # 3. Step-size evolution path (CSA).
        #    p_sigma is a low-pass filter of C^{-1/2} * <y>_w.
        #    Note: C^{-1/2} * y = B * D^{-1} * B^T * (B * D * z) = B * z.
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + \
                       math.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * (self.B @ z_w)

        # 4. Step-size update.
        #    If recent steps tend to align (||p_sigma|| > E||N(0,I)||), the
        #    landscape is consistent -> increase sigma. Otherwise decrease it.
        ps_norm = np.linalg.norm(self.p_sigma)
        self.sigma = self.sigma * math.exp((self.c_sigma / self.d_sigma) * (ps_norm / self.chi_n - 1))

        # 5. Heaviside switch: temporarily freeze the rank-1 path when sigma
        #    grows much faster than expected (prevents "axis-collapse" of C).
        h_sigma = float(
            ps_norm / math.sqrt(1 - (1 - self.c_sigma) ** (2 * (self.gen + 1)))
            < (1.4 + 2 / (self.d + 1)) * self.chi_n
        )

        # 6. Rank-1 evolution path: low-pass filter of <y>_w in original space.
        self.p_c = (1 - self.c_c) * self.p_c + \
                   h_sigma * math.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * y_w

        # 7. Build the per-candidate weights for the rank-mu update.
        #    For NEGATIVE weights (worst mu candidates) we apply the
        #    Mahalanobis rescaling from Hansen 2016 eq. 46:
        #        w_i^o = w_i * d / ||C^{-1/2} y_i||^2  for w_i < 0
        #    This guarantees C remains positive definite even when subtracting
        #    very long y-vectors. ||C^{-1/2} y_i|| = ||B z_i|| = ||z_i||
        #    because B is orthogonal.
        weights_circle = np.empty(self.lam)
        weights_circle[:self.mu] = self.weights_pos
        neg_norms_sq = np.sum(z_sorted[self.mu:] ** 2, axis=1)
        weights_circle[self.mu:] = self.weights_neg * self.d / (neg_norms_sq + 1e-30)

        # 8. Update C: decay term + rank-1 (from p_c) + rank-mu (from y_sorted).
        delta_h = (1 - h_sigma) * self.c_c * (2 - self.c_c)  # small Heaviside correction
        c_decay = (1 - self.c_1 - self.c_mu * self.weights.sum()) + self.c_1 * delta_h
        rank_one = self.c_1 * np.outer(self.p_c, self.p_c)
        # Memory-efficient rank-mu sum: weighted_y^T @ y_sorted produces (d, d)
        # without materializing the (lam, d, d) intermediate.
        weighted_y = weights_circle[:, None] * y_sorted
        rank_mu = self.c_mu * (weighted_y.T @ y_sorted)
        self.C = c_decay * self.C + rank_one + rank_mu

        # 9. Lazy eigendecomposition refresh.
        if self.gen - self.eigen_eval > self.eigen_update_interval:
            self._update_eigen()

    def _update_eigen(self):
        """Refresh C = B * D^2 * B^T. Done lazily because eigendecomp is O(d^3)."""
        self.C = (self.C + self.C.T) / 2  # enforce numerical symmetry
        eigvals, B = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-20)  # clamp tiny negative values from roundoff
        self.D = np.sqrt(eigvals)
        self.B = B
        self.eigen_eval = self.gen

    def condition_number(self):
        """Ratio of largest to smallest eigenvalue of C.

        Diagnostic indicator: a very large value (>1e14) means C has degenerated
        to a near-singular matrix and the search has effectively collapsed
        to a low-dimensional subspace. IPOP uses this as a restart trigger.
        """
        return (self.D.max() / self.D.min()) ** 2

    def state_dict(self):
        return {
            "mean": self.mean, "sigma": self.sigma,
            "C": self.C, "B": self.B, "D": self.D,
            "p_sigma": self.p_sigma, "p_c": self.p_c,
            "gen": self.gen, "eigen_eval": self.eigen_eval,
            "lam": self.lam,
        }

    def load_state_dict(self, sd):
        self.mean = sd["mean"]; self.sigma = sd["sigma"]
        self.C = sd["C"]; self.B = sd["B"]; self.D = sd["D"]
        self.p_sigma = sd["p_sigma"]; self.p_c = sd["p_c"]
        self.gen = sd["gen"]; self.eigen_eval = sd["eigen_eval"]


# ============================================================================
# 4. PARALLEL ROLLOUT WORKER (multiprocessing.Pool)
# ============================================================================
#
# Each worker process keeps its own gym env and policy network alive between
# tasks (via the pool initializer). The main process sends (params, obs_mean,
# obs_std, ...) for each candidate; the worker writes params into its policy,
# rolls out one episode, and returns the fitness plus partial Welford stats
# of the observations it saw. We then merge all partial stats into the global
# observation normalizer in the main process.

# Globals set inside each worker by _init_worker.
_W_ENV = None
_W_NET = None


def _init_worker(env_id, hidden_sizes, seed_base):
    """Pool initializer: create env + policy once per worker process."""
    global _W_ENV, _W_NET
    pid = os.getpid()
    _W_ENV = gym.make(env_id)
    _W_ENV.reset(seed=seed_base + pid)
    obs_dim = _W_ENV.observation_space.shape[0]
    act_dim = _W_ENV.action_space.shape[0]
    _W_NET = PolicyNet(obs_dim, act_dim, hidden=tuple(hidden_sizes))
    _W_NET.eval()
    # Each worker uses a single torch thread; otherwise N workers x M threads
    # oversubscribe CPU and become slower than serial.
    torch.set_num_threads(1)


def _evaluate_one(args):
    """Evaluate a single candidate: write its params, run an episode, return stats.

    Returns:
        fitness         : scalar to MINIMIZE (= -episode_return + l2_penalty)
        ep_return       : raw cumulative environment reward (for logging)
        ep_steps        : episode length (for env-step accounting)
        partial_mean    : Welford mean over the observations seen this episode
        partial_M2      : Welford sum of squared deviations
        partial_count   : number of observations in this episode
    """
    flat, obs_mean, obs_std, l2_coef, max_steps = args
    set_flat_params(_W_NET, flat)

    obs, _ = _W_ENV.reset()
    ep_return = 0.0
    obs_buffer = []
    steps = 0
    while True:
        # Apply running-stats normalization before passing the observation to the net.
        obs_norm = (obs - obs_mean) / (obs_std + 1e-8)
        with torch.no_grad():
            a = _W_NET(torch.from_numpy(obs_norm.astype(np.float32))).numpy()
        a = np.clip(a, -1.0, 1.0)
        next_obs, r, term, trunc, _ = _W_ENV.step(a)

        ep_return += float(r)
        obs_buffer.append(obs)
        steps += 1
        obs = next_obs
        if term or trunc or steps >= max_steps:
            break

    obs_arr = np.asarray(obs_buffer, dtype=np.float64)
    p_mean = obs_arr.mean(axis=0)
    p_var = obs_arr.var(axis=0)
    p_count = len(obs_arr)
    p_M2 = p_var * p_count

    # Mean-based L2 penalty: independent of network size, so this hyperparameter
    # transfers cleanly when you change the architecture.
    l2_pen = l2_coef * float(np.mean(flat ** 2))

    # CMA-ES minimizes; we want to MAXIMIZE return; so negate.
    fitness = -ep_return + l2_pen
    return fitness, ep_return, steps, p_mean, p_M2, p_count


# ============================================================================
# 5. CONFIG
# ============================================================================

@dataclass
class Args:
    exp_name: str = "CMA-ES_HalfCheetah_v5"
    env_id: str = "HalfCheetah-v5"
    total_timesteps: int = 20_000_000

    # --- Policy network ---
    hidden_sizes: tuple = (32, 32)
    init_param_std: float = 0.1  # std of initial random weights (small -> near-zero policy)

    # --- CMA-ES ---
    sigma_init: float = 0.5
    pop_size: int = 32  # lambda; will be auto-bumped to even number

    # --- Fitness ---
    n_episodes_per_candidate: int = 1  # >1 averages out env stochasticity
    max_episode_steps: int = 1000
    l2_coef: float = 0.001  # weight decay coefficient applied to mean(theta^2)

    # --- Honest final selection ---
    # During training "best" is chosen from a single noisy episode, which is
    # optimistically biased (it picks the luckiest rollout). At the end we
    # re-evaluate the top finalists AND the distribution mean over many episodes
    # and keep whichever has the best AVERAGED return.
    final_eval_episodes: int = 30  # episodes used to re-rank finalists at the end
    n_finalists: int = 10          # how many top single-episode candidates to re-evaluate

    # --- Parallelism ---
    n_workers: int = 8

    # --- IPOP-CMA-ES restart ---
    use_ipop: bool = True
    ipop_factor: int = 2          # population size multiplier per restart
    ipop_tol_gen: int = 30        # restart if best return stalls for this many generations
    ipop_tol_sigma: float = 1e-10 # restart if step size collapses
    ipop_tol_cond: float = 1e14   # restart if covariance becomes near-singular
    ipop_max_pop: int = 256       # cap on population size after several restarts

    # --- Logging / checkpoints ---
    save_every_gens: int = 50
    log_every_gens: int = 1
    seed: int = 1


# ============================================================================
# 6. TRAINING LOOP
# ============================================================================

def evaluate_population(pool, candidates, obs_norm, l2_coef, max_steps, n_episodes):
    """Run one full evaluation of the candidate population.

    If n_episodes > 1, each candidate is rolled out multiple times and its
    fitness is the average across rollouts. This reduces noise in the fitness
    estimate at the cost of more env steps per generation.
    """
    obs_mean = obs_norm.mean.astype(np.float32)
    obs_std = obs_norm.std.astype(np.float32)
    lam = len(candidates)

    # Build the flat task list: each candidate is repeated n_episodes times.
    tasks = []
    for c in candidates:
        for _ in range(n_episodes):
            tasks.append((c, obs_mean, obs_std, l2_coef, max_steps))

    raw = pool.map(_evaluate_one, tasks)

    # Aggregate per-candidate (average of fitness/return, sum of steps, all stats merged).
    fitnesses = np.zeros(lam)
    returns = np.zeros(lam)
    total_steps = 0
    all_partials = []
    for i in range(lam):
        slice_ = raw[i * n_episodes:(i + 1) * n_episodes]
        fits = np.array([r[0] for r in slice_])
        rets = np.array([r[1] for r in slice_])
        fitnesses[i] = fits.mean()
        returns[i] = rets.mean()
        for r in slice_:
            total_steps += r[2]
            all_partials.append((r[3], r[4], r[5]))

    return fitnesses, returns, total_steps, all_partials


def reevaluate_params(pool, param_list, obs_norm, l2_coef, max_steps, n_eval):
    """Re-evaluate a small set of parameter vectors over many episodes each.

    Returns (mean_returns, std_returns) aligned with param_list. Used at the end
    of training to select the genuinely best policy by AVERAGED return, removing
    the optimistic bias of picking the luckiest single-episode candidate.
    """
    obs_mean = obs_norm.mean.astype(np.float32)
    obs_std = obs_norm.std.astype(np.float32)
    tasks = []
    for p in param_list:
        for _ in range(n_eval):
            tasks.append((p, obs_mean, obs_std, l2_coef, max_steps))
    raw = pool.map(_evaluate_one, tasks)
    mean_returns = np.zeros(len(param_list))
    std_returns = np.zeros(len(param_list))
    for i in range(len(param_list)):
        rets = np.array([raw[i * n_eval + j][1] for j in range(n_eval)])
        mean_returns[i] = rets.mean()
        std_returns[i] = rets.std()
    return mean_returns, std_returns


def save_checkpoint(path, cma, obs_norm, args, best_params, best_return, total_steps, restart_idx):
    path = model_path(HALFCHEETAH_EVOLUTION_MODELS, path)
    ensure_dir(path.parent)
    np.savez(
        path,
        cma_state=np.array([cma.state_dict()], dtype=object),
        obs_norm=np.array([obs_norm.state_dict()], dtype=object),
        args=np.array([vars(args)], dtype=object),
        best_params=best_params,
        best_return=best_return,
        total_steps=total_steps,
        restart_idx=restart_idx,
    )


def train(resume_from=None):
    args = Args()
    args = apply_cma_es_smoke(args)
    if smoke_mode():
        print(
            f"Smoke mode: {args.total_timesteps} timesteps, "
            f"pop_size={args.pop_size}, workers={args.n_workers}"
        )
    init_wandb(args.exp_name, args, tags=["cma-es", "evolution", "halfcheetah"])

    # Build a template network only to determine d (parameter dimension) and
    # the initial mean vector. The actual rollouts use per-worker networks.
    template_env = gym.make(args.env_id)
    obs_dim = template_env.observation_space.shape[0]
    act_dim = template_env.action_space.shape[0]
    template_env.close()

    template_net = PolicyNet(obs_dim, act_dim, hidden=tuple(args.hidden_sizes))
    init_mean = init_random_params(template_net, std=args.init_param_std, seed=args.seed)
    d = len(init_mean)

    # Global running observation normalizer (persists across IPOP restarts).
    obs_norm = RunningStats(obs_dim)

    # Worker pool: each process gets its own env + policy via the initializer.
    pool = mp.Pool(
        processes=args.n_workers,
        initializer=_init_worker,
        initargs=(args.env_id, args.hidden_sizes, args.seed),
    )

    # Initial CMA-ES instance.
    cma = CMAES(init_mean, args.sigma_init, pop_size=args.pop_size, seed=args.seed)

    # Tracking the best policy ever seen across all restarts.
    best_overall_return = -float("inf")
    best_overall_params = init_mean.copy()

    total_steps = 0
    restart_idx = 0
    stagnation_counter = 0
    last_best_for_stagnation = -float("inf")

    # Buffer of top candidates (by noisy single-episode return) kept for the
    # honest re-evaluation at the end of training.
    finalists = []  # list of (single_episode_return, params)

    if resume_from is not None:
        ckpt = np.load(resume_from, allow_pickle=True)
        cma.load_state_dict(ckpt["cma_state"][0])
        obs_norm.load_state_dict(ckpt["obs_norm"][0])
        best_overall_params = ckpt["best_params"]
        best_overall_return = float(ckpt["best_return"])
        total_steps = int(ckpt["total_steps"])
        restart_idx = int(ckpt["restart_idx"])
        print(f"Resumed from {resume_from} at step {total_steps}, gen {cma.gen}")

    print(f"Training HalfCheetah CMA-ES")
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}, d (theta)={d}")
    print(f"  lambda={cma.lam}, mu={cma.mu}, mu_eff={cma.mu_eff:.2f}")
    print(f"  sigma_init={cma.sigma}, c_sigma={cma.c_sigma:.4f}, c_c={cma.c_c:.4f}")
    print(f"  c_1={cma.c_1:.6f}, c_mu={cma.c_mu:.6f}, lazy_eigen_every={cma.eigen_update_interval}")
    print(f"  workers={args.n_workers}, total_timesteps={args.total_timesteps:,}")

    pbar = tqdm(total=args.total_timesteps, initial=total_steps)

    while total_steps < args.total_timesteps:
        gen_start = time.time()

        # --- ASK ---
        candidates = cma.ask()

        # --- EVALUATE in parallel ---
        fitnesses, returns, gen_steps, partials = evaluate_population(
            pool, candidates, obs_norm,
            args.l2_coef, args.max_episode_steps, args.n_episodes_per_candidate,
        )

        # --- Merge per-rollout obs stats into the global normalizer ---
        for p_mean, p_M2, p_count in partials:
            obs_norm.merge(p_mean, p_M2, p_count)

        # --- TELL ---
        # Note: CMA-ES update is intrinsically rank-based (only argsort matters),
        # so passing raw fitnesses or any monotonic transform of them is equivalent.
        # No explicit "centered rank shaping" is needed here, unlike in OpenAI ES
        # where the gradient estimate uses fitness magnitudes directly.
        cma.tell(fitnesses)

        total_steps += gen_steps
        pbar.update(gen_steps)

        # --- Track best ---
        gen_best_idx = int(np.argmax(returns))
        gen_best_return = float(returns[gen_best_idx])
        if gen_best_return > best_overall_return:
            best_overall_return = gen_best_return
            best_overall_params = candidates[gen_best_idx].copy()

        # Keep a top-K buffer of candidates for the honest final re-evaluation.
        finalists.append((gen_best_return, candidates[gen_best_idx].copy()))
        finalists.sort(key=lambda t: -t[0])
        del finalists[args.n_finalists:]

        # --- Logging ---
        if cma.gen % args.log_every_gens == 0:
            wandb.log({
                METRIC_BEST_RETURN: gen_best_return,
                METRIC_MEAN_RETURN: float(returns.mean()),
                METRIC_WORST_RETURN: float(returns.min()),
                METRIC_BEST_OVERALL_RETURN: best_overall_return,
                "cma/sigma": cma.sigma,
                "cma/condition_number": cma.condition_number(),
                "cma/p_sigma_norm": float(np.linalg.norm(cma.p_sigma)),
                "cma/p_c_norm": float(np.linalg.norm(cma.p_c)),
                "cma/lambda": cma.lam,
                METRIC_TOTAL_ENV_STEPS: total_steps,
                METRIC_GENERATION: cma.gen,
                "charts/restart_idx": restart_idx,
                METRIC_GEN_SECONDS: time.time() - gen_start,
                METRIC_OBS_NORM_COUNT: obs_norm.count,
            }, step=total_steps)

        # --- IPOP restart logic ---
        if args.use_ipop:
            # Stagnation: track whether the BEST in the current generation
            # improves the running best by at least 1.0 reward.
            if gen_best_return > last_best_for_stagnation + 1.0:
                last_best_for_stagnation = gen_best_return
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            should_restart = (
                stagnation_counter >= args.ipop_tol_gen
                or cma.sigma < args.ipop_tol_sigma
                or cma.condition_number() > args.ipop_tol_cond
            )

            if should_restart and cma.lam < args.ipop_max_pop:
                restart_idx += 1
                new_lam = min(cma.lam * args.ipop_factor, args.ipop_max_pop)
                # Warm restart: start from the best policy ever found and reset
                # sigma to its initial value. The classic IPOP-CMA-ES paper does
                # a fully RANDOM restart, but for RL warm-starting from the best
                # so far is much more sample-efficient and standard practice.
                cma = CMAES(
                    best_overall_params.copy(),
                    args.sigma_init,
                    pop_size=new_lam,
                    seed=args.seed + restart_idx,
                )
                stagnation_counter = 0
                last_best_for_stagnation = -float("inf")
                tqdm.write(
                    f"[IPOP] restart #{restart_idx}: new lambda={new_lam}, "
                    f"resuming from best return {best_overall_return:.1f}"
                )

        # --- Periodic checkpoint ---
        if cma.gen % args.save_every_gens == 0:
            ckpt_path = model_path(
                HALFCHEETAH_EVOLUTION_MODELS,
                f"cmaes_checkpoint_gen{cma.gen}.npz",
            )
            save_checkpoint(ckpt_path, cma, obs_norm, args,
                            best_overall_params, best_overall_return,
                            total_steps, restart_idx)

    pbar.close()

    # --- Honest final selection -------------------------------------------------
    # The "best" tracked during training came from a single noisy episode, so it
    # is optimistically biased. Re-evaluate the top finalists AND the current
    # distribution mean over many episodes, then keep the best AVERAGED return.
    n_eval = 2 if smoke_mode() else args.final_eval_episodes
    finalist_params = [p for _, p in finalists]
    finalist_params.append(cma.mean.copy())  # the distribution mean is often the most robust
    labels = [f"finalist_{i}" for i in range(len(finalists))] + ["dist_mean"]
    mean_returns, std_returns = reevaluate_params(
        pool, finalist_params, obs_norm, args.l2_coef, args.max_episode_steps, n_eval,
    )
    best_i = int(np.argmax(mean_returns))
    best_overall_params = finalist_params[best_i]
    best_overall_return = float(mean_returns[best_i])
    best_overall_return_std = float(std_returns[best_i])

    pool.close()
    pool.join()

    print(f"\nFinal re-evaluation over {n_eval} episodes "
          f"(top {len(finalists)} finalists + distribution mean):")
    for j, (lab, m, s) in enumerate(zip(labels, mean_returns, std_returns)):
        marker = "  <-- selected" if j == best_i else ""
        print(f"  {lab:>12}: {m:8.1f} +/- {s:6.1f}{marker}")

    # Final save: best params + obs normalizer (everything needed to replay the policy).
    final_path = CMAES_FINAL
    ensure_dir(final_path.parent)
    np.savez(
        final_path,
        params=best_overall_params,
        obs_mean=obs_norm.mean,
        obs_std=obs_norm.std,
        hidden_sizes=np.array(args.hidden_sizes),
        env_id=args.env_id,
        best_return=best_overall_return,
        best_return_std=best_overall_return_std,
    )
    print(f"\nTraining finished. Honest best return: "
          f"{best_overall_return:.1f} +/- {best_overall_return_std:.1f} "
          f"(over {n_eval} episodes)")
    print(f"Saved final model to {final_path}")
    finish_wandb()


# ============================================================================
# 7. EVALUATION
# ============================================================================

def evaluate(model_path=CMAES_FINAL, n_episodes=10, render=False):
    """Load a trained policy and play n_episodes deterministic rollouts."""
    data = np.load(model_path, allow_pickle=True)
    flat = data["params"]
    obs_mean = data["obs_mean"]
    obs_std = data["obs_std"]
    hidden = tuple(int(x) for x in data["hidden_sizes"])
    env_id = str(data["env_id"])

    env = gym.make(env_id, render_mode="human" if render else None)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    net = PolicyNet(obs_dim, act_dim, hidden=hidden)
    set_flat_params(net, flat)
    net.eval()

    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        steps = 0
        while True:
            obs_n = (obs - obs_mean) / (obs_std + 1e-8)
            with torch.no_grad():
                a = net(torch.from_numpy(obs_n.astype(np.float32))).numpy()
            a = np.clip(a, -1.0, 1.0)
            obs, r, term, trunc, _ = env.step(a)
            ep_return += float(r)
            steps += 1
            if term or trunc:
                break
        returns.append(ep_return)
        print(f"Episode {ep + 1:2d} | steps={steps:4d} | return={ep_return:8.1f}")

    env.close()
    print(f"\nMean return over {n_episodes} eps: {np.mean(returns):.1f} +/- {np.std(returns):.1f}")


# ============================================================================
# 8. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # mp uses 'fork' by default on Linux, which works fine here.
    # On macOS/Windows you would need mp.set_start_method('spawn') and the
    # worker globals would need to be re-initialized differently.
    train(resume_from=None)
