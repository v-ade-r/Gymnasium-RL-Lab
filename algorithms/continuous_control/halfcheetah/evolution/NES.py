"""
xNES (Exponential Natural Evolution Strategies) for MuJoCo HalfCheetah-v5.

Background — what NES does:
    Natural Evolution Strategies (Wierstra et al. 2014) maximize the expected
    fitness J(theta) = E_{x ~ N(mu, Sigma)}[f(x)] by following the NATURAL
    gradient of J with respect to the parameters of the search distribution
    (mu, Sigma). The natural gradient (Amari 1998) is the inverse Fisher
    information matrix times the plain gradient; it makes the update
    INVARIANT to how the distribution is parameterized, which is what
    "natural" means here.

    For Gaussian search distributions, the natural gradient has a clean
    closed form. xNES (Glasmachers et al. 2010) is the version that
    represents the covariance via an "exponential map" of the log-Cholesky
    factor, which guarantees positive-definiteness of the covariance
    automatically and is the most principled / canonical NES variant.

    The other member of the NES family implemented here is:
        - sNES (Schaul 2011) : diagonal covariance, scales to d > 1e5

This implementation includes:
    - Rank-based utility shaping (the canonical NES "fitness function")
      which makes updates robust to outlier rewards
    - Mirrored / antithetic sampling for variance reduction of gradient estimates
    - Welford running observation normalizer (critical on MuJoCo)
    - Mean-based L2 weight decay in the fitness
    - Parallel candidate evaluation via multiprocessing
    - Restart-on-stagnation with population doubling (NES analogue of IPOP)

References:
    Wierstra et al., "Natural Evolution Strategies" (JMLR 2014)
    Glasmachers et al., "Exponential Natural Evolution Strategies" (GECCO 2010)
    Schaul et al., "High Dimensions and Heavy Tails for Natural Evolution Strategies" (GECCO 2011)
"""
import os
import math
import time
import sys
import multiprocessing as mp
from dataclasses import dataclass
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
from smoke_config import apply_nes_smoke, smoke_mode
from repo_paths import HALFCHEETAH_EVOLUTION_MODELS, SNES_FINAL, XNES_FINAL, ensure_dir, model_path
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

    The network is intentionally small because xNES maintains a d x d
    transformation matrix. With hidden = (32, 32) the parameter dimension is
    d = 17*32 + 32 + 32*32 + 32 + 32*6 + 6 = 1830, which is comfortable for
    full-covariance NES.

    There is no exploration noise inside the policy itself — exploration
    comes entirely from sampling weights theta from the search distribution
    N(mu, A A^T).
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
    """Concatenate all parameters of `net` into one 1D float64 numpy array.

    float64 is used for the optimizer math (the d x d matrix B is sensitive
    to roundoff); we cast back to float32 when writing into the network.
    """
    return torch.cat([p.data.view(-1) for p in net.parameters()]).cpu().numpy().astype(np.float64)


def set_flat_params(net, flat):
    """Inverse of flat_params: write the 1D vector back into the network."""
    flat_t = torch.from_numpy(flat.astype(np.float32))
    idx = 0
    for p in net.parameters():
        n = p.numel()
        p.data.copy_(flat_t[idx:idx + n].view_as(p))
        idx += n


def init_random_params(net, std=0.1, seed=0):
    """Reinitialize the network with N(0, std^2) weights and return the flat vector.

    A small std keeps the initial policy close to "do nothing" (tanh(0) = 0)
    while still breaking symmetry between hidden units. The optimizer expands
    sigma during the first generations and discovers useful behaviour.
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
    """Numerically stable online mean and variance for observations.

    Raw MuJoCo observations have wildly different scales across dimensions
    (joint angles, angular velocities, body positions, ...). Normalizing them
    is essential for evolution-strategy-style optimizers, otherwise the
    effective fitness landscape is so anisotropic that a single global step
    size cannot adapt and the search stalls.

    Welford's algorithm maintains the running mean and the running sum of
    squared deviations (M2) in a numerically stable way. Chan et al. 1979
    provides the formula for combining two partial sets of statistics, which
    we use to merge the per-rollout stats produced by worker processes back
    into the global normalizer.
    """

    def __init__(self, dim):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)
        self.count = 0

    def update_batch(self, x):
        """Incorporate a batch of observations of shape (n, dim)."""
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
# 3. xNES CORE
# ============================================================================

class xNES:
    """Exponential Natural Evolution Strategies.

    Search distribution:    pi(x) = N(mu, A A^T)   with   A = sigma * B

        mu     :  R^d         distribution mean
        sigma  :  scalar      global step size
        B      :  R^{d x d}   shape matrix (det(B) preserved by the update)

    Per-generation algorithm:
        1. Sample lambda noise vectors s_k ~ N(0, I_d) and form candidates
               x_k = mu + sigma * B @ s_k
        2. Evaluate f(x_k) in the environment.
        3. Sort by fitness (best first) and assign rank-based utilities u_k
           that sum to zero (positive for top half, negative for bottom half).
        4. Build natural-gradient estimates IN s-COORDINATES:
               G_delta = sum_k u_k * s_k                            (R^d)
               G_M     = sum_k u_k * (s_k s_k^T - I)                (R^{d x d})
                       = sum_k u_k * s_k s_k^T   (the -I cancels: sum u_k = 0)
               G_sigma = trace(G_M) / d                             (scalar)
               G_B     = G_M - G_sigma * I                          (traceless symmetric)
           Trace and traceless components correspond to "global scale" and
           "shape" directions of the natural gradient on the covariance.
        5. Apply with separate learning rates:
               mu    <- mu + eta_mu * sigma * (B @ G_delta)
               sigma <- sigma * exp(eta_sigma / 2 * G_sigma)
               B     <- B * exp(eta_B / 2 * G_B)

    The matrix exponential exp(eta/2 * G_B) is replaced here by its
    first-order Taylor expansion (I + eta/2 * G_B). For the small default
    learning rates (eta ~ 1e-4 for d=1830), this is essentially exact:
    the omitted O(eta^2) term contributes ~1e-8 per element per generation,
    far below the precision of float64 numerics.

    Convention: TELL receives fitness values where LOWER is BETTER. For RL
    callers should pass -episode_return so that maximization of return
    becomes minimization of fitness.
    """

    def __init__(self, mean_init, sigma_init, pop_size=None, seed=0,
                 eta_mu=1.0, eta_sigma=None, eta_B=None):
        self.d = len(mean_init)
        self.mu = np.array(mean_init, dtype=np.float64)
        self.sigma = float(sigma_init)
        self.B = np.eye(self.d, dtype=np.float64)  # initially isotropic shape
        self.rng = np.random.default_rng(seed)

        # Population size. Forced even so that mirrored sampling is symmetric.
        if pop_size is None:
            pop_size = 4 + int(3 * np.log(self.d))
        if pop_size % 2 == 1:
            pop_size += 1
        self.lam = pop_size

        # Default learning rates from Glasmachers et al. 2010, Table 1.
        # eta_mu = 1: the natural gradient direction on the mean is exactly
        # the optimal step length under the Gaussian assumption, no damping needed.
        # eta_sigma = eta_B: the same default for both the global-scale and
        # the shape part of the covariance gradient.
        default_eta_cov = (9.0 + 3.0 * math.log(self.d)) / (5.0 * self.d * math.sqrt(self.d))
        self.eta_mu = eta_mu
        self.eta_sigma = eta_sigma if eta_sigma is not None else default_eta_cov
        self.eta_B = eta_B if eta_B is not None else default_eta_cov

        # Rank-based utility function (Wierstra et al. 2014, eq. 12).
        # For lambda samples sorted from best (index 0) to worst (index lambda-1):
        #     raw_k = max(0, log(lambda/2 + 1) - log(k+1))
        # Then divide by sum and subtract 1/lambda to make sum-to-zero.
        # Only the top half receive positive raw values; the bottom half is
        # uniformly penalized. After centering the utilities sum to zero,
        # which means the natural-gradient estimate is unbiased.
        log_term = math.log(self.lam / 2 + 1)
        u_raw = np.maximum(0.0, log_term - np.log(np.arange(1, self.lam + 1)))
        self.utilities = u_raw / u_raw.sum() - 1.0 / self.lam

        self.gen = 0
        self._last_s = None

    def ask(self):
        """Sample lambda candidate parameter vectors x_k = mu + sigma * B @ s_k.

        Mirrored / antithetic sampling: half of the noise vectors s_k are
        negated. For every direction s explored we also explore -s, which
        halves the variance of the natural-gradient estimate and helps escape
        narrow basins of attraction.
        """
        half = self.lam // 2
        s_half = self.rng.standard_normal((half, self.d))
        s = np.vstack([s_half, -s_half])  # shape (lam, d), antithetic pairs
        # Vectorized x = mu + sigma * (s @ B^T) since B @ s_k for each row.
        x = self.mu + self.sigma * (s @ self.B.T)
        self._last_s = s
        return x

    def tell(self, fitnesses):
        """Update mu, sigma, B given fitnesses for the last ask() batch.

        fitnesses[i] corresponds to self._last_s[i]. LOWER is BETTER.
        """
        if self._last_s is None:
            raise RuntimeError("tell() called before ask()")
        self.gen += 1

        # 1. Sort samples by fitness. argsort ascending: best (lowest) first.
        order = np.argsort(fitnesses)
        s_sorted = self._last_s[order]  # (lam, d) ranked best -> worst
        u = self.utilities              # already best -> worst

        # 2. Natural gradient on the mean (in s-coordinates).
        #    G_delta is a weighted sum of noise vectors; positive utilities
        #    pull mu toward "good" directions, negative utilities push it away
        #    from "bad" ones.
        G_delta = (u[:, None] * s_sorted).sum(axis=0)  # R^d

        # 3. Natural gradient on the covariance.
        #    Mathematically G_M = sum_k u_k * (s_k s_k^T - I), but since the
        #    utilities sum to zero the -I term vanishes identically. The line
        #    below is the memory-efficient form (no (lam, d, d) intermediate).
        weighted_s = u[:, None] * s_sorted     # (lam, d)
        G_M = weighted_s.T @ s_sorted          # (d, d), symmetric

        # 4. Split into the two natural-gradient directions:
        #       G_sigma : isotropic scale change   (trace component)
        #       G_B     : pure shape change         (traceless component)
        G_sigma = float(np.trace(G_M)) / self.d
        G_B = G_M - G_sigma * np.eye(self.d)   # traceless symmetric matrix

        # 5. Apply the natural-gradient updates.
        # 5a. Mean update. Lifting from s-coordinates back to x-coordinates
        #     introduces the factor sigma * B = A.
        self.mu = self.mu + self.eta_mu * self.sigma * (self.B @ G_delta)

        # 5b. Step size update. Multiplicative form keeps sigma > 0 forever.
        self.sigma = self.sigma * math.exp(self.eta_sigma / 2.0 * G_sigma)

        # 5c. Shape update. Exact form is B <- B @ exp(eta_B/2 * G_B), but
        #     the matrix exponential of a d x d matrix costs O(d^3) per
        #     generation. We use the first-order Taylor expansion
        #         exp(M) ~ I + M
        #     The omitted O(M^2) term has Frobenius norm of order eta_B^2
        #     ~ 1e-8 per generation for our d, which is negligible. For
        #     applications that need the exact map, replace this line with
        #     an eigendecomposition: G_B = V D V^T -> B @ V @ diag(exp(eta_B/2 * D)) @ V^T.
        self.B = self.B + (self.eta_B / 2.0) * (self.B @ G_B)

    def covariance_diag_extremes(self):
        """Diagnostic: smallest and largest variance among the d marginals.

        The marginal variances are the diagonal of A A^T = sigma^2 * (B B^T).
        The ratio max/min indicates how anisotropic the search distribution
        has become; large ratio means some axes are being explored much more
        than others (xNES analogue of "condition number" in CMA-ES).
        """
        # diag(B B^T) = row-wise squared norm of B
        diag_var = (self.sigma ** 2) * np.sum(self.B ** 2, axis=1)
        return float(diag_var.min()), float(diag_var.max())

    def state_dict(self):
        return {
            "mu": self.mu, "sigma": self.sigma, "B": self.B,
            "gen": self.gen, "lam": self.lam,
        }

    def load_state_dict(self, sd):
        self.mu = sd["mu"]; self.sigma = sd["sigma"]; self.B = sd["B"]
        self.gen = sd["gen"]


# ============================================================================
# 3b. sNES CORE (separable / diagonal NES)
# ============================================================================

class sNES:
    """Separable (diagonal) Natural Evolution Strategies (Schaul et al. 2011).

    Search distribution:  pi(x) = N(mu, diag(sigma_vec^2))

        mu        : R^d   distribution mean
        sigma_vec : R^d   per-coordinate standard deviation (the diagonal)

    sNES is the NES variant designed for HIGH dimension / FEW generations,
    which is exactly the RL regime here (d ~ 1830, only a few hundred
    generations affordable). It differs from full xNES in two decisive ways:

        1. Memory: it keeps only a length-d std vector instead of a d x d
           shape matrix B, dropping memory from O(d^2) to O(d) and removing
           the O(d^2)/O(d^3) per-generation matrix algebra.

        2. Learning rate: the diagonal NES rate has NO 1/d factor,
               eta_sigma = (3 + ln d) / (5 sqrt(d))   (~0.05 at d = 1830)
           versus xNES's (9 + 3 ln d) / (5 d sqrt(d)) (~8e-5). That is ~600x
           larger, so the per-coordinate variances actually ADAPT within the
           generation budget instead of staying frozen at their initial value.

    Per-generation algorithm:
        1. Sample s_k ~ N(0, I_d) and form candidates x_k = mu + sigma_vec * s_k
        2. Evaluate f(x_k), sort, assign rank-based utilities u_k (sum to zero)
        3. Natural-gradient estimates in s-coordinates (separable case):
               grad_mu        = sum_k u_k * s_k                 (R^d)
               grad_log_sigma = sum_k u_k * (s_k^2 - 1)         (R^d)
        4. Apply:
               mu        <- mu + eta_mu * sigma_vec * grad_mu
               sigma_vec <- sigma_vec * exp(eta_sigma / 2 * grad_log_sigma)

    Convention: TELL receives fitness values where LOWER is BETTER. RL callers
    pass -episode_return so maximization becomes minimization.
    """

    def __init__(self, mean_init, sigma_init, pop_size=None, seed=0,
                 eta_mu=1.0, eta_sigma=None):
        self.d = len(mean_init)
        self.mu = np.array(mean_init, dtype=np.float64)
        self.sigma_vec = np.full(self.d, float(sigma_init), dtype=np.float64)
        self.rng = np.random.default_rng(seed)

        # Population size, forced even for symmetric mirrored sampling.
        if pop_size is None:
            pop_size = 4 + int(3 * np.log(self.d))
        if pop_size % 2 == 1:
            pop_size += 1
        self.lam = pop_size

        # Diagonal NES learning rate (Schaul et al. 2011). No 1/d factor, so it
        # stays usefully large at high d, unlike the full-covariance xNES rate.
        default_eta_sigma = (3.0 + math.log(self.d)) / (5.0 * math.sqrt(self.d))
        self.eta_mu = eta_mu
        self.eta_sigma = eta_sigma if eta_sigma is not None else default_eta_sigma

        # Rank-based utilities, identical to the xNES utilities (sum to zero).
        log_term = math.log(self.lam / 2 + 1)
        u_raw = np.maximum(0.0, log_term - np.log(np.arange(1, self.lam + 1)))
        self.utilities = u_raw / u_raw.sum() - 1.0 / self.lam

        self.gen = 0
        self._last_s = None

    def ask(self):
        """Sample lambda candidates x_k = mu + sigma_vec * s_k (mirrored pairs)."""
        half = self.lam // 2
        s_half = self.rng.standard_normal((half, self.d))
        s = np.vstack([s_half, -s_half])  # antithetic pairs
        x = self.mu + self.sigma_vec * s
        self._last_s = s
        return x

    def tell(self, fitnesses):
        """Update mu and sigma_vec given fitnesses (LOWER is BETTER)."""
        if self._last_s is None:
            raise RuntimeError("tell() called before ask()")
        self.gen += 1

        order = np.argsort(fitnesses)
        s_sorted = self._last_s[order]  # best -> worst
        u = self.utilities              # best -> worst

        # Natural gradient on the mean and on the log-std (diagonal).
        grad_mu = (u[:, None] * s_sorted).sum(axis=0)
        grad_log_sigma = (u[:, None] * (s_sorted ** 2 - 1.0)).sum(axis=0)

        self.mu = self.mu + self.eta_mu * self.sigma_vec * grad_mu
        self.sigma_vec = self.sigma_vec * np.exp((self.eta_sigma / 2.0) * grad_log_sigma)

    def covariance_diag_extremes(self):
        """Smallest and largest marginal variance (diag of the covariance)."""
        var = self.sigma_vec ** 2
        return float(var.min()), float(var.max())

    @property
    def sigma(self):
        """Scalar step-size summary (geometric mean of the per-coordinate stds).

        Provided so the training loop, logging and restart checks can treat
        sNES and xNES uniformly through a single `nes.sigma` value.
        """
        return float(np.exp(np.mean(np.log(self.sigma_vec))))

    def state_dict(self):
        return {"mu": self.mu, "sigma_vec": self.sigma_vec,
                "gen": self.gen, "lam": self.lam}

    def load_state_dict(self, sd):
        self.mu = sd["mu"]; self.sigma_vec = sd["sigma_vec"]; self.gen = sd["gen"]


def make_optimizer(variant, mean, sigma_init, pop_size, seed, args):
    """Construct the requested NES variant with a uniform interface.

    Both variants expose ask()/tell()/state_dict()/load_state_dict(), a `.mu`
    distribution mean, a scalar `.sigma`, and covariance_diag_extremes(), so the
    training loop is variant-agnostic.
    """
    if variant == "snes":
        return sNES(mean, sigma_init, pop_size=pop_size, seed=seed,
                    eta_mu=args.eta_mu, eta_sigma=args.eta_sigma)
    if variant == "xnes":
        return xNES(mean, sigma_init, pop_size=pop_size, seed=seed,
                    eta_mu=args.eta_mu, eta_sigma=args.eta_sigma, eta_B=args.eta_B)
    raise ValueError(f"Unknown nes_variant: {variant!r} (expected 'snes' or 'xnes')")


# ============================================================================
# 4. PARALLEL ROLLOUT WORKER (multiprocessing.Pool)
# ============================================================================
#
# Each worker owns its own gym env and policy network (created once via the
# pool initializer). For every candidate the main process sends the flat
# parameter vector plus the current observation normalization stats. The
# worker overwrites its policy weights, runs one episode, and returns the
# fitness together with the partial Welford statistics of the observations
# seen during the episode. The main process merges all partial stats into
# the global normalizer.

_W_ENV = None
_W_NET = None


def _init_worker(env_id, hidden_sizes, seed_base):
    """Pool initializer: create one env + one policy net per worker process."""
    global _W_ENV, _W_NET
    pid = os.getpid()
    _W_ENV = gym.make(env_id)
    _W_ENV.reset(seed=seed_base + pid)
    obs_dim = _W_ENV.observation_space.shape[0]
    act_dim = _W_ENV.action_space.shape[0]
    _W_NET = PolicyNet(obs_dim, act_dim, hidden=tuple(hidden_sizes))
    _W_NET.eval()
    # One torch thread per worker process; otherwise N workers x M threads
    # oversubscribe the CPU and wall-clock time gets worse than serial.
    torch.set_num_threads(1)


def _evaluate_one(args):
    """Evaluate a single candidate: load weights, run one episode, return fitness.

    Returns:
        fitness         : scalar to MINIMIZE (= -episode_return + l2_penalty)
        ep_return       : raw cumulative environment reward (logging only)
        ep_steps        : episode length (for env-step accounting)
        partial_mean    : Welford mean over observations seen this episode
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
        # Always normalize the observation before feeding it to the policy.
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

    # L2 penalty on the mean of squared parameters: independent of the
    # number of network parameters, so the same coefficient generalizes
    # across architecture changes.
    l2_pen = l2_coef * float(np.mean(flat ** 2))

    # The optimizer minimizes; we want to maximize return; so negate.
    fitness = -ep_return + l2_pen
    return fitness, ep_return, steps, p_mean, p_M2, p_count


# ============================================================================
# 5. CONFIG
# ============================================================================

@dataclass
class Args:
    exp_name: str = "xNES_HalfCheetah_v5"
    env_id: str = "HalfCheetah-v5"
    total_timesteps: int = 20_000_000

    # --- Policy network ---
    hidden_sizes: tuple = (32, 32)
    init_param_std: float = 0.1  # std of initial random weights (small -> near-zero policy)

    # --- NES variant ---
    # "snes": separable/diagonal NES (Schaul 2011) - RECOMMENDED for this
    #         problem. At d ~ 1830 with only a few hundred affordable
    #         generations, full xNES's covariance learning rate (~8e-5) is far
    #         too small to adapt, and its d x d matrix is memory-heavy. sNES
    #         keeps only a length-d std vector and uses a ~600x larger learning
    #         rate, so the per-coordinate scales actually adapt.
    # "xnes": full-covariance exponential NES (Glasmachers 2010), kept for
    #         comparison / completeness.
    nes_variant: str = "snes"

    # --- NES hyperparameters ---
    sigma_init: float = 0.5
    pop_size: int = 32           # lambda; auto-bumped to even number
    eta_mu: float = 1.0          # natural gradient on mean: full step
    # eta_sigma: None -> per-variant default
    #   xNES: (9 + 3 ln d) / (5 d sqrt(d))   sNES: (3 + ln d) / (5 sqrt(d))
    # eta_B: full-xNES shape-matrix rate; ignored by sNES.
    eta_sigma: float = None
    eta_B: float = None

    # --- Fitness ---
    n_episodes_per_candidate: int = 1   # >1 averages out env stochasticity
    max_episode_steps: int = 1000
    l2_coef: float = 0.001              # weight decay coefficient on mean(theta^2)

    # --- Honest final selection ---
    # During training "best" is chosen from a single noisy episode, which is
    # optimistically biased (it picks the luckiest rollout). At the end we
    # re-evaluate the top finalists AND the distribution mean over many episodes
    # and keep whichever has the best AVERAGED return.
    final_eval_episodes: int = 30       # episodes used to re-rank finalists at the end
    n_finalists: int = 10               # how many top single-episode candidates to re-evaluate

    # --- Parallelism ---
    n_workers: int = 8

    # --- Restart-on-stagnation (NES analogue of IPOP) ---
    use_restart: bool = True
    restart_factor: int = 2             # population size multiplier per restart
    restart_tol_gen: int = 30           # restart if best return stalls for this many generations
    restart_tol_sigma: float = 1e-10    # restart if step size collapses
    restart_max_pop: int = 256          # cap on population size after several restarts

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
    fitness is the average across rollouts. Useful when the env reward has
    high variance (initial-state randomness, noisy dynamics).
    """
    obs_mean = obs_norm.mean.astype(np.float32)
    obs_std = obs_norm.std.astype(np.float32)
    lam = len(candidates)

    # Build a flat task list: each candidate is repeated n_episodes times.
    tasks = []
    for c in candidates:
        for _ in range(n_episodes):
            tasks.append((c, obs_mean, obs_std, l2_coef, max_steps))

    raw = pool.map(_evaluate_one, tasks)

    # Aggregate per-candidate (averages of fitness/return, sum of steps,
    # all partial Welford stats collected for the merge step).
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


def save_checkpoint(path, nes, obs_norm, args, best_params, best_return, total_steps, restart_idx):
    path = model_path(HALFCHEETAH_EVOLUTION_MODELS, path)
    ensure_dir(path.parent)
    np.savez(
        path,
        nes_state=np.array([nes.state_dict()], dtype=object),
        obs_norm=np.array([obs_norm.state_dict()], dtype=object),
        args=np.array([vars(args)], dtype=object),
        best_params=best_params,
        best_return=best_return,
        total_steps=total_steps,
        restart_idx=restart_idx,
    )


def train(resume_from=None):
    args = Args()
    args = apply_nes_smoke(args)
    variant_label = "sNES" if args.nes_variant == "snes" else "xNES"
    args.exp_name = f"{variant_label}_HalfCheetah_v5"
    if smoke_mode():
        print(
            f"Smoke mode: {args.total_timesteps} timesteps, "
            f"pop_size={args.pop_size}, workers={args.n_workers}"
        )
    init_wandb(args.exp_name, args, tags=[args.nes_variant, "nes", "evolution", "halfcheetah"])

    # Build a template network only to determine d (parameter dimension) and
    # the initial mean vector. The actual rollouts use per-worker networks.
    template_env = gym.make(args.env_id)
    obs_dim = template_env.observation_space.shape[0]
    act_dim = template_env.action_space.shape[0]
    template_env.close()

    template_net = PolicyNet(obs_dim, act_dim, hidden=tuple(args.hidden_sizes))
    init_mean = init_random_params(template_net, std=args.init_param_std, seed=args.seed)
    d = len(init_mean)

    # Global running observation normalizer (persists across restarts).
    obs_norm = RunningStats(obs_dim)

    # Worker pool: each process gets its own env + policy via the initializer.
    pool = mp.Pool(
        processes=args.n_workers,
        initializer=_init_worker,
        initargs=(args.env_id, args.hidden_sizes, args.seed),
    )

    # Initial NES instance (variant selected by args.nes_variant).
    nes = make_optimizer(
        args.nes_variant, init_mean, args.sigma_init,
        pop_size=args.pop_size, seed=args.seed, args=args,
    )

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
        nes.load_state_dict(ckpt["nes_state"][0])
        obs_norm.load_state_dict(ckpt["obs_norm"][0])
        best_overall_params = ckpt["best_params"]
        best_overall_return = float(ckpt["best_return"])
        total_steps = int(ckpt["total_steps"])
        restart_idx = int(ckpt["restart_idx"])
        print(f"Resumed from {resume_from} at step {total_steps}, gen {nes.gen}")

    print(f"Training HalfCheetah {variant_label}")
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}, d (theta)={d}")
    if args.nes_variant == "xnes":
        print(f"  lambda={nes.lam}, eta_mu={nes.eta_mu}, "
              f"eta_sigma={nes.eta_sigma:.2e}, eta_B={nes.eta_B:.2e}")
    else:
        print(f"  lambda={nes.lam}, eta_mu={nes.eta_mu}, "
              f"eta_sigma={nes.eta_sigma:.2e} (diagonal)")
    print(f"  sigma_init={nes.sigma}")
    print(f"  utilities (top 4 / bottom 4): {nes.utilities[:4].round(3)} ... {nes.utilities[-4:].round(3)}")
    print(f"  utilities sum (should be 0): {nes.utilities.sum():.2e}")
    print(f"  workers={args.n_workers}, total_timesteps={args.total_timesteps:,}")

    pbar = tqdm(total=args.total_timesteps, initial=total_steps)

    while total_steps < args.total_timesteps:
        gen_start = time.time()

        # --- ASK ---
        candidates = nes.ask()

        # --- EVALUATE in parallel ---
        fitnesses, returns, gen_steps, partials = evaluate_population(
            pool, candidates, obs_norm,
            args.l2_coef, args.max_episode_steps, args.n_episodes_per_candidate,
        )

        # --- Merge per-rollout obs stats into the global normalizer ---
        for p_mean, p_M2, p_count in partials:
            obs_norm.merge(p_mean, p_M2, p_count)

        # --- TELL ---
        # The xNES update uses utilities derived from RANK only, so passing
        # raw fitnesses or any monotonic transform of them is equivalent.
        # The shaping (rank -> utility) happens internally inside xNES.
        nes.tell(fitnesses)

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
        if nes.gen % args.log_every_gens == 0:
            var_min, var_max = nes.covariance_diag_extremes()
            wandb.log({
                METRIC_BEST_RETURN: gen_best_return,
                METRIC_MEAN_RETURN: float(returns.mean()),
                METRIC_WORST_RETURN: float(returns.min()),
                METRIC_BEST_OVERALL_RETURN: best_overall_return,
                "nes/sigma": nes.sigma,
                "nes/var_diag_min": var_min,
                "nes/var_diag_max": var_max,
                "nes/anisotropy_ratio": var_max / max(var_min, 1e-30),
                "nes/lambda": nes.lam,
                METRIC_TOTAL_ENV_STEPS: total_steps,
                METRIC_GENERATION: nes.gen,
                "charts/restart_idx": restart_idx,
                METRIC_GEN_SECONDS: time.time() - gen_start,
                METRIC_OBS_NORM_COUNT: obs_norm.count,
            }, step=total_steps)

        # --- Restart-on-stagnation (NES analogue of IPOP-CMA-ES) ---
        if args.use_restart:
            # Track whether the best in this generation improves on the
            # running best by at least 1.0 reward unit.
            if gen_best_return > last_best_for_stagnation + 1.0:
                last_best_for_stagnation = gen_best_return
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            should_restart = (
                stagnation_counter >= args.restart_tol_gen
                or nes.sigma < args.restart_tol_sigma
            )

            if should_restart and nes.lam < args.restart_max_pop:
                restart_idx += 1
                new_lam = min(nes.lam * args.restart_factor, args.restart_max_pop)
                # Warm restart: reseed the search from the best policy ever
                # found and reset sigma to its initial value. Population is
                # doubled to widen the search and escape the current basin.
                nes = make_optimizer(
                    args.nes_variant,
                    best_overall_params.copy(),
                    args.sigma_init,
                    pop_size=new_lam,
                    seed=args.seed + restart_idx,
                    args=args,
                )
                stagnation_counter = 0
                last_best_for_stagnation = -float("inf")
                tqdm.write(
                    f"[restart] #{restart_idx}: new lambda={new_lam}, "
                    f"resuming from best return {best_overall_return:.1f}"
                )

        # --- Periodic checkpoint ---
        if nes.gen % args.save_every_gens == 0:
            ckpt_path = model_path(
                HALFCHEETAH_EVOLUTION_MODELS,
                f"{args.nes_variant}_checkpoint_gen{nes.gen}.npz",
            )
            save_checkpoint(ckpt_path, nes, obs_norm, args,
                            best_overall_params, best_overall_return,
                            total_steps, restart_idx)

    pbar.close()

    # --- Honest final selection -------------------------------------------------
    # The "best" tracked during training came from a single noisy episode, so it
    # is optimistically biased. Re-evaluate the top finalists AND the current
    # distribution mean over many episodes, then keep the best AVERAGED return.
    n_eval = 2 if smoke_mode() else args.final_eval_episodes
    finalist_params = [p for _, p in finalists]
    finalist_params.append(nes.mu.copy())  # the distribution mean is often the most robust
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

    # Final save: best params + obs normalizer (everything needed to replay the
    # policy). One file per variant so comparative runs don't overwrite each other.
    final_path = SNES_FINAL if args.nes_variant == "snes" else XNES_FINAL
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

def evaluate(model_path=SNES_FINAL, n_episodes=10, render=False):
    """Load a trained policy and play n_episodes deterministic rollouts.

    Defaults to the sNES model (the default trainer variant). Pass
    model_path=XNES_FINAL to evaluate an xNES run instead.
    """
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
    # Multiprocessing uses 'fork' by default on Linux, which works fine here.
    # On macOS/Windows you would need mp.set_start_method('spawn') and the
    # worker globals would have to be re-initialized differently.
    train(resume_from=None)
