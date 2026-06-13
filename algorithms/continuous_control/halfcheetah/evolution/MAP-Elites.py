"""
CMA-ME (Covariance Matrix Adaptation MAP-Elites Emitters) for MuJoCo HalfCheetah-v5.

Background — Quality-Diversity and MAP-Elites:
    Standard optimizers (CMA-ES, xNES, gradient descent, ...) seek ONE
    solution that maximizes a single objective. Quality-Diversity (QD)
    algorithms seek a COLLECTION of high-performing solutions that are
    DIVERSE in some user-specified "behavior space".

    MAP-Elites (Mouret & Clune 2015) is the canonical QD algorithm:
        1. Define a low-dimensional behavior descriptor b(x) for each
           candidate solution x.  For HalfCheetah we use
               b(x) = (mean forward velocity, mean control effort)
           so the behavior space is 2-dimensional.
        2. Discretize the behavior space into a regular grid (the
           "archive").  Each grid cell stores ONE solution: the best
           one ever found whose b(x) falls inside that cell.
        3. Iterate: pick a solution from the archive, perturb it,
           evaluate its (fitness, behavior descriptor), and try to
           insert it into the archive.

    The output is a 2D map of high-performing policies, one per cell.
    For HalfCheetah you end up with a portfolio: fast-and-energetic,
    fast-and-efficient, slow-and-quiet, backwards-walking, ... — all
    from a single training run.  Pure single-objective optimizers
    only ever return ONE policy.

CMA-ME (Fontaine et al. 2020) — the top-performing CMA variant of MAP-Elites:
    Vanilla MAP-Elites uses simple Gaussian mutations to generate
    new candidates.  CMA-ME replaces them with CMA-ES "emitters":
    each emitter is a small CMA-ES instance that adapts its own
    search distribution.

    The crucial trick is the IMPROVEMENT RANKING used as the fitness
    signal sent back to the inner CMA-ES:
        - Solutions that filled an EMPTY cell rank highest
        - Solutions that IMPROVED an existing cell rank middle
        - Solutions that didn't change the archive rank lowest
    Because CMA-ES is intrinsically rank-based, this re-ranks the
    selection toward QD-relevant directions: the emitter learns to
    move toward unexplored or under-optimized regions of behavior
    space, not just toward higher raw fitness.

CMA-MAE (Fontaine & Nikolaidis 2023) — what this file actually implements:
    Plain CMA-ME only rewards a candidate when it fills an EMPTY cell or
    strictly beats the elite already in its cell.  On a noisy, hard task
    like HalfCheetah that signal is sparse: once the reachable cells are
    filled, almost every candidate is "rejected", every emitter restarts
    constantly (we observed ~1000 restarts in a 20M-step run), and the
    archive never concentrates optimization pressure on quality — the best
    elite stalled around a return of ~220.

    CMA-MAE fixes this with a SOFT archive.  Each cell keeps a running
    acceptance THRESHOLD t_e (not just the elite's fitness).  A candidate
    with objective f is ranked by its improvement over that threshold,
    delta = f - t_e, and whenever f > t_e the threshold is annealed toward
    f with an archive learning rate alpha:
        t_e <- (1 - alpha) * t_e + alpha * f
    The cell still stores the genuine best elite separately (for output).

    The learning rate interpolates between two classic algorithms:
        alpha -> 1 : thresholds jump straight to f         => recovers CMA-ME
        alpha -> 0 : thresholds frozen at the floor min_f   => recovers a
                     pure CMA-ES optimizer (ranks by raw fitness)
    A small alpha (we default to 0.05) therefore gives the emitters a dense,
    CMA-ES-like quality gradient *and* still illuminates the behavior space,
    which is exactly what was missing before: best elite jumps from the
    low hundreds to the low thousands while coverage stays high.

    Two more changes matter for HalfCheetah specifically:
        - Each candidate is evaluated over several episodes and its fitness
          and behavior descriptor are AVERAGED.  Single-episode evaluation
          let lucky rollouts occupy cells with descriptors they cannot
          reproduce (stored vx 0.69 vs measured 0.44 on re-eval).
        - Behavior bounds and grid resolution are sized to the regime a
          small 32x32 policy can actually reach under an ES budget.

References:
    Mouret & Clune, "Illuminating search spaces by mapping elites" (2015)
    Fontaine et al., "Covariance Matrix Adaptation for the Rapid
        Illumination of Behavior Space" (GECCO 2020)
    Fontaine & Nikolaidis, "Covariance Matrix Adaptation MAP-Annealing"
        (GECCO 2023)
    Cully et al., "Robots that can adapt like animals" (Nature 2015)
"""
import os
import io
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HC_ROOT))
sys.path.insert(0, str(_HC_ROOT.parents[2]))
from smoke_config import apply_map_elites_smoke, smoke_mode
from repo_paths import (
    HALFCHEETAH_EVOLUTION,
    HALFCHEETAH_EVOLUTION_MODELS,
    MAP_ELITES_FINAL,
    ensure_dir,
    model_path,
)
from wandb_utils import (
    finish_wandb,
    init_wandb,
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

    The network is intentionally small because each emitter maintains its
    own d x d covariance matrix.  With hidden = (32, 32) we get
    d = 17*32 + 32 + 32*32 + 32 + 32*6 + 6 = 1830 parameters, which fits
    comfortably for several CMA-ES emitters at once.

    There is no exploration noise inside the policy itself — exploration
    comes entirely from the emitters sampling parameter vectors theta from
    their search distributions.  The output tanh squashes actions into
    the valid range.
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
    """Concatenate all parameters of `net` into a single 1D float64 numpy array.

    float64 is used for the optimizer math (covariance matrices are sensitive
    to roundoff); we cast back to float32 only when writing into the network.
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
    while still breaking symmetry between hidden units.  Each emitter starts
    near this point with a small extra perturbation, then expands its sigma
    during the first generations to discover useful behaviour.
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
    (joint angles, angular velocities, body positions, ...).  Normalizing
    them is essential for evolution-strategy-style optimizers, otherwise
    the effective fitness landscape is so anisotropic that the global
    step size cannot adapt and the search stalls.

    Welford's algorithm maintains the running mean and the running sum
    of squared deviations (M2) in a numerically stable way.  Chan et al.
    1979 provides the formula for combining two partial sets of
    statistics, which we use to merge the per-rollout stats produced by
    worker processes back into the global normalizer.
    """

    def __init__(self, dim):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)
        self.count = 0

    def update_batch(self, x):
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
        self.mean = sd["mean"]; self.M2 = sd["M2"]; self.count = sd["count"]


# ============================================================================
# 3. CMA-ES (used as MAP-Elites emitter)
# ============================================================================

class CMAES:
    """Standard CMA-ES with rank-1 + rank-mu update and CSA step-size adaptation.

    This is a slightly slimmed-down version of the full CMA-ES (no active
    update with negative weights, no IPOP, no lazy-eigen scheduling kept
    explicit).  At the level of a MAP-Elites emitter this is sufficient:
    the diversity machinery is provided by the outer MAP-Elites loop,
    so the inner optimizer just needs to be a competent local-search
    method that adapts its sampling distribution.

    Convention: TELL receives fitness values where LOWER is BETTER.
    MAP-Elites maximizes the raw return; the conversion to a CMA-ES
    minimization signal happens inside the emitter, not here.
    """

    def __init__(self, mean_init, sigma_init, pop_size, seed=0):
        self.d = len(mean_init)
        self.mean = np.array(mean_init, dtype=np.float64)
        self.sigma = float(sigma_init)
        self.rng = np.random.default_rng(seed)

        if pop_size % 2 == 1:
            pop_size += 1
        self.lam = pop_size
        self.mu = self.lam // 2

        # Recombination weights: positive only here (no active update).
        weights_raw = math.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights_raw / weights_raw.sum()
        self.mu_eff = 1.0 / (self.weights ** 2).sum()

        # Strategy parameters from Hansen 2016.
        self.c_sigma = (self.mu_eff + 2) / (self.d + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0.0, math.sqrt((self.mu_eff - 1) / (self.d + 1)) - 1) + self.c_sigma
        self.c_c = (4 + self.mu_eff / self.d) / (self.d + 4 + 2 * self.mu_eff / self.d)
        self.c_1 = 2 / ((self.d + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1,
                       2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.d + 2) ** 2 + self.mu_eff))

        # Evolution paths and covariance state.
        self.p_sigma = np.zeros(self.d)
        self.p_c = np.zeros(self.d)
        self.C = np.eye(self.d)
        self.B = np.eye(self.d)
        self.D = np.ones(self.d)

        self.gen = 0
        self.eigen_eval = 0
        self.eigen_update_interval = max(1, int(1 / ((self.c_1 + self.c_mu) * self.d * 10)))
        self.chi_n = math.sqrt(self.d) * (1 - 1 / (4 * self.d) + 1 / (21 * self.d ** 2))

        self._last_z = None
        self._last_y = None

    def ask(self):
        """Sample lambda candidates: x_i = mean + sigma * B * D * z_i."""
        z = self.rng.standard_normal((self.lam, self.d))
        BD = self.B * self.D
        y = z @ BD.T
        x = self.mean + self.sigma * y
        self._last_z = z
        self._last_y = y
        return x

    def tell(self, fitnesses):
        """Update mean, sigma, C using the fitnesses (LOWER is BETTER)."""
        self.gen += 1
        order = np.argsort(fitnesses)
        y_sorted = self._last_y[order]
        z_sorted = self._last_z[order]

        # Weighted recombination of the best mu samples drives the mean shift.
        y_w = (self.weights[:, None] * y_sorted[:self.mu]).sum(axis=0)
        z_w = (self.weights[:, None] * z_sorted[:self.mu]).sum(axis=0)
        self.mean = self.mean + self.sigma * y_w

        # CSA: update the step-size evolution path and then the step size.
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + \
                       math.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * (self.B @ z_w)
        ps_norm = float(np.linalg.norm(self.p_sigma))
        self.sigma = self.sigma * math.exp((self.c_sigma / self.d_sigma) * (ps_norm / self.chi_n - 1))

        # Heaviside switch to dampen the rank-1 path when sigma grows too fast.
        h_sigma = float(
            ps_norm / math.sqrt(1 - (1 - self.c_sigma) ** (2 * (self.gen + 1)))
            < (1.4 + 2 / (self.d + 1)) * self.chi_n
        )

        # Rank-1 evolution path and covariance updates.
        self.p_c = (1 - self.c_c) * self.p_c + \
                   h_sigma * math.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * y_w

        delta_h = (1 - h_sigma) * self.c_c * (2 - self.c_c)
        c_decay = (1 - self.c_1 - self.c_mu) + self.c_1 * delta_h
        rank_one = self.c_1 * np.outer(self.p_c, self.p_c)
        weighted_y = self.weights[:, None] * y_sorted[:self.mu]
        rank_mu = self.c_mu * (weighted_y.T @ y_sorted[:self.mu])
        self.C = c_decay * self.C + rank_one + rank_mu

        # Lazy eigendecomposition: only refresh B and D every K generations
        # because the O(d^3) cost dominates otherwise.
        if self.gen - self.eigen_eval > self.eigen_update_interval:
            self.C = (self.C + self.C.T) / 2
            eigvals, B = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-20)
            self.D = np.sqrt(eigvals)
            self.B = B
            self.eigen_eval = self.gen


# ============================================================================
# 4. ARCHIVE (the central data structure of MAP-Elites)
# ============================================================================

class Archive:
    """A grid in behavior-descriptor (BD) space, each cell holding one elite.

    The archive is the MAIN OUTPUT of MAP-Elites: it represents the
    portfolio of diverse, high-performing solutions discovered during
    training.  After training, you query it by indexing the cell that
    corresponds to the behavior you want, e.g. "the fastest elite at
    medium energy" -> cell (vx_high, |a|_mid).

    Storage is a sparse dictionary keyed by integer cell indices, which
    is much more memory-efficient than a dense grid because most cells
    are empty for a long time during training.
    """

    def __init__(self, bd_bounds, resolution, param_dim,
                 learning_rate=0.05, min_f=-1000.0):
        self.bd_bounds = np.array(bd_bounds, dtype=np.float64)  # (n_bd, 2)
        self.resolution = np.array(resolution, dtype=int)        # (n_bd,)
        self.n_bd = len(resolution)
        self.param_dim = param_dim
        # cell_index_tuple -> (params, fitness, bd_array)
        self.elites = {}
        # CMA-MAE soft archive: cell_index_tuple -> acceptance threshold.
        # Unseen cells implicitly sit at `min_f`. A candidate is "accepted"
        # (and contributes a positive ranking signal to its emitter) when its
        # fitness exceeds this threshold, even if it does not beat the stored
        # elite. The threshold then anneals toward the fitness at `learning_rate`.
        self.thresholds = {}
        self.learning_rate = float(learning_rate)
        self.min_f = float(min_f)

    def threshold_of(self, cell):
        """Current acceptance threshold for a cell (min_f if never visited)."""
        return self.thresholds.get(cell, self.min_f)

    def index(self, bd):
        """Map a behavior descriptor to a discrete cell index tuple."""
        bd = np.asarray(bd, dtype=np.float64)
        normalized = (bd - self.bd_bounds[:, 0]) / (self.bd_bounds[:, 1] - self.bd_bounds[:, 0])
        cell = np.floor(normalized * self.resolution).astype(int)
        # Clip BDs that fall outside the user-defined range to the boundary cells.
        cell = np.clip(cell, 0, self.resolution - 1)
        return tuple(cell.tolist())

    def add(self, params, fitness, bd):
        """CMA-MAE insertion against the soft per-cell threshold.

        A candidate is accepted when its fitness beats the cell's current
        threshold (which starts at `min_f`). On acceptance the threshold is
        annealed toward the fitness, and the stored elite is replaced only if
        the candidate also beats the genuine best elite seen in that cell.

        Returns (status, delta) where:
            delta  = fitness - threshold   (the emitter's ranking signal)
            status = 'new'      : accepted into a previously empty cell
                     'improved' : accepted and beat the stored elite
                     'accepted' : beat the threshold but not the stored elite
                     'rejected' : did not beat the threshold
        """
        cell = self.index(bd)
        threshold = self.thresholds.get(cell, self.min_f)
        delta = float(fitness) - threshold

        if fitness <= threshold:
            return "rejected", delta

        # Accepted: anneal the threshold toward this fitness.
        self.thresholds[cell] = (1.0 - self.learning_rate) * threshold + self.learning_rate * float(fitness)

        if cell not in self.elites:
            self.elites[cell] = (params.copy(), float(fitness), np.asarray(bd, dtype=np.float64))
            return "new", delta
        if fitness > self.elites[cell][1]:
            self.elites[cell] = (params.copy(), float(fitness), np.asarray(bd, dtype=np.float64))
            return "improved", delta
        return "accepted", delta

    def random_elite(self, rng):
        """Sample one elite uniformly at random from the filled cells."""
        if not self.elites:
            return None
        keys = list(self.elites.keys())
        idx = int(rng.integers(len(keys)))
        return self.elites[keys[idx]]

    @property
    def coverage(self):
        """Fraction of grid cells that contain at least one elite."""
        n_total = int(np.prod(self.resolution))
        return len(self.elites) / n_total

    @property
    def qd_score(self):
        """QD-score: sum of fitnesses across all filled cells.

        The canonical aggregate QD metric.  Improves when (a) new cells
        are filled or (b) existing cells get better elites.  A pure-quality
        method (CMA-ES) maxes out one cell only; a pure-diversity method
        fills many cells with low fitness; QD-score rewards filling many
        cells with HIGH-QUALITY elites.
        """
        if not self.elites:
            return 0.0
        return float(sum(e[1] for e in self.elites.values()))

    @property
    def best_fitness(self):
        if not self.elites:
            return -float("inf")
        return float(max(e[1] for e in self.elites.values()))

    def heatmap_grid(self):
        """Return a 2D ndarray of fitness values per cell (NaN for empty)."""
        if self.n_bd != 2:
            raise ValueError("heatmap_grid only supported for 2D archives")
        grid = np.full(tuple(self.resolution), np.nan, dtype=np.float64)
        for cell, (_, fit, _) in self.elites.items():
            grid[cell] = fit
        return grid


# ============================================================================
# 5. IMPROVEMENT EMITTER (the CMA-ME core idea)
# ============================================================================

class ImprovementEmitter:
    """A CMA-ES emitter driven by the CMA-MAE soft-threshold improvement signal.

    Vanilla MAP-Elites uses isotropic Gaussian mutation. CMA-ME drives each
    emitter with the improvement it produces in the archive; CMA-MAE makes
    that signal CONTINUOUS by ranking each candidate by

        delta = fitness - threshold(cell)

    where threshold(cell) is the soft acceptance threshold maintained by the
    archive (see Archive.add). Candidates that beat their cell's threshold get
    the best (lowest, since CMA-ES minimizes) ranks, ordered by how much they
    beat it; everything below threshold is tied at the bottom.

    Because the inner CMA-ES is rank-based, only this ordering matters. With a
    small archive learning rate the thresholds lag well behind the elites, so
    delta stays informative for many generations even in already-filled cells
    — this is what gives CMA-MAE a dense, CMA-ES-like quality gradient that
    plain CMA-ME lacks.

    Restart logic (the QD analogue of IPOP-CMA-ES):
        If a whole generation produces zero ACCEPTED candidates (none beat
        their threshold), OR the inner sigma collapses, the emitter restarts
        from a random elite drawn from the current archive. Because the
        threshold is soft, this triggers far less often than CMA-ME's
        beat-the-elite rule, so emitters keep optimizing instead of thrashing.
    """

    def __init__(self, archive, sigma_init, pop_size, seed=0):
        self.archive = archive
        self.sigma_init = float(sigma_init)
        self.pop_size = int(pop_size)
        self._seed = int(seed)
        self.cma = None
        self.restart_count = 0

    def initialize(self, init_params):
        """Create the underlying CMA-ES instance at the given starting point."""
        self.cma = CMAES(init_params, self.sigma_init, self.pop_size,
                         seed=self._seed + self.restart_count)

    def emit(self):
        """Sample a population of candidate parameter vectors."""
        return self.cma.ask()

    def update(self, candidates, fitnesses, bds, rng):
        """Insert candidates into the soft archive, then update the inner CMA-ES.

        Returns (n_new, n_improved, n_accepted, n_rejected) where n_accepted
        counts candidates that beat the threshold without beating the elite.
        """
        n = len(candidates)
        # Step 1. Insert each candidate and collect its (status, delta).
        # NOTE: later candidates in the same batch see an archive already
        # modified by earlier ones; this matches the reference CMA-ME/CMA-MAE
        # implementations and gives a slightly stronger gradient than
        # snapshotting the archive.
        deltas = np.empty(n, dtype=np.float64)
        n_new = n_improved = n_accepted = n_rejected = 0
        for i in range(n):
            status, delta = self.archive.add(candidates[i], fitnesses[i], bds[i])
            deltas[i] = delta
            if status == "new":
                n_new += 1
            elif status == "improved":
                n_improved += 1
            elif status == "accepted":
                n_accepted += 1
            else:
                n_rejected += 1

        # Step 2. Rank by improvement over threshold. CMA-ES minimizes, so we
        # feed -delta: the largest improvement gets the most-negative value and
        # therefore the best rank. CMA-ES only uses the argsort, so the
        # continuous magnitudes are a convenience, not a requirement.
        self.cma.tell(-deltas)

        # Step 3. Restart logic. Only when NOTHING beat its threshold this
        # generation (the region is saturated) or the step size collapsed.
        n_above_threshold = n_new + n_improved + n_accepted
        if n_above_threshold == 0 or self.cma.sigma < 1e-10:
            elite = self.archive.random_elite(rng)
            if elite is not None:
                self.restart_count += 1
                self.cma = CMAES(elite[0].copy(), self.sigma_init, self.pop_size,
                                 seed=self._seed + 1000 * self.restart_count)

        return n_new, n_improved, n_accepted, n_rejected


# ============================================================================
# 6. PARALLEL ROLLOUT WORKER (multiprocessing.Pool)
# ============================================================================
#
# Each worker owns its own gym env and policy network (created once by
# the pool initializer).  For every candidate the main process sends the
# flat parameter vector + observation normalization stats.  The worker
# overwrites its policy weights, runs one episode, and returns:
#     fitness, episode return, episode length,
#     partial Welford observation stats,
#     behavior descriptor (BD).
# The main process merges all partial obs stats into the global normalizer
# and feeds (fitness, BD) to the appropriate emitter.

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
    # One torch thread per worker process to avoid CPU oversubscription.
    torch.set_num_threads(1)


def _evaluate_one(args):
    """Evaluate one candidate: run an episode, return fitness + BD + obs stats.

    HalfCheetah-v5 observation layout (17 dims):
        obs[0]   : root z (height)
        obs[1]   : root y (pitch angle)
        obs[2:8] : 6 joint angles
        obs[8]   : root x velocity (forward speed) <- BD1 source
        obs[9]   : root z velocity
        obs[10]  : root y angular velocity
        obs[11:] : 6 joint angular velocities
    """
    flat, obs_mean, obs_std, l2_coef, max_steps = args
    set_flat_params(_W_NET, flat)

    obs, _ = _W_ENV.reset()
    ep_return = 0.0
    obs_buffer = []
    velocities = []     # for BD1: mean forward velocity
    actions_abs = []    # for BD2: mean control effort
    steps = 0
    while True:
        obs_norm = (obs - obs_mean) / (obs_std + 1e-8)
        with torch.no_grad():
            a = _W_NET(torch.from_numpy(obs_norm.astype(np.float32))).numpy()
        a = np.clip(a, -1.0, 1.0)
        next_obs, r, term, trunc, _ = _W_ENV.step(a)

        ep_return += float(r)
        obs_buffer.append(obs)
        velocities.append(float(obs[8]))           # forward velocity at THIS state
        actions_abs.append(float(np.abs(a).mean()))  # control effort for THIS step
        steps += 1
        obs = next_obs
        if term or trunc or steps >= max_steps:
            break

    obs_arr = np.asarray(obs_buffer, dtype=np.float64)
    p_mean = obs_arr.mean(axis=0)
    p_var = obs_arr.var(axis=0)
    p_count = len(obs_arr)
    p_M2 = p_var * p_count

    # Behavior descriptor for this candidate.
    bd = (
        float(np.mean(velocities)),
        float(np.mean(actions_abs)),
    )

    # MAP-Elites maximizes fitness directly (the rank conversion to "lower is
    # better" happens later inside the emitter).  We subtract a small L2
    # penalty on the parameters as a mild regularization that discourages
    # extreme weight magnitudes; this is independent of network size because
    # we use the MEAN of squared parameters, not the sum.
    l2_pen = l2_coef * float(np.mean(flat ** 2))
    fitness = ep_return - l2_pen
    return fitness, ep_return, steps, p_mean, p_M2, p_count, bd


def evaluate_candidates(pool, candidates, obs_norm, l2_coef, max_steps, n_episodes):
    """Evaluate candidates in parallel, AVERAGING fitness/return/BD over episodes.

    Single-episode evaluation is the main reason the old archive was unreliable:
    HalfCheetah's random initial state makes one rollout a noisy estimate, so a
    lucky episode could occupy a cell with a fitness and behavior descriptor it
    could not reproduce. Averaging over `n_episodes` rollouts per candidate
    de-noises both the fitness and the BD before the candidate is ever inserted,
    so cells are claimed on the basis of a policy's typical behavior.

    Returns:
        fitnesses : (N,) mean fitness per candidate
        returns   : (N,) mean raw return per candidate
        bds       : list of N averaged (vx, |a|) behavior descriptors
        total_steps : total environment steps consumed (all episodes)
        partials  : list of (mean, M2, count) Welford stats to merge globally
    """
    obs_mean = obs_norm.mean.astype(np.float32)
    obs_std = obs_norm.std.astype(np.float32)
    n = len(candidates)
    tasks = []
    for c in candidates:
        for _ in range(n_episodes):
            tasks.append((c, obs_mean, obs_std, l2_coef, max_steps))
    raw = pool.map(_evaluate_one, tasks)

    fitnesses = np.zeros(n, dtype=np.float64)
    returns = np.zeros(n, dtype=np.float64)
    bds = []
    total_steps = 0
    partials = []
    for i in range(n):
        block = raw[i * n_episodes:(i + 1) * n_episodes]
        fitnesses[i] = float(np.mean([b[0] for b in block]))
        returns[i] = float(np.mean([b[1] for b in block]))
        bds.append((
            float(np.mean([b[6][0] for b in block])),
            float(np.mean([b[6][1] for b in block])),
        ))
        for b in block:
            total_steps += b[2]
            partials.append((b[3], b[4], b[5]))
    return fitnesses, returns, bds, total_steps, partials


# ============================================================================
# 7. CONFIG
# ============================================================================

@dataclass
class Args:
    exp_name: str = "CMA-MAE_HalfCheetah_v5"
    env_id: str = "HalfCheetah-v5"
    # CMA-MAE illuminates a whole archive, so it needs more rollouts than a
    # single-objective optimizer. 50M is an honest budget for comparison with
    # the 20M CMA-ES / NES runs (still far cheaper than SAC's 3M *gradient*
    # steps, which see vastly more data per step). Raise to 100M for more
    # coverage/quality if you have the wall-clock budget.
    total_timesteps: int = 50_000_000

    # --- Policy network ---
    hidden_sizes: tuple = (32, 32)
    init_param_std: float = 0.1

    # --- Behavior descriptors and archive ---
    # BD1 = mean forward velocity (m/s), BD2 = mean |action|.
    # Bounds are sized to the regime ACTUALLY reachable by a small 32x32 policy
    # under an evolutionary budget. A good policy here averages ~3-4 m/s, so the
    # old upper bound of 8 m/s left most of the grid permanently empty and
    # diluted optimization pressure. A coarser grid over a tighter, reachable
    # range concentrates emitters on cells they can actually fill well.
    # Out-of-range BDs are clipped to the boundary cells (see Archive.index).
    bd_bounds: list = field(default_factory=lambda: [(-1.0, 5.0), (0.0, 1.0)])
    archive_resolution: tuple = (25, 25)   # 625 cells total

    # --- CMA-MAE soft archive ---
    # archive_learning_rate (alpha) interpolates between a pure CMA-ES optimizer
    # (alpha->0) and plain CMA-ME (alpha->1). A small value keeps a dense,
    # quality-seeking gradient while still illuminating behavior space.
    archive_learning_rate: float = 0.05
    # Threshold floor: the objective below which a cell is considered "empty".
    # Sized just under a do-nothing policy's return so weak-but-novel behaviors
    # can still seed cells early in training.
    archive_min_fitness: float = -1000.0

    # --- Emitters ---
    n_emitters: int = 5
    emitter_pop_size: int = 36             # lambda per emitter; bumped to even
    sigma_init: float = 0.5

    # --- Fitness ---
    max_episode_steps: int = 1000
    l2_coef: float = 0.001                 # weight decay coefficient on mean(theta^2)
    # Episodes averaged per candidate DURING search to de-noise fitness and BD
    # before insertion (the single biggest reliability fix over the old run).
    n_episodes_per_candidate: int = 2

    # --- Honest final archive ---
    # Even with multi-episode search, do a heavier re-evaluation of every elite
    # before the final save and overwrite its stored fitness/BD with the mean.
    final_eval_episodes: int = 10

    # --- Parallelism ---
    n_workers: int = 12

    # --- Logging / checkpoints ---
    save_every_gens: int = 50
    log_every_gens: int = 1
    print_every_gens: int = 25             # console QD-metric line (works without W&B)
    heatmap_every_gens: int = 10           # render archive heatmap to wandb
    seed: int = 1


# ============================================================================
# 8. TRAINING LOOP
# ============================================================================

def render_archive_heatmap(archive, step, generation, save_path=None):
    """Render the current archive as a 2D heatmap; optionally save PNG to disk."""
    if archive.n_bd != 2:
        return None
    grid = archive.heatmap_grid()
    fig, ax = plt.subplots(figsize=(7, 5))
    # imshow draws rows top-to-bottom; we want BD1 on x and BD2 on y, so
    # transpose and use origin='lower' to match (BD1, BD2) -> (x, y).
    im = ax.imshow(
        grid.T,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[archive.bd_bounds[0, 0], archive.bd_bounds[0, 1],
                archive.bd_bounds[1, 0], archive.bd_bounds[1, 1]],
    )
    ax.set_xlabel("BD1: mean forward velocity (m/s)")
    ax.set_ylabel("BD2: mean |action| (control effort)")
    ax.set_title(f"Archive @ gen {generation} | "
                 f"coverage={archive.coverage:.1%} | best={archive.best_fitness:.0f}")
    plt.colorbar(im, ax=ax, label="Fitness (episode return)")
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def reevaluate_archive(pool, archive, obs_norm, l2_coef, max_steps, n_eval):
    """Re-evaluate every stored elite over multiple episodes and de-noise it.

    Single-episode evaluation lets a lucky rollout permanently occupy a cell with
    an inflated fitness/BD ("archive corruption"). Here each elite is re-run
    n_eval times and its stored fitness and BD are overwritten with the averaged
    values. Elites keep their ORIGINAL cell (no re-binning) so that re-evaluation
    cannot create collisions; only the stored values are corrected.
    """
    if not archive.elites:
        return
    obs_mean = obs_norm.mean.astype(np.float32)
    obs_std = obs_norm.std.astype(np.float32)
    cells = list(archive.elites.keys())
    tasks = []
    for cell in cells:
        params = archive.elites[cell][0]
        for _ in range(n_eval):
            tasks.append((params, obs_mean, obs_std, l2_coef, max_steps))
    raw = pool.map(_evaluate_one, tasks)
    for ci, cell in enumerate(cells):
        block = raw[ci * n_eval:(ci + 1) * n_eval]
        fit = float(np.mean([b[0] for b in block]))
        bd = (
            float(np.mean([b[6][0] for b in block])),
            float(np.mean([b[6][1] for b in block])),
        )
        params = archive.elites[cell][0]
        archive.elites[cell] = (params, fit, np.asarray(bd, dtype=np.float64))


def save_archive(path, archive, obs_norm, args, total_steps, generation):
    path = model_path(HALFCHEETAH_EVOLUTION_MODELS, path)
    ensure_dir(path.parent)
    np.savez(
        path,
        elites=np.array([archive.elites], dtype=object),
        thresholds=np.array([archive.thresholds], dtype=object),
        archive_learning_rate=archive.learning_rate,
        archive_min_fitness=archive.min_f,
        bd_bounds=archive.bd_bounds,
        resolution=archive.resolution,
        obs_mean=obs_norm.mean,
        obs_std=obs_norm.std,
        hidden_sizes=np.array(args.hidden_sizes),
        env_id=args.env_id,
        total_steps=total_steps,
        generation=generation,
    )


def train(resume_from=None):
    args = Args()
    args = apply_map_elites_smoke(args)
    if smoke_mode():
        print(
            f"Smoke mode: {args.total_timesteps} timesteps, "
            f"{args.n_emitters} emitters x pop {args.emitter_pop_size}, "
            f"workers={args.n_workers}"
        )
    init_wandb(args.exp_name, args, tags=["map-elites", "cma-me", "evolution", "halfcheetah"])

    # Build a template network only to determine d (parameter dimension)
    # and the initial parameter vector for emitters.
    template_env = gym.make(args.env_id)
    obs_dim = template_env.observation_space.shape[0]
    act_dim = template_env.action_space.shape[0]
    template_env.close()

    template_net = PolicyNet(obs_dim, act_dim, hidden=tuple(args.hidden_sizes))
    init_mean = init_random_params(template_net, std=args.init_param_std, seed=args.seed)
    d = len(init_mean)

    obs_norm = RunningStats(obs_dim)

    pool = mp.Pool(
        processes=args.n_workers,
        initializer=_init_worker,
        initargs=(args.env_id, args.hidden_sizes, args.seed),
    )

    archive = Archive(
        args.bd_bounds, args.archive_resolution, d,
        learning_rate=args.archive_learning_rate,
        min_f=args.archive_min_fitness,
    )

    # Initialize emitters.  Each one starts at init_mean + a small random
    # perturbation, so they don't all sample identical batches in their
    # first generation.  As the archive fills up, restarted emitters seed
    # themselves from random elites instead.
    main_rng = np.random.default_rng(args.seed)
    emitters = []
    for i in range(args.n_emitters):
        em = ImprovementEmitter(
            archive, sigma_init=args.sigma_init,
            pop_size=args.emitter_pop_size,
            seed=args.seed + 100 * i,
        )
        perturb = main_rng.standard_normal(d) * args.init_param_std
        em.initialize(init_mean + perturb)
        emitters.append(em)

    total_steps = 0
    generation = 0

    if resume_from is not None:
        ckpt = np.load(resume_from, allow_pickle=True)
        archive.elites = dict(ckpt["elites"].item())
        if "thresholds" in ckpt.files:
            archive.thresholds = dict(ckpt["thresholds"].item())
        obs_norm.mean = ckpt["obs_mean"]
        obs_norm.M2 = (ckpt["obs_std"] ** 2) * max(1, len(archive.elites)) * args.max_episode_steps
        obs_norm.count = max(1, len(archive.elites)) * args.max_episode_steps
        total_steps = int(ckpt["total_steps"])
        generation = int(ckpt["generation"])
        # Re-seed every emitter from a random elite on resume.
        for em in emitters:
            elite = archive.random_elite(main_rng)
            if elite is not None:
                em.initialize(elite[0].copy())
        print(f"Resumed from {resume_from} at step {total_steps}, gen {generation}")

    print(f"Training HalfCheetah CMA-MAE")
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}, d (theta)={d}")
    print(f"  archive: {tuple(args.archive_resolution)} cells over BDs "
          f"{[tuple(b) for b in args.bd_bounds]}")
    print(f"  soft archive: learning_rate(alpha)={args.archive_learning_rate}, "
          f"min_f={args.archive_min_fitness}")
    print(f"  {args.n_emitters} emitters x lambda={emitters[0].cma.lam} = "
          f"{args.n_emitters * emitters[0].cma.lam} candidates per generation, "
          f"{args.n_episodes_per_candidate} episode(s) each")
    print(f"  workers={args.n_workers}, total_timesteps={args.total_timesteps:,}")

    pbar = tqdm(total=args.total_timesteps, initial=total_steps)

    while total_steps < args.total_timesteps:
        gen_start = time.time()
        generation += 1

        # --- 1. EMIT: each emitter samples its own population ---
        emitter_candidates = [em.emit() for em in emitters]
        all_candidates_flat = np.vstack(emitter_candidates)

        # --- 2. EVALUATE all candidates in parallel (averaged over episodes) ---
        all_fitnesses, all_returns, all_bds, gen_steps, partials = evaluate_candidates(
            pool, all_candidates_flat, obs_norm,
            args.l2_coef, args.max_episode_steps, args.n_episodes_per_candidate,
        )

        # --- 3. Merge obs stats into the global normalizer ---
        for p_mean, p_M2, p_count in partials:
            obs_norm.merge(p_mean, p_M2, p_count)

        # --- 4. UPDATE each emitter (insert into archive + adapt CMA-ES) ---
        cursor = 0
        n_new_total = n_improved_total = n_accepted_total = n_rejected_total = 0
        for ei, em in enumerate(emitters):
            sub_n = len(emitter_candidates[ei])
            sub_cands = emitter_candidates[ei]
            sub_fits = all_fitnesses[cursor:cursor + sub_n]
            sub_bds = all_bds[cursor:cursor + sub_n]
            n_new, n_imp, n_acc, n_rej = em.update(sub_cands, sub_fits, sub_bds, main_rng)
            n_new_total += n_new
            n_improved_total += n_imp
            n_accepted_total += n_acc
            n_rejected_total += n_rej
            cursor += sub_n

        total_steps += gen_steps
        pbar.update(gen_steps)

        # --- 5. Logging ---
        if generation % args.log_every_gens == 0:
            log_dict = {
                # Quality-Diversity metrics:
                "qd/coverage": archive.coverage,
                "qd/qd_score": archive.qd_score,
                "qd/best_fitness": archive.best_fitness,
                "qd/n_filled_cells": len(archive.elites),
                # Per-generation archive deltas:
                "gen/n_new": n_new_total,
                "gen/n_improved": n_improved_total,
                "gen/n_accepted": n_accepted_total,
                "gen/n_rejected": n_rejected_total,
                # Raw return summary for this generation:
                METRIC_BEST_RETURN: float(all_returns.max()),
                METRIC_MEAN_RETURN: float(all_returns.mean()),
                METRIC_WORST_RETURN: float(all_returns.min()),
                METRIC_TOTAL_ENV_STEPS: total_steps,
                METRIC_GENERATION: generation,
                METRIC_GEN_SECONDS: time.time() - gen_start,
                METRIC_OBS_NORM_COUNT: obs_norm.count,
                # Emitter health:
                "emitters/total_restarts": int(sum(em.restart_count for em in emitters)),
            }
            for i, em in enumerate(emitters):
                log_dict[f"emitters/sigma_{i}"] = float(em.cma.sigma)
            wandb.log(log_dict, step=total_steps)

        # Console progress (visible without W&B, e.g. WANDB_MODE=disabled).
        if generation % args.print_every_gens == 0:
            tqdm.write(
                f"gen {generation:5d} | steps {total_steps:>10,} | "
                f"coverage {archive.coverage:5.1%} ({len(archive.elites):4d}) | "
                f"best_fit {archive.best_fitness:8.1f} | "
                f"qd {archive.qd_score:11.1f} | "
                f"gen_ret max {all_returns.max():7.1f}"
            )

        # --- 6. Periodic heatmap of the archive ---
        if generation % args.heatmap_every_gens == 0:
            heatmap_path = HALFCHEETAH_EVOLUTION / f"map_elites_heatmap_gen{generation:04d}.png"
            img = render_archive_heatmap(archive, total_steps, generation, save_path=heatmap_path)
            if img is not None:
                wandb.log({"qd/archive_heatmap": img}, step=total_steps)

        # --- 7. Periodic checkpoint ---
        if generation % args.save_every_gens == 0:
            save_archive(f"cma_me_archive_gen{generation}.npz",
                         archive, obs_norm, args, total_steps, generation)

    pbar.close()

    # --- De-noise the archive before the final save -----------------------------
    # Re-evaluate every elite over several episodes and overwrite its stored
    # fitness/BD with the averaged values, removing the lucky-episode bias that
    # single-episode insertion introduces.
    n_eval = 2 if smoke_mode() else args.final_eval_episodes
    print(f"Re-evaluating {len(archive.elites)} elites over {n_eval} episodes each "
          f"(de-noising archive)...")
    reevaluate_archive(pool, archive, obs_norm, args.l2_coef, args.max_episode_steps, n_eval)

    pool.close()
    pool.join()

    save_archive(
        MAP_ELITES_FINAL.name,
        archive, obs_norm, args, total_steps, generation,
    )
    final_heatmap_path = HALFCHEETAH_EVOLUTION / "map_elites_heatmap_final.png"
    img = render_archive_heatmap(archive, total_steps, generation, save_path=final_heatmap_path)
    if img is not None:
        wandb.log({"qd/archive_heatmap_final": img}, step=total_steps)

    print(f"\nTraining finished.")
    print(f"  Coverage  : {archive.coverage:.1%} ({len(archive.elites)} / "
          f"{int(np.prod(archive.resolution))} cells)")
    print(f"  QD-score  : {archive.qd_score:.1f}")
    print(f"  Best fit. : {archive.best_fitness:.1f}")
    print(f"Saved final archive to {MAP_ELITES_FINAL}")
    print(f"Saved final heatmap to {final_heatmap_path}")
    finish_wandb()


# ============================================================================
# 9. EVALUATION
# ============================================================================

def evaluate(archive_path=MAP_ELITES_FINAL, cell=None,
             best=False, n_episodes=5, render=False):
    """Load the archive and evaluate one elite policy.

    Args:
        cell      : tuple (i, j) selecting a specific archive cell
        best      : if True, ignore `cell` and pick the cell with the
                    highest fitness in the entire archive
        n_episodes: number of rollouts to average over
        render    : if True, open the MuJoCo viewer
    """
    data = np.load(archive_path, allow_pickle=True)
    elites = dict(data["elites"].item())
    obs_mean = data["obs_mean"]
    obs_std = data["obs_std"]
    bd_bounds = data["bd_bounds"]
    resolution = data["resolution"]
    hidden = tuple(int(x) for x in data["hidden_sizes"])
    env_id = str(data["env_id"])

    if not elites:
        print("Archive is empty.")
        return

    if best:
        cell = max(elites.keys(), key=lambda k: elites[k][1])
        print(f"Using best cell: {cell}")
    if cell is None:
        cell = max(elites.keys(), key=lambda k: elites[k][1])
        print(f"No cell specified — defaulting to best cell {cell}")
    if cell not in elites:
        print(f"Cell {cell} not in archive. Total filled cells: {len(elites)}")
        # Suggest the closest filled cell
        nearest = min(elites.keys(),
                      key=lambda k: (k[0] - cell[0]) ** 2 + (k[1] - cell[1]) ** 2)
        print(f"Nearest filled cell: {nearest}")
        return

    params, fitness, bd = elites[cell]
    bd_size = (bd_bounds[:, 1] - bd_bounds[:, 0]) / resolution
    cell_center = bd_bounds[:, 0] + (np.array(cell) + 0.5) * bd_size
    print(f"Cell {cell}: stored fitness={fitness:.1f}, BD={bd}, "
          f"cell center=({cell_center[0]:.2f}, {cell_center[1]:.3f})")

    env = gym.make(env_id, render_mode="human" if render else None)
    net = PolicyNet(env.observation_space.shape[0],
                    env.action_space.shape[0], hidden=hidden)
    set_flat_params(net, params)
    net.eval()

    returns = []
    measured_bds = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        steps = 0
        velocities = []
        actions_abs = []
        while True:
            obs_n = (obs - obs_mean) / (obs_std + 1e-8)
            with torch.no_grad():
                a = net(torch.from_numpy(obs_n.astype(np.float32))).numpy()
            a = np.clip(a, -1.0, 1.0)
            obs, r, term, trunc, _ = env.step(a)
            ep_return += float(r)
            velocities.append(float(obs[8]))
            actions_abs.append(float(np.abs(a).mean()))
            steps += 1
            if term or trunc:
                break
        returns.append(ep_return)
        measured_bds.append((float(np.mean(velocities)), float(np.mean(actions_abs))))
        print(f"Episode {ep + 1:2d} | steps={steps:4d} | return={ep_return:8.1f} | "
              f"BD={measured_bds[-1]}")

    env.close()
    print(f"\nMean return over {n_episodes} eps: {np.mean(returns):.1f} +/- {np.std(returns):.1f}")
    bd_arr = np.array(measured_bds)
    print(f"Mean measured BD: ({bd_arr[:, 0].mean():.2f}, {bd_arr[:, 1].mean():.3f})")


def list_top_elites(archive_path=MAP_ELITES_FINAL, n=10):
    """Print the top-n elites in the archive sorted by fitness."""
    data = np.load(archive_path, allow_pickle=True)
    elites = dict(data["elites"].item())
    sorted_cells = sorted(elites.keys(), key=lambda k: -elites[k][1])
    print(f"Top {n} elites in archive:")
    print(f"{'cell':>12} {'fitness':>10} {'BD (vel, |a|)':>20}")
    for cell in sorted_cells[:n]:
        _, fit, bd = elites[cell]
        print(f"{str(cell):>12} {fit:>10.1f} ({bd[0]:>5.2f}, {bd[1]:>5.3f})")


# ============================================================================
# 10. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Multiprocessing uses 'fork' by default on Linux, which works fine here.
    # On macOS / Windows you would need mp.set_start_method('spawn') and the
    # worker globals would have to be re-initialized differently.
    train(resume_from=None)
