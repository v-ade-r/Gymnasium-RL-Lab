"""
Unified evaluator for the three evolutionary HalfCheetah agents:
    - CMA-ES   (CMA-ES.py     -> cmaes_final_model.npz)
    - NES      (NES.py        -> snes_final_model.npz / xnes_final_model.npz)
    - MAP-Elites/CMA-ME (MAP-Elites.py -> cma_me_final_archive.npz)

All three trainers save the policy weights as a single flat float64 numpy
vector together with the matching observation-normalization statistics
(mean, std).  The network architecture (hidden sizes) is also stored.
The evaluator rebuilds the policy network, loads the weights, applies
the same observation normalization that was active during training, and
plays deterministic rollouts.

CMA-ES and NES return ONE policy (the best one found).  MAP-Elites returns
an ARCHIVE — a 2D map of policies indexed by behavior descriptors
(mean forward velocity, mean control effort).

CLI examples:
    python test_all.py cmaes
    python test_all.py nes --variant snes
    python test_all.py map-elites list-top --n 20
    python test_all.py map-elites evaluate --best
    python test_all.py map-elites evaluate --cell 45 5
    python test_all.py map-elites heatmap --save-path archive_heatmap.png
    python test_all.py map-elites heatmap --show
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repo_paths import (
    CMAES_FINAL,
    HALFCHEETAH_EVOLUTION_RESULTS,
    MAP_ELITES_FINAL,
    SNES_FINAL,
    XNES_FINAL,
    ensure_dir,
)
from utils.recording import CONTINUOUS_GIF_KWARGS, record_episode_gif


# ============================================================================
# 1. POLICY NETWORK (must match the architecture used during training)
# ============================================================================

class PolicyNet(nn.Module):
    """Small deterministic MLP: obs -> action in [-1, 1] via final tanh.

    Identical to the network used in all three training scripts so that
    flat parameter vectors can be loaded directly.
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


def set_flat_params(net, flat):
    """Write a 1D float numpy parameter vector into `net` in declaration order.

    The flat vector is laid out exactly as torch's Module.parameters() yields
    the tensors, which is the same order used at save time.
    """
    flat_t = torch.from_numpy(np.asarray(flat).astype(np.float32))
    idx = 0
    for p in net.parameters():
        n = p.numel()
        p.data.copy_(flat_t[idx:idx + n].view_as(p))
        idx += n


# ============================================================================
# 2. CORE ROLLOUT — used by every evaluator
# ============================================================================

def _rollout_one_episode(env, net, obs_mean, obs_std, max_steps=1000, reset_seed=None):
    """Play one episode with a deterministic policy and return diagnostics.

    Returns a dict with episode return, episode length, mean forward
    velocity (BD1) and mean control effort (BD2).  These are the same
    quantities MAP-Elites uses as behavior descriptors, so they make a
    convenient way to verify that an elite reproduces its claimed BDs.
    """
    if reset_seed is not None:
        obs, _ = env.reset(seed=reset_seed)
    else:
        obs, _ = env.reset()
    ep_return = 0.0
    velocities = []
    actions_abs = []
    steps = 0
    while True:
        obs_n = (obs - obs_mean) / (obs_std + 1e-8)
        with torch.no_grad():
            a = net(torch.from_numpy(obs_n.astype(np.float32))).numpy()
        a = np.clip(a, -1.0, 1.0)
        next_obs, r, term, trunc, _ = env.step(a)
        ep_return += float(r)
        # HalfCheetah-v5 obs[8] = root x velocity (forward speed)
        velocities.append(float(obs[8]))
        actions_abs.append(float(np.abs(a).mean()))
        steps += 1
        obs = next_obs
        if term or trunc or steps >= max_steps:
            break
    return {
        "return": ep_return,
        "steps": steps,
        "mean_vx": float(np.mean(velocities)),
        "mean_abs_a": float(np.mean(actions_abs)),
    }


def _build_env_and_net(model_npz, render=False, record_video=False):
    """Reconstruct the env and the policy net from a saved npz checkpoint.

    render=True opens the MuJoCo viewer. record_video=True uses rgb_array
    for GIF export via utils.recording. Both False = headless eval.
    """
    env_id = str(model_npz["env_id"])
    hidden = tuple(int(x) for x in model_npz["hidden_sizes"])
    obs_mean = model_npz["obs_mean"]
    obs_std = model_npz["obs_std"]

    if render and record_video:
        raise ValueError("Use either render=True or record_video=True, not both.")

    if render:
        render_mode = "human"
    elif record_video:
        render_mode = "rgb_array"
    else:
        render_mode = None

    env = gym.make(env_id, render_mode=render_mode)
    net = PolicyNet(env.observation_space.shape[0],
                    env.action_space.shape[0], hidden=hidden)
    net.eval()
    return env, net, obs_mean, obs_std, env_id


def _run_eval_episode(
    env, net, obs_mean, obs_std, max_steps, ep, record_video, gif_path, seed,
):
    reset_seed = (seed + ep) if seed is not None else None
    if record_video and ep == 0:
        def rollout(capture_env):
            return _rollout_one_episode(
                capture_env, net, obs_mean, obs_std,
                max_steps=max_steps, reset_seed=reset_seed,
            )

        result = record_episode_gif(env, rollout, gif_path, **CONTINUOUS_GIF_KWARGS)
        print(f"Saved demo GIF to {gif_path}")
        return result
    return _rollout_one_episode(
        env, net, obs_mean, obs_std, max_steps=max_steps, reset_seed=reset_seed,
    )


def _print_summary(label, returns, lengths, vxs, accs):
    """Pretty-print rollout statistics."""
    print(f"\n{label}")
    print(f"  episodes      : {len(returns)}")
    print(f"  return        : {np.mean(returns):8.1f}  +/- {np.std(returns):6.1f}   "
          f"(min {np.min(returns):.1f}, max {np.max(returns):.1f})")
    print(f"  episode length: {np.mean(lengths):8.1f}  +/- {np.std(lengths):6.1f}")
    print(f"  mean fwd vel  : {np.mean(vxs):8.3f}  +/- {np.std(vxs):6.3f}")
    print(f"  mean |action| : {np.mean(accs):8.3f}  +/- {np.std(accs):6.3f}")


# ============================================================================
# 3. CMA-ES EVALUATOR
# ============================================================================

def evaluate_cmaes(model_path=CMAES_FINAL,
                   n_episodes=10, max_steps=1000, render=False,
                   record_video=False, seed=None):
    """Load the best CMA-ES policy and play n_episodes rollouts.

    Args:
        model_path : path to the .npz produced by gym_MuJoCO_HalfCheetah_CMA-ES.py
        n_episodes : how many rollouts to average over
        max_steps  : episode horizon (HalfCheetah default is 1000)
        render     : if True, opens the MuJoCo viewer (requires display)
        record_video : if True, saves a demo GIF to results/continuous_control/halfcheetah/evolution/
        seed       : optional master seed; episodes are seeded as seed+i
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CMA-ES checkpoint not found: {model_path}")

    data = np.load(model_path, allow_pickle=True)
    params = data["params"]
    best_return_train = float(data["best_return"]) if "best_return" in data.files else None

    env, net, obs_mean, obs_std, env_id = _build_env_and_net(
        data, render=render, record_video=record_video,
    )
    set_flat_params(net, params)
    gif_path = ensure_dir(HALFCHEETAH_EVOLUTION_RESULTS) / "cmaes.gif"

    print(f"\nLoaded CMA-ES policy from '{model_path}'")
    print(f"  env_id            : {env_id}")
    print(f"  parameter dim     : {len(params):,}")
    if best_return_train is not None:
        print(f"  reported train best return: {best_return_train:.1f}")
    print(f"  running {n_episodes} eval episodes (max_steps={max_steps})...")

    returns, lengths, vxs, accs = [], [], [], []
    for ep in range(n_episodes):
        r = _run_eval_episode(
            env, net, obs_mean, obs_std, max_steps, ep,
            record_video, gif_path, seed,
        )
        returns.append(r["return"]); lengths.append(r["steps"])
        vxs.append(r["mean_vx"]); accs.append(r["mean_abs_a"])
        print(f"  ep {ep + 1:2d} | steps={r['steps']:4d} | return={r['return']:8.1f} | "
              f"vx={r['mean_vx']:5.2f} | |a|={r['mean_abs_a']:.3f}")

    env.close()
    _print_summary("CMA-ES evaluation summary:", returns, lengths, vxs, accs)
    return {"returns": returns, "lengths": lengths, "mean_vx": vxs, "mean_abs_a": accs}


# ============================================================================
# 4. xNES EVALUATOR
# ============================================================================

def evaluate_nes(model_path=SNES_FINAL,
                 n_episodes=10, max_steps=1000, render=False,
                 record_video=False, seed=None):
    """Load the best NES policy and play n_episodes rollouts.

    Defaults to the sNES model (the default trainer variant). To evaluate an
    xNES run instead, pass model_path=XNES_FINAL.

    Identical in spirit to evaluate_cmaes: both NES variants save a flat
    parameter vector + observation normalizer + network architecture, exactly
    the fields the rollout needs (the optimizer variant is irrelevant at
    evaluation time).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"NES checkpoint not found: {model_path}")

    data = np.load(model_path, allow_pickle=True)
    params = data["params"]
    best_return_train = float(data["best_return"]) if "best_return" in data.files else None

    env, net, obs_mean, obs_std, env_id = _build_env_and_net(
        data, render=render, record_video=record_video,
    )
    set_flat_params(net, params)
    variant = "snes" if "snes" in os.path.basename(str(model_path)).lower() else "xnes"
    gif_path = ensure_dir(HALFCHEETAH_EVOLUTION_RESULTS) / f"{variant}.gif"

    print(f"\nLoaded {variant} policy from '{model_path}'")
    print(f"  env_id            : {env_id}")
    print(f"  parameter dim     : {len(params):,}")
    if best_return_train is not None:
        print(f"  reported train best return: {best_return_train:.1f}")
    print(f"  running {n_episodes} eval episodes (max_steps={max_steps})...")

    returns, lengths, vxs, accs = [], [], [], []
    for ep in range(n_episodes):
        r = _run_eval_episode(
            env, net, obs_mean, obs_std, max_steps, ep,
            record_video, gif_path, seed,
        )
        returns.append(r["return"]); lengths.append(r["steps"])
        vxs.append(r["mean_vx"]); accs.append(r["mean_abs_a"])
        print(f"  ep {ep + 1:2d} | steps={r['steps']:4d} | return={r['return']:8.1f} | "
              f"vx={r['mean_vx']:5.2f} | |a|={r['mean_abs_a']:.3f}")

    env.close()
    _print_summary(f"{variant} evaluation summary:", returns, lengths, vxs, accs)
    return {"returns": returns, "lengths": lengths, "mean_vx": vxs, "mean_abs_a": accs}


# ============================================================================
# 5. MAP-ELITES (CMA-ME) EVALUATOR
# ============================================================================
#
# MAP-Elites doesn't save a single best policy — it saves an ARCHIVE: a
# dict mapping cell index (i, j) -> (params, fitness, bd).  The cells live
# in a 2D behavior space (BD1 = mean forward velocity, BD2 = mean |action|).
#
# Three convenience entry points:
#   list_top_elites_map_elites(...) : print the top-N elites by fitness
#   show_archive_heatmap(...)       : visualize the full archive
#   evaluate_map_elites(...)        : roll out one specific cell (or best)


def _load_archive(archive_path):
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"MAP-Elites archive not found: {archive_path}")
    data = np.load(archive_path, allow_pickle=True)
    elites = dict(data["elites"].item())
    return data, elites


def list_top_elites_map_elites(archive_path=MAP_ELITES_FINAL, n=15):
    """Print the top-n cells in the archive ranked by stored fitness."""
    data, elites = _load_archive(archive_path)
    bd_bounds = data["bd_bounds"]
    resolution = data["resolution"]
    bd_size = (bd_bounds[:, 1] - bd_bounds[:, 0]) / resolution

    if not elites:
        print("Archive is empty.")
        return []

    sorted_cells = sorted(elites.keys(), key=lambda k: -elites[k][1])
    print(f"\nTop {n} elites in '{archive_path}'  (out of {len(elites)} filled cells):")
    print(f"  {'cell (i,j)':>12}  {'fitness':>10}  {'BD vx':>7}  {'BD |a|':>7}  "
          f"{'cell-center vx':>14}  {'cell-center |a|':>15}")
    rows = []
    for cell in sorted_cells[:n]:
        _, fit, bd = elites[cell]
        center = bd_bounds[:, 0] + (np.array(cell) + 0.5) * bd_size
        print(f"  {str(cell):>12}  {fit:>10.1f}  {bd[0]:>7.2f}  {bd[1]:>7.3f}  "
              f"{center[0]:>14.2f}  {center[1]:>15.3f}")
        rows.append({"cell": cell, "fitness": float(fit), "bd": tuple(map(float, bd))})
    return rows


def show_archive_heatmap(archive_path=MAP_ELITES_FINAL,
                         save_path=None, show=True):
    """Render the archive as a 2D heatmap (BD1 x BD2 colored by fitness).

    If save_path is given, also dumps a PNG.  If show is False (e.g. on a
    headless machine), only saves to disk.  This matches the heatmap the
    trainer logs to wandb every few generations, but you can call it
    on any saved archive.
    """
    data, elites = _load_archive(archive_path)
    bd_bounds = data["bd_bounds"]
    resolution = data["resolution"]

    grid = np.full(tuple(resolution), np.nan, dtype=np.float64)
    for cell, (_, fit, _) in elites.items():
        grid[cell] = fit

    if not show:
        matplotlib.use("Agg")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        grid.T,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[bd_bounds[0, 0], bd_bounds[0, 1],
                bd_bounds[1, 0], bd_bounds[1, 1]],
    )
    ax.set_xlabel("BD1: mean forward velocity (m/s)")
    ax.set_ylabel("BD2: mean |action| (control effort)")
    ax.set_title(f"MAP-Elites archive  |  {len(elites)} / "
                 f"{int(np.prod(resolution))} cells filled  "
                 f"({len(elites) / np.prod(resolution):.1%} coverage)")
    plt.colorbar(im, ax=ax, label="Fitness (episode return)")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        print(f"Saved heatmap to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def evaluate_map_elites(archive_path=MAP_ELITES_FINAL,
                        cell=None, best=False,
                        n_episodes=5, max_steps=1000, render=False,
                        record_video=False, seed=None):
    """Evaluate one elite policy from the MAP-Elites archive.

    Args:
        cell      : tuple (i, j) selecting a specific archive cell; ignored
                    if `best=True`
        best      : if True, pick the cell with the highest stored fitness
        n_episodes: rollouts to average over
        max_steps : per-episode step horizon
        render    : if True opens the MuJoCo viewer
        record_video : if True, saves a demo GIF to results/continuous_control/halfcheetah/evolution/

    If `cell` is not in the archive, the nearest filled cell (in cell-index
    Euclidean distance) is suggested and evaluation aborts.
    """
    data, elites = _load_archive(archive_path)
    if not elites:
        print("Archive is empty.")
        return None

    bd_bounds = data["bd_bounds"]
    resolution = data["resolution"]
    bd_size = (bd_bounds[:, 1] - bd_bounds[:, 0]) / resolution

    if best or cell is None:
        cell = max(elites.keys(), key=lambda k: elites[k][1])
        print(f"Using best cell in archive: {cell}")
    else:
        cell = tuple(cell)
        if cell not in elites:
            nearest = min(elites.keys(),
                          key=lambda k: (k[0] - cell[0]) ** 2 + (k[1] - cell[1]) ** 2)
            print(f"Cell {cell} is empty.  Nearest filled cell: {nearest}")
            print(f"Re-run with cell={nearest} or use best=True.")
            return None

    params, stored_fit, stored_bd = elites[cell]
    cell_center = bd_bounds[:, 0] + (np.array(cell) + 0.5) * bd_size
    print(f"\nLoaded elite from '{archive_path}', cell {cell}")
    print(f"  stored fitness    : {stored_fit:.1f}")
    print(f"  stored BD         : (vx={stored_bd[0]:.3f}, |a|={stored_bd[1]:.3f})")
    print(f"  cell center       : (vx={cell_center[0]:.2f}, |a|={cell_center[1]:.3f})")
    print(f"  parameter dim     : {len(params):,}")

    env, net, obs_mean, obs_std, env_id = _build_env_and_net(
        data, render=render, record_video=record_video,
    )
    set_flat_params(net, params)
    gif_path = ensure_dir(HALFCHEETAH_EVOLUTION_RESULTS) / "map_elites.gif"
    print(f"  env_id            : {env_id}")
    print(f"  running {n_episodes} eval episodes (max_steps={max_steps})...")

    returns, lengths, vxs, accs = [], [], [], []
    for ep in range(n_episodes):
        r = _run_eval_episode(
            env, net, obs_mean, obs_std, max_steps, ep,
            record_video, gif_path, seed,
        )
        returns.append(r["return"]); lengths.append(r["steps"])
        vxs.append(r["mean_vx"]); accs.append(r["mean_abs_a"])
        print(f"  ep {ep + 1:2d} | steps={r['steps']:4d} | return={r['return']:8.1f} | "
              f"vx={r['mean_vx']:5.2f} | |a|={r['mean_abs_a']:.3f}")

    env.close()
    _print_summary(f"MAP-Elites cell {cell} evaluation summary:",
                   returns, lengths, vxs, accs)
    print(f"  measured BD vs stored BD:")
    print(f"     measured (vx={np.mean(vxs):.3f}, |a|={np.mean(accs):.3f})  "
          f"vs stored (vx={stored_bd[0]:.3f}, |a|={stored_bd[1]:.3f})")
    return {"returns": returns, "lengths": lengths,
            "mean_vx": vxs, "mean_abs_a": accs,
            "cell": cell, "stored_fitness": float(stored_fit)}


# ============================================================================
# 6. CLI
# ============================================================================

def _add_rollout_args(parser):
    """Arguments shared by CMA-ES, NES, and MAP-Elites rollout evaluation."""
    parser.add_argument(
        "--n-episodes", type=int, default=10,
        help="number of evaluation rollouts (default: 10)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=1000,
        help="episode horizon (default: 1000)",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="open the MuJoCo viewer (requires a display)",
    )
    parser.add_argument(
        "--record-video", action="store_true",
        help="save a demo GIF under results/continuous_control/halfcheetah/evolution/",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="master seed; episode i uses seed+i when set",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate saved HalfCheetah evolution policies and MAP-Elites archives.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python test_all.py cmaes
  python test_all.py nes --variant xnes --n-episodes 5 --seed 42
  python test_all.py map-elites list-top --n 20
  python test_all.py map-elites evaluate --best
  python test_all.py map-elites evaluate --cell 45 5 --record-video
  python test_all.py map-elites heatmap --save-path archive_heatmap.png
  python test_all.py map-elites heatmap --save-path archive_heatmap.png --show
""",
    )
    subparsers = parser.add_subparsers(dest="algorithm", required=True)

    cmaes = subparsers.add_parser(
        "cmaes", help="evaluate the best CMA-ES policy",
        description="Roll out the saved CMA-ES checkpoint.",
    )
    _add_rollout_args(cmaes)
    cmaes.add_argument(
        "--model-path", type=Path, default=CMAES_FINAL,
        help=f"path to cmaes_final_model.npz (default: {CMAES_FINAL})",
    )

    nes = subparsers.add_parser(
        "nes", help="evaluate the best NES policy (sNES or xNES)",
        description="Roll out a saved NES checkpoint.",
    )
    _add_rollout_args(nes)
    nes.add_argument(
        "--variant", choices=("snes", "xnes"), default="snes",
        help="which NES trainer variant to load (default: snes)",
    )
    nes.add_argument(
        "--model-path", type=Path, default=None,
        help="override checkpoint path (default: derived from --variant)",
    )

    map_elites = subparsers.add_parser(
        "map-elites", help="inspect or evaluate the MAP-Elites archive",
        description=(
            "MAP-Elites saves a 2D archive of elites indexed by behavior "
            "descriptors (mean forward velocity, mean |action|). "
            "Cells are (i, j) in a 50x50 grid by default: "
            "i=0 slowest (BD1=-3 m/s) .. i=49 fastest (+15 m/s); "
            "j=0 lowest control effort .. j=49 highest."
        ),
    )
    map_elites.add_argument(
        "--archive-path", type=Path, default=MAP_ELITES_FINAL,
        help=f"path to cma_me_final_archive.npz (default: {MAP_ELITES_FINAL})",
    )
    map_actions = map_elites.add_subparsers(dest="map_action", required=True)

    list_top = map_actions.add_parser(
        "list-top", help="print the top-N elites ranked by stored fitness",
    )
    list_top.add_argument(
        "--n", type=int, default=15,
        help="how many elites to list (default: 15)",
    )

    evaluate = map_actions.add_parser(
        "evaluate", help="roll out one archive cell (best or specific)",
    )
    _add_rollout_args(evaluate)
    evaluate.set_defaults(n_episodes=5)
    cell_group = evaluate.add_mutually_exclusive_group(required=True)
    cell_group.add_argument(
        "--best", action="store_true",
        help="evaluate the cell with the highest stored fitness",
    )
    cell_group.add_argument(
        "--cell", nargs=2, type=int, metavar=("I", "J"),
        help="evaluate archive cell (i, j); aborts with a hint if the cell is empty",
    )

    heatmap = map_actions.add_parser(
        "heatmap", help="render the archive fitness heatmap",
    )
    heatmap.add_argument(
        "--save-path", type=Path, default=None,
        help="optional PNG output path (relative paths go under evolution results/)",
    )
    heatmap.add_argument(
        "--show", action="store_true",
        help="open an interactive matplotlib window (default: headless, no window)",
    )

    return parser


def _resolve_nes_model_path(args):
    if args.model_path is not None:
        return args.model_path
    return SNES_FINAL if args.variant == "snes" else XNES_FINAL


def _resolve_heatmap_save_path(args):
    if args.save_path is None:
        return None
    if args.save_path.is_absolute():
        return args.save_path
    return ensure_dir(HALFCHEETAH_EVOLUTION_RESULTS) / args.save_path


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.algorithm == "cmaes":
        evaluate_cmaes(
            model_path=args.model_path,
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            render=args.render,
            record_video=args.record_video,
            seed=args.seed,
        )
        return

    if args.algorithm == "nes":
        evaluate_nes(
            model_path=_resolve_nes_model_path(args),
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            render=args.render,
            record_video=args.record_video,
            seed=args.seed,
        )
        return

    archive_path = args.archive_path
    if args.map_action == "list-top":
        list_top_elites_map_elites(archive_path=archive_path, n=args.n)
        return

    if args.map_action == "evaluate":
        evaluate_map_elites(
            archive_path=archive_path,
            cell=tuple(args.cell) if args.cell is not None else None,
            best=args.best,
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            render=args.render,
            record_video=args.record_video,
            seed=args.seed,
        )
        return

    show_archive_heatmap(
        archive_path=archive_path,
        save_path=_resolve_heatmap_save_path(args),
        show=args.show,
    )


if __name__ == "__main__":
    main()
