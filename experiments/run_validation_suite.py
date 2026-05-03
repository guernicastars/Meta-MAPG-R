"""Validation experiments for the equilibrium-selection / basin-entry framing.

Phases (run independently or in sequence):

    A   expected-update arrows (PG vs Meta-MAPG)
    B   4-method basin atlas at 51x51
    C   stochastic basin maps with multiple seeds per cell
    D   4-arm shape-then-cool ablation including the warm-MetaMAPG -> pure-PG arm
    E   coordination-game family sweep over the temptation parameter T
    F   threshold robustness over tau in {0.75, 0.82, 0.90}

Re-uses primitives from run_meta_mapg_experiments.py without modifying it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from run_meta_mapg_experiments import (  # noqa: E402
    Game,
    GradientComponents,
    cooperation_probs,
    estimate_components,
    estimate_components_L,
    expected_return,
    is_success,
    logit,
    perturb_theta,
    run_rollout,
    run_rollout_asymmetric,
    run_rollout_with_checkpoints,
    select_checkpoint,
    sigmoid,
    stag_hunt,
    update_from_components,
    update_from_components_asymmetric,
)


METHODS_FULL = ["standard_pg", "meta_pg", "lola_style", "meta_mapg"]
METHOD_LABELS = {
    "standard_pg": "PG",
    "meta_pg": "Meta-PG",
    "lola_style": "Peer only",
    "meta_mapg": "Meta-MAPG",
}
METHOD_COLORS = {
    "standard_pg": "#4c78a8",
    "meta_pg": "#f58518",
    "lola_style": "#54a24b",
    "meta_mapg": "#b279a2",
}


def coordination_game(temptation: float) -> Game:
    """Stag-Hunt-like 2x2 with R=4, P=2, S=0, T=temptation. Action 0 = C, 1 = D."""
    p1 = np.array([[4.0, 0.0], [temptation, 2.0]], dtype=float)
    return Game(f"coord_T{temptation:.2f}", p1, p1.T.copy())


@dataclass
class Config:
    outdir: Path
    fig_outdir: Path
    seed_base: int = 700_000
    success_threshold: float = 0.82
    inner_lr: float = 0.55
    lr: float = 0.9
    lr_power: float = 0.24
    peer_coef: float = 1.5
    own_coef: float = 0.35


def now_str() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(values.mean()), float(lo), float(hi)


def exact_components_horizon1(theta: np.ndarray, game: Game, inner_lr: float) -> GradientComponents:
    p = sigmoid(theta[:, 0])
    base = np.zeros((2, game.n_states), dtype=float)
    own = np.zeros((2, game.n_states), dtype=float)
    peer = np.zeros((2, game.n_states), dtype=float)
    reward = np.zeros(2, dtype=float)
    outcomes = []
    for c0 in (False, True):
        for c1 in (False, True):
            prob = (p[0] if c0 else 1.0 - p[0]) * (p[1] if c1 else 1.0 - p[1])
            idx0 = 0 if c0 else 1
            idx1 = 0 if c1 else 1
            returns = np.array([game.payoff_p1[idx0, idx1], game.payoff_p2[idx0, idx1]])
            scores = np.array([[float(c0) - p[0]], [float(c1) - p[1]]])
            hess_diag = np.array([[-p[0] * (1.0 - p[0])], [-p[1] * (1.0 - p[1])]])
            outcomes.append((prob, returns, scores, hess_diag))
            reward += prob * returns

    for player in range(2):
        opp = 1 - player
        g_self = sum(prob * returns[player] * scores[player] for prob, returns, scores, _ in outcomes)
        q_self_wrt_opp = sum(prob * returns[player] * scores[opp] for prob, returns, scores, _ in outcomes)
        h_self = sum(
            prob * returns[player] * np.outer(scores[player], scores[player])
            for prob, returns, scores, _ in outcomes
        )
        h_self += np.diag(
            sum(prob * returns[player] * hess_diag[player] for prob, returns, _, hess_diag in outcomes)
        )
        cross_opp_self = sum(
            prob * returns[opp] * np.outer(scores[opp], scores[player])
            for prob, returns, scores, _ in outcomes
        )
        base[player] = g_self
        own[player] = inner_lr * h_self.T @ g_self
        peer[player] = inner_lr * cross_opp_self.T @ q_self_wrt_opp
    return GradientComponents(base=base, own=own, peer=peer, reward_estimate=reward)


def probability_delta(theta: np.ndarray, update: np.ndarray) -> np.ndarray:
    p = sigmoid(theta[:, 0])
    return update[:, 0] * p * (1.0 - p)


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def step_lr(cfg: Config, step_index: int) -> float:
    return cfg.lr / ((step_index + 10.0) ** cfg.lr_power)


# -------------------------------------------------------------------------
# Phase A : expected-update arrows
# -------------------------------------------------------------------------
def expected_update_at(
    game: Game,
    p1: float,
    p2: float,
    method: str,
    cfg: Config,
    batch_size: int,
    seed: int,
) -> tuple[float, float]:
    theta = np.zeros((2, game.n_states), dtype=float)
    theta[0, 0] = logit(p1)
    theta[1, 0] = logit(p2)
    rng = np.random.default_rng(seed)
    comps = estimate_components(theta, game, batch_size, rng, cfg.inner_lr)
    update = update_from_components(comps, method, cfg.peer_coef, cfg.own_coef)
    # Convert dtheta to dp via chain rule p = sigmoid(theta).
    p_now = sigmoid(theta[:, 0])
    dp_dtheta = p_now * (1.0 - p_now)
    return float(update[0, 0] * dp_dtheta[0]), float(update[1, 0] * dp_dtheta[1])


def run_phase_a(cfg: Config, grid_size: int, batch_size: int, seed_reps: int) -> Path:
    game = stag_hunt()
    grid = np.linspace(0.05, 0.95, grid_size)
    rows: list[dict] = []
    for method in ["standard_pg", "meta_mapg"]:
        for i, p1 in enumerate(grid):
            for j, p2 in enumerate(grid):
                dp1_acc = 0.0
                dp2_acc = 0.0
                for r in range(seed_reps):
                    seed = cfg.seed_base + 1_000 * (0 if method == "standard_pg" else 1) + 41 * i + 7 * j + r
                    dp1, dp2 = expected_update_at(game, p1, p2, method, cfg, batch_size, seed)
                    dp1_acc += dp1
                    dp2_acc += dp2
                rows.append(
                    {
                        "phase": "A",
                        "method": method,
                        "p1": float(p1),
                        "p2": float(p2),
                        "dp1": dp1_acc / seed_reps,
                        "dp2": dp2_acc / seed_reps,
                    }
                )
    df = pd.DataFrame(rows)
    out = cfg.outdir / "phase_a_arrows.csv"
    df.to_csv(out, index=False)
    plot_phase_a(df, cfg)
    return out


def plot_phase_a(df: pd.DataFrame, cfg: Config) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    for ax, method in zip(axes, ["standard_pg", "meta_mapg"]):
        sub = df[df["method"] == method]
        p1 = sub["p1"].to_numpy()
        p2 = sub["p2"].to_numpy()
        u = sub["dp1"].to_numpy()
        v = sub["dp2"].to_numpy()
        norm = np.maximum(np.sqrt(u**2 + v**2), 1e-8)
        scale = 0.04 / norm.max() if norm.max() > 0 else 1.0
        ax.quiver(
            p1,
            p2,
            u * scale,
            v * scale,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.005,
            color="#222222",
            alpha=0.8,
        )
        ax.scatter([1.0], [1.0], marker="*", s=110, color="#2fbf71", zorder=5, label="(C,C)")
        ax.scatter([0.0], [0.0], marker="X", s=70, color="#e45756", zorder=5, label="(D,D)")
        ax.plot([0, 1], [1, 0], color="grey", linewidth=0.6, linestyle=":")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$p_1$ (cooperate)")
        ax.set_ylabel(r"$p_2$ (cooperate)")
        ax.set_title(f"Expected update : {METHOD_LABELS[method]}", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_pdf = cfg.fig_outdir / "phase_a_expected_arrows.pdf"
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=180)
    plt.close(fig)


# -------------------------------------------------------------------------
# Phase B : 4-method basin atlas + frontier overlay
# -------------------------------------------------------------------------
def run_basin_grid(
    game: Game,
    method: str,
    grid: np.ndarray,
    steps: int,
    batch_size: int,
    cfg: Config,
    seed_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    G = grid.size
    success = np.zeros((G, G), dtype=int)
    coop_min = np.zeros((G, G), dtype=float)
    for i, p1 in enumerate(grid):
        for j, p2 in enumerate(grid):
            init_theta = np.zeros((2, game.n_states), dtype=float)
            init_theta[0, 0] = logit(float(p1))
            init_theta[1, 0] = logit(float(p2))
            theta, _ = run_rollout(
                game=game,
                method=method,
                seed=seed_offset + 101 * i + 13 * j,
                steps=steps,
                batch_size=batch_size,
                lr=cfg.lr,
                inner_lr=cfg.inner_lr,
                peer_coef=cfg.peer_coef,
                own_coef=cfg.own_coef,
                init_theta=init_theta,
                lr_power=cfg.lr_power,
                lambda_power=0.0,
                log_every=steps + 1,
            )
            coop = cooperation_probs(theta, game)
            success[i, j] = int(is_success(theta, game, threshold=cfg.success_threshold))
            coop_min[i, j] = float(np.min(coop))
    return success, coop_min


def run_phase_b(cfg: Config, grid_size: int, steps: int, batch_size: int) -> Path:
    game = stag_hunt()
    grid = np.linspace(0.05, 0.95, grid_size)
    rows: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    for k, method in enumerate(METHODS_FULL):
        success, coop_min = run_basin_grid(
            game=game,
            method=method,
            grid=grid,
            steps=steps,
            batch_size=batch_size,
            cfg=cfg,
            seed_offset=cfg.seed_base + 200_000 + 50_000 * k,
        )
        arrays[method] = success
        np.save(cfg.outdir / f"phase_b_basin_{method}.npy", success)
        np.save(cfg.outdir / f"phase_b_coopmin_{method}.npy", coop_min)
        for i, p1 in enumerate(grid):
            for j, p2 in enumerate(grid):
                rows.append(
                    {
                        "phase": "B",
                        "method": method,
                        "init_p1": float(p1),
                        "init_p2": float(p2),
                        "success": int(success[i, j]),
                        "final_coop_min": float(coop_min[i, j]),
                    }
                )
    df = pd.DataFrame(rows)
    out = cfg.outdir / "phase_b_basin.csv"
    df.to_csv(out, index=False)
    np.save(cfg.outdir / "phase_b_grid.npy", grid)
    plot_phase_b_atlas(arrays, grid, cfg)
    plot_phase_b_frontier(arrays, grid, cfg)
    return out


def _basin_fraction(success: np.ndarray) -> float:
    return float(np.mean(success))


def plot_phase_b_atlas(arrays: dict[str, np.ndarray], grid: np.ndarray, cfg: Config) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.6))
    for ax, method in zip(axes.flat, METHODS_FULL):
        success = arrays[method]
        ax.imshow(
            success.T,
            origin="lower",
            extent=(grid[0], grid[-1], grid[0], grid[-1]),
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            aspect="equal",
            interpolation="nearest",
        )
        ax.scatter([1.0], [1.0], marker="*", s=110, color="black", zorder=5)
        ax.scatter([0.0], [0.0], marker="X", s=70, color="black", zorder=5)
        frac = _basin_fraction(success)
        ax.set_title(f"{METHOD_LABELS[method]} : coop basin = {frac*100:.1f}%", fontsize=10)
        ax.set_xlabel(r"$p_1^0$")
        ax.set_ylabel(r"$p_2^0$")
    fig.tight_layout()
    out = cfg.fig_outdir / "phase_b_basin_atlas.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


def plot_phase_b_frontier(arrays: dict[str, np.ndarray], grid: np.ndarray, cfg: Config) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    XX, YY = np.meshgrid(grid, grid, indexing="ij")
    for method in METHODS_FULL:
        success = arrays[method].astype(float)
        cs = ax.contour(
            XX,
            YY,
            success,
            levels=[0.5],
            colors=[METHOD_COLORS[method]],
            linewidths=2.0,
        )
        if cs.allsegs and cs.allsegs[0]:
            seg = cs.allsegs[0][0]
            ax.plot(
                seg[:, 0],
                seg[:, 1],
                color=METHOD_COLORS[method],
                linewidth=2.0,
                label=METHOD_LABELS[method],
            )
    ax.scatter([1.0], [1.0], marker="*", s=120, color="black", zorder=5)
    ax.scatter([0.0], [0.0], marker="X", s=80, color="black", zorder=5)
    ax.text(0.85, 0.85, "(C,C)", fontsize=9)
    ax.text(0.05, 0.05, "(D,D)", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$p_1^0$")
    ax.set_ylabel(r"$p_2^0$")
    ax.set_title("Empirical separatrix (success = 0.5)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = cfg.fig_outdir / "phase_b_frontier_overlay.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


# -------------------------------------------------------------------------
# Phase C : stochastic basin map with S seeds per cell
# -------------------------------------------------------------------------
def run_phase_c(cfg: Config, grid_size: int, steps: int, batch_size: int, seeds_per_cell: int) -> Path:
    game = stag_hunt()
    grid = np.linspace(0.05, 0.95, grid_size)
    rows: list[dict] = []
    coop_probs: dict[str, np.ndarray] = {}
    for k, method in enumerate(METHODS_FULL):
        cprob = np.zeros((grid_size, grid_size), dtype=float)
        for i, p1 in enumerate(grid):
            for j, p2 in enumerate(grid):
                init_theta = np.zeros((2, game.n_states), dtype=float)
                init_theta[0, 0] = logit(float(p1))
                init_theta[1, 0] = logit(float(p2))
                hits = 0
                for s in range(seeds_per_cell):
                    seed = cfg.seed_base + 700_000 + 100_000 * k + 503 * i + 71 * j + s
                    theta, _ = run_rollout(
                        game=game,
                        method=method,
                        seed=seed,
                        steps=steps,
                        batch_size=batch_size,
                        lr=cfg.lr,
                        inner_lr=cfg.inner_lr,
                        peer_coef=cfg.peer_coef,
                        own_coef=cfg.own_coef,
                        init_theta=init_theta,
                        lr_power=cfg.lr_power,
                        lambda_power=0.0,
                        log_every=steps + 1,
                    )
                    if is_success(theta, game, threshold=cfg.success_threshold):
                        hits += 1
                p_hit = hits / seeds_per_cell
                cprob[i, j] = p_hit
                rows.append(
                    {
                        "phase": "C",
                        "method": method,
                        "init_p1": float(p1),
                        "init_p2": float(p2),
                        "n_seeds": int(seeds_per_cell),
                        "n_success": int(hits),
                        "p_success": float(p_hit),
                    }
                )
        coop_probs[method] = cprob
        np.save(cfg.outdir / f"phase_c_pcoop_{method}.npy", cprob)
    df = pd.DataFrame(rows)
    out = cfg.outdir / "phase_c_stochastic_basin.csv"
    df.to_csv(out, index=False)
    np.save(cfg.outdir / "phase_c_grid.npy", grid)
    plot_phase_c(coop_probs, grid, cfg)
    return out


def plot_phase_c(coop_probs: dict[str, np.ndarray], grid: np.ndarray, cfg: Config) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.8))
    for ax, method in zip(axes.flat, METHODS_FULL):
        cprob = coop_probs[method]
        im = ax.imshow(
            cprob.T,
            origin="lower",
            extent=(grid[0], grid[-1], grid[0], grid[-1]),
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
            interpolation="bilinear",
        )
        frac = float(np.mean(cprob))
        ax.set_title(f"{METHOD_LABELS[method]} : E[p_coop] = {frac:.3f}", fontsize=10)
        ax.scatter([1.0], [1.0], marker="*", s=80, color="white", zorder=5)
        ax.scatter([0.0], [0.0], marker="X", s=50, color="white", zorder=5)
        ax.set_xlabel(r"$p_1^0$")
        ax.set_ylabel(r"$p_2^0$")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label=r"$\hat P(\text{coop})$")
    out = cfg.fig_outdir / "phase_c_stochastic_basin.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


# -------------------------------------------------------------------------
# Phase D : 4-arm shape-then-cool ablation
# -------------------------------------------------------------------------
def _two_phase_lambda(step: int, n0: int, lam_c: float, scale: int, q: float) -> float:
    if step < n0:
        return lam_c
    return lam_c / ((1.0 + (step - n0) / float(max(1, scale))) ** q)


def run_phase_d(
    cfg: Config,
    n_seeds: int,
    n0: int,
    total_steps: int,
    scale: int,
    q: float,
    batch_size: int,
) -> Path:
    game = stag_hunt()
    rng_master = np.random.default_rng(cfg.seed_base + 990_000)
    init_thetas = [
        rng_master.normal(loc=0.0, scale=1.35, size=(2, game.n_states))
        for _ in range(n_seeds)
    ]
    arms = [
        ("pg", "constant_pg"),
        ("meta_mapg_constant", "constant"),
        ("meta_mapg_two_phase", "two_phase"),
        ("warm_metamapg_pure_pg", "warm_then_pg"),
    ]
    arm_seed_offsets = {
        "pg": 0,
        "meta_mapg_constant": 1,
        "meta_mapg_two_phase": 2,
        "warm_metamapg_pure_pg": 3,
    }
    rows: list[dict] = []
    for label, schedule in arms:
        for seed in range(n_seeds):
            theta = init_thetas[seed].copy()
            rng = np.random.default_rng(cfg.seed_base + 991_000 + 71 * seed + arm_seed_offsets[label])
            coop_trace: list[float] = []
            lambda_trace: list[float] = []
            for step in range(total_steps):
                comps = estimate_components(theta, game, batch_size, rng, cfg.inner_lr)
                lr_step = cfg.lr / ((step + 10.0) ** cfg.lr_power)
                if schedule == "constant_pg":
                    method = "standard_pg"
                    lam_step = 0.0
                elif schedule == "constant":
                    method = "meta_mapg"
                    lam_step = cfg.peer_coef
                elif schedule == "two_phase":
                    method = "meta_mapg"
                    lam_step = _two_phase_lambda(step, n0, cfg.peer_coef, scale, q)
                elif schedule == "warm_then_pg":
                    if step < n0:
                        method = "meta_mapg"
                        lam_step = cfg.peer_coef
                    else:
                        method = "standard_pg"
                        lam_step = 0.0
                else:
                    raise ValueError(schedule)
                update = update_from_components(comps, method, lam_step, cfg.own_coef)
                theta = np.clip(theta + lr_step * update, -8.0, 8.0)
                coop_trace.append(float(np.min(cooperation_probs(theta, game))))
                lambda_trace.append(float(lam_step))
            arr = np.array(coop_trace)
            second_half = arr[total_steps // 2 :]
            rows.append(
                {
                    "phase": "D",
                    "label": label,
                    "schedule": schedule,
                    "seed": seed,
                    "final_coop_min": float(arr[-1]),
                    "second_half_coop_mean": float(np.mean(second_half)),
                    "second_half_coop_std": float(np.std(second_half, ddof=1)) if second_half.size > 1 else 0.0,
                    "success": int(arr[-1] >= cfg.success_threshold),
                    "final_lambda": float(lambda_trace[-1]),
                }
            )
    df = pd.DataFrame(rows)
    out = cfg.outdir / "phase_d_anneal_4arm.csv"
    df.to_csv(out, index=False)
    plot_phase_d(df, cfg, n0=n0, total_steps=total_steps, scale=scale, q=q)
    return out


def plot_phase_d(df: pd.DataFrame, cfg: Config, n0: int, total_steps: int, scale: int, q: float) -> None:
    arm_order = ["pg", "meta_mapg_constant", "meta_mapg_two_phase", "warm_metamapg_pure_pg"]
    pretty = {
        "pg": "PG",
        "meta_mapg_constant": "Meta-MAPG (const $\\lambda$)",
        "meta_mapg_two_phase": "Meta-MAPG (two-phase)",
        "warm_metamapg_pure_pg": "warm-Meta-MAPG $\\to$ pure-PG",
    }
    colors = {
        "pg": "#4c78a8",
        "meta_mapg_constant": "#b279a2",
        "meta_mapg_two_phase": "#2fbf71",
        "warm_metamapg_pure_pg": "#e45756",
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    ax = axes[0]
    for lbl in arm_order:
        sub = df[df["label"] == lbl]
        rate = float(sub["success"].mean())
        sem = float(sub["success"].std(ddof=1) / np.sqrt(len(sub))) if len(sub) > 1 else 0.0
        ax.bar(pretty[lbl], rate, yerr=1.96 * sem, color=colors[lbl], alpha=0.85)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Cooperative success rate")
    ax.tick_params(axis="x", labelrotation=18, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Outcome", fontsize=10)

    ax = axes[1]
    steps = np.arange(total_steps)
    lam_const = float(cfg.peer_coef) * np.ones_like(steps, dtype=float)
    lam_two = np.where(
        steps < n0,
        float(cfg.peer_coef),
        float(cfg.peer_coef)
        / np.maximum(1.0, 1.0 + (steps - n0) / float(scale)) ** q,
    )
    lam_warm_pg = np.where(steps < n0, float(cfg.peer_coef), 0.0)
    ax.plot(steps, lam_const, color=colors["meta_mapg_constant"], linewidth=1.7, label="const $\\lambda$")
    ax.plot(steps, lam_two, color=colors["meta_mapg_two_phase"], linewidth=1.7, label="two-phase")
    ax.plot(steps, lam_warm_pg, color=colors["warm_metamapg_pure_pg"], linewidth=1.7, label="warm $\\to$ PG")
    ax.axvline(n0, color="grey", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$\lambda_n$")
    ax.set_title("Schedule", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = cfg.fig_outdir / "phase_d_anneal_4arm.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


# -------------------------------------------------------------------------
# Phase E : coordination-game family sweep
# -------------------------------------------------------------------------
def run_phase_e(
    cfg: Config,
    temptations: list[float],
    methods: list[str],
    grid_size: int,
    steps: int,
    batch_size: int,
) -> Path:
    grid = np.linspace(0.05, 0.95, grid_size)
    rows: list[dict] = []
    for ti, T in enumerate(temptations):
        game = coordination_game(T)
        for mi, method in enumerate(methods):
            success_grid, _ = run_basin_grid(
                game=game,
                method=method,
                grid=grid,
                steps=steps,
                batch_size=batch_size,
                cfg=cfg,
                seed_offset=cfg.seed_base + 400_000 + 1_000 * ti + 50_000 * mi,
            )
            frac = float(np.mean(success_grid))
            rows.append(
                {
                    "phase": "E",
                    "T": float(T),
                    "method": method,
                    "coop_basin_fraction": frac,
                    "n_cells": int(grid_size * grid_size),
                }
            )
    df = pd.DataFrame(rows)
    out = cfg.outdir / "phase_e_game_family.csv"
    df.to_csv(out, index=False)
    plot_phase_e(df, cfg)
    return out


def plot_phase_e(df: pd.DataFrame, cfg: Config) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for method in df["method"].unique():
        sub = df[df["method"] == method].sort_values("T")
        ax.plot(
            sub["T"],
            sub["coop_basin_fraction"],
            marker="o",
            linewidth=1.8,
            color=METHOD_COLORS.get(method, "#333333"),
            label=METHOD_LABELS.get(method, method),
        )
    ax.set_xlabel("Temptation T")
    ax.set_ylabel("Cooperative basin fraction")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = cfg.fig_outdir / "phase_e_game_family.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


# -------------------------------------------------------------------------
# Phase F : threshold robustness (re-classification)
# -------------------------------------------------------------------------
def run_phase_f(cfg: Config, thresholds: list[float]) -> Path:
    """Reclassify endpoints from existing artifacts at multiple thresholds."""
    rows: list[dict] = []
    sources = {
        "phase_b_full_grid": cfg.outdir / "phase_b_basin.csv",
        "main_basin": EXPERIMENTS_DIR.parent / "artifacts" / "main" / "basin_map.csv",
    }
    for source_name, path in sources.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for tau in thresholds:
            for method, sub in df.groupby("method"):
                successes = (sub["final_coop_min"] >= tau).astype(int)
                rate = float(successes.mean())
                rows.append(
                    {
                        "phase": "F",
                        "source": source_name,
                        "method": str(method),
                        "tau": float(tau),
                        "coop_basin_fraction": rate,
                        "n_cells": int(len(sub)),
                    }
                )
    df_out = pd.DataFrame(rows)
    out = cfg.outdir / "phase_f_threshold_robustness.csv"
    df_out.to_csv(out, index=False)
    return out


# -------------------------------------------------------------------------
# Shared helpers for new phases
# -------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


# -------------------------------------------------------------------------
# Phase A2 : angle-tilt heatmap + Meta-MAPG minus PG difference quiver
# -------------------------------------------------------------------------
def run_phase_a2(cfg: Config) -> Path:
    csv_a = cfg.outdir / "phase_a_arrows.csv"
    if not csv_a.exists():
        return csv_a
    df = pd.read_csv(csv_a)
    sub_pg = df[df["method"] == "standard_pg"].sort_values(["p1", "p2"]).reset_index(drop=True)
    sub_mm = df[df["method"] == "meta_mapg"].sort_values(["p1", "p2"]).reset_index(drop=True)
    angle_pg = np.arctan2(sub_pg["dp2"].to_numpy(), sub_pg["dp1"].to_numpy())
    angle_mm = np.arctan2(sub_mm["dp2"].to_numpy(), sub_mm["dp1"].to_numpy())
    angle_diff = ((angle_mm - angle_pg) + np.pi) % (2 * np.pi) - np.pi
    out_df = pd.DataFrame({
        "phase": "A2",
        "p1": sub_pg["p1"].to_numpy(),
        "p2": sub_pg["p2"].to_numpy(),
        "angle_diff_rad": angle_diff,
        "dp1_diff": sub_mm["dp1"].to_numpy() - sub_pg["dp1"].to_numpy(),
        "dp2_diff": sub_mm["dp2"].to_numpy() - sub_pg["dp2"].to_numpy(),
    })
    out = cfg.outdir / "phase_a2_angle_diff.csv"
    out_df.to_csv(out, index=False)
    plot_phase_a2(out_df, cfg)
    return out


def plot_phase_a2(df: pd.DataFrame, cfg: Config) -> None:
    p1 = df["p1"].to_numpy()
    p2 = df["p2"].to_numpy()
    n = int(round(np.sqrt(len(p1))))
    grid_vals = np.sort(np.unique(p1))
    angle_diff = df["angle_diff_rad"].to_numpy().reshape(n, n)
    dp1_diff = df["dp1_diff"].to_numpy()
    dp2_diff = df["dp2_diff"].to_numpy()
    extent = (grid_vals[0], grid_vals[-1], grid_vals[0], grid_vals[-1])

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8))

    ax = axes[0]
    im = ax.imshow(angle_diff.T, origin="lower", extent=extent, cmap="RdBu",
                   vmin=-np.pi, vmax=np.pi, aspect="equal", interpolation="bilinear")
    plt.colorbar(im, ax=ax, label="angle tilt (rad)")
    ax.plot([0, 1], [1, 0], color="grey", linewidth=0.6, linestyle=":")
    ax.set_xlabel(r"$p_1^0$")
    ax.set_ylabel(r"$p_2^0$")
    ax.set_title("Meta-MAPG vs PG: angle tilt", fontsize=10)

    ax = axes[1]
    norm_d = max(np.sqrt(dp1_diff**2 + dp2_diff**2).max(), 1e-8)
    scale = 0.04 / norm_d
    ax.quiver(p1, p2, dp1_diff * scale, dp2_diff * scale,
              angles="xy", scale_units="xy", scale=1.0, width=0.005, color="#b279a2", alpha=0.8)
    ax.scatter([1.0], [1.0], marker="*", s=110, color="#2fbf71", zorder=5)
    ax.scatter([0.0], [0.0], marker="X", s=70, color="#e45756", zorder=5)
    ax.plot([0, 1], [1, 0], color="grey", linewidth=0.6, linestyle=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$p_1^0$")
    ax.set_ylabel(r"$p_2^0$")
    ax.set_title("Meta-MAPG $-$ PG (difference quiver)", fontsize=10)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    out = cfg.fig_outdir / "phase_a2_angle_diff.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


# -------------------------------------------------------------------------
# Phase G : multi-seed T-sweep with Wilson CIs
# -------------------------------------------------------------------------
def run_phase_g(
    cfg: Config,
    temptations: list[float],
    methods: list[str],
    grid_size: int,
    steps: int,
    batch_size: int,
    seeds_per_cell: int,
) -> Path:
    grid = np.linspace(0.05, 0.95, grid_size)
    rows: list[dict] = []
    for ti, T in enumerate(temptations):
        game = coordination_game(T)
        for mi, method in enumerate(methods):
            k, n = 0, 0
            for i, p1 in enumerate(grid):
                for j, p2 in enumerate(grid):
                    init_theta = np.zeros((2, game.n_states), dtype=float)
                    init_theta[0, 0] = logit(float(p1))
                    init_theta[1, 0] = logit(float(p2))
                    for s in range(seeds_per_cell):
                        seed = cfg.seed_base + 600_000 + 10_000 * ti + 50_000 * mi + 503 * i + 71 * j + s
                        theta, _ = run_rollout(
                            game=game, method=method, seed=seed, steps=steps, batch_size=batch_size,
                            lr=cfg.lr, inner_lr=cfg.inner_lr, peer_coef=cfg.peer_coef, own_coef=cfg.own_coef,
                            init_theta=init_theta, lr_power=cfg.lr_power, lambda_power=0.0, log_every=steps + 1,
                        )
                        k += int(is_success(theta, game, threshold=cfg.success_threshold))
                        n += 1
            p_hat, lo, hi = wilson_ci(k, n)
            rows.append({
                "phase": "G", "T": float(T), "method": method,
                "coop_basin_fraction": p_hat, "ci_lo": lo, "ci_hi": hi,
                "n_success": k, "n_trials": n,
            })
    df = pd.DataFrame(rows)
    out = cfg.outdir / "phase_g_tsweep_multiseed.csv"
    df.to_csv(out, index=False)
    plot_phase_g(df, cfg)
    return out


def plot_phase_g(df: pd.DataFrame, cfg: Config) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for method in df["method"].unique():
        sub = df[df["method"] == method].sort_values("T")
        T_vals = sub["T"].to_numpy()
        frac = sub["coop_basin_fraction"].to_numpy()
        lo = sub["ci_lo"].to_numpy()
        hi = sub["ci_hi"].to_numpy()
        color = METHOD_COLORS.get(method, "#333333")
        ax.plot(T_vals, frac, marker="o", linewidth=1.8, color=color, label=METHOD_LABELS.get(method, method))
        ax.fill_between(T_vals, lo, hi, alpha=0.2, color=color)
    ax.set_xlabel("Temptation T")
    ax.set_ylabel("Cooperative basin fraction")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = cfg.fig_outdir / "phase_g_tsweep_multiseed.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


# -------------------------------------------------------------------------
# Phase H : resolution robustness
# -------------------------------------------------------------------------
def run_phase_h(cfg: Config, grid_sizes: list[int], steps: int, batch_size: int) -> Path:
    game = stag_hunt()
    rows: list[dict] = []
    for gi, gs in enumerate(grid_sizes):
        grid = np.linspace(0.05, 0.95, gs)
        for mi, method in enumerate(METHODS_FULL):
            success, _ = run_basin_grid(
                game=game, method=method, grid=grid, steps=steps, batch_size=batch_size,
                cfg=cfg, seed_offset=cfg.seed_base + 500_000 + 10_000 * gi + 50_000 * mi,
            )
            rows.append({
                "phase": "H", "grid_size": int(gs), "method": method,
                "coop_basin_fraction": float(np.mean(success)), "n_cells": int(gs * gs),
            })
    df = pd.DataFrame(rows)
    out = cfg.outdir / "phase_h_resolution.csv"
    df.to_csv(out, index=False)
    plot_phase_h(df, cfg)
    return out


def plot_phase_h(df: pd.DataFrame, cfg: Config) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for method in METHODS_FULL:
        sub = df[df["method"] == method].sort_values("grid_size")
        ax.plot(sub["grid_size"], sub["coop_basin_fraction"], marker="o", linewidth=1.8,
                color=METHOD_COLORS[method], label=METHOD_LABELS[method])
    ax.set_xlabel("Grid resolution $N$ ($N\\times N$)")
    ax.set_ylabel("Cooperative basin fraction")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = cfg.fig_outdir / "phase_h_resolution.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


# -------------------------------------------------------------------------
# Phase I : first-hit-time atlas
# -------------------------------------------------------------------------
def first_hit_step(
    game: Game, method: str, init_theta: np.ndarray,
    total_steps: int, batch_size: int, cfg: Config, seed: int,
) -> int:
    theta = init_theta.copy()
    rng = np.random.default_rng(seed)
    for step in range(total_steps):
        comps = estimate_components(theta, game, batch_size, rng, cfg.inner_lr)
        lr_step = cfg.lr / ((step + 10.0) ** cfg.lr_power)
        update = update_from_components(comps, method, cfg.peer_coef, cfg.own_coef)
        theta = np.clip(theta + lr_step * update, -8.0, 8.0)
        if is_success(theta, game, threshold=cfg.success_threshold):
            return step + 1
    return -1


def run_phase_i(cfg: Config, grid_size: int, total_steps: int, batch_size: int) -> Path:
    game = stag_hunt()
    grid = np.linspace(0.05, 0.95, grid_size)
    rows: list[dict] = []
    npy_data: dict[str, np.ndarray] = {}
    for mi, method in enumerate(METHODS_FULL):
        hit_grid = np.full((grid_size, grid_size), -1, dtype=int)
        for i, p1 in enumerate(grid):
            for j, p2 in enumerate(grid):
                init_theta = np.zeros((2, game.n_states), dtype=float)
                init_theta[0, 0] = logit(float(p1))
                init_theta[1, 0] = logit(float(p2))
                seed = cfg.seed_base + 800_000 + 50_000 * mi + 101 * i + 13 * j
                t_hit = first_hit_step(game, method, init_theta, total_steps, batch_size, cfg, seed)
                hit_grid[i, j] = t_hit
                rows.append({
                    "phase": "I", "method": method,
                    "init_p1": float(p1), "init_p2": float(p2), "first_hit_step": int(t_hit),
                })
        npy_data[method] = hit_grid
        np.save(cfg.outdir / f"phase_i_fht_{method}.npy", hit_grid)
    df = pd.DataFrame(rows)
    out = cfg.outdir / "phase_i_fht.csv"
    df.to_csv(out, index=False)
    np.save(cfg.outdir / "phase_i_grid.npy", grid)
    plot_phase_i(npy_data, grid, total_steps, cfg)
    return out


def plot_phase_i(npy_data: dict[str, np.ndarray], grid: np.ndarray, total_steps: int, cfg: Config) -> None:
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgrey")
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.6))
    for ax, method in zip(axes.flat, METHODS_FULL):
        hit = npy_data[method].astype(float)
        hit[hit < 0] = float("nan")
        log_hit = np.log1p(hit)
        im = ax.imshow(log_hit.T, origin="lower",
                       extent=(grid[0], grid[-1], grid[0], grid[-1]),
                       cmap=cmap, aspect="equal", interpolation="nearest")
        plt.colorbar(im, ax=ax, label="log(1+step)")
        valid = hit[~np.isnan(hit)]
        mean_t = float(np.mean(valid)) if valid.size > 0 else float("nan")
        ax.set_title(f"{METHOD_LABELS[method]} : mean $t^*$={mean_t:.1f}", fontsize=10)
        ax.set_xlabel(r"$p_1^0$")
        ax.set_ylabel(r"$p_2^0$")
    fig.tight_layout()
    out = cfg.fig_outdir / "phase_i_fht.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


# -------------------------------------------------------------------------
# Phase L : diagonal threshold from Phase C data (no new compute)
# -------------------------------------------------------------------------
def run_phase_l(cfg: Config) -> Path:
    csv = cfg.outdir / "phase_c_stochastic_basin.csv"
    if not csv.exists():
        return csv
    df = pd.read_csv(csv)
    diag = df[np.isclose(df["init_p1"], df["init_p2"])].copy()
    rows: list[dict] = []
    for method in METHODS_FULL:
        sub = diag[diag["method"] == method].sort_values("init_p1")
        p_vals = sub["init_p1"].to_numpy()
        s_vals = sub["p_success"].to_numpy()
        crosses = np.where(np.diff(np.sign(s_vals - 0.5)))[0]
        if len(crosses) > 0:
            idx = crosses[0]
            x0, x1, y0, y1 = p_vals[idx], p_vals[idx + 1], s_vals[idx], s_vals[idx + 1]
            p_star = x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0) if abs(y1 - y0) > 1e-8 else float(x0)
        else:
            p_star = float("nan")
        rows.append({"phase": "L", "method": method, "p_diag_star": p_star})
    df_out = pd.DataFrame(rows)
    out = cfg.outdir / "phase_l_diagonal.csv"
    df_out.to_csv(out, index=False)
    plot_phase_l(diag, df_out, cfg)
    return out


def plot_phase_l(diag: pd.DataFrame, summary: pd.DataFrame, cfg: Config) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for method in METHODS_FULL:
        sub = diag[diag["method"] == method].sort_values("init_p1")
        ax.plot(sub["init_p1"], sub["p_success"], marker="o", linewidth=1.8, markersize=4,
                color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        row = summary[summary["method"] == method]
        if not row.empty:
            val = float(row["p_diag_star"].iloc[0])
            if not np.isnan(val):
                ax.axvline(val, color=METHOD_COLORS[method], linewidth=0.8, linestyle="--")
    ax.axhline(0.5, color="grey", linewidth=0.6, linestyle=":")
    ax.set_xlabel(r"$p_0$ (init coop on diagonal $p_1^0=p_2^0$)")
    ax.set_ylabel(r"$\hat P(\mathrm{coop})$")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = cfg.fig_outdir / "phase_l_diagonal.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


# -------------------------------------------------------------------------
# Phase D2 : independent-seed ablation audit
# -------------------------------------------------------------------------
def run_phase_d2(
    cfg: Config,
    n_seeds: int,
    n0: int,
    total_steps: int,
    scale: int,
    q: float,
    batch_size: int,
) -> Path:
    game = stag_hunt()
    arms = [
        ("pg", "constant_pg"),
        ("meta_mapg_constant", "constant"),
        ("meta_mapg_two_phase", "two_phase"),
        ("warm_metamapg_pure_pg", "warm_then_pg"),
    ]
    arm_seed_offsets = {"pg": 0, "meta_mapg_constant": 1, "meta_mapg_two_phase": 2, "warm_metamapg_pure_pg": 3}
    rows: list[dict] = []
    for label, schedule in arms:
        rng_init = np.random.default_rng(cfg.seed_base + 980_000 + arm_seed_offsets[label])
        init_thetas = [rng_init.normal(0.0, 1.35, size=(2, game.n_states)) for _ in range(n_seeds)]
        for seed_idx in range(n_seeds):
            theta = init_thetas[seed_idx].copy()
            rng = np.random.default_rng(cfg.seed_base + 981_000 + 71 * seed_idx + arm_seed_offsets[label])
            coop_trace: list[float] = []
            for step in range(total_steps):
                comps = estimate_components(theta, game, batch_size, rng, cfg.inner_lr)
                lr_step = cfg.lr / ((step + 10.0) ** cfg.lr_power)
                if schedule == "constant_pg":
                    method, lam_step = "standard_pg", 0.0
                elif schedule == "constant":
                    method, lam_step = "meta_mapg", cfg.peer_coef
                elif schedule == "two_phase":
                    method = "meta_mapg"
                    lam_step = _two_phase_lambda(step, n0, cfg.peer_coef, scale, q)
                elif schedule == "warm_then_pg":
                    method = "meta_mapg" if step < n0 else "standard_pg"
                    lam_step = cfg.peer_coef if step < n0 else 0.0
                else:
                    raise ValueError(schedule)
                update = update_from_components(comps, method, lam_step, cfg.own_coef)
                theta = np.clip(theta + lr_step * update, -8.0, 8.0)
                coop_trace.append(float(np.min(cooperation_probs(theta, game))))
            arr = np.array(coop_trace)
            second_half = arr[total_steps // 2:]
            rows.append({
                "phase": "D2", "label": label, "schedule": schedule, "seed": seed_idx,
                "final_coop_min": float(arr[-1]),
                "second_half_coop_mean": float(np.mean(second_half)),
                "success": int(arr[-1] >= cfg.success_threshold),
            })
    df = pd.DataFrame(rows)
    out = cfg.outdir / "phase_d2_audit.csv"
    df.to_csv(out, index=False)
    plot_phase_d2(df, cfg)
    return out


def plot_phase_d2(df: pd.DataFrame, cfg: Config) -> None:
    arm_order = ["pg", "meta_mapg_constant", "meta_mapg_two_phase", "warm_metamapg_pure_pg"]
    pretty = {
        "pg": "PG",
        "meta_mapg_constant": "Meta-MAPG (const $\\lambda$)",
        "meta_mapg_two_phase": "Meta-MAPG (two-phase)",
        "warm_metamapg_pure_pg": "warm-MM $\\to$ PG",
    }
    colors = {
        "pg": "#4c78a8",
        "meta_mapg_constant": "#b279a2",
        "meta_mapg_two_phase": "#2fbf71",
        "warm_metamapg_pure_pg": "#e45756",
    }
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for lbl in arm_order:
        sub = df[df["label"] == lbl]
        rate = float(sub["success"].mean())
        sem = float(sub["success"].std(ddof=1) / np.sqrt(len(sub))) if len(sub) > 1 else 0.0
        ax.bar(pretty[lbl], rate, yerr=1.96 * sem, color=colors[lbl], alpha=0.85)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Cooperative success rate")
    ax.tick_params(axis="x", labelrotation=18, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("D2: independent init per arm", fontsize=10)
    fig.tight_layout()
    out = cfg.fig_outdir / "phase_d2_audit.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


# -------------------------------------------------------------------------
# Phase M : peer-coefficient sweep
# -------------------------------------------------------------------------
def run_phase_m(
    cfg: Config,
    lambdas: list[float],
    grid_size: int,
    steps: int,
    batch_size: int,
) -> Path:
    game = stag_hunt()
    grid = np.linspace(0.05, 0.95, grid_size)
    rows: list[dict] = []
    pg_success, _ = run_basin_grid(
        game=game,
        method="standard_pg",
        grid=grid,
        steps=steps,
        batch_size=batch_size,
        cfg=cfg,
        seed_offset=cfg.seed_base + 1_100_000,
    )
    rows.append(
        {
            "phase": "M",
            "method": "standard_pg",
            "peer_coef": 0.0,
            "coop_basin_fraction": float(np.mean(pg_success)),
            "n_cells": int(grid_size * grid_size),
        }
    )
    for li, lam in enumerate(lambdas):
        cfg_lam = Config(
            outdir=cfg.outdir,
            fig_outdir=cfg.fig_outdir,
            seed_base=cfg.seed_base,
            success_threshold=cfg.success_threshold,
            inner_lr=cfg.inner_lr,
            lr=cfg.lr,
            lr_power=cfg.lr_power,
            peer_coef=float(lam),
            own_coef=cfg.own_coef,
        )
        success, _ = run_basin_grid(
            game=game,
            method="meta_mapg",
            grid=grid,
            steps=steps,
            batch_size=batch_size,
            cfg=cfg_lam,
            seed_offset=cfg.seed_base + 1_120_000 + 10_000 * li,
        )
        rows.append(
            {
                "phase": "M",
                "method": "meta_mapg",
                "peer_coef": float(lam),
                "coop_basin_fraction": float(np.mean(success)),
                "n_cells": int(grid_size * grid_size),
            }
        )
    df = pd.DataFrame(rows)
    out = cfg.outdir / "phase_m_lambda_sweep.csv"
    df.to_csv(out, index=False)
    plot_phase_m(df, cfg)
    return out


def plot_phase_m(df: pd.DataFrame, cfg: Config) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    pg = df[df["method"] == "standard_pg"]
    mm = df[df["method"] == "meta_mapg"].sort_values("peer_coef")
    if not pg.empty:
        ax.axhline(
            float(pg["coop_basin_fraction"].iloc[0]),
            color=METHOD_COLORS["standard_pg"],
            linestyle="--",
            linewidth=1.2,
            label="PG baseline",
        )
    ax.plot(
        mm["peer_coef"],
        mm["coop_basin_fraction"],
        marker="o",
        linewidth=1.8,
        color=METHOD_COLORS["meta_mapg"],
        label="Meta-MAPG",
    )
    ax.set_xlabel(r"Peer coefficient $\lambda$")
    ax.set_ylabel("Cooperative basin fraction")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = cfg.fig_outdir / "phase_m_lambda_sweep.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


# -------------------------------------------------------------------------
# Phase N : summary metrics and endpoint welfare
# -------------------------------------------------------------------------
def run_phase_n(cfg: Config, grid_size: int, steps: int, batch_size: int) -> Path:
    game = stag_hunt()
    grid = np.linspace(0.05, 0.95, grid_size)
    pg_basin = np.load(cfg.outdir / "phase_b_basin_standard_pg.npy")
    basin_by_method = {
        method: np.load(cfg.outdir / f"phase_b_basin_{method}.npy")
        for method in METHODS_FULL
    }
    welfare_rows: list[dict] = []
    for mi, method in enumerate(METHODS_FULL):
        welfare = []
        success = []
        for i, p1 in enumerate(grid):
            for j, p2 in enumerate(grid):
                init_theta = np.zeros((2, game.n_states), dtype=float)
                init_theta[0, 0] = logit(float(p1))
                init_theta[1, 0] = logit(float(p2))
                theta, _ = run_rollout(
                    game=game,
                    method=method,
                    seed=cfg.seed_base + 1_200_000 + 50_000 * mi + 101 * i + 13 * j,
                    steps=steps,
                    batch_size=batch_size,
                    lr=cfg.lr,
                    inner_lr=cfg.inner_lr,
                    peer_coef=cfg.peer_coef,
                    own_coef=cfg.own_coef,
                    init_theta=init_theta,
                    lr_power=cfg.lr_power,
                    lambda_power=0.0,
                    log_every=steps + 1,
                )
                welfare.append(float(expected_return(theta, game).sum()))
                success.append(int(is_success(theta, game, threshold=cfg.success_threshold)))
        welfare_rows.append(
            {
                "method": method,
                "welfare_grid_size": int(grid_size),
                "mean_terminal_welfare": float(np.mean(welfare)),
                "terminal_success_fraction": float(np.mean(success)),
            }
        )

    rows: list[dict] = []
    pg_frac = float(pg_basin.mean())
    welfare_df = pd.DataFrame(welfare_rows).set_index("method")
    for method in METHODS_FULL:
        basin = basin_by_method[method]
        gained = np.logical_and(basin == 1, pg_basin == 0)
        lost = np.logical_and(basin == 0, pg_basin == 1)
        rows.append(
            {
                "phase": "N",
                "method": method,
                "phase_b_basin_fraction": float(basin.mean()),
                "expansion_vs_pg": float(basin.mean() / max(pg_frac, 1e-12)),
                "gained_area_vs_pg": float(gained.mean()),
                "lost_area_vs_pg": float(lost.mean()),
                "net_area_vs_pg": float(gained.mean() - lost.mean()),
                "mean_terminal_welfare": float(welfare_df.loc[method, "mean_terminal_welfare"]),
                "welfare_grid_success_fraction": float(welfare_df.loc[method, "terminal_success_fraction"]),
                "welfare_grid_size": int(grid_size),
            }
        )
    df = pd.DataFrame(rows)
    out = cfg.outdir / "phase_n_summary_metrics.csv"
    df.to_csv(out, index=False)
    return out


# -------------------------------------------------------------------------
# Phase O : basin-gain masks relative to PG
# -------------------------------------------------------------------------
def run_phase_o(cfg: Config) -> Path:
    pg = np.load(cfg.outdir / "phase_b_basin_standard_pg.npy")
    grid = np.load(cfg.outdir / "phase_b_grid.npy")
    rows: list[dict] = []
    masks: dict[str, np.ndarray] = {}
    for method in ["meta_pg", "lola_style", "meta_mapg"]:
        basin = np.load(cfg.outdir / f"phase_b_basin_{method}.npy")
        mask = np.zeros_like(basin, dtype=int)
        mask[np.logical_and(pg == 1, basin == 1)] = 1
        mask[np.logical_and(pg == 0, basin == 1)] = 2
        mask[np.logical_and(pg == 1, basin == 0)] = 3
        masks[method] = mask
        rows.append(
            {
                "phase": "O",
                "method": method,
                "shared_success_area": float((mask == 1).mean()),
                "gained_area_vs_pg": float((mask == 2).mean()),
                "lost_area_vs_pg": float((mask == 3).mean()),
                "net_area_vs_pg": float(((mask == 2).mean()) - ((mask == 3).mean())),
            }
        )
    df = pd.DataFrame(rows)
    out = cfg.outdir / "phase_o_gain_masks.csv"
    df.to_csv(out, index=False)
    plot_phase_o(masks, grid, cfg)
    return out


def plot_phase_o(masks: dict[str, np.ndarray], grid: np.ndarray, cfg: Config) -> None:
    cmap = ListedColormap(["#f2f2f2", "#2fbf71", "#4c78a8", "#e45756"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.1))
    for ax, method in zip(axes, ["meta_pg", "lola_style", "meta_mapg"]):
        mask = masks[method]
        ax.imshow(
            mask.T,
            origin="lower",
            extent=(grid[0], grid[-1], grid[0], grid[-1]),
            cmap=cmap,
            norm=norm,
            aspect="equal",
            interpolation="nearest",
        )
        ax.scatter([1.0], [1.0], marker="*", s=80, color="black", zorder=5)
        ax.scatter([0.0], [0.0], marker="X", s=50, color="black", zorder=5)
        ax.set_title(METHOD_LABELS[method], fontsize=10)
        ax.set_xlabel(r"$p_1^0$")
        ax.set_ylabel(r"$p_2^0$")
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#f2f2f2", markersize=8, label="fail for both"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#2fbf71", markersize=8, label="shared success"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#4c78a8", markersize=8, label="gained vs PG"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#e45756", markersize=8, label="lost vs PG"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    out = cfg.fig_outdir / "phase_o_gain_masks.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


# -------------------------------------------------------------------------
# Phase P : paper-grade peer-correction projection
# -------------------------------------------------------------------------
def run_phase_p(cfg: Config) -> Path:
    csv_a = cfg.outdir / "phase_a_arrows.csv"
    if not csv_a.exists():
        return csv_a
    df = pd.read_csv(csv_a)
    sub_pg = df[df["method"] == "standard_pg"].sort_values(["p1", "p2"]).reset_index(drop=True)
    sub_mm = df[df["method"] == "meta_mapg"].sort_values(["p1", "p2"]).reset_index(drop=True)
    p1 = sub_pg["p1"].to_numpy()
    p2 = sub_pg["p2"].to_numpy()
    diff1 = sub_mm["dp1"].to_numpy() - sub_pg["dp1"].to_numpy()
    diff2 = sub_mm["dp2"].to_numpy() - sub_pg["dp2"].to_numpy()
    target1 = 1.0 - p1
    target2 = 1.0 - p2
    target_norm = np.maximum(np.sqrt(target1**2 + target2**2), 1e-12)
    projection = (diff1 * target1 + diff2 * target2) / target_norm
    angle_pg = np.arctan2(sub_pg["dp2"].to_numpy(), sub_pg["dp1"].to_numpy())
    angle_mm = np.arctan2(sub_mm["dp2"].to_numpy(), sub_mm["dp1"].to_numpy())
    angle_diff = ((angle_mm - angle_pg) + np.pi) % (2 * np.pi) - np.pi
    out_df = pd.DataFrame(
        {
            "phase": "P",
            "p1": p1,
            "p2": p2,
            "toward_cc_projection": projection,
            "angle_diff_rad": angle_diff,
            "dp1_diff": diff1,
            "dp2_diff": diff2,
        }
    )
    out = cfg.outdir / "phase_p_projection.csv"
    out_df.to_csv(out, index=False)
    plot_phase_p(out_df, cfg)
    return out


def plot_phase_p(df: pd.DataFrame, cfg: Config) -> None:
    p1 = df["p1"].to_numpy()
    p2 = df["p2"].to_numpy()
    n = int(round(np.sqrt(len(p1))))
    grid_vals = np.sort(np.unique(p1))
    projection = df["toward_cc_projection"].to_numpy().reshape(n, n)
    diff1 = df["dp1_diff"].to_numpy()
    diff2 = df["dp2_diff"].to_numpy()
    proj_flat = df["toward_cc_projection"].to_numpy()
    vmax = max(float(np.nanmax(np.abs(projection))), 1e-10)
    extent = (grid_vals[0], grid_vals[-1], grid_vals[0], grid_vals[-1])

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
    ax = axes[0]
    im = ax.imshow(
        projection.T,
        origin="lower",
        extent=extent,
        cmap="RdYlGn",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
        aspect="equal",
        interpolation="bilinear",
    )
    plt.colorbar(im, ax=ax, label=r"$(\Delta p^{MM}-\Delta p^{PG})\cdot \widehat{(C,C)}$")
    ax.plot([0, 1], [1, 0], color="grey", linewidth=0.7, linestyle=":")
    ax.scatter([1.0], [1.0], marker="*", s=90, color="black", zorder=5)
    ax.scatter([0.0], [0.0], marker="X", s=55, color="black", zorder=5)
    ax.set_xlabel(r"$p_1^0$")
    ax.set_ylabel(r"$p_2^0$")
    ax.set_title("Total correction toward $(C,C)$", fontsize=10)

    ax = axes[1]
    norm_d = max(np.sqrt(diff1**2 + diff2**2).max(), 1e-8)
    scale = 0.04 / norm_d
    q = ax.quiver(
        p1,
        p2,
        diff1 * scale,
        diff2 * scale,
        proj_flat,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.005,
        cmap="RdYlGn",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
        alpha=0.9,
    )
    plt.colorbar(q, ax=ax, label="toward $(C,C)$ component")
    ax.plot([0, 1], [1, 0], color="grey", linewidth=0.7, linestyle=":")
    ax.text(0.43, 0.52, "saddle rotation", fontsize=8, color="#333333")
    ax.scatter([1.0], [1.0], marker="*", s=90, color="black", zorder=5)
    ax.scatter([0.0], [0.0], marker="X", s=55, color="black", zorder=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$p_1^0$")
    ax.set_ylabel(r"$p_2^0$")
    ax.set_title("MM$-$PG difference field", fontsize=10)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    out = cfg.fig_outdir / "phase_p_projection.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


# -------------------------------------------------------------------------
# Phase Q : cell-bootstrap T-sweep with all four methods
# -------------------------------------------------------------------------
def run_phase_q(
    cfg: Config,
    temptations: list[float],
    methods: list[str],
    grid_size: int,
    steps: int,
    batch_size: int,
    seeds_per_cell: int,
) -> Path:
    grid = np.linspace(0.05, 0.95, grid_size)
    summary_rows: list[dict] = []
    cell_rows: list[dict] = []
    for ti, T in enumerate(temptations):
        game = coordination_game(T)
        for mi, method in enumerate(methods):
            cell_probs = []
            for i, p1 in enumerate(grid):
                for j, p2 in enumerate(grid):
                    init_theta = np.zeros((2, game.n_states), dtype=float)
                    init_theta[0, 0] = logit(float(p1))
                    init_theta[1, 0] = logit(float(p2))
                    hits = 0
                    for s in range(seeds_per_cell):
                        seed = cfg.seed_base + 1_300_000 + 10_000 * ti + 50_000 * mi + 503 * i + 71 * j + s
                        theta, _ = run_rollout(
                            game=game,
                            method=method,
                            seed=seed,
                            steps=steps,
                            batch_size=batch_size,
                            lr=cfg.lr,
                            inner_lr=cfg.inner_lr,
                            peer_coef=cfg.peer_coef,
                            own_coef=cfg.own_coef,
                            init_theta=init_theta,
                            lr_power=cfg.lr_power,
                            lambda_power=0.0,
                            log_every=steps + 1,
                        )
                        hits += int(is_success(theta, game, threshold=cfg.success_threshold))
                    p_cell = hits / seeds_per_cell
                    cell_probs.append(p_cell)
                    cell_rows.append(
                        {
                            "phase": "Q",
                            "T": float(T),
                            "method": method,
                            "init_p1": float(p1),
                            "init_p2": float(p2),
                            "n_success": int(hits),
                            "n_seeds": int(seeds_per_cell),
                            "p_success": float(p_cell),
                        }
                    )
            values = np.array(cell_probs, dtype=float)
            mean, lo, hi = bootstrap_mean_ci(values, seed=cfg.seed_base + 1_390_000 + 101 * ti + mi)
            summary_rows.append(
                {
                    "phase": "Q",
                    "T": float(T),
                    "method": method,
                    "coop_basin_fraction": mean,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "n_cells": int(values.size),
                    "seeds_per_cell": int(seeds_per_cell),
                    "n_success": int(round(values.sum() * seeds_per_cell)),
                    "n_trials": int(values.size * seeds_per_cell),
                }
            )
    summary = pd.DataFrame(summary_rows)
    cells = pd.DataFrame(cell_rows)
    out = cfg.outdir / "phase_q_tsweep_bootstrap.csv"
    summary.to_csv(out, index=False)
    cells.to_csv(cfg.outdir / "phase_q_tsweep_cells.csv", index=False)
    plot_phase_q(summary, cfg)
    return out


def plot_phase_q(df: pd.DataFrame, cfg: Config) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for method in METHODS_FULL:
        sub = df[df["method"] == method].sort_values("T")
        if sub.empty:
            continue
        color = METHOD_COLORS.get(method, "#333333")
        x = sub["T"].to_numpy()
        y = sub["coop_basin_fraction"].to_numpy()
        lo = sub["ci_lo"].to_numpy()
        hi = sub["ci_hi"].to_numpy()
        ax.plot(x, y, marker="o", linewidth=1.8, color=color, label=METHOD_LABELS[method])
        ax.fill_between(x, lo, hi, color=color, alpha=0.18)
    ax.set_xlabel("Temptation T")
    ax.set_ylabel("Cooperative basin fraction")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = cfg.fig_outdir / "phase_q_tsweep_bootstrap.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


# -------------------------------------------------------------------------
# Phase R : alignment-to-entry regression
# -------------------------------------------------------------------------
def run_phase_r(cfg: Config) -> Path:
    game = stag_hunt()
    grid = np.load(cfg.outdir / "phase_b_grid.npy")
    pg = np.load(cfg.outdir / "phase_b_basin_standard_pg.npy")
    mm = np.load(cfg.outdir / "phase_b_basin_meta_mapg.npy")
    rows: list[dict] = []
    for i, p1 in enumerate(grid):
        for j, p2 in enumerate(grid):
            theta = np.zeros((2, game.n_states), dtype=float)
            theta[0, 0] = logit(float(p1))
            theta[1, 0] = logit(float(p2))
            comps = exact_components_horizon1(theta, game, cfg.inner_lr)
            pg_update = comps.base
            peer_update = cfg.peer_coef * comps.peer
            own_update = cfg.own_coef * comps.own
            mm_update = pg_update + own_update + peer_update
            pg_dp = probability_delta(theta, pg_update)
            peer_dp = probability_delta(theta, peer_update)
            own_dp = probability_delta(theta, own_update)
            mm_dp = probability_delta(theta, mm_update)
            target = np.array([1.0 - p1, 1.0 - p2], dtype=float)
            target_norm = max(float(np.linalg.norm(target)), 1e-12)
            target_unit = target / target_norm
            pg_decrease = float(np.dot(pg_dp, target_unit))
            peer_projection = float(np.dot(peer_dp, target_unit))
            own_projection = float(np.dot(own_dp, target_unit))
            mm_decrease = float(np.dot(mm_dp, target_unit))
            pg_success = int(pg[i, j])
            mm_success = int(mm[i, j])
            if pg_success and mm_success:
                group = "shared_success"
            elif (not pg_success) and mm_success:
                group = "gained"
            elif pg_success and (not mm_success):
                group = "lost"
            else:
                group = "shared_failure"
            rows.append(
                {
                    "phase": "R",
                    "init_p1": float(p1),
                    "init_p2": float(p2),
                    "pg_success": pg_success,
                    "meta_mapg_success": mm_success,
                    "gain_group": group,
                    "peer_projection_to_cc": peer_projection,
                    "own_projection_to_cc": own_projection,
                    "pg_projection_to_cc": pg_decrease,
                    "meta_mapg_projection_to_cc": mm_decrease,
                    "projection_improvement": mm_decrease - pg_decrease,
                }
            )
    df = pd.DataFrame(rows)
    fail = df[df["pg_success"] == 0].copy()
    labels = fail["meta_mapg_success"].to_numpy(dtype=int)
    summary = pd.DataFrame(
        [
            {
                "phase": "R",
                "n_cells": int(len(df)),
                "n_pg_fail_cells": int(len(fail)),
                "n_gained_cells": int((fail["meta_mapg_success"] == 1).sum()),
                "auc_peer_projection": auc_score(fail["peer_projection_to_cc"].to_numpy(), labels),
                "auc_projection_improvement": auc_score(fail["projection_improvement"].to_numpy(), labels),
                "spearman_peer_gain": float(fail["peer_projection_to_cc"].corr(fail["meta_mapg_success"], method="spearman")),
                "spearman_improvement_gain": float(fail["projection_improvement"].corr(fail["meta_mapg_success"], method="spearman")),
                "mean_peer_projection_gained": float(fail.loc[fail["meta_mapg_success"] == 1, "peer_projection_to_cc"].mean()),
                "mean_peer_projection_not_gained": float(fail.loc[fail["meta_mapg_success"] == 0, "peer_projection_to_cc"].mean()),
            }
        ]
    )
    out = cfg.outdir / "phase_r_alignment_cells.csv"
    summary_out = cfg.outdir / "phase_r_alignment_summary.csv"
    df.to_csv(out, index=False)
    summary.to_csv(summary_out, index=False)
    plot_phase_r(df, cfg)
    return summary_out


def plot_phase_r(df: pd.DataFrame, cfg: Config) -> None:
    grid = np.sort(df["init_p1"].unique())
    n = grid.size
    proj = df.sort_values(["init_p1", "init_p2"])["peer_projection_to_cc"].to_numpy().reshape(n, n)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
    ax = axes[0]
    im = ax.imshow(
        proj.T,
        origin="lower",
        extent=(grid[0], grid[-1], grid[0], grid[-1]),
        cmap="YlGn",
        vmin=0.0,
        aspect="equal",
        interpolation="bilinear",
    )
    plt.colorbar(im, ax=ax, label="peer projection magnitude")
    ax.set_title("Peer term toward $(C,C)$", fontsize=10)
    ax.set_xlabel(r"$p_1^0$")
    ax.set_ylabel(r"$p_2^0$")

    ax = axes[1]
    order = ["shared_failure", "gained", "shared_success"]
    labels = ["fail", "gained", "shared"]
    data = [df[df["gain_group"] == group]["peer_projection_to_cc"].to_numpy() for group in order]
    counts = [len(d) for d in data]
    n_lost = int((df["gain_group"] == "lost").sum())
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False)
    for i, cnt in enumerate(counts):
        ax.text(i + 1, ax.get_ylim()[1] * 0.95, f"n={cnt}", ha="center", fontsize=7, color="#555555")
    if n_lost > 0:
        ax.text(0.98, 0.02, f"lost: n={n_lost} (omitted)", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="#999999")
    ax.set_ylabel("Peer projection magnitude")
    ax.set_title("Projection by basin outcome", fontsize=10)
    ax.tick_params(axis="x", labelrotation=15)

    ax = axes[2]
    fail = df[df["pg_success"] == 0].copy()
    fail["bin"] = pd.qcut(fail["peer_projection_to_cc"], q=6, duplicates="drop")
    binned = fail.groupby("bin", observed=True).agg(
        x=("peer_projection_to_cc", "mean"),
        p_gain=("meta_mapg_success", "mean"),
        n=("meta_mapg_success", "size"),
    )
    ax.plot(binned["x"], binned["p_gain"], marker="o", linewidth=1.8, color=METHOD_COLORS["meta_mapg"])
    for _, row in binned.iterrows():
        ax.text(row["x"], row["p_gain"] + 0.025, str(int(row["n"])), ha="center", fontsize=7)
    ax.set_xlabel("Peer projection among PG-fail cells")
    ax.set_ylabel(r"$P(\mathrm{MM\ succeeds})$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.set_title("Binned gain probability", fontsize=10)

    fig.tight_layout()
    out = cfg.fig_outdir / "phase_r_alignment_regression.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


# -------------------------------------------------------------------------
# Phase T : warm-up handoff into ordinary PG
# -------------------------------------------------------------------------
def deterministic_rollout_schedule(
    game: Game,
    init_theta: np.ndarray,
    cfg: Config,
    total_steps: int,
    warm_steps: int,
    lambda_value: float,
    clip: float | None = 8.0,
) -> tuple[np.ndarray, float, bool]:
    theta = init_theta.copy()
    max_abs = float(np.max(np.abs(theta)))
    diverged = False
    for step in range(total_steps):
        method = "meta_mapg" if step < warm_steps else "standard_pg"
        lam = lambda_value if step < warm_steps else 0.0
        comps = exact_components_horizon1(theta, game, cfg.inner_lr)
        update = update_from_components(comps, method, lam, cfg.own_coef)
        theta = theta + cfg.lr / ((step + 10.0) ** cfg.lr_power) * update
        if clip is not None:
            theta = np.clip(theta, -clip, clip)
        max_abs = max(max_abs, float(np.max(np.abs(theta))))
        if not np.all(np.isfinite(theta)) or max_abs > 50.0:
            diverged = True
            break
    return theta, max_abs, diverged


def run_phase_t(cfg: Config, grid_size: int, total_steps: int, warm_steps_list: list[int]) -> Path:
    game = stag_hunt()
    grid = np.linspace(0.05, 0.95, grid_size)
    rows: list[dict] = []
    schedules = [("pg", 0, 0.0), ("constant_meta_mapg", total_steps, cfg.peer_coef)]
    schedules += [(f"warm_{warm_steps}_then_pg", warm_steps, cfg.peer_coef) for warm_steps in warm_steps_list]
    for label, warm_steps, lam in schedules:
        successes = 0
        for p1 in grid:
            for p2 in grid:
                init_theta = np.array([[logit(float(p1))], [logit(float(p2))]], dtype=float)
                theta, _, diverged = deterministic_rollout_schedule(
                    game=game,
                    init_theta=init_theta,
                    cfg=cfg,
                    total_steps=total_steps,
                    warm_steps=warm_steps,
                    lambda_value=lam,
                    clip=8.0,
                )
                successes += int((not diverged) and is_success(theta, game, threshold=cfg.success_threshold))
        rows.append(
            {
                "phase": "T",
                "schedule": label,
                "warm_steps": int(warm_steps),
                "total_steps": int(total_steps),
                "coop_basin_fraction": successes / float(grid_size * grid_size),
                "n_cells": int(grid_size * grid_size),
            }
        )
    df = pd.DataFrame(rows)
    out = cfg.outdir / "phase_t_handoff_curve.csv"
    df.to_csv(out, index=False)
    plot_phase_t(df, cfg)
    return out


def plot_phase_t(df: pd.DataFrame, cfg: Config) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    pg = df[df["schedule"] == "pg"]
    const = df[df["schedule"] == "constant_meta_mapg"]
    warm = df[df["schedule"].str.startswith("warm_")].sort_values("warm_steps")
    if not pg.empty:
        ax.axhline(float(pg["coop_basin_fraction"].iloc[0]), color=METHOD_COLORS["standard_pg"], linestyle="--", label="PG")
    if not const.empty:
        ax.axhline(float(const["coop_basin_fraction"].iloc[0]), color=METHOD_COLORS["meta_mapg"], linestyle=":", label="constant Meta-MAPG")
    ax.plot(warm["warm_steps"], warm["coop_basin_fraction"], marker="o", linewidth=1.8, color="#2fbf71", label="warm -> PG")
    ax.set_xlabel("Meta-MAPG warm-up steps before PG handoff")
    ax.set_ylabel("Cooperative basin fraction")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = cfg.fig_outdir / "phase_t_handoff_curve.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


# -------------------------------------------------------------------------
# Phase U : high-lambda no-clip stress test
# -------------------------------------------------------------------------
def run_phase_u(
    cfg: Config,
    lambdas: list[float],
    grid_size: int,
    total_steps: int,
) -> Path:
    game = stag_hunt()
    grid = np.linspace(0.05, 0.95, grid_size)
    rows: list[dict] = []
    for lam in lambdas:
        successes = 0
        diverged = 0
        max_abs_values = []
        for p1 in grid:
            for p2 in grid:
                init_theta = np.array([[logit(float(p1))], [logit(float(p2))]], dtype=float)
                theta, max_abs, did_diverge = deterministic_rollout_schedule(
                    game=game,
                    init_theta=init_theta,
                    cfg=cfg,
                    total_steps=total_steps,
                    warm_steps=total_steps,
                    lambda_value=float(lam),
                    clip=None,
                )
                diverged += int(did_diverge)
                max_abs_values.append(max_abs)
                successes += int((not did_diverge) and is_success(theta, game, threshold=cfg.success_threshold))
        rows.append(
            {
                "phase": "U",
                "peer_coef": float(lam),
                "coop_basin_fraction": successes / float(grid_size * grid_size),
                "divergence_fraction": diverged / float(grid_size * grid_size),
                "median_max_abs_logit": float(np.median(max_abs_values)),
                "max_abs_logit": float(np.max(max_abs_values)),
                "n_cells": int(grid_size * grid_size),
            }
        )
    df = pd.DataFrame(rows)
    out = cfg.outdir / "phase_u_high_lambda_stress.csv"
    df.to_csv(out, index=False)
    plot_phase_u(df, cfg)
    return out


def plot_phase_u(df: pd.DataFrame, cfg: Config) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    ax = axes[0]
    ax.plot(df["peer_coef"], df["coop_basin_fraction"], marker="o", linewidth=1.8, color=METHOD_COLORS["meta_mapg"], label="coop basin")
    ax.plot(df["peer_coef"], df["divergence_fraction"], marker="s", linewidth=1.5, color="#e45756", label="divergence")
    ax.set_xlabel(r"Peer coefficient $\lambda$")
    ax.set_ylabel("Fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(df["peer_coef"], df["median_max_abs_logit"], marker="o", linewidth=1.8, label="median max |logit|")
    ax.plot(df["peer_coef"], df["max_abs_logit"], marker="s", linewidth=1.5, label="max |logit|")
    ax.axhline(8.0, color="grey", linestyle=":", linewidth=1.0, label="old clip")
    ax.set_xlabel(r"Peer coefficient $\lambda$")
    ax.set_ylabel("Logit magnitude")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = cfg.fig_outdir / "phase_u_high_lambda_stress.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


# -------------------------------------------------------------------------
# Phase V : 2x2 payoff-geometry counterexample search
# -------------------------------------------------------------------------
def vectorized_rollout_2x2_final_probs(
    R: float,
    S: float,
    T: float,
    P: float,
    method: str,
    cfg: Config,
    grid_size: int,
    steps: int,
    peer_coef: float,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0.05, 0.95, grid_size)
    theta1, theta2 = np.meshgrid([logit(float(x)) for x in grid], [logit(float(x)) for x in grid], indexing="ij")
    payoff = np.array([[R, S], [T, P]], dtype=float)
    payoff_t = payoff.T
    for step in range(steps):
        p1 = sigmoid(theta1)
        p2 = sigmoid(theta2)
        base1 = np.zeros_like(p1)
        base2 = np.zeros_like(p1)
        q1 = np.zeros_like(p1)
        q2 = np.zeros_like(p1)
        h11 = np.zeros_like(p1)
        h22 = np.zeros_like(p1)
        cross21 = np.zeros_like(p1)
        cross12 = np.zeros_like(p1)
        for c1 in (0, 1):
            for c2 in (0, 1):
                prob = (p1 if c1 else 1.0 - p1) * (p2 if c2 else 1.0 - p2)
                idx1 = 0 if c1 else 1
                idx2 = 0 if c2 else 1
                r1 = payoff[idx1, idx2]
                r2 = payoff_t[idx1, idx2]
                score1 = c1 - p1
                score2 = c2 - p2
                hdiag1 = -p1 * (1.0 - p1)
                hdiag2 = -p2 * (1.0 - p2)
                base1 += prob * r1 * score1
                base2 += prob * r2 * score2
                q1 += prob * r1 * score2
                q2 += prob * r2 * score1
                h11 += prob * r1 * (score1 * score1 + hdiag1)
                h22 += prob * r2 * (score2 * score2 + hdiag2)
                cross21 += prob * r2 * score2 * score1
                cross12 += prob * r1 * score1 * score2
        own1 = cfg.inner_lr * h11 * base1
        own2 = cfg.inner_lr * h22 * base2
        peer1 = cfg.inner_lr * cross21 * q1
        peer2 = cfg.inner_lr * cross12 * q2
        update1 = base1.copy()
        update2 = base2.copy()
        if method in {"meta_pg", "meta_mapg"}:
            update1 += cfg.own_coef * own1
            update2 += cfg.own_coef * own2
        if method in {"lola_style", "meta_mapg"}:
            update1 += peer_coef * peer1
            update2 += peer_coef * peer2
        lr_step = cfg.lr / ((step + 10.0) ** cfg.lr_power)
        theta1 = np.clip(theta1 + lr_step * update1, -8.0, 8.0)
        theta2 = np.clip(theta2 + lr_step * update2, -8.0, 8.0)
    return sigmoid(theta1), sigmoid(theta2)


def vectorized_rollout_2x2(
    R: float,
    S: float,
    T: float,
    P: float,
    method: str,
    cfg: Config,
    grid_size: int,
    steps: int,
    peer_coef: float,
) -> float:
    p1, p2 = vectorized_rollout_2x2_final_probs(R, S, T, P, method, cfg, grid_size, steps, peer_coef)
    return float(np.mean((p1 >= cfg.success_threshold) & (p2 >= cfg.success_threshold)))


def run_phase_v(
    cfg: Config,
    s_values: list[float],
    t_values: list[float],
    grid_size: int,
    steps: int,
) -> Path:
    rows: list[dict] = []
    for S in s_values:
        for T in t_values:
            R = 4.0
            P = 2.0
            if 2.0 * R <= T + S:
                continue
            pg = vectorized_rollout_2x2(R, S, T, P, "standard_pg", cfg, grid_size, steps, cfg.peer_coef)
            mm = vectorized_rollout_2x2(R, S, T, P, "meta_mapg", cfg, grid_size, steps, cfg.peer_coef)
            rows.append(
                {
                    "phase": "V",
                    "R": R,
                    "S": float(S),
                    "T": float(T),
                    "P": P,
                    "pg_basin_fraction": pg,
                    "meta_mapg_basin_fraction": mm,
                    "delta_meta_minus_pg": mm - pg,
                    "social_cc_minus_dd": 2.0 * (R - P),
                    "social_cc_minus_miscoord": 2.0 * R - (T + S),
                }
            )
    df = pd.DataFrame(rows)
    out = cfg.outdir / "phase_v_payoff_geometry_search.csv"
    df.to_csv(out, index=False)
    plot_phase_v(df, cfg)
    return out


def plot_phase_v(df: pd.DataFrame, cfg: Config) -> None:
    pivot = df.pivot_table(index="S", columns="T", values="delta_meta_minus_pg")
    worst = df.nsmallest(8, "delta_meta_minus_pg")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    ax = axes[0]
    data = pivot.to_numpy()
    vmax = max(abs(float(np.nanmin(data))), abs(float(np.nanmax(data))), 1e-12)
    im = ax.imshow(
        data,
        origin="lower",
        extent=(pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()),
        cmap="RdYlGn",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
        aspect="auto",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="Meta-MAPG - PG basin")
    ax.set_xlabel("Temptation payoff T")
    ax.set_ylabel("Sucker payoff S")
    ax.set_title("Payoff-geometry search", fontsize=10)
    ax = axes[1]
    labels = [f"S={float(r['S']):.1f}\nT={float(r['T']):.1f}" for _, r in worst.iterrows()]
    ax.bar(labels, worst["delta_meta_minus_pg"], color="#e45756")
    ax.axhline(0.0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_ylabel("Meta-MAPG - PG basin")
    ax.set_title("Worst discovered cells", fontsize=10)
    ax.tick_params(axis="x", labelrotation=0, labelsize=7)
    fig.tight_layout()
    out = cfg.fig_outdir / "phase_v_payoff_geometry_search.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


# -------------------------------------------------------------------------
# Phase W : separatrix margin certificate
# -------------------------------------------------------------------------
def signed_grid_margin(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    coords = np.array([(float(p1), float(p2)) for p1 in grid for p2 in grid], dtype=float)
    labels = mask.astype(bool).ravel()
    success = coords[labels]
    failure = coords[~labels]
    dist_success = np.sqrt(((coords[:, None, :] - success[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    dist_failure = np.sqrt(((coords[:, None, :] - failure[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    return np.where(labels, dist_failure, -dist_success).reshape(mask.shape)


def nearest_mask_value(mask: np.ndarray, grid: np.ndarray, p1: float, p2: float) -> bool:
    i = int(np.argmin(np.abs(grid - p1)))
    j = int(np.argmin(np.abs(grid - p2)))
    return bool(mask[i, j])


def run_phase_w(cfg: Config) -> Path:
    grid = np.load(cfg.outdir / "phase_b_grid.npy")
    pg = np.load(cfg.outdir / "phase_b_basin_standard_pg.npy")
    cells = pd.read_csv(cfg.outdir / "phase_r_alignment_cells.csv")
    margin = signed_grid_margin(grid, pg)
    cells = cells.sort_values(["init_p1", "init_p2"]).copy()
    first_lr = step_lr(cfg, 0)
    cells["pg_signed_margin"] = margin.ravel()
    cells["boundary_distance"] = np.where(cells["pg_success"] == 1, cells["pg_signed_margin"], -cells["pg_signed_margin"])
    cells["peer_step_projection_to_cc"] = first_lr * cells["peer_projection_to_cc"]
    cells["improvement_step_projection_to_cc"] = first_lr * cells["projection_improvement"]
    cells["peer_margin_ratio"] = cells["peer_step_projection_to_cc"] / np.maximum(cells["boundary_distance"], 1e-12)
    cells["improvement_margin_ratio"] = cells["improvement_step_projection_to_cc"] / np.maximum(cells["boundary_distance"], 1e-12)
    fail = cells[cells["pg_success"] == 0]
    labels = fail["meta_mapg_success"].to_numpy(dtype=int)
    summary = pd.DataFrame(
        [
            {
                "phase": "W",
                "first_step_lr": float(first_lr),
                "n_pg_fail_cells": int(len(fail)),
                "n_gained_cells": int((fail["meta_mapg_success"] == 1).sum()),
                "auc_peer_margin_ratio": auc_score(fail["peer_margin_ratio"].to_numpy(), labels),
                "auc_improvement_margin_ratio": auc_score(fail["improvement_margin_ratio"].to_numpy(), labels),
                "spearman_peer_margin_ratio": float(fail["peer_margin_ratio"].corr(fail["meta_mapg_success"], method="spearman")),
                "spearman_improvement_margin_ratio": float(fail["improvement_margin_ratio"].corr(fail["meta_mapg_success"], method="spearman")),
                "median_boundary_distance_gained": float(fail.loc[fail["meta_mapg_success"] == 1, "boundary_distance"].median()),
                "median_boundary_distance_not_gained": float(fail.loc[fail["meta_mapg_success"] == 0, "boundary_distance"].median()),
                "mean_peer_margin_ratio_gained": float(fail.loc[fail["meta_mapg_success"] == 1, "peer_margin_ratio"].mean()),
                "mean_peer_margin_ratio_not_gained": float(fail.loc[fail["meta_mapg_success"] == 0, "peer_margin_ratio"].mean()),
            }
        ]
    )
    cells_out = cfg.outdir / "phase_w_margin_cells.csv"
    summary_out = cfg.outdir / "phase_w_margin_summary.csv"
    cells.to_csv(cells_out, index=False)
    summary.to_csv(summary_out, index=False)
    plot_phase_w(cells, grid, margin, cfg)
    return summary_out


def plot_phase_w(cells: pd.DataFrame, grid: np.ndarray, margin: np.ndarray, cfg: Config) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))
    ax = axes[0]
    vmax = max(abs(float(np.nanmin(margin))), abs(float(np.nanmax(margin))), 1e-12)
    im = ax.imshow(
        margin.T,
        origin="lower",
        extent=(grid[0], grid[-1], grid[0], grid[-1]),
        cmap="RdYlGn",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
        aspect="equal",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="signed PG margin")
    ax.set_xlabel(r"$p_1^0$")
    ax.set_ylabel(r"$p_2^0$")
    ax.set_title("PG basin separatrix margin", fontsize=10)

    fail = cells[cells["pg_success"] == 0].copy()
    gained = fail["meta_mapg_success"] == 1
    ax = axes[1]
    ax.scatter(
        fail.loc[~gained, "boundary_distance"],
        fail.loc[~gained, "peer_step_projection_to_cc"],
        s=8,
        alpha=0.45,
        color="#777777",
        label="not gained",
    )
    ax.scatter(
        fail.loc[gained, "boundary_distance"],
        fail.loc[gained, "peer_step_projection_to_cc"],
        s=10,
        alpha=0.7,
        color=METHOD_COLORS["meta_mapg"],
        label="gained",
    )
    ax.set_xlabel("distance to PG basin")
    ax.set_ylabel("first-step peer displacement")
    ax.set_title("Margin versus push", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    ax = axes[2]
    fail["bin"] = pd.qcut(fail["peer_margin_ratio"], q=6, duplicates="drop")
    binned = fail.groupby("bin", observed=True).agg(
        x=("peer_margin_ratio", "mean"),
        p_gain=("meta_mapg_success", "mean"),
        n=("meta_mapg_success", "size"),
    )
    ax.plot(binned["x"], binned["p_gain"], marker="o", linewidth=1.8, color=METHOD_COLORS["meta_mapg"])
    for _, row in binned.iterrows():
        ax.text(row["x"], row["p_gain"] + 0.025, str(int(row["n"])), ha="center", fontsize=7)
    ax.set_xlabel("peer projection / margin")
    ax.set_ylabel(r"$P(\mathrm{MM\ succeeds})$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.set_title("Margin-normalized gain", fontsize=10)
    fig.tight_layout()
    out = cfg.fig_outdir / "phase_w_separatrix_margin.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


# -------------------------------------------------------------------------
# Phase X : cumulative early alignment
# -------------------------------------------------------------------------
def run_phase_x(cfg: Config, k_values: list[int]) -> Path:
    game = stag_hunt()
    grid = np.load(cfg.outdir / "phase_b_grid.npy")
    pg = np.load(cfg.outdir / "phase_b_basin_standard_pg.npy")
    mm = np.load(cfg.outdir / "phase_b_basin_meta_mapg.npy")
    checkpoints = sorted(set(k_values))
    rows: list[dict] = []
    for i, p1 in enumerate(grid):
        for j, p2 in enumerate(grid):
            theta = np.array([[logit(float(p1))], [logit(float(p2))]], dtype=float)
            target = np.array([1.0 - p1, 1.0 - p2], dtype=float)
            target_unit = target / max(float(np.linalg.norm(target)), 1e-12)
            peer_projection = 0.0
            improvement_projection = 0.0
            full_projection = 0.0
            for step in range(1, max(checkpoints) + 1):
                comps = exact_components_horizon1(theta, game, cfg.inner_lr)
                base_update = comps.base
                improvement_update = cfg.own_coef * comps.own + cfg.peer_coef * comps.peer
                full_update = base_update + improvement_update
                lr_step = step_lr(cfg, step - 1)
                peer_projection += lr_step * float(np.dot(probability_delta(theta, cfg.peer_coef * comps.peer), target_unit))
                improvement_projection += lr_step * float(np.dot(probability_delta(theta, improvement_update), target_unit))
                full_projection += lr_step * float(np.dot(probability_delta(theta, full_update), target_unit))
                theta = np.clip(theta + lr_step * full_update, -8.0, 8.0)
                if step in checkpoints:
                    probs = cooperation_probs(theta, game)
                    rows.append(
                        {
                            "phase": "X",
                            "k": int(step),
                            "init_p1": float(p1),
                            "init_p2": float(p2),
                            "pg_success": int(pg[i, j]),
                            "meta_mapg_success": int(mm[i, j]),
                            "entered_pg_basin": int(nearest_mask_value(pg, grid, float(probs[0]), float(probs[1]))),
                            "cumulative_peer_projection": peer_projection,
                            "cumulative_improvement_projection": improvement_projection,
                            "cumulative_full_projection": full_projection,
                        }
                    )
    df = pd.DataFrame(rows)
    summary_rows = []
    for k, sub in df.groupby("k"):
        fail = sub[sub["pg_success"] == 0]
        labels = fail["meta_mapg_success"].to_numpy(dtype=int)
        gained = fail[fail["meta_mapg_success"] == 1]
        summary_rows.append(
            {
                "phase": "X",
                "k": int(k),
                "n_pg_fail_cells": int(len(fail)),
                "n_gained_cells": int(len(gained)),
                "auc_cumulative_peer": auc_score(fail["cumulative_peer_projection"].to_numpy(), labels),
                "auc_cumulative_improvement": auc_score(fail["cumulative_improvement_projection"].to_numpy(), labels),
                "pg_basin_entry_rate_pg_fail": float(fail["entered_pg_basin"].mean()),
                "pg_basin_entry_rate_gained": float(gained["entered_pg_basin"].mean()) if len(gained) else float("nan"),
            }
        )
    cells_out = cfg.outdir / "phase_x_cumulative_alignment_cells.csv"
    summary_out = cfg.outdir / "phase_x_cumulative_alignment_summary.csv"
    summary = pd.DataFrame(summary_rows)
    df.to_csv(cells_out, index=False)
    summary.to_csv(summary_out, index=False)
    plot_phase_x(summary, cfg)
    return summary_out


def plot_phase_x(summary: pd.DataFrame, cfg: Config) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    ax = axes[0]
    ax.plot(summary["k"], summary["auc_cumulative_peer"], marker="o", linewidth=1.8, label="peer")
    ax.plot(summary["k"], summary["auc_cumulative_improvement"], marker="s", linewidth=1.6, label="own + peer")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.9)
    ax.set_xlabel("Meta-MAPG prefix length K")
    ax.set_ylabel("AUC on PG-fail cells")
    ax.set_ylim(0.45, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(summary["k"], summary["pg_basin_entry_rate_pg_fail"], marker="o", linewidth=1.8, label="all PG-fail")
    ax.plot(summary["k"], summary["pg_basin_entry_rate_gained"], marker="s", linewidth=1.6, label="gained")
    ax.set_xlabel("Meta-MAPG prefix length K")
    ax.set_ylabel("nearest-cell PG basin entry")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = cfg.fig_outdir / "phase_x_cumulative_alignment.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


# -------------------------------------------------------------------------
# Phase Y : minimal warm-up crossing time
# -------------------------------------------------------------------------
def run_phase_y(cfg: Config, max_steps: int) -> Path:
    game = stag_hunt()
    grid = np.load(cfg.outdir / "phase_b_grid.npy")
    pg = np.load(cfg.outdir / "phase_b_basin_standard_pg.npy")
    mm = np.load(cfg.outdir / "phase_b_basin_meta_mapg.npy")
    rows: list[dict] = []
    for i, p1 in enumerate(grid):
        for j, p2 in enumerate(grid):
            if pg[i, j]:
                rows.append(
                    {
                        "phase": "Y",
                        "init_p1": float(p1),
                        "init_p2": float(p2),
                        "pg_success": 1,
                        "meta_mapg_success": int(mm[i, j]),
                        "first_pg_basin_step": 0,
                    }
                )
                continue
            theta = np.array([[logit(float(p1))], [logit(float(p2))]], dtype=float)
            first_step = -1
            for step in range(1, max_steps + 1):
                comps = exact_components_horizon1(theta, game, cfg.inner_lr)
                update = update_from_components(comps, "meta_mapg", cfg.peer_coef, cfg.own_coef)
                theta = np.clip(theta + step_lr(cfg, step - 1) * update, -8.0, 8.0)
                probs = cooperation_probs(theta, game)
                if nearest_mask_value(pg, grid, float(probs[0]), float(probs[1])):
                    first_step = step
                    break
            rows.append(
                {
                    "phase": "Y",
                    "init_p1": float(p1),
                    "init_p2": float(p2),
                    "pg_success": 0,
                    "meta_mapg_success": int(mm[i, j]),
                    "first_pg_basin_step": int(first_step),
                }
            )
    df = pd.DataFrame(rows)
    fail = df[df["pg_success"] == 0]
    gained = fail[fail["meta_mapg_success"] == 1]
    crossed = fail[fail["first_pg_basin_step"] > 0]
    summary = pd.DataFrame(
        [
            {
                "phase": "Y",
                "max_steps": int(max_steps),
                "n_pg_fail_cells": int(len(fail)),
                "n_gained_cells": int(len(gained)),
                "n_crossed_cells": int(len(crossed)),
                "n_gained_crossed_cells": int(((gained["first_pg_basin_step"] > 0)).sum()),
                "cross_rate_pg_fail": float((fail["first_pg_basin_step"] > 0).mean()),
                "cross_rate_gained": float((gained["first_pg_basin_step"] > 0).mean()),
                "median_cross_step_gained": float(gained.loc[gained["first_pg_basin_step"] > 0, "first_pg_basin_step"].median()),
                "median_cross_step_not_gained": float(fail.loc[(fail["meta_mapg_success"] == 0) & (fail["first_pg_basin_step"] > 0), "first_pg_basin_step"].median()),
            }
        ]
    )
    cells_out = cfg.outdir / "phase_y_min_warmup_cells.csv"
    summary_out = cfg.outdir / "phase_y_min_warmup_summary.csv"
    df.to_csv(cells_out, index=False)
    summary.to_csv(summary_out, index=False)
    plot_phase_y(df, grid, cfg)
    return summary_out


def plot_phase_y(df: pd.DataFrame, grid: np.ndarray, cfg: Config) -> None:
    heat = df.sort_values(["init_p1", "init_p2"])["first_pg_basin_step"].to_numpy().reshape(grid.size, grid.size).astype(float)
    heat[heat < 0] = np.nan
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.2))
    ax = axes[0]
    im = ax.imshow(
        heat.T,
        origin="lower",
        extent=(grid[0], grid[-1], grid[0], grid[-1]),
        cmap=cmap,
        aspect="equal",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="first PG-basin step")
    ax.set_xlabel(r"$p_1^0$")
    ax.set_ylabel(r"$p_2^0$")
    ax.set_title("Minimal Meta-MAPG warm-up", fontsize=10)
    ax.text(0.07, 0.08, "grey = no crossing", color="#555555", fontsize=7)
    ax = axes[1]
    fail = df[df["pg_success"] == 0]
    bins = np.arange(1, max(2, int(fail["first_pg_basin_step"].max()) + 2)) - 0.5
    ax.hist(
        fail.loc[(fail["meta_mapg_success"] == 1) & (fail["first_pg_basin_step"] > 0), "first_pg_basin_step"],
        bins=bins,
        alpha=0.75,
        label="gained",
        color=METHOD_COLORS["meta_mapg"],
    )
    ax.hist(
        fail.loc[(fail["meta_mapg_success"] == 0) & (fail["first_pg_basin_step"] > 0), "first_pg_basin_step"],
        bins=bins,
        alpha=0.55,
        label="not gained",
        color="#777777",
    )
    ax.set_xlabel("first PG-basin step")
    ax.set_ylabel("cells")
    ax.set_title("Crossing-time distribution", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = cfg.fig_outdir / "phase_y_min_warmup.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


# -------------------------------------------------------------------------
# Phase Z : adversarial alignment search over payoff geometry
# -------------------------------------------------------------------------
def mean_peer_projection_2x2(R: float, S: float, T: float, P: float, cfg: Config, grid_size: int, pg_mask: np.ndarray) -> tuple[float, float, int]:
    game = Game("payoff_geometry", np.array([[R, S], [T, P]], dtype=float), np.array([[R, T], [S, P]], dtype=float))
    grid = np.linspace(0.05, 0.95, grid_size)
    all_values = []
    fail_values = []
    first_lr = step_lr(cfg, 0)
    for i, p1 in enumerate(grid):
        for j, p2 in enumerate(grid):
            theta = np.array([[logit(float(p1))], [logit(float(p2))]], dtype=float)
            target = np.array([1.0 - p1, 1.0 - p2], dtype=float)
            target_unit = target / max(float(np.linalg.norm(target)), 1e-12)
            comps = exact_components_horizon1(theta, game, cfg.inner_lr)
            projection = first_lr * float(np.dot(probability_delta(theta, cfg.peer_coef * comps.peer), target_unit))
            all_values.append(projection)
            if not pg_mask[i, j]:
                fail_values.append(projection)
    return float(np.mean(all_values)), float(np.mean(fail_values)) if fail_values else float("nan"), len(fail_values)


def run_phase_z(
    cfg: Config,
    s_values: list[float],
    t_values: list[float],
    grid_size: int,
    steps: int,
) -> Path:
    rows: list[dict] = []
    for S in s_values:
        for T in t_values:
            R = 4.0
            P = 2.0
            if 2.0 * R <= T + S:
                continue
            pg_p1, pg_p2 = vectorized_rollout_2x2_final_probs(R, S, T, P, "standard_pg", cfg, grid_size, steps, cfg.peer_coef)
            mm_p1, mm_p2 = vectorized_rollout_2x2_final_probs(R, S, T, P, "meta_mapg", cfg, grid_size, steps, cfg.peer_coef)
            pg_mask = (pg_p1 >= cfg.success_threshold) & (pg_p2 >= cfg.success_threshold)
            mm_mask = (mm_p1 >= cfg.success_threshold) & (mm_p2 >= cfg.success_threshold)
            mean_all, mean_fail, n_fail = mean_peer_projection_2x2(R, S, T, P, cfg, grid_size, pg_mask)
            delta = float(mm_mask.mean() - pg_mask.mean())
            rows.append(
                {
                    "phase": "Z",
                    "R": R,
                    "S": float(S),
                    "T": float(T),
                    "P": P,
                    "pg_basin_fraction": float(pg_mask.mean()),
                    "meta_mapg_basin_fraction": float(mm_mask.mean()),
                    "delta_meta_minus_pg": delta,
                    "mean_peer_projection_all": mean_all,
                    "mean_peer_projection_pg_fail": mean_fail,
                    "n_pg_fail_cells": int(n_fail),
                    "positive_all_projection_but_hurts": int(mean_all > 0.0 and delta < 0.0),
                    "positive_fail_projection_but_hurts": int(mean_fail > 0.0 and delta < 0.0),
                    "negative_all_projection_but_helps": int(mean_all < 0.0 and delta > 0.0),
                }
            )
    df = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "phase": "Z",
                "n_games": int(len(df)),
                "n_hurt_games": int((df["delta_meta_minus_pg"] < 0).sum()),
                "n_positive_all_projection_but_hurts": int(df["positive_all_projection_but_hurts"].sum()),
                "n_positive_fail_projection_but_hurts": int(df["positive_fail_projection_but_hurts"].sum()),
                "n_negative_all_projection_but_helps": int(df["negative_all_projection_but_helps"].sum()),
                "corr_mean_all_delta": float(df["mean_peer_projection_all"].corr(df["delta_meta_minus_pg"], method="spearman")),
                "corr_mean_fail_delta": float(df["mean_peer_projection_pg_fail"].corr(df["delta_meta_minus_pg"], method="spearman")),
            }
        ]
    )
    out = cfg.outdir / "phase_z_alignment_adversary.csv"
    summary_out = cfg.outdir / "phase_z_alignment_adversary_summary.csv"
    df.to_csv(out, index=False)
    summary.to_csv(summary_out, index=False)
    plot_phase_z(df, cfg)
    return summary_out


def plot_phase_z(df: pd.DataFrame, cfg: Config) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    ax = axes[0]
    colors = np.where(df["delta_meta_minus_pg"] < 0.0, "#e45756", METHOD_COLORS["meta_mapg"])
    ax.scatter(df["mean_peer_projection_all"], df["delta_meta_minus_pg"], s=18, alpha=0.75, c=colors)
    ax.axhline(0.0, color="grey", linestyle=":", linewidth=0.9)
    ax.axvline(0.0, color="grey", linestyle=":", linewidth=0.9)
    ax.set_xlabel("mean first-step peer displacement")
    ax.set_ylabel("Meta-MAPG - PG basin")
    ax.set_title("Naive alignment stress test", fontsize=10)
    ax.grid(alpha=0.25)
    ax = axes[1]
    pivot = df.pivot_table(index="S", columns="T", values="positive_all_projection_but_hurts")
    z_cmap = ListedColormap(["#eeeeee", "#e45756"])
    z_cmap.set_bad("#ffffff")
    ax.imshow(
        pivot.to_numpy(),
        origin="lower",
        extent=(pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()),
        cmap=z_cmap,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xlabel("Temptation payoff T")
    ax.set_ylabel("Sucker payoff S")
    ax.set_title("positive local projection, negative basin delta", fontsize=10)
    ax.text(-0.9, -5.4, "red = counterexample, white = outside search constraint", fontsize=7)
    fig.tight_layout()
    out = cfg.fig_outdir / "phase_z_alignment_adversary.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


# =========================================================================
# Phase AA : asymmetric learner pairings (Meta-MAPG x PG)
# =========================================================================
def run_phase_aa(
    cfg: Config,
    grid_size: int,
    steps: int,
    batch_size: int,
) -> Path:
    game = stag_hunt()
    grid = np.linspace(0.05, 0.95, grid_size)
    pairings = [
        ("PG_PG", "standard_pg", "standard_pg"),
        ("MM_PG", "meta_mapg", "standard_pg"),
        ("PG_MM", "standard_pg", "meta_mapg"),
        ("MM_MM", "meta_mapg", "meta_mapg"),
    ]
    rows: list[dict] = []
    arrays_success: dict[str, np.ndarray] = {}
    arrays_welfare: dict[str, np.ndarray] = {}
    arrays_fairness: dict[str, np.ndarray] = {}
    for k, (label, m0, m1) in enumerate(pairings):
        success = np.zeros((grid_size, grid_size), dtype=int)
        welfare = np.zeros((grid_size, grid_size), dtype=float)
        fairness = np.zeros((grid_size, grid_size), dtype=float)
        for i, p1 in enumerate(grid):
            for j, p2 in enumerate(grid):
                init_theta = np.zeros((2, game.n_states), dtype=float)
                init_theta[0, 0] = logit(float(p1))
                init_theta[1, 0] = logit(float(p2))
                theta_final, _ = run_rollout_asymmetric(
                    game=game,
                    methods=(m0, m1),
                    seed=cfg.seed_base + 2_000_000 + 100_000 * k + 101 * i + 13 * j,
                    steps=steps,
                    batch_size=batch_size,
                    lr=cfg.lr,
                    inner_lr=cfg.inner_lr,
                    peer_coefs=(cfg.peer_coef, cfg.peer_coef),
                    own_coefs=(cfg.own_coef, cfg.own_coef),
                    init_theta=init_theta,
                    lr_power=cfg.lr_power,
                    lambda_power=0.0,
                    log_every=steps + 1,
                )
                ret = expected_return(theta_final, game)
                coop = cooperation_probs(theta_final, game)
                success[i, j] = int(is_success(theta_final, game, threshold=cfg.success_threshold))
                welfare[i, j] = float(np.sum(ret))
                fairness[i, j] = float(abs(ret[0] - ret[1]))
                rows.append(
                    {
                        "phase": "AA",
                        "pair_label": label,
                        "method0": m0,
                        "method1": m1,
                        "init_p1": float(p1),
                        "init_p2": float(p2),
                        "success": int(success[i, j]),
                        "final_p1": float(coop[0]),
                        "final_p2": float(coop[1]),
                        "welfare": float(welfare[i, j]),
                        "fairness_gap": float(fairness[i, j]),
                        "reward_p1": float(ret[0]),
                        "reward_p2": float(ret[1]),
                    }
                )
        arrays_success[label] = success
        arrays_welfare[label] = welfare
        arrays_fairness[label] = fairness
        np.save(cfg.outdir / f"phase_aa_success_{label}.npy", success)
        np.save(cfg.outdir / f"phase_aa_welfare_{label}.npy", welfare)
        np.save(cfg.outdir / f"phase_aa_fairness_{label}.npy", fairness)
    df = pd.DataFrame(rows)
    out = cfg.outdir / "phase_aa_pairings.csv"
    df.to_csv(out, index=False)
    np.save(cfg.outdir / "phase_aa_grid.npy", grid)
    plot_phase_aa(arrays_success, arrays_welfare, arrays_fairness, grid, df, cfg)
    return out


def plot_phase_aa(
    success: dict[str, np.ndarray],
    welfare: dict[str, np.ndarray],
    fairness: dict[str, np.ndarray],
    grid: np.ndarray,
    df: pd.DataFrame,
    cfg: Config,
) -> None:
    pretty = {
        "PG_PG": r"PG $\times$ PG",
        "MM_PG": r"Meta-MAPG $\times$ PG",
        "PG_MM": r"PG $\times$ Meta-MAPG",
        "MM_MM": r"Meta-MAPG $\times$ Meta-MAPG",
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.6))
    for ax, label in zip(axes.flat, ["PG_PG", "MM_PG", "PG_MM", "MM_MM"]):
        ax.imshow(
            success[label].T,
            origin="lower",
            extent=(grid[0], grid[-1], grid[0], grid[-1]),
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            aspect="equal",
            interpolation="nearest",
        )
        ax.scatter([1.0], [1.0], marker="*", s=110, color="black", zorder=5)
        ax.scatter([0.0], [0.0], marker="X", s=70, color="black", zorder=5)
        frac = float(np.mean(success[label]))
        ax.set_title(f"{pretty[label]}: coop = {frac*100:.1f}%", fontsize=10)
        ax.set_xlabel(r"$p_1^0$")
        ax.set_ylabel(r"$p_2^0$")
    fig.tight_layout()
    out = cfg.fig_outdir / "phase_aa_basin_atlas.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))
    labels = ["PG_PG", "MM_PG", "PG_MM", "MM_MM"]
    colors = ["#4c78a8", "#b279a2", "#f58518", "#54a24b"]
    coop_rates = [float(df[df.pair_label == l]["success"].mean()) for l in labels]
    welfares = [float(df[df.pair_label == l]["welfare"].mean()) for l in labels]
    fairnesses = [float(df[df.pair_label == l]["fairness_gap"].mean()) for l in labels]

    axes[0].bar(labels, coop_rates, color=colors)
    axes[0].set_ylabel("Cooperative success rate")
    axes[0].set_ylim(0, 1.02)
    axes[0].tick_params(axis="x", labelrotation=20, labelsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_title("Cooperation", fontsize=10)

    axes[1].bar(labels, welfares, color=colors)
    axes[1].set_ylabel("Mean joint welfare")
    axes[1].tick_params(axis="x", labelrotation=20, labelsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_title("Welfare", fontsize=10)

    axes[2].bar(labels, fairnesses, color=colors)
    axes[2].set_ylabel(r"Mean $|u_1 - u_2|$")
    axes[2].tick_params(axis="x", labelrotation=20, labelsize=8)
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].set_title("Exploitation gap", fontsize=10)

    fig.tight_layout()
    out = cfg.fig_outdir / "phase_aa_metrics.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


# =========================================================================
# Phase BB : heterogeneous peer coefficients (lambda_1, lambda_2)
# =========================================================================
def run_phase_bb(
    cfg: Config,
    lambdas: list[float],
    grid_size: int,
    steps: int,
    batch_size: int,
) -> Path:
    game = stag_hunt()
    grid = np.linspace(0.05, 0.95, grid_size)
    rows: list[dict] = []
    coop_grid = np.zeros((len(lambdas), len(lambdas)), dtype=float)
    welfare_grid = np.zeros_like(coop_grid)
    fairness_grid = np.zeros_like(coop_grid)
    for li, l0 in enumerate(lambdas):
        for lj, l1 in enumerate(lambdas):
            n_succ = 0
            sum_w = 0.0
            sum_f = 0.0
            n_total = 0
            for i, p1 in enumerate(grid):
                for j, p2 in enumerate(grid):
                    init_theta = np.zeros((2, game.n_states), dtype=float)
                    init_theta[0, 0] = logit(float(p1))
                    init_theta[1, 0] = logit(float(p2))
                    theta_final, _ = run_rollout_asymmetric(
                        game=game,
                        methods=("meta_mapg", "meta_mapg"),
                        seed=cfg.seed_base + 2_100_000
                        + 30_000 * li + 1_000 * lj + 101 * i + 13 * j,
                        steps=steps,
                        batch_size=batch_size,
                        lr=cfg.lr,
                        inner_lr=cfg.inner_lr,
                        peer_coefs=(float(l0), float(l1)),
                        own_coefs=(cfg.own_coef, cfg.own_coef),
                        init_theta=init_theta,
                        lr_power=cfg.lr_power,
                        lambda_power=0.0,
                        log_every=steps + 1,
                    )
                    ret = expected_return(theta_final, game)
                    n_succ += int(is_success(theta_final, game, threshold=cfg.success_threshold))
                    sum_w += float(np.sum(ret))
                    sum_f += float(abs(ret[0] - ret[1]))
                    n_total += 1
            coop_grid[li, lj] = n_succ / n_total
            welfare_grid[li, lj] = sum_w / n_total
            fairness_grid[li, lj] = sum_f / n_total
            rows.append(
                {
                    "phase": "BB",
                    "lambda0": float(l0),
                    "lambda1": float(l1),
                    "coop_basin_fraction": coop_grid[li, lj],
                    "mean_welfare": welfare_grid[li, lj],
                    "fairness_gap": fairness_grid[li, lj],
                    "n_cells": int(n_total),
                }
            )
    df = pd.DataFrame(rows)
    out = cfg.outdir / "phase_bb_hetero.csv"
    df.to_csv(out, index=False)
    np.save(cfg.outdir / "phase_bb_lambdas.npy", np.array(lambdas, dtype=float))
    np.save(cfg.outdir / "phase_bb_coop_grid.npy", coop_grid)
    np.save(cfg.outdir / "phase_bb_welfare_grid.npy", welfare_grid)
    np.save(cfg.outdir / "phase_bb_fairness_grid.npy", fairness_grid)
    plot_phase_bb(coop_grid, welfare_grid, fairness_grid, lambdas, cfg)
    return out


def plot_phase_bb(
    coop: np.ndarray,
    welfare: np.ndarray,
    fairness: np.ndarray,
    lambdas: list[float],
    cfg: Config,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2))
    for ax, mat, title, cmap, vmin, vmax in [
        (axes[0], coop, "Cooperative basin fraction", "viridis", 0.0, 1.0),
        (axes[1], welfare, "Mean joint welfare", "magma", None, None),
        (axes[2], fairness, r"Mean $|u_1 - u_2|$", "Reds", 0.0, None),
    ]:
        im = ax.imshow(
            mat,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_xticks(range(len(lambdas)))
        ax.set_yticks(range(len(lambdas)))
        ax.set_xticklabels([f"{l:g}" for l in lambdas], fontsize=8)
        ax.set_yticklabels([f"{l:g}" for l in lambdas], fontsize=8)
        ax.set_xlabel(r"$\lambda_2$")
        ax.set_ylabel(r"$\lambda_1$")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    out = cfg.fig_outdir / "phase_bb_hetero.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


# =========================================================================
# Phase CC : unroll-length sweep L in {1, 3, 5} (tabular IPD + MLP IPD)
# =========================================================================
def run_phase_cc_tabular(
    cfg: Config,
    L_values: list[int],
    n_seeds: int,
    n_steps: int,
    batch_size: int,
) -> Path:
    from run_meta_mapg_experiments import prisoners_dilemma  # local import

    game = prisoners_dilemma()
    methods = ["standard_pg", "meta_pg", "lola_style", "meta_mapg"]
    rows: list[dict] = []
    for L in L_values:
        for method in methods:
            for seed in range(n_seeds):
                rng = np.random.default_rng(
                    cfg.seed_base + 2_200_000 + 1_000 * int(L) + 100 * methods.index(method) + seed
                )
                theta = rng.normal(loc=0.0, scale=1.35, size=(2, game.n_states))
                eff_L = 1 if method == "standard_pg" else int(L)
                for step in range(n_steps):
                    comps = estimate_components_L(theta, game, batch_size, rng, cfg.inner_lr, eff_L)
                    lr_step = cfg.lr / ((step + 10.0) ** cfg.lr_power)
                    update = update_from_components(comps, method, cfg.peer_coef, cfg.own_coef)
                    theta = np.clip(theta + lr_step * update, -8.0, 8.0)
                ret = expected_return(theta, game)
                coop = cooperation_probs(theta, game)
                rows.append(
                    {
                        "phase": "CC_tab",
                        "L": int(L),
                        "method": method,
                        "seed": seed,
                        "success": int(is_success(theta, game, threshold=cfg.success_threshold)),
                        "final_coop_min": float(np.min(coop)),
                        "final_welfare": float(np.sum(ret)),
                    }
                )
    df = pd.DataFrame(rows)
    out = cfg.outdir / "phase_cc_unroll_tab.csv"
    df.to_csv(out, index=False)
    plot_phase_cc_tabular(df, cfg)
    return out


def plot_phase_cc_tabular(df: pd.DataFrame, cfg: Config) -> None:
    methods = ["standard_pg", "meta_pg", "lola_style", "meta_mapg"]
    L_values = sorted(df["L"].unique().tolist())
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for method in methods:
        rates = []
        ci_lo = []
        ci_hi = []
        for L in L_values:
            sub = df[(df["method"] == method) & (df["L"] == L)]
            k = int(sub["success"].sum())
            n = int(len(sub))
            if n == 0:
                rates.append(float("nan"))
                ci_lo.append(0.0)
                ci_hi.append(0.0)
                continue
            from run_meta_mapg_experiments import _wilson_ci
            p, lo, hi = _wilson_ci(k, n)
            rates.append(p)
            ci_lo.append(p - lo)
            ci_hi.append(hi - p)
        ax.errorbar(
            L_values,
            rates,
            yerr=[ci_lo, ci_hi],
            marker="o",
            capsize=3,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            linewidth=1.8,
        )
    ax.set_xlabel("Inner unroll length L")
    ax.set_ylabel("Cooperative success rate")
    ax.set_xticks(L_values)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_title(r"Tabular IPD: own-vs-peer separation across $L$", fontsize=10)
    fig.tight_layout()
    out = cfg.fig_outdir / "phase_cc_unroll_tab.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


def run_phase_cc_mlp(
    cfg: Config,
    L_values: list[int],
    n_seeds: int,
    n_steps: int,
    batch_size: int,
) -> Path:
    from run_mlp_ipd import train_one_seed  # deferred: heavy torch import

    methods = ["standard_pg", "meta_pg", "lola_style", "meta_mapg"]
    rows: list[dict] = []
    for L in L_values:
        for method in methods:
            for seed in range(n_seeds):
                eff_L = 1 if method == "standard_pg" else int(L)
                r = train_one_seed(
                    method=method,
                    seed=seed + 10_000 * int(L),
                    n_steps=n_steps,
                    batch_size=batch_size,
                    peer_coef=cfg.peer_coef,
                    own_coef=cfg.own_coef,
                    lr=cfg.lr,
                    lr_power=cfg.lr_power,
                    inner_lr=cfg.inner_lr,
                    inner_unroll=eff_L,
                )
                rows.append(
                    {
                        "phase": "CC_mlp",
                        "L": int(L),
                        "method": method,
                        "seed": seed,
                        "success": int(r["success"]),
                        "final_coop": float(r["final_coop"]),
                        "final_return": float((r["final_return0"] + r["final_return1"]) / 2.0),
                    }
                )
    df = pd.DataFrame(rows)
    out = cfg.outdir / "phase_cc_unroll_mlp.csv"
    df.to_csv(out, index=False)
    plot_phase_cc_mlp(df, cfg)
    return out


def plot_phase_cc_mlp(df: pd.DataFrame, cfg: Config) -> None:
    methods = ["standard_pg", "meta_pg", "lola_style", "meta_mapg"]
    L_values = sorted(df["L"].unique().tolist())
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for method in methods:
        rates = []
        ci_lo = []
        ci_hi = []
        for L in L_values:
            sub = df[(df["method"] == method) & (df["L"] == L)]
            k = int(sub["success"].sum())
            n = int(len(sub))
            if n == 0:
                rates.append(float("nan"))
                ci_lo.append(0.0)
                ci_hi.append(0.0)
                continue
            from run_meta_mapg_experiments import _wilson_ci
            p, lo, hi = _wilson_ci(k, n)
            rates.append(p)
            ci_lo.append(p - lo)
            ci_hi.append(hi - p)
        ax.errorbar(
            L_values,
            rates,
            yerr=[ci_lo, ci_hi],
            marker="o",
            capsize=3,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            linewidth=1.8,
        )
    ax.set_xlabel("Inner unroll length L")
    ax.set_ylabel("Cooperative success rate")
    ax.set_xticks(L_values)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_title(r"MLP IPD: own-vs-peer separation across $L$", fontsize=10)
    fig.tight_layout()
    out = cfg.fig_outdir / "phase_cc_unroll_mlp.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


# =========================================================================
# Phase DD : cooldown annealing-rate q sweep
# =========================================================================
def run_phase_dd(
    cfg: Config,
    q_values: list[float],
    n_seeds: int,
    n0: int,
    total_steps: int,
    scale: int,
    batch_size: int,
) -> Path:
    game = stag_hunt()
    rng_master = np.random.default_rng(cfg.seed_base + 2_300_000)
    init_thetas = [
        rng_master.normal(loc=0.0, scale=1.35, size=(2, game.n_states))
        for _ in range(n_seeds)
    ]
    rows: list[dict] = []
    for qi, q in enumerate(q_values):
        for seed in range(n_seeds):
            theta = init_thetas[seed].copy()
            rng = np.random.default_rng(cfg.seed_base + 2_310_000 + 1_000 * qi + seed)
            coop_trace: list[float] = []
            for step in range(total_steps):
                comps = estimate_components(theta, game, batch_size, rng, cfg.inner_lr)
                lr_step = cfg.lr / ((step + 10.0) ** cfg.lr_power)
                if step < n0:
                    lam_step = cfg.peer_coef
                else:
                    lam_step = cfg.peer_coef / (
                        (1.0 + (step - n0) / float(max(1, scale))) ** float(q)
                    )
                update = update_from_components(comps, "meta_mapg", lam_step, cfg.own_coef)
                theta = np.clip(theta + lr_step * update, -8.0, 8.0)
                coop_trace.append(float(np.min(cooperation_probs(theta, game))))
            arr = np.array(coop_trace)
            second_half = arr[total_steps // 2:]
            rows.append(
                {
                    "phase": "DD",
                    "q": float(q),
                    "seed": seed,
                    "final_coop_min": float(arr[-1]),
                    "second_half_coop_mean": float(np.mean(second_half)),
                    "second_half_coop_std": float(np.std(second_half, ddof=1)) if second_half.size > 1 else 0.0,
                    "success": int(arr[-1] >= cfg.success_threshold),
                }
            )
    df = pd.DataFrame(rows)
    out = cfg.outdir / "phase_dd_qsweep.csv"
    df.to_csv(out, index=False)
    plot_phase_dd(df, cfg)
    return out


def plot_phase_dd(df: pd.DataFrame, cfg: Config) -> None:
    from run_meta_mapg_experiments import _wilson_ci
    qs = sorted(df["q"].unique().tolist())
    succ_p = []
    succ_lo = []
    succ_hi = []
    stab = []
    for q in qs:
        sub = df[df["q"] == q]
        k = int(sub["success"].sum())
        n = int(len(sub))
        p, lo, hi = _wilson_ci(k, n)
        succ_p.append(p)
        succ_lo.append(p - lo)
        succ_hi.append(hi - p)
        stab.append(float(sub["second_half_coop_std"].mean()))
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    ax = axes[0]
    ax.errorbar(qs, succ_p, yerr=[succ_lo, succ_hi], marker="o", capsize=3,
                color=METHOD_COLORS["meta_mapg"], linewidth=1.8)
    ax.set_xlabel(r"Annealing exponent $q$")
    ax.set_ylabel("Cooperative success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Phase 2 robustness to $q$", fontsize=10)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(qs, stab, marker="s", color="#2fbf71", linewidth=1.8)
    ax.set_xlabel(r"Annealing exponent $q$")
    ax.set_ylabel(r"Second-half min-coop std")
    ax.set_title("Endpoint stability vs $q$", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = cfg.fig_outdir / "phase_dd_qsweep.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


# =========================================================================
# Phase EE : checkpoint-selection rules during warm-up
# =========================================================================
def run_phase_ee(
    cfg: Config,
    rules: list[str],
    n_seeds: int,
    warm_steps: int,
    cool_steps: int,
    batch_size: int,
    checkpoint_every: int,
) -> Path:
    # warm-up and cooldown seeds are disjoint (warm: +2_400_000, cool: +2_450_000)
    # so checkpoint-rule selection cannot overfit to the cooldown evaluation noise.
    game = stag_hunt()
    rng_master = np.random.default_rng(cfg.seed_base + 2_400_000)
    init_thetas = [
        rng_master.normal(loc=0.0, scale=1.35, size=(2, game.n_states))
        for _ in range(n_seeds)
    ]
    rows: list[dict] = []
    for seed in range(n_seeds):
        warm_seed = cfg.seed_base + 2_400_000 + 100 * seed
        cool_seed = cfg.seed_base + 2_450_000 + 100 * seed
        _, ckpts = run_rollout_with_checkpoints(
            game=game,
            method="meta_mapg",
            seed=warm_seed,
            steps=warm_steps,
            batch_size=batch_size,
            lr=cfg.lr,
            inner_lr=cfg.inner_lr,
            peer_coef=cfg.peer_coef,
            own_coef=cfg.own_coef,
            init_theta=init_thetas[seed],
            lr_power=cfg.lr_power,
            lambda_power=0.0,
            checkpoint_every=checkpoint_every,
        )
        rule_rng = np.random.default_rng(warm_seed + 7919)
        for rule in rules:
            init_cool = select_checkpoint(ckpts, rule, rng=rule_rng)
            theta_final, _ = run_rollout(
                game=game,
                method="standard_pg",
                seed=cool_seed,
                steps=cool_steps,
                batch_size=batch_size,
                lr=cfg.lr,
                inner_lr=cfg.inner_lr,
                peer_coef=0.0,
                own_coef=cfg.own_coef,
                init_theta=init_cool,
                lr_power=cfg.lr_power,
                lambda_power=0.0,
                log_every=cool_steps + 1,
            )
            ret = expected_return(theta_final, game)
            coop = cooperation_probs(theta_final, game)
            rows.append(
                {
                    "phase": "EE",
                    "rule": rule,
                    "seed": seed,
                    "post_cooldown_coop_min": float(np.min(coop)),
                    "post_cooldown_welfare": float(np.sum(ret)),
                    "post_cooldown_success": int(is_success(theta_final, game, threshold=cfg.success_threshold)),
                }
            )
    df = pd.DataFrame(rows)
    out = cfg.outdir / "phase_ee_checkpoint.csv"
    df.to_csv(out, index=False)
    plot_phase_ee(df, cfg)
    return out


def plot_phase_ee(df: pd.DataFrame, cfg: Config) -> None:
    from run_meta_mapg_experiments import _wilson_ci
    rules = list(df["rule"].unique())
    short_labels = {
        "final": "final",
        "best_coopmin": "best\ncoopmin",
        "best_welfare": "best\nwelfare",
        "lowest_update_norm_high_welfare": "low-norm\nhigh-welf",
        "random": "random",
    }
    xlabels = [short_labels.get(r, r) for r in rules]
    rates, ci_lo, ci_hi, welfares = [], [], [], []
    for r in rules:
        sub = df[df["rule"] == r]
        k = int(sub["post_cooldown_success"].sum())
        n = int(len(sub))
        p, lo, hi = _wilson_ci(k, n)
        rates.append(p)
        ci_lo.append(p - lo)
        ci_hi.append(hi - p)
        welfares.append(float(sub["post_cooldown_welfare"].mean()))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"][: len(rules)]
    xs = range(len(rules))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))

    ax = axes[0]
    ax.bar(xs, rates, yerr=[ci_lo, ci_hi], color=colors, capsize=4, width=0.6)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(xlabels, fontsize=8.5)
    ax.set_ylabel("Post-cooldown success rate")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Selection-rule effect on basin entry", fontsize=10)

    ax = axes[1]
    w_min = min(welfares) - 0.05
    w_max = max(welfares) + 0.05
    ax.bar(xs, welfares, color=colors, width=0.6)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(xlabels, fontsize=8.5)
    ax.set_ylabel("Mean post-cooldown welfare")
    ax.set_ylim(w_min, w_max)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Selection-rule effect on welfare", fontsize=10)

    fig.tight_layout()
    out = cfg.fig_outdir / "phase_ee_checkpoint.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


# =========================================================================
# Phase FF : policy perturbation vs environment reset
# =========================================================================
def run_phase_ff(
    cfg: Config,
    n_seeds: int,
    steps_per_attempt: int,
    K_budget: int,
    batch_size: int,
    perturb_low: float,
    perturb_high: float,
) -> Path:
    game = stag_hunt()
    rng_master = np.random.default_rng(cfg.seed_base + 2_500_000)
    init_thetas = [
        rng_master.normal(loc=0.0, scale=1.35, size=(2, game.n_states))
        for _ in range(n_seeds)
    ]
    conditions = ["episode_only", "perturb_low", "perturb_high", "reinit", "ckpt_warm"]
    rows: list[dict] = []
    for seed in range(n_seeds):
        for cond in conditions:
            rng = np.random.default_rng(cfg.seed_base + 2_510_000 + 100 * seed + conditions.index(cond))
            ever_success = False
            first_hit = K_budget + 1
            theta = init_thetas[seed].copy()

            if cond == "ckpt_warm":
                _, ckpts = run_rollout_with_checkpoints(
                    game=game,
                    method="meta_mapg",
                    seed=cfg.seed_base + 2_520_000 + 100 * seed,
                    steps=steps_per_attempt,
                    batch_size=batch_size,
                    lr=cfg.lr,
                    inner_lr=cfg.inner_lr,
                    peer_coef=cfg.peer_coef,
                    own_coef=cfg.own_coef,
                    init_theta=init_thetas[seed],
                    lr_power=cfg.lr_power,
                    lambda_power=0.0,
                    checkpoint_every=max(1, steps_per_attempt // 20),
                )
                warm_theta = select_checkpoint(ckpts, "best_coopmin")

            for k in range(1, K_budget + 1):
                if cond == "episode_only":
                    theta_attempt = theta
                elif cond == "perturb_low":
                    theta_attempt = perturb_theta(theta, rng, perturb_low) if k > 1 else theta
                elif cond == "perturb_high":
                    theta_attempt = perturb_theta(theta, rng, perturb_high) if k > 1 else theta
                elif cond == "reinit":
                    theta_attempt = rng.uniform(low=-3.0, high=3.0, size=(2, game.n_states))
                elif cond == "ckpt_warm":
                    theta_attempt = warm_theta + rng.normal(0.0, 0.05, size=warm_theta.shape) if k > 1 else warm_theta
                else:
                    raise ValueError(cond)
                theta_after, _ = run_rollout(
                    game=game,
                    method="meta_mapg",
                    seed=int(rng.integers(0, 2**31 - 1)),
                    steps=steps_per_attempt,
                    batch_size=batch_size,
                    lr=cfg.lr,
                    inner_lr=cfg.inner_lr,
                    peer_coef=cfg.peer_coef,
                    own_coef=cfg.own_coef,
                    init_theta=theta_attempt,
                    lr_power=cfg.lr_power,
                    lambda_power=0.0,
                    log_every=steps_per_attempt + 1,
                )
                theta = theta_after  # episode_only path uses this for next iteration
                if is_success(theta_after, game, threshold=cfg.success_threshold):
                    ever_success = True
                    if first_hit == K_budget + 1:
                        first_hit = k
                rows.append(
                    {
                        "phase": "FF",
                        "condition": cond,
                        "seed": seed,
                        "k": int(k),
                        "success_so_far": int(ever_success),
                        "first_hit": int(first_hit),
                    }
                )
    df = pd.DataFrame(rows)
    out = cfg.outdir / "phase_ff_restart.csv"
    df.to_csv(out, index=False)
    plot_phase_ff(df, K_budget, cfg)
    return out


def plot_phase_ff(df: pd.DataFrame, K_budget: int, cfg: Config) -> None:
    conditions = ["episode_only", "perturb_low", "perturb_high", "reinit", "ckpt_warm"]
    pretty = {
        "episode_only": "episode reset only",
        "perturb_low": r"perturb $\sigma=0.1$",
        "perturb_high": r"perturb $\sigma=0.5$",
        "reinit": "full reinit",
        "ckpt_warm": "ckpt warm-start",
    }
    colors = {
        "episode_only": "#e45756",
        "perturb_low": "#f58518",
        "perturb_high": "#b279a2",
        "reinit": "#4c78a8",
        "ckpt_warm": "#2fbf71",
    }
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for cond in conditions:
        sub = df[df["condition"] == cond]
        rates = []
        for k in range(1, K_budget + 1):
            sub_k = sub[sub["k"] == k]
            n = sub_k["seed"].nunique()
            if n == 0:
                rates.append(float("nan"))
                continue
            ever = sub_k.groupby("seed")["success_so_far"].max()
            rates.append(float(ever.mean()))
        ax.plot(range(1, K_budget + 1), rates, marker="o",
                color=colors[cond], linewidth=1.8, label=pretty[cond])
    ax.set_xlabel("Restart budget K")
    ax.set_ylabel("P(reach cooperation by k)")
    ax.set_xlim(1, K_budget)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.25)
    ax.set_title("Episode reset is not policy restart", fontsize=10)
    fig.tight_layout()
    out = cfg.fig_outdir / "phase_ff_restart.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)


# -------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=[
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "L", "A2", "D2",
            "M", "N", "O", "P", "Q", "R", "T", "U", "V", "W", "X", "Y", "Z",
            "AA", "BB", "CC", "DD", "EE", "FF",
            "all",
        ],
    )
    p.add_argument("--outdir", type=Path, default=Path("artifacts/validation"))
    p.add_argument("--fig-outdir", type=Path, default=Path("figures/validation"))
    # Sizing
    p.add_argument("--phase-a-grid", type=int, default=15)
    p.add_argument("--phase-a-batch", type=int, default=8192)
    p.add_argument("--phase-a-reps", type=int, default=4)
    p.add_argument("--phase-b-grid", type=int, default=51)
    p.add_argument("--phase-b-steps", type=int, default=140)
    p.add_argument("--phase-b-batch", type=int, default=192)
    p.add_argument("--phase-c-grid", type=int, default=21)
    p.add_argument("--phase-c-steps", type=int, default=140)
    p.add_argument("--phase-c-batch", type=int, default=128)
    p.add_argument("--phase-c-seeds", type=int, default=10)
    p.add_argument("--phase-d-seeds", type=int, default=80)
    p.add_argument("--phase-d-n0", type=int, default=100)
    p.add_argument("--phase-d-total", type=int, default=260)
    p.add_argument("--phase-d-scale", type=int, default=30)
    p.add_argument("--phase-d-q", type=float, default=0.7)
    p.add_argument("--phase-d-batch", type=int, default=256)
    p.add_argument(
        "--phase-e-temptations",
        type=float,
        nargs="+",
        default=[2.2, 2.5, 3.0, 3.5, 3.8],
    )
    p.add_argument(
        "--phase-e-methods",
        type=str,
        nargs="+",
        default=["standard_pg", "lola_style", "meta_mapg"],
    )
    p.add_argument("--phase-e-grid", type=int, default=21)
    p.add_argument("--phase-e-steps", type=int, default=140)
    p.add_argument("--phase-e-batch", type=int, default=192)
    p.add_argument("--phase-f-thresholds", type=float, nargs="+", default=[0.75, 0.82, 0.90])
    # Phase G
    p.add_argument("--phase-g-seeds", type=int, default=5)
    p.add_argument("--phase-g-grid", type=int, default=21)
    p.add_argument("--phase-g-steps", type=int, default=140)
    p.add_argument("--phase-g-batch", type=int, default=192)
    p.add_argument("--phase-g-temptations", type=float, nargs="+", default=[2.2, 2.5, 3.0, 3.5, 3.8])
    p.add_argument("--phase-g-methods", type=str, nargs="+", default=["standard_pg", "lola_style", "meta_mapg"])
    # Phase H
    p.add_argument("--phase-h-grids", type=int, nargs="+", default=[11, 21, 51])
    p.add_argument("--phase-h-steps", type=int, default=140)
    p.add_argument("--phase-h-batch", type=int, default=192)
    # Phase I
    p.add_argument("--phase-i-grid", type=int, default=51)
    p.add_argument("--phase-i-steps", type=int, default=140)
    p.add_argument("--phase-i-batch", type=int, default=192)
    # Phase D2
    p.add_argument("--phase-d2-seeds", type=int, default=80)
    p.add_argument("--phase-d2-n0", type=int, default=100)
    p.add_argument("--phase-d2-total", type=int, default=260)
    p.add_argument("--phase-d2-scale", type=int, default=30)
    p.add_argument("--phase-d2-q", type=float, default=0.7)
    p.add_argument("--phase-d2-batch", type=int, default=256)
    # Phase M
    p.add_argument("--phase-m-lambdas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    p.add_argument("--phase-m-grid", type=int, default=21)
    p.add_argument("--phase-m-steps", type=int, default=140)
    p.add_argument("--phase-m-batch", type=int, default=192)
    # Phase N
    p.add_argument("--phase-n-grid", type=int, default=21)
    p.add_argument("--phase-n-steps", type=int, default=140)
    p.add_argument("--phase-n-batch", type=int, default=192)
    # Phase Q
    p.add_argument("--phase-q-seeds", type=int, default=5)
    p.add_argument("--phase-q-grid", type=int, default=21)
    p.add_argument("--phase-q-steps", type=int, default=140)
    p.add_argument("--phase-q-batch", type=int, default=192)
    p.add_argument("--phase-q-temptations", type=float, nargs="+", default=[2.2, 2.5, 3.0, 3.5, 3.8])
    p.add_argument("--phase-q-methods", type=str, nargs="+", default=METHODS_FULL)
    # Phase T
    p.add_argument("--phase-t-grid", type=int, default=21)
    p.add_argument("--phase-t-total", type=int, default=140)
    p.add_argument("--phase-t-warm-steps", type=int, nargs="+", default=[5, 10, 25, 50, 100])
    # Phase U
    p.add_argument("--phase-u-lambdas", type=float, nargs="+", default=[0.0, 1.5, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0, 80.0])
    p.add_argument("--phase-u-grid", type=int, default=21)
    p.add_argument("--phase-u-total", type=int, default=200)
    # Phase V
    p.add_argument("--phase-v-s-values", type=float, nargs="+", default=[-6.0, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    p.add_argument("--phase-v-t-values", type=float, nargs="+", default=[-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5])
    p.add_argument("--phase-v-grid", type=int, default=15)
    p.add_argument("--phase-v-steps", type=int, default=100)
    # Phases X/Y/Z
    p.add_argument("--phase-x-k-values", type=int, nargs="+", default=[1, 5, 10, 25, 50])
    p.add_argument("--phase-y-max-steps", type=int, default=100)
    p.add_argument("--phase-z-s-values", type=float, nargs="+", default=[-6.0, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    p.add_argument("--phase-z-t-values", type=float, nargs="+", default=[-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5])
    p.add_argument("--phase-z-grid", type=int, default=15)
    p.add_argument("--phase-z-steps", type=int, default=100)
    # Phase AA
    p.add_argument("--phase-aa-grid", type=int, default=21)
    p.add_argument("--phase-aa-steps", type=int, default=1000)
    p.add_argument("--phase-aa-batch", type=int, default=192)
    # Phase BB
    p.add_argument("--phase-bb-lambdas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 5.0])
    p.add_argument("--phase-bb-grid", type=int, default=11)
    p.add_argument("--phase-bb-steps", type=int, default=1000)
    p.add_argument("--phase-bb-batch", type=int, default=192)
    # Phase CC
    p.add_argument("--phase-cc-Ls", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--phase-cc-tabular-seeds", type=int, default=40)
    p.add_argument("--phase-cc-tabular-steps", type=int, default=500)
    p.add_argument("--phase-cc-tabular-batch", type=int, default=192)
    p.add_argument("--phase-cc-mlp-seeds", type=int, default=30)
    p.add_argument("--phase-cc-mlp-steps", type=int, default=500)
    p.add_argument("--phase-cc-mlp-batch", type=int, default=64)
    # Phase DD
    p.add_argument("--phase-dd-qs", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0])
    p.add_argument("--phase-dd-seeds", type=int, default=80)
    p.add_argument("--phase-dd-n0", type=int, default=100)
    p.add_argument("--phase-dd-total", type=int, default=2000)
    p.add_argument("--phase-dd-scale", type=int, default=30)
    p.add_argument("--phase-dd-batch", type=int, default=256)
    # Phase EE
    p.add_argument(
        "--phase-ee-rules",
        type=str,
        nargs="+",
        default=["final", "best_coopmin", "best_welfare", "lowest_update_norm_high_welfare", "random"],
    )
    p.add_argument("--phase-ee-seeds", type=int, default=80)
    p.add_argument("--phase-ee-warm-steps", type=int, default=1000)
    p.add_argument("--phase-ee-cool-steps", type=int, default=500)
    p.add_argument("--phase-ee-batch", type=int, default=192)
    p.add_argument("--phase-ee-checkpoint-every", type=int, default=50)
    # Phase FF
    p.add_argument("--phase-ff-seeds", type=int, default=80)
    p.add_argument("--phase-ff-steps", type=int, default=500)
    p.add_argument("--phase-ff-K", type=int, default=8)
    p.add_argument("--phase-ff-batch", type=int, default=192)
    p.add_argument("--phase-ff-perturb-low", type=float, default=0.1)
    p.add_argument("--phase-ff-perturb-high", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(outdir=args.outdir, fig_outdir=args.fig_outdir)
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    cfg.fig_outdir.mkdir(parents=True, exist_ok=True)

    log_path = cfg.outdir / "run_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(event: dict) -> None:
        event["t"] = now_str()
        with log_path.open("a") as fh:
            fh.write(json.dumps(event) + "\n")

    phases = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "L", "A2", "D2",
        "M", "N", "O", "P", "Q", "R", "T", "U", "V", "W", "X", "Y", "Z",
        "AA", "BB", "CC", "DD", "EE", "FF",
    ] if args.phase == "all" else [args.phase]
    log({"event": "start", "phases": phases, "args": vars(args) | {"outdir": str(args.outdir), "fig_outdir": str(args.fig_outdir)}})

    if "A" in phases:
        log({"event": "phase_start", "phase": "A"})
        out = run_phase_a(cfg, args.phase_a_grid, args.phase_a_batch, args.phase_a_reps)
        log({"event": "phase_done", "phase": "A", "csv": str(out)})

    if "B" in phases:
        log({"event": "phase_start", "phase": "B"})
        out = run_phase_b(cfg, args.phase_b_grid, args.phase_b_steps, args.phase_b_batch)
        log({"event": "phase_done", "phase": "B", "csv": str(out)})

    if "C" in phases:
        log({"event": "phase_start", "phase": "C"})
        out = run_phase_c(cfg, args.phase_c_grid, args.phase_c_steps, args.phase_c_batch, args.phase_c_seeds)
        log({"event": "phase_done", "phase": "C", "csv": str(out)})

    if "D" in phases:
        log({"event": "phase_start", "phase": "D"})
        out = run_phase_d(
            cfg,
            args.phase_d_seeds,
            args.phase_d_n0,
            args.phase_d_total,
            args.phase_d_scale,
            args.phase_d_q,
            args.phase_d_batch,
        )
        log({"event": "phase_done", "phase": "D", "csv": str(out)})

    if "E" in phases:
        log({"event": "phase_start", "phase": "E"})
        out = run_phase_e(
            cfg,
            args.phase_e_temptations,
            args.phase_e_methods,
            args.phase_e_grid,
            args.phase_e_steps,
            args.phase_e_batch,
        )
        log({"event": "phase_done", "phase": "E", "csv": str(out)})

    if "F" in phases:
        log({"event": "phase_start", "phase": "F"})
        out = run_phase_f(cfg, args.phase_f_thresholds)
        log({"event": "phase_done", "phase": "F", "csv": str(out)})

    if "A2" in phases:
        log({"event": "phase_start", "phase": "A2"})
        out = run_phase_a2(cfg)
        log({"event": "phase_done", "phase": "A2", "csv": str(out)})

    if "G" in phases:
        log({"event": "phase_start", "phase": "G"})
        out = run_phase_g(
            cfg, args.phase_g_temptations, args.phase_g_methods,
            args.phase_g_grid, args.phase_g_steps, args.phase_g_batch, args.phase_g_seeds,
        )
        log({"event": "phase_done", "phase": "G", "csv": str(out)})

    if "H" in phases:
        log({"event": "phase_start", "phase": "H"})
        out = run_phase_h(cfg, args.phase_h_grids, args.phase_h_steps, args.phase_h_batch)
        log({"event": "phase_done", "phase": "H", "csv": str(out)})

    if "I" in phases:
        log({"event": "phase_start", "phase": "I"})
        out = run_phase_i(cfg, args.phase_i_grid, args.phase_i_steps, args.phase_i_batch)
        log({"event": "phase_done", "phase": "I", "csv": str(out)})

    if "L" in phases:
        log({"event": "phase_start", "phase": "L"})
        out = run_phase_l(cfg)
        log({"event": "phase_done", "phase": "L", "csv": str(out)})

    if "D2" in phases:
        log({"event": "phase_start", "phase": "D2"})
        out = run_phase_d2(
            cfg, args.phase_d2_seeds, args.phase_d2_n0, args.phase_d2_total,
            args.phase_d2_scale, args.phase_d2_q, args.phase_d2_batch,
        )
        log({"event": "phase_done", "phase": "D2", "csv": str(out)})

    if "M" in phases:
        log({"event": "phase_start", "phase": "M"})
        out = run_phase_m(cfg, args.phase_m_lambdas, args.phase_m_grid, args.phase_m_steps, args.phase_m_batch)
        log({"event": "phase_done", "phase": "M", "csv": str(out)})

    if "N" in phases:
        log({"event": "phase_start", "phase": "N"})
        out = run_phase_n(cfg, args.phase_n_grid, args.phase_n_steps, args.phase_n_batch)
        log({"event": "phase_done", "phase": "N", "csv": str(out)})

    if "O" in phases:
        log({"event": "phase_start", "phase": "O"})
        out = run_phase_o(cfg)
        log({"event": "phase_done", "phase": "O", "csv": str(out)})

    if "P" in phases:
        log({"event": "phase_start", "phase": "P"})
        out = run_phase_p(cfg)
        log({"event": "phase_done", "phase": "P", "csv": str(out)})

    if "Q" in phases:
        log({"event": "phase_start", "phase": "Q"})
        out = run_phase_q(
            cfg, args.phase_q_temptations, args.phase_q_methods,
            args.phase_q_grid, args.phase_q_steps, args.phase_q_batch, args.phase_q_seeds,
        )
        log({"event": "phase_done", "phase": "Q", "csv": str(out)})

    if "R" in phases:
        log({"event": "phase_start", "phase": "R"})
        out = run_phase_r(cfg)
        log({"event": "phase_done", "phase": "R", "csv": str(out)})

    if "T" in phases:
        log({"event": "phase_start", "phase": "T"})
        out = run_phase_t(cfg, args.phase_t_grid, args.phase_t_total, args.phase_t_warm_steps)
        log({"event": "phase_done", "phase": "T", "csv": str(out)})

    if "U" in phases:
        log({"event": "phase_start", "phase": "U"})
        out = run_phase_u(cfg, args.phase_u_lambdas, args.phase_u_grid, args.phase_u_total)
        log({"event": "phase_done", "phase": "U", "csv": str(out)})

    if "V" in phases:
        log({"event": "phase_start", "phase": "V"})
        out = run_phase_v(cfg, args.phase_v_s_values, args.phase_v_t_values, args.phase_v_grid, args.phase_v_steps)
        log({"event": "phase_done", "phase": "V", "csv": str(out)})

    if "W" in phases:
        log({"event": "phase_start", "phase": "W"})
        out = run_phase_w(cfg)
        log({"event": "phase_done", "phase": "W", "csv": str(out)})

    if "X" in phases:
        log({"event": "phase_start", "phase": "X"})
        out = run_phase_x(cfg, args.phase_x_k_values)
        log({"event": "phase_done", "phase": "X", "csv": str(out)})

    if "Y" in phases:
        log({"event": "phase_start", "phase": "Y"})
        out = run_phase_y(cfg, args.phase_y_max_steps)
        log({"event": "phase_done", "phase": "Y", "csv": str(out)})

    if "Z" in phases:
        log({"event": "phase_start", "phase": "Z"})
        out = run_phase_z(cfg, args.phase_z_s_values, args.phase_z_t_values, args.phase_z_grid, args.phase_z_steps)
        log({"event": "phase_done", "phase": "Z", "csv": str(out)})

    if "AA" in phases:
        log({"event": "phase_start", "phase": "AA"})
        out = run_phase_aa(cfg, args.phase_aa_grid, args.phase_aa_steps, args.phase_aa_batch)
        log({"event": "phase_done", "phase": "AA", "csv": str(out)})

    if "BB" in phases:
        log({"event": "phase_start", "phase": "BB"})
        out = run_phase_bb(cfg, args.phase_bb_lambdas, args.phase_bb_grid, args.phase_bb_steps, args.phase_bb_batch)
        log({"event": "phase_done", "phase": "BB", "csv": str(out)})

    if "CC" in phases:
        log({"event": "phase_start", "phase": "CC (tabular)"})
        out_tab = run_phase_cc_tabular(
            cfg, args.phase_cc_Ls, args.phase_cc_tabular_seeds,
            args.phase_cc_tabular_steps, args.phase_cc_tabular_batch,
        )
        log({"event": "phase_done", "phase": "CC (tabular)", "csv": str(out_tab)})
        log({"event": "phase_start", "phase": "CC (MLP)"})
        out_mlp = run_phase_cc_mlp(
            cfg, args.phase_cc_Ls, args.phase_cc_mlp_seeds,
            args.phase_cc_mlp_steps, args.phase_cc_mlp_batch,
        )
        log({"event": "phase_done", "phase": "CC (MLP)", "csv": str(out_mlp)})

    if "DD" in phases:
        log({"event": "phase_start", "phase": "DD"})
        out = run_phase_dd(
            cfg, args.phase_dd_qs, args.phase_dd_seeds, args.phase_dd_n0,
            args.phase_dd_total, args.phase_dd_scale, args.phase_dd_batch,
        )
        log({"event": "phase_done", "phase": "DD", "csv": str(out)})

    if "EE" in phases:
        log({"event": "phase_start", "phase": "EE"})
        out = run_phase_ee(
            cfg, args.phase_ee_rules, args.phase_ee_seeds, args.phase_ee_warm_steps,
            args.phase_ee_cool_steps, args.phase_ee_batch, args.phase_ee_checkpoint_every,
        )
        log({"event": "phase_done", "phase": "EE", "csv": str(out)})

    if "FF" in phases:
        log({"event": "phase_start", "phase": "FF"})
        out = run_phase_ff(
            cfg, args.phase_ff_seeds, args.phase_ff_steps, args.phase_ff_K,
            args.phase_ff_batch, args.phase_ff_perturb_low, args.phase_ff_perturb_high,
        )
        log({"event": "phase_done", "phase": "FF", "csv": str(out)})

    log({"event": "end"})


if __name__ == "__main__":
    main()
