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
    expected_return,
    is_success,
    logit,
    run_rollout,
    sigmoid,
    stag_hunt,
    update_from_components,
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
    ax.set_title("Projection toward $(C,C)$", fontsize=10)

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
    ax.set_title("Peer-correction difference field", fontsize=10)
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
# Driver
# -------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["A", "B", "C", "D", "E", "F", "G", "H", "I", "L", "A2", "D2", "M", "N", "O", "P", "Q", "all"],
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

    phases = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "L", "A2", "D2", "M", "N", "O", "P", "Q"] if args.phase == "all" else [args.phase]
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

    log({"event": "end"})


if __name__ == "__main__":
    main()
