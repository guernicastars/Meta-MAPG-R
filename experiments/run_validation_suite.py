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
    rows: list[dict] = []
    for label, schedule in arms:
        for seed in range(n_seeds):
            theta = init_thetas[seed].copy()
            rng = np.random.default_rng(cfg.seed_base + 991_000 + 71 * seed + hash(label) % 17)
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
# Driver
# -------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=str, default="all", choices=["A", "B", "C", "D", "E", "F", "all"])
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
    p.add_argument(
        "--phase-f-thresholds",
        type=float,
        nargs="+",
        default=[0.75, 0.82, 0.90],
    )
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

    phases = ["A", "B", "C", "D", "E", "F"] if args.phase == "all" else [args.phase]
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

    log({"event": "end"})


if __name__ == "__main__":
    main()
