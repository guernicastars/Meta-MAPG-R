"""Figures 2 / 3 / 4 + a peer-coefficient sweep plot.

Each function reads JSONL artefacts produced by `train.py` and writes
PDF + PNG to ``out_dir``.  All figures use the shared palette below so the
neural section reads as a coherent suite.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..algos.arms import ARMS
from ..utils import bootstrap_ci, wilson_ci


PALETTE = {
    "ippo":      "#7A7A7A",
    "own_only":  "#5B7393",
    "peer_only": "#E76F51",
    "meta_mapg": "#2A9D8F",
    "handoff":   "#264653",
}
LINESTYLE = {
    "ippo":      "dashed",
    "own_only":  "dotted",
    "peer_only": "solid",
    "meta_mapg": "solid",
    "handoff":   "dashdot",
}
LINEWIDTH = {"ippo": 1.5, "own_only": 1.7, "peer_only": 2.2,
             "meta_mapg": 2.4, "handoff": 2.4}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_eval_jsonl(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "eval.jsonl"
    if not p.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def _load_summary(run_dir: Path) -> dict:
    p = run_dir / "summary.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def collect_runs(root: Path, *, benchmark: str, env_id: str) -> dict[str, dict[int, pd.DataFrame]]:
    """Returns {arm: {seed: eval_df}}."""
    base = Path(root) / benchmark / env_id
    out: dict[str, dict[int, pd.DataFrame]] = {}
    if not base.exists():
        return out
    for arm_dir in sorted(base.iterdir()):
        if not arm_dir.is_dir() or arm_dir.name not in ARMS:
            continue
        seeds = {}
        for seed_dir in sorted(arm_dir.iterdir()):
            if not seed_dir.name.startswith("seed_"):
                continue
            sd = int(seed_dir.name.removeprefix("seed_"))
            df = _load_eval_jsonl(seed_dir)
            if not df.empty:
                seeds[sd] = df
        if seeds:
            out[arm_dir.name] = seeds
    return out


# ---------------------------------------------------------------------------
# Figure 2: basin-entry probability vs training budget
# ---------------------------------------------------------------------------

def _success_curve(arm_runs: dict[int, pd.DataFrame], step_grid: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Empirical fraction of seeds that have crossed `success` by step ≤ s."""
    n = len(arm_runs)
    if n == 0:
        return step_grid, np.zeros_like(step_grid, dtype=float), np.zeros_like(step_grid, dtype=float)
    # For each seed, find first hit step (or +inf).
    first_hit = np.full(n, np.inf)
    for k, (sd, df) in enumerate(arm_runs.items()):
        suc = df[df["success"]]
        if not suc.empty:
            first_hit[k] = suc["step"].min()
    succ = np.zeros_like(step_grid, dtype=float)
    lo   = np.zeros_like(step_grid, dtype=float)
    hi   = np.zeros_like(step_grid, dtype=float)
    for i, s in enumerate(step_grid):
        succeeded = int((first_hit <= s).sum())
        p, l, h = wilson_ci(succeeded, n)
        succ[i], lo[i], hi[i] = p, l, h
    return succ, lo, hi


def figure_2_basin_entry(root: Path, *, panels: list[tuple[str, str, str]],
                         out_dir: Path, max_step: int | None = None,
                         title: str | None = None) -> None:
    """Three (or N) -panel figure: success probability vs training budget.

    `panels` is a list of (label, benchmark, env_id) tuples — one per panel."""
    n_p = len(panels)
    fig, axes = plt.subplots(1, n_p, figsize=(4.6 * n_p, 4.2),
                              constrained_layout=True, squeeze=False)
    axes = axes[0]
    for ax, (label, benchmark, env_id) in zip(axes, panels):
        runs = collect_runs(root, benchmark=benchmark, env_id=env_id)
        if not runs:
            ax.set_title(f"{label}\n(no runs)")
            continue

        # Pick a step grid spanning all seeds.
        all_steps = []
        for arm_runs in runs.values():
            for df in arm_runs.values():
                all_steps.extend(df["step"].tolist())
        if not all_steps:
            ax.set_title(f"{label}\n(no eval data)"); continue
        gmax = max_step or int(max(all_steps))
        step_grid = np.linspace(0, gmax, 60)

        for arm in ARMS:
            if arm not in runs:
                continue
            succ, lo, hi = _success_curve(runs[arm], step_grid)
            ax.plot(step_grid / 1e3, 100 * succ,
                    label=ARMS[arm].label,
                    color=PALETTE[arm], linestyle=LINESTYLE[arm],
                    linewidth=LINEWIDTH[arm])
            ax.fill_between(step_grid / 1e3, 100 * lo, 100 * hi,
                            color=PALETTE[arm], alpha=0.12)
        ax.set_title(f"{label}", fontweight="semibold")
        ax.set_xlabel("Training steps (×1k)")
        ax.set_ylabel("Basin-entry success (% of seeds)")
        ax.set_ylim(-2, 102)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right", frameon=True)

    fig.suptitle(title or "Neural basin-entry probability across benchmarks",
                 fontsize=13, fontweight="semibold")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figure_2_basin_entry.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "figure_2_basin_entry.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: shape-then-cool handoff
# ---------------------------------------------------------------------------

def figure_3_handoff(root: Path, *, benchmark: str, env_id: str,
                     out_dir: Path, T_warm: int) -> None:
    runs = collect_runs(root, benchmark=benchmark, env_id=env_id)
    arms_to_show = ["ippo", "meta_mapg", "handoff"]
    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    for arm in arms_to_show:
        if arm not in runs:
            continue
        # Mean ± 95% bootstrap CI of eval return per training step.
        rows = []
        for sd, df in runs[arm].items():
            for _, r in df.iterrows():
                rows.append((r["step"], r["eval_return_mean"]))
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["step", "ret"])
        df = df.groupby("step").agg(list)
        steps = np.array(df.index.tolist(), dtype=float)
        means = np.array([np.mean(r) for r in df["ret"].values])
        cis = np.array([bootstrap_ci(np.asarray(r))[1:] for r in df["ret"].values])
        ax.plot(steps / 1e3, means, label=ARMS[arm].label,
                color=PALETTE[arm], linestyle=LINESTYLE[arm], linewidth=LINEWIDTH[arm])
        ax.fill_between(steps / 1e3, cis[:, 0], cis[:, 1],
                        color=PALETTE[arm], alpha=0.18)
    ax.axvline(T_warm / 1e3, color="black", linestyle=":", alpha=0.6,
               label=f"T_warm = {T_warm/1e3:.0f}k")
    ax.set_xlabel("Training steps (×1k)")
    ax.set_ylabel("Eval return (mean ± 95% bootstrap)")
    ax.set_title(f"Shape-then-cool handoff — {benchmark} / {env_id}",
                 fontweight="semibold")
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figure_3_handoff.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "figure_3_handoff.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: peer ablation (per-arm bar across benchmarks)
# ---------------------------------------------------------------------------

def figure_4_peer_ablation(root: Path, *, panels: list[tuple[str, str, str]],
                           out_dir: Path) -> None:
    """Bar plot: success-fraction-at-final-eval per arm × benchmark."""
    arms = list(ARMS.keys())
    width = 0.16
    fig, ax = plt.subplots(figsize=(max(7, 1.2 + 1.0 * len(panels)), 5),
                            constrained_layout=True)
    for i, (label, benchmark, env_id) in enumerate(panels):
        runs = collect_runs(root, benchmark=benchmark, env_id=env_id)
        for j, arm in enumerate(arms):
            if arm not in runs:
                continue
            successes = sum(int(df["success"].iloc[-1]) for df in runs[arm].values()
                            if not df.empty)
            n = len(runs[arm])
            if n == 0:
                continue
            p, lo, hi = wilson_ci(successes, n)
            x = i + (j - 2) * width
            ax.bar(x, 100 * p, width=width, color=PALETTE[arm],
                   edgecolor="black", linewidth=0.4,
                   label=ARMS[arm].label if i == 0 else None)
            yerr_lo = max(0.0, 100.0 * (p - lo)) if not (p != p) else 0.0
            yerr_hi = max(0.0, 100.0 * (hi - p)) if not (p != p) else 0.0
            ax.errorbar(x, 100 * p,
                        yerr=[[yerr_lo], [yerr_hi]],
                        ecolor="black", elinewidth=1.0, capsize=2.5)
    ax.set_xticks(range(len(panels)))
    ax.set_xticklabels([p[0] for p in panels])
    ax.set_ylabel("Final basin-entry success (% of seeds)")
    ax.set_title("Peer-ablation across neural benchmarks",
                  fontweight="semibold")
    ax.set_ylim(0, 105); ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=9)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figure_4_peer_ablation.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "figure_4_peer_ablation.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Lambda sweep figure
# ---------------------------------------------------------------------------

def figure_lambda_sweep(root: Path, *, benchmark: str, env_id: str,
                        out_dir: Path) -> None:
    sweep_root = Path(root) / benchmark / env_id
    if not sweep_root.exists():
        return
    rows = []
    for d in sorted(sweep_root.glob("lam_*")):
        try:
            lam = float(d.name.removeprefix("lam_"))
        except ValueError:
            continue
        runs = collect_runs(root, benchmark=benchmark, env_id=f"{env_id}/{d.name}")
        if not runs:
            continue
        for arm, sds in runs.items():
            successes = sum(int(df["success"].iloc[-1]) for df in sds.values())
            n = len(sds)
            p, lo, hi = wilson_ci(successes, n)
            rows.append({"lambda": lam, "arm": arm, "p": p, "lo": lo, "hi": hi, "n": n})
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["arm", "lambda"])
    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    for arm, sub in df.groupby("arm"):
        ax.errorbar(sub["lambda"], 100 * sub["p"],
                    yerr=[100 * (sub["p"] - sub["lo"]), 100 * (sub["hi"] - sub["p"])],
                    label=ARMS[arm].label, color=PALETTE[arm],
                    marker="o", linewidth=1.8, capsize=3)
    ax.set_xlabel(r"Peer coefficient $\lambda_\mathrm{peer}$")
    ax.set_ylabel("Final basin-entry success (%)")
    ax.set_title(f"Peer-coefficient sweep — {benchmark} / {env_id}",
                 fontweight="semibold")
    ax.grid(True, alpha=0.3); ax.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figure_lambda_sweep.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "figure_lambda_sweep.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------

def write_metrics_table(root: Path, *, panels: list[tuple[str, str, str]],
                        out_path: Path) -> None:
    rows = []
    for label, benchmark, env_id in panels:
        runs = collect_runs(root, benchmark=benchmark, env_id=env_id)
        for arm, sds in runs.items():
            ret_finals = []
            successes  = 0
            first_hits = []
            for sd, df in sds.items():
                if df.empty:
                    continue
                ret_finals.append(float(df["eval_return_mean"].iloc[-1]))
                successes += int(df["success"].iloc[-1])
                hits = df.loc[df["success"], "step"].tolist()
                if hits:
                    first_hits.append(min(hits))
            n = len(sds)
            p, lo, hi = wilson_ci(successes, n)
            mean_ret, ret_lo, ret_hi = bootstrap_ci(np.asarray(ret_finals)) if ret_finals \
                                           else (float("nan"),)*3
            rows.append({
                "benchmark": label, "arm": arm,
                "n_seeds": n,
                "success_pct": 100 * p,
                "success_lo":  100 * lo,
                "success_hi":  100 * hi,
                "mean_final_return":   mean_ret,
                "final_return_lo":     ret_lo,
                "final_return_hi":     ret_hi,
                "median_first_hit":    float(np.median(first_hits)) if first_hits else float("nan"),
            })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
