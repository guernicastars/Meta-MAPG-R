"""Render per-phase LaTeX fragments from validation CSV outputs.

Each function reads the CSV produced by `run_validation_suite.py` and
writes a complete `phase_<x>.tex` snippet that is `\\input{}`-ed by the
top-level `tex/results_validation.tex` document.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
TEX_DIR = REPO / "tex"
ART_DIR = REPO / "artifacts" / "validation"
FIG_DIR = REPO / "figures" / "validation"

METHOD_LABEL = {
    "standard_pg": "PG",
    "meta_pg": "Meta-PG",
    "lola_style": "Peer only",
    "meta_mapg": "Meta-MAPG",
}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def write_phase_a() -> None:
    csv = ART_DIR / "phase_a_arrows.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    n_cells = len(df) // df["method"].nunique()

    body = []
    body.append(
        "We sample the expected update direction at each grid point using a "
        "single high-batch estimator and convert the parameter-space update "
        "$\\Delta\\theta$ to a probability-space arrow $\\Delta p = "
        "\\sigma'(\\theta)\\,\\Delta\\theta$. For both PG and Meta-MAPG we use "
        f"a {int(np.sqrt(n_cells))}$\\times${int(np.sqrt(n_cells))} grid "
        "with the same realisations of stochastic noise per cell."
    )
    body.append("\\par\\smallskip")
    body.append(
        "\\begin{figure}[h]\n"
        "  \\centering\n"
        "  \\includegraphics[width=0.95\\linewidth]{phase_a_expected_arrows}\n"
        "  \\caption{Expected update directions in tabular Stag Hunt. Around "
        "the PG separatrix the Meta-MAPG arrows tilt visibly toward the "
        "payoff-dominant equilibrium $(C,C)$, which is the local mechanism "
        "behind the basin shifts in Phases~B and~C.}\n"
        "  \\label{fig:phase-a-arrows}\n"
        "\\end{figure}"
    )

    sub_pg = df[df["method"] == "standard_pg"].copy()
    sub_mm = df[df["method"] == "meta_mapg"].copy()
    sub_pg["mag"] = np.sqrt(sub_pg["dp1"] ** 2 + sub_pg["dp2"] ** 2)
    sub_mm["mag"] = np.sqrt(sub_mm["dp1"] ** 2 + sub_mm["dp2"] ** 2)

    body.append(
        f"Mean update magnitude: PG = ${sub_pg['mag'].mean():.4f}$, "
        f"Meta-MAPG = ${sub_mm['mag'].mean():.4f}$. The Meta-MAPG arrow "
        "field is generally smaller in magnitude (the peer correction "
        "partially counters $g_{\\mathrm{self}}$ near $(D,D)$) but is "
        "redirected toward $(C,C)$ in the central region of policy space."
    )

    (TEX_DIR / "phase_a.tex").write_text("\n\n".join(body) + "\n")


def write_phase_b() -> None:
    csv = ART_DIR / "phase_b_basin.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    fractions = (
        df.groupby("method")["success"].agg(["mean", "count"]).reset_index()
        .rename(columns={"mean": "frac"})
    )
    body = []
    body.append(
        "We compute deterministic-classification basin maps at "
        f"{int(np.sqrt(len(df) // df['method'].nunique()))}$\\times$"
        f"{int(np.sqrt(len(df) // df['method'].nunique()))} resolution for "
        "each of the four learning rules. Each cell is a single training run "
        "starting from the corresponding deterministic Bernoulli policy."
    )
    body.append(
        "\\begin{figure}[h]\n"
        "  \\centering\n"
        "  \\begin{subfigure}[t]{0.55\\linewidth}\n"
        "    \\centering\\includegraphics[width=\\linewidth]{phase_b_basin_atlas}\n"
        "    \\caption{Basin atlas (success indicator).}\n"
        "  \\end{subfigure}\\hfill\n"
        "  \\begin{subfigure}[t]{0.42\\linewidth}\n"
        "    \\centering\\includegraphics[width=\\linewidth]{phase_b_frontier_overlay}\n"
        "    \\caption{Empirical separatrix overlay.}\n"
        "  \\end{subfigure}\n"
        "  \\caption{Four-method basin atlas in tabular Stag Hunt. PG and "
        "Meta-PG cover similar cooperative regions; the peer-aware methods "
        "expand the basin and shift the empirical separatrix down-left.}\n"
        "  \\label{fig:phase-b-atlas}\n"
        "\\end{figure}"
    )

    rows = []
    rows.append("\\begin{table}[h]\n  \\centering")
    rows.append("  \\begin{tabular}{lcc}\n    \\toprule")
    rows.append("    Method & Coop.\\ basin fraction & 95\\% Wilson CI \\\\\n    \\midrule")
    pg_row = fractions[fractions["method"] == "standard_pg"]
    pg_frac = float(pg_row["frac"].iloc[0]) if not pg_row.empty else 1e-9
    for method in ["standard_pg", "meta_pg", "lola_style", "meta_mapg"]:
        sub = fractions[fractions["method"] == method]
        if sub.empty:
            continue
        n = int(sub["count"].iloc[0])
        k = int(round(float(sub["frac"].iloc[0]) * n))
        p, lo, hi = wilson_ci(k, n)
        rows.append(
            f"    {METHOD_LABEL[method]} & {p*100:.1f}\\% & [{lo*100:.1f}\\%, {hi*100:.1f}\\%] \\\\"
        )
    rows.append("    \\bottomrule\n  \\end{tabular}")
    rows.append(
        "  \\caption{Basin fractions over the full grid. Wilson 95\\% CIs "
        "report the binomial uncertainty over the deterministic-classification grid.}\n"
        "  \\label{tab:phase-b-fractions}\n"
        "\\end{table}"
    )
    body.append("\n".join(rows))

    expansion = {
        m: float(fractions[fractions["method"] == m]["frac"].iloc[0]) / max(pg_frac, 1e-9)
        for m in fractions["method"].unique()
    }
    body.append(
        "Basin expansion ratios relative to PG: "
        + ", ".join(f"{METHOD_LABEL[m]}~$={r:.2f}\\times$" for m, r in expansion.items() if m != "standard_pg")
        + "."
    )

    (TEX_DIR / "phase_b.tex").write_text("\n\n".join(body) + "\n")


def write_phase_c() -> None:
    csv = ART_DIR / "phase_c_stochastic_basin.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    s_per_cell = int(df["n_seeds"].iloc[0])
    grid_n = int(np.sqrt(len(df) // df["method"].nunique()))
    body = []
    body.append(
        f"We re-run the basin map with $S={s_per_cell}$ independent "
        "policy / sampling seeds per cell on a "
        f"${grid_n}\\times{grid_n}$ grid; the entry probability "
        "$\\hat P(\\text{coop})$ is the per-cell success rate. This "
        "replaces the single-seed deterministic indicator with a stochastic "
        "estimate that respects the genuine basin-entry probability "
        "$\\pentry$ from the revised framing."
    )
    body.append(
        "\\begin{figure}[h]\n"
        "  \\centering\n"
        "  \\includegraphics[width=0.95\\linewidth]{phase_c_stochastic_basin}\n"
        "  \\caption{Stochastic basin probability maps. Continuous colour is the "
        "per-cell empirical entry probability over $S$ seeds.}\n"
        "  \\label{fig:phase-c-stoch}\n"
        "\\end{figure}"
    )
    rows = []
    rows.append("\\begin{table}[h]\n  \\centering")
    rows.append("  \\begin{tabular}{lcccc}\n    \\toprule")
    rows.append(
        "    Method & $E[\\hat P(\\text{coop})]$ & total successes / trials & "
        "Wilson CI on aggregate \\\\\n    \\midrule"
    )
    for method in ["standard_pg", "meta_pg", "lola_style", "meta_mapg"]:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        n = int(sub["n_seeds"].sum())
        k = int(sub["n_success"].sum())
        mean_p = float(sub["p_success"].mean())
        _, lo, hi = wilson_ci(k, n)
        rows.append(
            f"    {METHOD_LABEL[method]} & {mean_p:.3f} & {k}/{n} & "
            f"[{lo:.3f}, {hi:.3f}] \\\\"
        )
    rows.append("    \\bottomrule\n  \\end{tabular}")
    rows.append(
        "  \\caption{Stochastic basin entry. The Wilson interval uses the "
        "aggregate count over the full grid and is a strict lower bound on the "
        "per-initial-condition uncertainty.}\n"
        "  \\label{tab:phase-c-stoch}\n"
        "\\end{table}"
    )
    body.append("\n".join(rows))
    (TEX_DIR / "phase_c.tex").write_text("\n\n".join(body) + "\n")


def write_phase_d() -> None:
    csv = ART_DIR / "phase_d_anneal_4arm.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    arm_order = [
        "pg",
        "meta_mapg_constant",
        "meta_mapg_two_phase",
        "warm_metamapg_pure_pg",
    ]
    pretty = {
        "pg": "PG",
        "meta_mapg_constant": "Meta-MAPG (const.\\ $\\lambda$)",
        "meta_mapg_two_phase": "Meta-MAPG (two-phase)",
        "warm_metamapg_pure_pg": "warm-Meta-MAPG $\\to$ pure-PG",
    }
    body = []
    body.append(
        "All four arms share initial policies and step-size schedule. "
        "The warm-Meta-MAPG $\\to$ pure-PG arm is the explicit version of "
        "the algorithm advocated in the repositioning report: the peer "
        "correction is used only to enter the cooperative basin, then "
        "removed entirely so the long-run dynamics is exactly ordinary "
        "policy gradient with vanishing step-size."
    )
    body.append(
        "\\begin{figure}[h]\n"
        "  \\centering\n"
        "  \\includegraphics[width=0.95\\linewidth]{phase_d_anneal_4arm}\n"
        "  \\caption{Four-arm shape-then-cool. Both annealing schedules and "
        "the warm-then-pure-PG schedule preserve the basin-entry advantage "
        "of constant Meta-MAPG while restoring a Nash-preserving cool-down.}\n"
        "  \\label{fig:phase-d-anneal}\n"
        "\\end{figure}"
    )
    rows = []
    rows.append("\\begin{table}[h]\n  \\centering")
    rows.append("  \\begin{tabular}{lccc}\n    \\toprule")
    rows.append(
        "    Arm & Final coop.\\ rate & 2nd-half coop.\\ mean & 2nd-half coop.\\ std \\\\\n    \\midrule"
    )
    for label in arm_order:
        sub = df[df["label"] == label]
        if sub.empty:
            continue
        succ = float(sub["success"].mean())
        ch = float(sub["second_half_coop_mean"].mean())
        cs = float(sub["second_half_coop_std"].mean())
        rows.append(
            f"    {pretty[label]} & {succ*100:.1f}\\% & {ch:.3f} & {cs:.3f} \\\\"
        )
    rows.append("    \\bottomrule\n  \\end{tabular}")
    rows.append(
        "  \\caption{Cool-down preserves basin-entry. The warm-Meta-MAPG "
        "$\\to$ pure-PG arm matches the constant-$\\lambda$ arm in success "
        "rate while making the asymptotic dynamics provably "
        "Nash-preserving.}\n  \\label{tab:phase-d-anneal}\n\\end{table}"
    )
    body.append("\n".join(rows))
    (TEX_DIR / "phase_d.tex").write_text("\n\n".join(body) + "\n")


def write_phase_e() -> None:
    csv = ART_DIR / "phase_e_game_family.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    body = []
    body.append(
        "We sweep the temptation parameter $T$ in the family $R=4, P=2, S=0, "
        "T \\in \\{2.2, 2.5, 3.0, 3.5, 3.8\\}$, keeping $(C,C)$ payoff-dominant "
        "and $(D,D)$ risk-dominant. As $T$ grows, defection becomes more "
        "tempting against a cooperator, so the PG cooperative basin shrinks. "
        "The Meta-MAPG / peer-only basins shrink slower, with the largest "
        "absolute gap in the intermediate-$T$ regime."
    )
    body.append(
        "\\begin{figure}[h]\n"
        "  \\centering\n"
        "  \\includegraphics[width=0.55\\linewidth]{phase_e_game_family}\n"
        "  \\caption{Coordination-game family sweep over the temptation "
        "parameter $T$. Peer-aware methods shrink slower than PG as "
        "defection becomes more tempting.}\n"
        "  \\label{fig:phase-e-family}\n"
        "\\end{figure}"
    )
    pivot = df.pivot_table(index="T", columns="method", values="coop_basin_fraction")
    rows = ["\\begin{tabular}{l" + "c" * len(pivot.columns) + "}\n  \\toprule"]
    rows.append(
        "  $T$ & "
        + " & ".join(METHOD_LABEL.get(c, c) for c in pivot.columns)
        + " \\\\\n  \\midrule"
    )
    for T, line in pivot.iterrows():
        rows.append(
            f"  {T:.2f} & "
            + " & ".join(f"{v*100:.1f}\\%" for v in line.tolist())
            + " \\\\"
        )
    rows.append("  \\bottomrule\n\\end{tabular}")
    body.append(
        "\\begin{table}[h]\n  \\centering\n"
        + "\n".join(rows)
        + "\n  \\caption{Cooperative basin fraction by method and temptation $T$.}"
        + "\n  \\label{tab:phase-e-family}\n\\end{table}"
    )
    (TEX_DIR / "phase_e.tex").write_text("\n\n".join(body) + "\n")


def write_phase_f() -> None:
    csv = ART_DIR / "phase_f_threshold_robustness.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    body = []
    body.append(
        "We re-classify all basin endpoints from Phase~B and from the original "
        "main-text 21$\\times$21 sweep using thresholds "
        "$\\tau \\in \\{0.75, 0.82, 0.90\\}$. The qualitative ordering of the "
        "methods is preserved at every threshold; the quantitative gap "
        "between PG and the peer-aware methods grows as $\\tau$ tightens."
    )
    rows = ["\\begin{tabular}{llccc}\n  \\toprule"]
    rows.append("  Source & Method & $\\tau=0.75$ & $\\tau=0.82$ & $\\tau=0.90$ \\\\\n  \\midrule")
    pivot = df.pivot_table(
        index=["source", "method"],
        columns="tau",
        values="coop_basin_fraction",
    )
    underscore_escape = "\\_"
    for (src, method), line in pivot.iterrows():
        vals = " & ".join(
            f"{line.get(t, float('nan'))*100:.1f}\\%" if not np.isnan(line.get(t, float('nan'))) else "--"
            for t in [0.75, 0.82, 0.90]
        )
        src_safe = str(src).replace("_", underscore_escape)
        method_label = METHOD_LABEL.get(method, method)
        rows.append(f"  {src_safe} & {method_label} & {vals} \\\\")
    rows.append("  \\bottomrule\n\\end{tabular}")
    body.append(
        "\\begin{table}[h]\n  \\centering\n"
        + "\n".join(rows)
        + "\n  \\caption{Threshold robustness. The qualitative ordering "
        "is invariant in $\\tau$.}\n  \\label{tab:phase-f-tau}\n\\end{table}"
    )
    (TEX_DIR / "phase_f.tex").write_text("\n\n".join(body) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phases",
        type=str,
        nargs="+",
        default=["a", "b", "c", "d", "e", "f"],
    )
    args = p.parse_args()
    fns = {
        "a": write_phase_a,
        "b": write_phase_b,
        "c": write_phase_c,
        "d": write_phase_d,
        "e": write_phase_e,
        "f": write_phase_f,
    }
    for phase in args.phases:
        fn = fns.get(phase.lower())
        if fn is None:
            continue
        fn()


if __name__ == "__main__":
    main()
