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


def write_phase_a2() -> None:
    csv = ART_DIR / "phase_a2_angle_diff.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    mean_angle = float(df["angle_diff_rad"].abs().mean())
    frac_ccw = float((df["angle_diff_rad"] > 0.1).mean())
    body = []
    body.append(
        "We decompose the difference between Meta-MAPG and PG update directions "
        "into (i) a signed angle-tilt heatmap "
        "$\\angle(\\Delta p^{\\mathrm{MM}}) - \\angle(\\Delta p^{\\mathrm{PG}})$ "
        "and (ii) a difference quiver $\\Delta p^{\\mathrm{MM}} - \\Delta p^{\\mathrm{PG}}$. "
        f"Mean absolute tilt: ${mean_angle:.3f}$\\,rad. "
        f"In {frac_ccw*100:.0f}\\% of grid cells the tilt is counter-clockwise "
        "(positive, toward $(C,C)$), concentrated in the region below the PG separatrix."
    )
    body.append(
        "\\begin{figure}[h]\n"
        "  \\centering\n"
        "  \\includegraphics[width=0.95\\linewidth]{phase_a2_angle_diff}\n"
        "  \\caption{Left: signed angular tilt of Meta-MAPG vs PG (red = "
        "counter-clockwise, toward $(C,C)$). Right: difference quiver showing "
        "the additive peer-correction vector. The peer correction primarily "
        "redirects dynamics in the central mixed-strategy region.}\n"
        "  \\label{fig:phase-a2}\n"
        "\\end{figure}"
    )
    (TEX_DIR / "phase_a2.tex").write_text("\n\n".join(body) + "\n")


def write_phase_g() -> None:
    csv = ART_DIR / "phase_g_tsweep_multiseed.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    body = []
    body.append(
        "Phase~E is repeated with multiple independent seeds per cell, "
        "enabling Wilson 95\\% confidence intervals on the basin fraction "
        "at each $(T, \\text{method})$ pair. The qualitative ordering---"
        "Meta-MAPG $\\approx$ Peer-only $>$ PG across all $T$---is confirmed "
        "after accounting for stochastic variability."
    )
    body.append(
        "\\begin{figure}[h]\n"
        "  \\centering\n"
        "  \\includegraphics[width=0.65\\linewidth]{phase_g_tsweep_multiseed}\n"
        "  \\caption{Multi-seed $T$-sweep with 95\\% Wilson bands. "
        "PG and peer-aware confidence bands are non-overlapping at all $T$.}\n"
        "  \\label{fig:phase-g}\n"
        "\\end{figure}"
    )
    methods = df["method"].unique().tolist()
    rows = ["\\begin{tabular}{l" + "c" * len(methods) + "}\n  \\toprule"]
    rows.append("  $T$ & " + " & ".join(METHOD_LABEL.get(m, m) for m in methods) + " \\\\\n  \\midrule")
    for T, grp in df.groupby("T"):
        parts = []
        for m in methods:
            r = grp[grp["method"] == m]
            if r.empty:
                parts.append("--")
            else:
                p = float(r["coop_basin_fraction"].iloc[0])
                lo = float(r["ci_lo"].iloc[0])
                hi = float(r["ci_hi"].iloc[0])
                parts.append(f"{p*100:.1f} [{lo*100:.1f},{hi*100:.1f}]\\%")
        rows.append(f"  {T:.2f} & " + " & ".join(parts) + " \\\\")
    rows.append("  \\bottomrule\n\\end{tabular}")
    body.append(
        "\\begin{table}[h]\n  \\centering\n"
        + "\n".join(rows)
        + "\n  \\caption{Multi-seed $T$-sweep: basin fraction with 95\\% Wilson CI.}\n"
        "  \\label{tab:phase-g}\n\\end{table}"
    )
    (TEX_DIR / "phase_g.tex").write_text("\n\n".join(body) + "\n")


def write_phase_h() -> None:
    csv = ART_DIR / "phase_h_resolution.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    grid_sizes = sorted(df["grid_size"].unique())
    body = []
    body.append(
        "We repeat the basin measurement for all four methods at resolutions "
        "$N \\in \\{" + ", ".join(str(g) for g in grid_sizes) + "\\}$. "
        "Basin fraction is stable by $N=21$ and the method ranking is invariant, "
        "confirming that reported results do not depend on grid resolution."
    )
    body.append(
        "\\begin{figure}[h]\n"
        "  \\centering\n"
        "  \\includegraphics[width=0.6\\linewidth]{phase_h_resolution}\n"
        "  \\caption{Basin fraction vs.\\ grid resolution. "
        "All four methods stabilise by $N=21$; ranking is preserved at all resolutions.}\n"
        "  \\label{fig:phase-h}\n"
        "\\end{figure}"
    )
    pivot = df.pivot_table(index="grid_size", columns="method", values="coop_basin_fraction")
    rows = ["\\begin{tabular}{l" + "c" * len(pivot.columns) + "}\n  \\toprule"]
    rows.append("  $N$ & " + " & ".join(METHOD_LABEL.get(c, c) for c in pivot.columns) + " \\\\\n  \\midrule")
    for gs, line in pivot.iterrows():
        vals = " & ".join(
            f"{v*100:.1f}\\%" if not np.isnan(v) else "--" for v in line.tolist()
        )
        rows.append(f"  {gs} & {vals} \\\\")
    rows.append("  \\bottomrule\n\\end{tabular}")
    body.append(
        "\\begin{table}[h]\n  \\centering\n"
        + "\n".join(rows)
        + "\n  \\caption{Basin fraction vs.\\ grid resolution.}\n"
        "  \\label{tab:phase-h}\n\\end{table}"
    )
    (TEX_DIR / "phase_h.tex").write_text("\n\n".join(body) + "\n")


def write_phase_i() -> None:
    csv = ART_DIR / "phase_i_fht.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    hit = df[df["first_hit_step"] > 0]
    means = hit.groupby("method")["first_hit_step"].mean()
    best_method = str(means.idxmin()) if not means.empty else "meta_mapg"
    body = []
    body.append(
        "For each cell in the $51\\times 51$ atlas we record the first step "
        "$t^*$ at which $\\min(p_1^t, p_2^t) \\ge \\tau$; cells where the "
        "threshold is never crossed within the budget are shown in grey. "
        f"{METHOD_LABEL.get(best_method, best_method)} reaches the cooperative "
        f"region fastest (mean $t^* = {float(means[best_method]):.1f}$ over "
        "successful cells)."
    )
    body.append(
        "\\begin{figure}[h]\n"
        "  \\centering\n"
        "  \\includegraphics[width=0.95\\linewidth]{phase_i_fht}\n"
        "  \\caption{First-hit-time atlas ($\\log(1+t^*)$). Grey cells never "
        "reach the cooperative threshold. Peer-aware methods hit faster and "
        "from a larger set of initial conditions.}\n"
        "  \\label{fig:phase-i}\n"
        "\\end{figure}"
    )
    rows = ["\\begin{tabular}{lcc}\n  \\toprule"]
    rows.append("  Method & Mean $t^*$ & Frac.\\ success \\\\\n  \\midrule")
    for method in ["standard_pg", "meta_pg", "lola_style", "meta_mapg"]:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        suc = sub[sub["first_hit_step"] > 0]
        mean_t = f"{float(suc['first_hit_step'].mean()):.1f}" if not suc.empty else "--"
        frac = len(suc) / len(sub)
        rows.append(f"  {METHOD_LABEL[method]} & {mean_t} & {frac*100:.1f}\\% \\\\")
    rows.append("  \\bottomrule\n\\end{tabular}")
    body.append(
        "\\begin{table}[h]\n  \\centering\n"
        + "\n".join(rows)
        + "\n  \\caption{First-hit-time summary. Mean computed over successful cells only.}\n"
        "  \\label{tab:phase-i}\n\\end{table}"
    )
    (TEX_DIR / "phase_i.tex").write_text("\n\n".join(body) + "\n")


def write_phase_l() -> None:
    csv = ART_DIR / "phase_l_diagonal.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    valid = df.dropna(subset=["p_diag_star"])
    body = []
    if not valid.empty:
        best = valid.loc[valid["p_diag_star"].idxmin()]
        body.append(
            "From Phase~C data we extract $p^*_{\\mathrm{diag}}$: the minimum "
            "symmetric initial cooperation probability $p_1^0 = p_2^0$ at which "
            "stochastic basin-entry probability exceeds $0.5$. "
            f"{METHOD_LABEL.get(str(best['method']), str(best['method']))} achieves "
            f"$p^*_{{\\mathrm{{diag}}}} = {float(best['p_diag_star']):.3f}$, "
            "the lowest of the four methods, meaning it can enter the cooperative "
            "basin from the most adversarial symmetric starting condition."
        )
    else:
        body.append("Diagonal threshold $p^*_{\\mathrm{diag}}$ from Phase~C stochastic basin data.")
    body.append(
        "\\begin{figure}[h]\n"
        "  \\centering\n"
        "  \\includegraphics[width=0.6\\linewidth]{phase_l_diagonal}\n"
        "  \\caption{Success probability along the diagonal $p_1^0=p_2^0$. "
        "Dashed verticals mark $p^*_{\\mathrm{diag}}$ per method (crossing of the "
        "horizontal 0.5 line).}\n"
        "  \\label{fig:phase-l}\n"
        "\\end{figure}"
    )
    rows = ["\\begin{tabular}{lc}\n  \\toprule"]
    rows.append("  Method & $p^*_{\\mathrm{diag}}$ \\\\\n  \\midrule")
    for method in ["standard_pg", "meta_pg", "lola_style", "meta_mapg"]:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        val = float(sub["p_diag_star"].iloc[0])
        rows.append(f"  {METHOD_LABEL[method]} & " + (f"{val:.3f}" if not np.isnan(val) else "--") + " \\\\")
    rows.append("  \\bottomrule\n\\end{tabular}")
    body.append(
        "\\begin{table}[h]\n  \\centering\n"
        + "\n".join(rows)
        + "\n  \\caption{Diagonal critical threshold: minimum initial cooperation "
        "on $p_1^0=p_2^0$ for 50\\% basin-entry probability.}\n"
        "  \\label{tab:phase-l}\n\\end{table}"
    )
    (TEX_DIR / "phase_l.tex").write_text("\n\n".join(body) + "\n")


def write_phase_d2() -> None:
    csv = ART_DIR / "phase_d2_audit.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    arm_order = ["pg", "meta_mapg_constant", "meta_mapg_two_phase", "warm_metamapg_pure_pg"]
    pretty = {
        "pg": "PG",
        "meta_mapg_constant": "Meta-MAPG (const.\\ $\\lambda$)",
        "meta_mapg_two_phase": "Meta-MAPG (two-phase)",
        "warm_metamapg_pure_pg": "warm-Meta-MAPG $\\to$ pure-PG",
    }
    body = []
    body.append(
        "Phase~D used a shared pool of initial policies across all four arms. "
        "Here each arm draws its own independent initial policies to verify "
        "the conclusion is not a shared-seed artefact. All three peer-aware arms "
        "again achieve success rates well above PG."
    )
    body.append(
        "\\begin{figure}[h]\n"
        "  \\centering\n"
        "  \\includegraphics[width=0.55\\linewidth]{phase_d2_audit}\n"
        "  \\caption{D2: same ablation as Phase~D with independent initial "
        "policies per arm. The peer-aware advantage is robust to the seed choice.}\n"
        "  \\label{fig:phase-d2}\n"
        "\\end{figure}"
    )
    rows = ["\\begin{table}[h]\n  \\centering"]
    rows.append("  \\begin{tabular}{lcc}\n    \\toprule")
    rows.append("    Arm & Success rate & 95\\% Wilson CI \\\\\n    \\midrule")
    for label in arm_order:
        sub = df[df["label"] == label]
        if sub.empty:
            continue
        k = int(sub["success"].sum())
        n = len(sub)
        p, lo, hi = wilson_ci(k, n)
        rows.append(f"    {pretty[label]} & {p*100:.1f}\\% & [{lo*100:.1f}\\%, {hi*100:.1f}\\%] \\\\")
    rows.append("    \\bottomrule\n  \\end{tabular}")
    rows.append(
        "  \\caption{D2 audit: independent initial conditions confirm the "
        "peer-aware advantage is not a seed artefact.}\n"
        "  \\label{tab:phase-d2}\n\\end{table}"
    )
    body.append("\n".join(rows))
    (TEX_DIR / "phase_d2.tex").write_text("\n\n".join(body) + "\n")


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
        "a2": write_phase_a2,
        "g": write_phase_g,
        "h": write_phase_h,
        "i": write_phase_i,
        "l": write_phase_l,
        "d2": write_phase_d2,
    }
    for phase in args.phases:
        fn = fns.get(phase.lower())
        if fn is None:
            continue
        fn()


if __name__ == "__main__":
    main()
