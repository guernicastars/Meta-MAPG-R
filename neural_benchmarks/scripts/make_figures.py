"""Generate Figures 2 / 3 / 4 + metrics CSV from artefacts.

Usage:
    python scripts/make_figures.py --runs-root artifacts/runs --out-dir artifacts/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from meta_mapg.plotting import (
    figure_2_basin_entry,
    figure_3_handoff,
    figure_4_peer_ablation,
    figure_lambda_sweep,
    write_metrics_table,
)


def _panel_list_from_artifacts(root: Path) -> list[tuple[str, str, str]]:
    """Auto-discover (label, benchmark, env_id) triples from runs_root."""
    panels = []
    for benchmark in sorted(p.name for p in root.iterdir() if p.is_dir()):
        for env_dir in sorted((root / benchmark).iterdir()):
            if not env_dir.is_dir() or env_dir.name.startswith("lam_"):
                continue
            label = f"{benchmark}/{env_dir.name}"
            panels.append((label, benchmark, env_dir.name))
    return panels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", type=Path, default=Path("artifacts/runs"))
    p.add_argument("--out-dir",   type=Path, default=Path("artifacts/figures"))
    p.add_argument("--config-dir", type=Path, default=Path("configs"))
    args = p.parse_args()

    panels = _panel_list_from_artifacts(args.runs_root)
    if not panels:
        print(f"[plots] no runs found under {args.runs_root}")
        return

    print(f"[plots] discovered {len(panels)} panels:")
    for lab, bm, eid in panels:
        print(f"  {lab}")

    # Figure 2 — basin-entry probability (one panel per (benchmark, env)).
    figure_2_basin_entry(args.runs_root, panels=panels, out_dir=args.out_dir,
                          title="Neural basin-entry probability across benchmarks")

    # Figure 3 — handoff retention.  Pick the first panel that has the handoff arm.
    for lab, bm, eid in panels:
        sub = args.runs_root / bm / eid
        if (sub / "handoff").exists():
            cfg_yaml = next((c for c in args.config_dir.glob(f"{bm}_*.yaml")
                             if eid in c.read_text()), None)
            T_warm = 200_000
            if cfg_yaml:
                T_warm = int(yaml.safe_load(cfg_yaml.read_text())["T_warm"])
            figure_3_handoff(args.runs_root, benchmark=bm, env_id=eid,
                              out_dir=args.out_dir / "handoff", T_warm=T_warm)
            break

    # Figure 4 — peer ablation.
    figure_4_peer_ablation(args.runs_root, panels=panels, out_dir=args.out_dir)

    # Lambda sweep figures (one per benchmark/env that has lam_* subdirs).
    for benchmark in sorted(p.name for p in args.runs_root.iterdir() if p.is_dir()):
        for env_dir in sorted((args.runs_root / benchmark).iterdir()):
            if any(c.name.startswith("lam_") for c in env_dir.iterdir() if c.is_dir()):
                figure_lambda_sweep(args.runs_root, benchmark=benchmark,
                                     env_id=env_dir.name,
                                     out_dir=args.out_dir / "lambda")

    # Metrics table.
    write_metrics_table(args.runs_root, panels=panels,
                         out_path=args.out_dir / "metrics.csv")
    print(f"[plots] wrote figures and metrics to {args.out_dir}")


if __name__ == "__main__":
    main()
