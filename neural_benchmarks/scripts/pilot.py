"""Pilot run: train a few IPPO seeds, inspect return distribution, recommend a
threshold (plan §5.4 / §6.4 / §7.4).

The pilot does NOT change the config in place — it prints a recommended
threshold which the user decides whether to update.

Usage:
    python scripts/pilot.py configs/mpe_simple_spread.yaml --pilot-seeds 4 --pilot-steps 250000
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path

import numpy as np
import yaml

from meta_mapg.algos.ippo import IPPOConfig
from meta_mapg.orchestrator import SweepConfig, run_sweep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--pilot-seeds", type=int, default=4)
    p.add_argument("--pilot-steps", type=int, default=None,
                   help="defaults to total_steps / 4 from config")
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/pilot"))
    p.add_argument("--max-parallel", type=int, default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    pilot_steps = args.pilot_steps or max(int(cfg["total_steps"]) // 4, 50_000)

    sweep = SweepConfig(
        benchmark=cfg["benchmark"],
        env_id=cfg["env_id"],
        arms=["ippo"],                                   # IPPO baseline only
        seeds=list(range(args.pilot_seeds)),
        total_steps=pilot_steps,
        rollout_len=int(cfg["rollout_len"]),
        eval_every=max(pilot_steps // 4, 10_000),
        eval_episodes=int(cfg["eval_episodes"]),
        T_warm=int(cfg["T_warm"]),
        threshold=float(cfg["threshold"]),
        output_dir=str(args.output_dir),
        ippo=IPPOConfig(**cfg.get("ippo", {})),
    )
    print(f"[pilot] {sweep.benchmark}/{sweep.env_id}: "
          f"{args.pilot_seeds} seeds × {pilot_steps} steps")
    run_sweep(sweep, max_parallel=args.max_parallel)

    # Aggregate final eval returns and propose a threshold at the 75th percentile
    # of IPPO finals — that's the "high-return basin" used by the plan.
    finals = []
    runs_dir = Path(args.output_dir) / sweep.benchmark / sweep.env_id / "ippo"
    for sd_dir in runs_dir.glob("seed_*"):
        sj = sd_dir / "summary.json"
        if sj.exists():
            data = json.loads(sj.read_text())
            if "final_eval" in data:
                finals.append(float(data["final_eval"]["eval_return_mean"]))
    if not finals:
        print("[pilot] no finals collected — something failed.")
        return

    pct = np.percentile(finals, [25, 50, 75, 90])
    suggested = float(pct[2])
    print(f"[pilot] IPPO final eval returns: "
          f"min={min(finals):.2f} med={stats.median(finals):.2f} max={max(finals):.2f}")
    print(f"[pilot] percentiles 25/50/75/90: {pct.tolist()}")
    print(f"[pilot] *** suggested threshold *** = {suggested:.4f}  (75th percentile)")

    # Persist the threshold so run_sweep.py can pick it up automatically.
    pilot_root = Path(args.output_dir) / sweep.benchmark / sweep.env_id
    threshold_file = pilot_root / "threshold.txt"
    threshold_file.write_text(f"{suggested:.6f}\n")
    print(f"[pilot] wrote {threshold_file}")
    print(f"[pilot] run_sweep.py picks this up automatically via --threshold-file "
          f"or PILOT_DIR env var.")


if __name__ == "__main__":
    main()
