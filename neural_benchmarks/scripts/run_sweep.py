"""Run a full benchmark sweep from a YAML config.

Usage:
    python scripts/run_sweep.py configs/mpe_simple_spread.yaml
    python scripts/run_sweep.py configs/overcooked_ring.yaml --max-parallel 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from meta_mapg.algos.ippo import IPPOConfig
from meta_mapg.orchestrator import SweepConfig, run_sweep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/runs"))
    p.add_argument("--max-parallel", type=int, default=None,
                   help="defaults to torch.cuda.device_count()")
    p.add_argument("--seeds", type=int, default=None,
                   help="override seed count from config (truncate to first N seeds)")
    p.add_argument("--arms", nargs="+", default=None,
                   help="override the arm list from config")
    p.add_argument("--threshold", type=float, default=None,
                   help="override threshold from config")
    p.add_argument("--total-steps", type=int, default=None,
                   help="override total training steps per run (smoke / debug)")
    p.add_argument("--eval-every", type=int, default=None,
                   help="override eval cadence")
    p.add_argument("--threshold-file", type=Path, default=None,
                   help="read a single float threshold from this file (e.g. pilot output)")
    p.add_argument("--pilot-dir", type=Path, default=None,
                   help="if set, look for {pilot-dir}/{benchmark}/{env}/threshold.txt and use it")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg_yaml = yaml.safe_load(args.config.read_text())
    n_seeds = args.seeds or cfg_yaml["seeds"]
    arms = args.arms or cfg_yaml["arms"]
    seeds = list(range(int(n_seeds)))
    ippo_cfg = IPPOConfig(**cfg_yaml.get("ippo", {}))

    # Threshold resolution priority: --threshold > --threshold-file > --pilot-dir > YAML.
    threshold = float(cfg_yaml["threshold"])
    pilot_threshold = None
    if args.pilot_dir is not None:
        candidate = args.pilot_dir / cfg_yaml["benchmark"] / cfg_yaml["env_id"] / "threshold.txt"
        if candidate.exists():
            pilot_threshold = float(candidate.read_text().strip())
            print(f"[run_sweep] using pilot threshold from {candidate}: {pilot_threshold}")
    if args.threshold_file is not None and args.threshold_file.exists():
        pilot_threshold = float(args.threshold_file.read_text().strip())
        print(f"[run_sweep] using threshold from {args.threshold_file}: {pilot_threshold}")
    if pilot_threshold is not None:
        threshold = pilot_threshold
    if args.threshold is not None:
        threshold = float(args.threshold)
        print(f"[run_sweep] using --threshold override: {threshold}")
    print(f"[run_sweep] effective threshold = {threshold}")

    total_steps = int(args.total_steps if args.total_steps is not None else cfg_yaml["total_steps"])
    eval_every  = int(args.eval_every  if args.eval_every  is not None else cfg_yaml["eval_every"])
    sweep = SweepConfig(
        benchmark=cfg_yaml["benchmark"],
        env_id=cfg_yaml["env_id"],
        arms=arms,
        seeds=seeds,
        total_steps=total_steps,
        rollout_len=int(cfg_yaml["rollout_len"]),
        eval_every=eval_every,
        eval_episodes=int(cfg_yaml["eval_episodes"]),
        T_warm=int(cfg_yaml["T_warm"]),
        threshold=threshold,
        output_dir=str(args.output_dir),
        ippo=ippo_cfg,
    )
    print(json.dumps({
        "benchmark": sweep.benchmark, "env_id": sweep.env_id,
        "arms": sweep.arms, "n_seeds": len(sweep.seeds),
        "total_steps": sweep.total_steps, "T_warm": sweep.T_warm,
    }, indent=2))
    run_sweep(sweep, max_parallel=args.max_parallel, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
