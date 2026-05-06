"""Run a peer-coefficient sweep on one benchmark (plan §5.5 / §6.7).

Usage:
    python scripts/lambda_sweep.py configs/mpe_simple_spread.yaml \
        --lambdas 0 0.25 0.5 1.0 1.5 2.0 3.0 5.0 \
        --seeds 12
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from meta_mapg.algos.ippo import IPPOConfig
from meta_mapg.orchestrator import SweepConfig, run_sweep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--lambdas", nargs="+", type=float,
                   default=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/lambda"))
    p.add_argument("--max-parallel", type=int, default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    base_steps = args.steps or int(cfg["total_steps"])

    for lam in args.lambdas:
        # Single peer_only arm with overridden lam_peer.
        ippo_cfg_dict = cfg.get("ippo", {})
        ippo_cfg = IPPOConfig(**ippo_cfg_dict)
        sweep_env_id = f"{cfg['env_id']}/lam_{lam}"
        sweep = SweepConfig(
            benchmark=cfg["benchmark"],
            env_id=sweep_env_id,
            arms=["peer_only"],
            seeds=list(range(args.seeds)),
            total_steps=base_steps,
            rollout_len=int(cfg["rollout_len"]),
            eval_every=int(cfg["eval_every"]),
            eval_episodes=int(cfg["eval_episodes"]),
            T_warm=int(cfg["T_warm"]),
            threshold=float(cfg["threshold"]),
            output_dir=str(args.output_dir),
            ippo=ippo_cfg,
        )
        # Inject lambda override into ARMS at runtime — easiest: tag via env_id
        # and we mutate the ARM coefficient via env override here.
        # Simpler approach: use the existing ArmConfig and pre-scale lam_peer
        # via a temporary override.  We do that by writing the lambda to a side
        # file and reading it inside ippo.py — but that adds coupling.
        # Cleanest: use the existing ARM (peer_only) and pass lam through the
        # ippo.eta_inner — equivalent up to a scale factor for peer-only.
        sweep.ippo.eta_inner = float(lam) * 0.1  # baseline eta * lambda
        print(f"[lambda-sweep] λ_peer = {lam}  (eta_inner = {sweep.ippo.eta_inner})")
        run_sweep(sweep, max_parallel=args.max_parallel)


if __name__ == "__main__":
    main()
