"""Multi-GPU orchestrator: dispatch (arm × seed) jobs across visible devices.

Implements the engineering plan §8 paired-seed comparison:
* same `seed` is used to initialise the policy across all five arms;
* each (arm, seed) is one process bound to one GPU.

Round-robins jobs across `torch.cuda.device_count()` GPUs.  Long-running
runs use `multiprocessing.spawn` (so each child gets its own CUDA context).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .algos.ippo import IPPOConfig
from .train import TrainConfig, train


@dataclass
class SweepConfig:
    benchmark: str
    env_id: str
    arms: list[str]
    seeds: list[int]
    total_steps: int    = 500_000
    eval_every:  int    = 50_000
    eval_episodes: int  = 50
    rollout_len: int    = 256
    T_warm:      int    = 100_000
    threshold:   float  = -25.0
    output_dir:  str    = "artifacts/runs"
    device_pool: list[int] | None = None
    ippo:        IPPOConfig = field(default_factory=IPPOConfig)


def _worker(payload: dict):
    """Subprocess entry point.  Pin to one GPU and call `train`."""
    gpu = payload["gpu"]
    cfg_dict = payload["cfg"]
    if gpu is not None and gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Re-import torch inside the subprocess so CUDA inits with the right env var.
    import torch as _torch
    cfg_dict["device"] = "cuda" if _torch.cuda.is_available() else "cpu"
    # Reconstruct dataclasses.
    ippo = IPPOConfig(**cfg_dict.pop("ippo", {}))
    cfg = TrainConfig(**cfg_dict, ippo=ippo)
    try:
        return train(cfg)
    except Exception as e:
        import sys, traceback
        tb = traceback.format_exc()
        # Loud error so a failing arm/seed is visible in nohup logs.
        print(f"[worker FAIL] {cfg.benchmark}/{cfg.env_id}/{cfg.arm}/seed_{cfg.seed} :: {e!r}",
              file=sys.stderr, flush=True)
        print(tb, file=sys.stderr, flush=True)
        return {"error": str(e), "trace": tb, "config": cfg_dict}


def _build_jobs(sweep: SweepConfig) -> list[dict]:
    jobs = []
    for arm in sweep.arms:
        for seed in sweep.seeds:
            jobs.append({
                "benchmark":     sweep.benchmark,
                "env_id":        sweep.env_id,
                "arm":           arm,
                "seed":          seed,
                "total_steps":   sweep.total_steps,
                "rollout_len":   sweep.rollout_len,
                "eval_every":    sweep.eval_every,
                "eval_episodes": sweep.eval_episodes,
                "T_warm":        sweep.T_warm,
                "threshold":     sweep.threshold,
                "output_dir":    sweep.output_dir,
                "device":        "cuda",
                "ippo":          asdict(sweep.ippo),
            })
    return jobs


def run_sweep(sweep: SweepConfig, *, max_parallel: int | None = None,
              dry_run: bool = False) -> list[dict]:
    """Run all (arm, seed) jobs.  GPU-round-robin across `device_pool`."""
    import torch
    pool = sweep.device_pool
    if pool is None:
        pool = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [-1]
    n_gpu = len(pool)
    if max_parallel is None:
        max_parallel = n_gpu

    jobs = _build_jobs(sweep)
    print(f"[orchestrator] {len(jobs)} jobs | {n_gpu} GPU(s) | parallel={max_parallel}")
    if dry_run:
        for j in jobs[:5]:
            print(" ", j["arm"], "seed", j["seed"])
        return []

    # Tag each job with a GPU.
    payloads = []
    for k, job in enumerate(jobs):
        gpu = pool[k % n_gpu]
        payloads.append({"cfg": job, "gpu": gpu})

    ctx = mp.get_context("spawn")
    results: list[dict] = []
    t0 = time.time()
    with ctx.Pool(max_parallel) as p:
        for k, res in enumerate(p.imap_unordered(_worker, payloads)):
            done = k + 1
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-3)
            eta = (len(payloads) - done) / max(rate, 1e-6)
            print(f"[orchestrator] {done}/{len(payloads)} done "
                  f"| {elapsed/60:.1f} min elapsed | ETA {eta/60:.1f} min")
            results.append(res)

    out = Path(sweep.output_dir) / sweep.benchmark / sweep.env_id / "sweep_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    return results
