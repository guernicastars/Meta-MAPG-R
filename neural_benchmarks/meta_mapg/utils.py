"""Shared utilities: paired seeds, logging, bootstrap CIs, JSON-line writers."""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ---------------- seeding -------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Make a run deterministic *enough* — exact reproducibility on GPU is
    not promised, but the same seed across arms gives paired comparisons."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def paired_seed_init(seed: int) -> None:
    """Call once at the start of every (arm, seed) run before constructing
    networks.  This guarantees identical initial weights across arms for the
    same seed — the §8 paired-seed protocol."""
    set_global_seed(seed)


# ---------------- structured logging -------------------------------------

class JsonlLogger:
    """Append-only JSON-Lines logger.  Safe under nohup."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", buffering=1)  # line-buffered

    def log(self, **kwargs: Any) -> None:
        record = {"_t": time.time(), **kwargs}
        self._fh.write(json.dumps(record, default=_json_default) + "\n")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


# ---------------- statistics ---------------------------------------------

def bootstrap_ci(samples: np.ndarray, *, alpha: float = 0.05,
                 n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap CI for the mean.  Returns (mean, low, high)."""
    samples = np.asarray(samples, dtype=float)
    if samples.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = samples.size
    idx = rng.integers(0, n, size=(n_boot, n))
    means = samples[idx].mean(axis=1)
    return (
        float(samples.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def wilson_ci(successes: int, n: int, *, alpha: float = 0.05) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.  Returns (p, lo, hi)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    from scipy import stats
    z = float(stats.norm.ppf(1 - alpha / 2))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


# ---------------- norm helpers -------------------------------------------

def grad_norm(grads) -> torch.Tensor:
    """L2 norm over a list of gradient tensors."""
    sqsum = sum((g.detach() ** 2).sum() for g in grads if g is not None)
    return sqsum.sqrt()


def clip_correction(pg_grads, corr_grads, *, c: float):
    """Clip ‖Δ_corr‖ ≤ c · ‖Δ_PG‖ (engineering plan §4)."""
    pg_n = grad_norm(pg_grads)
    cn = grad_norm(corr_grads)
    if cn > c * pg_n:
        scale = (c * pg_n) / (cn + 1e-12)
        return [g * scale for g in corr_grads], float(scale)
    return corr_grads, 1.0


def cosine(a_grads, b_grads) -> float:
    """Cosine similarity between two flat gradient lists."""
    fa = torch.cat([g.detach().flatten() for g in a_grads if g is not None])
    fb = torch.cat([g.detach().flatten() for g in b_grads if g is not None])
    n = (fa.norm() * fb.norm()).clamp_min(1e-12)
    return float((fa @ fb) / n)


# ---------------- run config ---------------------------------------------

@dataclass
class RunConfig:
    benchmark: str            # "mpe" | "overcooked" | "meltingpot"
    env_id: str
    arm: str
    seed: int
    total_steps: int
    T_warm: int = 0           # used by handoff arm
    output_dir: str = "artifacts"
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=_json_default)
