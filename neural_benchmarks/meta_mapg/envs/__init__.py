"""Environment factories — one entry per benchmark family."""
from __future__ import annotations

from typing import Any, Callable

from .mpe import make_mpe
from .overcooked import make_overcooked
from .meltingpot import make_meltingpot

EnvFactory = Callable[..., Any]


def make_env(benchmark: str, env_id: str, *, seed: int) -> Any:
    if benchmark == "mpe":
        return make_mpe(env_id, seed=seed)
    if benchmark == "overcooked":
        return make_overcooked(env_id, seed=seed)
    if benchmark == "meltingpot":
        return make_meltingpot(env_id, seed=seed)
    raise ValueError(f"unknown benchmark: {benchmark}")


__all__ = ["make_env", "make_mpe", "make_overcooked", "make_meltingpot"]
