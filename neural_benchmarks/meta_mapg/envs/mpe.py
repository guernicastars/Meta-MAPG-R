"""PettingZoo MPE wrappers (engineering plan §5)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Threshold definitions (see plan §5.4) — chosen post-pilot, fixed for paper.
# These reflect "high-return" classification used to compute basin-entry
# probability.  Actual values can be tuned per-pilot via configs/<env>.yaml.
DEFAULT_RETURN_THRESHOLDS = {
    "simple_spread":           -25.0,   # better (less negative) → all landmarks covered
    "simple_reference":         -10.0,
    "simple_speaker_listener":  -10.0,
}


@dataclass
class MPESpec:
    env_id: str
    n_agents: int
    obs_dims: list[int]
    n_actions: int
    max_cycles: int = 25


def _import_mpe(env_id: str):
    # Strip sweep tags like "simple_spread/lam_0.25" → "simple_spread".
    env_id = env_id.split("/", 1)[0]
    """Support both legacy ``pettingzoo.mpe`` (≤1.24) and modern ``mpe2`` (≥1.25)."""
    pkgs = ("mpe2", "pettingzoo.mpe")
    submod = {
        "simple_spread":            "simple_spread_v3",
        "simple_reference":         "simple_reference_v3",
        "simple_speaker_listener":  "simple_speaker_listener_v4",
    }.get(env_id)
    if submod is None:
        raise ValueError(f"unsupported MPE env: {env_id}")
    last_err = None
    for pkg in pkgs:
        try:
            return __import__(f"{pkg}.{submod}", fromlist=[submod])
        except ImportError as e:
            last_err = e
    raise ImportError(
        f"could not import MPE env {env_id} from any of {pkgs}.  "
        f"Install with `uv pip install mpe2` (recommended) or "
        f"`uv pip install 'pettingzoo[mpe]<1.25'`. Last error: {last_err!r}"
    )


class MPEEnv:
    """Thin numpy wrapper around PettingZoo's parallel MPE API."""

    def __init__(self, env_id: str, *, seed: int, max_cycles: int = 25):
        env_id = env_id.split("/", 1)[0]
        mod = _import_mpe(env_id)
        kwargs: dict = {"max_cycles": max_cycles, "continuous_actions": False}
        if env_id == "simple_spread":
            kwargs["N"] = 3
        if env_id == "simple_reference":
            kwargs["local_ratio"] = 0.5
        self.env = mod.parallel_env(**kwargs)
        self.env_id = env_id
        self.max_cycles = max_cycles
        self._seed = seed
        self._t = 0

        obs, _ = self.env.reset(seed=seed)
        self.agents = list(obs.keys())
        self.spec = MPESpec(
            env_id=env_id,
            n_agents=len(self.agents),
            obs_dims=[int(np.array(obs[a]).shape[0]) for a in self.agents],
            n_actions=int(self.env.action_space(self.agents[0]).n),
            max_cycles=max_cycles,
        )
        self._last_obs = obs
        self._episode_returns = {a: 0.0 for a in self.agents}

    # ---- API expected by train.py ----------------------------------------

    def reset(self, *, seed: int | None = None) -> list[np.ndarray]:
        obs, _ = self.env.reset(seed=seed)
        self._last_obs = obs
        self._t = 0
        self._episode_returns = {a: 0.0 for a in self.agents}
        return [np.array(obs[a], dtype=np.float32) for a in self.agents]

    def step(self, actions: list[int]) -> tuple[list[np.ndarray], list[float], list[bool], dict]:
        action_dict = {a: int(actions[i]) for i, a in enumerate(self.agents)}
        next_obs, rewards, terms, truncs, _info = self.env.step(action_dict)
        self._t += 1

        # PettingZoo returns empty dicts when the episode ends.  Use last_obs
        # to keep stable shapes; the trainer will reset on episode end.
        if not next_obs:
            obs_out = [np.zeros(self.spec.obs_dims[i], dtype=np.float32)
                       for i in range(self.spec.n_agents)]
            done_out = [True] * self.spec.n_agents
            r_out    = [0.0]  * self.spec.n_agents
        else:
            obs_out  = [np.array(next_obs[a], dtype=np.float32) for a in self.agents]
            r_out    = [float(rewards[a]) for a in self.agents]
            done_out = [bool(terms.get(a, False) or truncs.get(a, False))
                        for a in self.agents]
            for a, r in rewards.items():
                self._episode_returns[a] += float(r)
        self._last_obs = next_obs
        return obs_out, r_out, done_out, {"episode_returns": dict(self._episode_returns)}

    @property
    def episode_returns(self) -> list[float]:
        return [self._episode_returns[a] for a in self.agents]

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


def make_mpe(env_id: str, *, seed: int) -> MPEEnv:
    return MPEEnv(env_id, seed=seed)


__all__ = ["MPEEnv", "MPESpec", "make_mpe", "DEFAULT_RETURN_THRESHOLDS"]
