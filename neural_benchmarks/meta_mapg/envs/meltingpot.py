"""Melting Pot wrapper with PettingZoo fallback (engineering plan §7).

`dm_meltingpot` has heavy native dependencies (dmlab2d, bazel build).  When
it is *not* installed we fall back to a small PettingZoo classic social-
dilemma — `prison_dilemma_v3` extended to N players — which captures the
qualitative cooperation/defection equilibrium-selection axis required by the
plan §7 narrative.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_RETURN_THRESHOLDS = {
    # dm_meltingpot substrates
    "prisoners_dilemma_in_the_matrix__repeated":  3.0,    # social welfare per step
    "commons_harvest__open":                      8.0,
    "clean_up":                                   5.0,
    "collaborative_cooking__cramped":             1.0,
    # PettingZoo fallback
    "ipd":                                        0.5,    # average per-step return
}


@dataclass
class MeltingPotSpec:
    substrate: str
    n_agents: int
    obs_dim: int
    n_actions: int
    max_cycles: int


# ---------------------------------------------------------------------------
# Native dm_meltingpot path
# ---------------------------------------------------------------------------

class MeltingPotNative:
    """Adapter for `dm_meltingpot` substrates."""

    def __init__(self, substrate: str, *, seed: int):
        from meltingpot import substrate as _substrate
        from ml_collections import config_dict

        cfg = _substrate.get_config(substrate)
        self._env = _substrate.build(substrate, roles=cfg.default_player_roles, scenario=None)
        self.substrate = substrate
        self._seed = seed

        ts = self._env.reset()
        n_agents = len(ts.observation)
        first_obs = ts.observation[0]
        # Use the RGB observation channel and flatten — small substrates only.
        flat = np.asarray(first_obs.get("RGB", first_obs)).flatten()
        self.spec = MeltingPotSpec(
            substrate=substrate,
            n_agents=n_agents,
            obs_dim=int(flat.shape[0]),
            n_actions=int(self._env.action_spec()[0].num_values),
            max_cycles=1000,
        )
        self._episode_returns = [0.0] * n_agents

    def reset(self, *, seed: int | None = None) -> list[np.ndarray]:
        ts = self._env.reset()
        self._episode_returns = [0.0] * self.spec.n_agents
        return [_obs_flat(o) for o in ts.observation]

    def step(self, actions):
        ts = self._env.step(list(map(int, actions)))
        rewards = [float(r) for r in ts.reward] if ts.reward is not None else [0.0]*self.spec.n_agents
        for i, r in enumerate(rewards):
            self._episode_returns[i] += r
        done = ts.last()
        return ([_obs_flat(o) for o in ts.observation], rewards,
                [bool(done)]*self.spec.n_agents,
                {"episode_returns": list(self._episode_returns)})

    @property
    def episode_returns(self):
        return list(self._episode_returns)

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass


def _obs_flat(o):
    if isinstance(o, dict):
        v = o.get("RGB", next(iter(o.values())))
    else:
        v = o
    return np.asarray(v, dtype=np.float32).flatten() / 255.0


# ---------------------------------------------------------------------------
# PettingZoo fallback: iterated prisoner's dilemma with N players
# ---------------------------------------------------------------------------

class IPDFallback:
    """N-player iterated prisoner's dilemma (matrix-style social dilemma).

    Used when dm_meltingpot is unavailable.  Captures the equilibrium-
    selection / basin-entry axis the plan §7 narrative needs."""

    def __init__(self, *, n_agents: int = 2, max_cycles: int = 100, seed: int = 0):
        self.spec = MeltingPotSpec(
            substrate="ipd",
            n_agents=n_agents,
            # Observation = prev joint actions one-hot encoded.
            obs_dim=2 * n_agents,
            n_actions=2,
            max_cycles=max_cycles,
        )
        self._t = 0
        self._prev_actions = np.zeros(n_agents, dtype=np.int64)
        self._episode_returns = [0.0] * n_agents
        self._rng = np.random.default_rng(seed)

    def _obs(self) -> list[np.ndarray]:
        n = self.spec.n_agents
        oh = np.zeros((n, 2 * n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                oh[i, 2 * j + int(self._prev_actions[j])] = 1.0
        return [oh[i] for i in range(n)]

    def reset(self, *, seed: int | None = None) -> list[np.ndarray]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._prev_actions = np.zeros(self.spec.n_agents, dtype=np.int64)
        self._episode_returns = [0.0] * self.spec.n_agents
        return self._obs()

    def step(self, actions):
        # 0 = cooperate, 1 = defect.  Symmetric N-player payoff:
        # r_i = 3 * (#coops by others) - 5 * a_i + 5 * (a_i == 1) * 0
        # Equivalent to standard PD pairwise averaged.
        a = np.asarray(actions, dtype=np.int64)
        n = self.spec.n_agents
        r = np.zeros(n, dtype=np.float32)
        for i in range(n):
            others_coop = int(((a == 0).sum() - (a[i] == 0)))
            # cooperate (a=0): get 3 per cooperating peer, lose 1 (cost)
            # defect    (a=1): get 5 per cooperating peer, lose 0
            if a[i] == 0:
                r[i] = 3.0 * others_coop / max(n - 1, 1) - 1.0
            else:
                r[i] = 5.0 * others_coop / max(n - 1, 1)
        self._prev_actions = a
        self._t += 1
        for i in range(n):
            self._episode_returns[i] += float(r[i])
        done = self._t >= self.spec.max_cycles
        return (self._obs(), [float(x) for x in r], [done]*n,
                {"episode_returns": list(self._episode_returns),
                 "cooperation_rate": float((a == 0).mean())})

    @property
    def episode_returns(self):
        return list(self._episode_returns)

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_meltingpot(substrate: str, *, seed: int):
    """Try the native substrate; fall back to IPD on ImportError."""
    if substrate in ("ipd", "fallback_ipd"):
        return IPDFallback(seed=seed)
    try:
        return MeltingPotNative(substrate, seed=seed)
    except Exception as e:
        import warnings
        warnings.warn(f"dm_meltingpot unavailable ({e!r}) — falling back to IPD.")
        return IPDFallback(seed=seed)


__all__ = ["MeltingPotSpec", "MeltingPotNative", "IPDFallback",
           "make_meltingpot", "DEFAULT_RETURN_THRESHOLDS"]
