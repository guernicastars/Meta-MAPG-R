"""Overcooked-AI wrapper (engineering plan §6).

Uses Stanford's overcooked-ai-py.  Layouts are exposed by name:
    "cramped_room"
    "coordination_ring"
    "forced_coordination"
    "asymmetric_advantages"
    "counter_circuit_o_1order"

Plan §6.4 thresholds are layout-dependent; we expose defaults that should be
re-tuned with a pilot run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_RETURN_THRESHOLDS = {
    "cramped_room":            10.0,    # reward per soup ≈ 20, threshold = 1 soup avg
    "coordination_ring":        5.0,
    "forced_coordination":      3.0,
    "asymmetric_advantages":   10.0,
    "counter_circuit_o_1order": 3.0,
}


@dataclass
class OvercookedSpec:
    layout: str
    n_agents: int
    obs_dim: int
    n_actions: int
    max_cycles: int


class OvercookedEnv:
    """Wrap overcooked-ai's MDP into a PettingZoo-style parallel API."""

    def __init__(self, layout: str, *, seed: int, horizon: int = 400):
        try:
            from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
            from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv as _OEnv
            from overcooked_ai_py.mdp.actions import Action
        except ImportError as e:
            raise ImportError(
                "overcooked-ai is not installed.  Run:  uv pip install overcooked-ai"
            ) from e

        mdp = OvercookedGridworld.from_layout_name(layout)
        self._inner = _OEnv.from_mdp(mdp, horizon=horizon)
        self._Action = Action
        self.layout = layout
        self._seed = seed
        self._t = 0
        self._horizon = horizon

        # Featurise to lossless flat vector ("featurize_state_mdp" if present,
        # otherwise lossless_state_encoding flattened).
        self._inner.reset()
        try:
            feats = mdp.featurize_state_mdp(self._inner.state)
            self._featurise = lambda s: mdp.featurize_state_mdp(s)
            obs0 = feats[0]
        except Exception:
            enc = mdp.lossless_state_encoding(self._inner.state)
            self._featurise = lambda s: mdp.lossless_state_encoding(s)
            obs0 = np.array(enc[0]).flatten()

        self.spec = OvercookedSpec(
            layout=layout,
            n_agents=2,
            obs_dim=int(np.asarray(obs0).flatten().shape[0]),
            n_actions=len(self._Action.ALL_ACTIONS),
            max_cycles=horizon,
        )
        self._episode_returns = [0.0, 0.0]

    def _flatten(self, ob) -> np.ndarray:
        return np.asarray(ob, dtype=np.float32).flatten()

    def reset(self, *, seed: int | None = None) -> list[np.ndarray]:
        if seed is not None:
            np.random.seed(seed)
        self._inner.reset()
        feats = self._featurise(self._inner.state)
        self._t = 0
        self._episode_returns = [0.0, 0.0]
        return [self._flatten(feats[i]) for i in range(2)]

    def step(self, actions: list[int]):
        a_tuple = (self._Action.ALL_ACTIONS[int(actions[0])],
                   self._Action.ALL_ACTIONS[int(actions[1])])
        next_state, reward, done, info = self._inner.step(a_tuple)
        feats = self._featurise(next_state)
        obs = [self._flatten(feats[i]) for i in range(2)]
        # Overcooked sparse reward (soup delivery = +20) + shaped reward
        # (onion pickup, dish handling, etc.) — without shaping, vanilla PPO
        # cannot learn coordination_ring at 1M steps.  Shaping coef 0.5 per
        # Overcooked-AI defaults; sparse reward split equally between agents.
        shaped = info.get("shaped_r_by_agent", [0.0, 0.0])
        per_agent_rew = [float(reward) / 2.0 + 0.5 * float(shaped[i]) for i in range(2)]
        self._episode_returns = [self._episode_returns[i] + per_agent_rew[i] for i in range(2)]
        self._t += 1
        d = bool(done) or self._t >= self._horizon
        return obs, per_agent_rew, [d, d], {
            "episode_returns": list(self._episode_returns),
            "soups_delivered": float(info.get("episode", {}).get("ep_sparse_r", 0)) / 20.0
                if info else 0.0,
            "shaped_reward":   float(info.get("shaped_r_by_agent", [0, 0])[0])
                if info else 0.0,
        }

    @property
    def episode_returns(self) -> list[float]:
        return list(self._episode_returns)

    def close(self):
        pass


def make_overcooked(layout: str, *, seed: int) -> OvercookedEnv:
    return OvercookedEnv(layout, seed=seed)


__all__ = ["OvercookedEnv", "OvercookedSpec", "make_overcooked", "DEFAULT_RETURN_THRESHOLDS"]
