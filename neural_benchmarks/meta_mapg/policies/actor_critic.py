"""Actor + critic networks (separate per agent for clean LOLA gradients).

We deliberately do *not* share parameters across agents.  Sharing is fine
for vanilla IPPO but breaks the algebra of the Meta-MAPG corrections,
which require ``∇_{θ_j} J_i`` with ``θ_j`` distinct from ``θ_i``.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import set_global_seed


def _orthogonal(layer: nn.Linear, gain: float = math.sqrt(2)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class MLPActor(nn.Module):
    """Discrete-action categorical policy with optional continuous output."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple[int, ...] = (64, 64)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden:
            layers += [_orthogonal(nn.Linear(prev, h)), nn.Tanh()]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.head = _orthogonal(nn.Linear(prev, n_actions), gain=0.01)

    def logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(obs))

    def dist(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.logits(obs))

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.dist(obs).log_prob(action)

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        return self.dist(obs).entropy()

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.dist(obs)
        a = d.sample()
        return a, d.log_prob(a)


class MLPCritic(nn.Module):
    """Per-agent value head V(o_i)."""

    def __init__(self, obs_dim: int, hidden: tuple[int, ...] = (64, 64)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden:
            layers += [_orthogonal(nn.Linear(prev, h)), nn.Tanh()]
            prev = h
        layers += [_orthogonal(nn.Linear(prev, 1), gain=1.0)]
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def init_paired_actors(
    seed: int,
    n_agents: int,
    obs_dim: int | list[int],
    n_actions: int,
    *,
    hidden: tuple[int, ...] = (64, 64),
    device: str = "cpu",
) -> tuple[list[MLPActor], list[MLPCritic]]:
    """Construct (actors, critics) with deterministic init from `seed`.

    The same seed *must* yield the same initial state-dict, so different arms
    can be compared head-to-head from identical starting policies (engineering
    plan §8).

    `obs_dim` can be a single int (homogeneous agents — simple_spread) or a
    list of per-agent ints (heterogeneous — simple_speaker_listener has
    speaker obs_dim=3, listener obs_dim=11).
    """
    set_global_seed(seed)
    if isinstance(obs_dim, int):
        dims = [obs_dim] * n_agents
    else:
        dims = list(obs_dim)
        assert len(dims) == n_agents, f"obs_dim len {len(dims)} != n_agents {n_agents}"
    actors = [MLPActor(dims[i], n_actions, hidden).to(device) for i in range(n_agents)]
    critics = [MLPCritic(dims[i], hidden).to(device) for i in range(n_agents)]
    return actors, critics


def actor_params(actor: MLPActor) -> list[torch.Tensor]:
    return [p for p in actor.parameters() if p.requires_grad]
