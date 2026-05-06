"""Sanity tests for the LOLA-PG / Meta-MAPG correction module.

These tests are *not* a substitute for empirical validation but they catch
the kind of math bugs that broke the previous repo (peer term computing
``g_peer · θ_peer`` instead of the cross-policy gradient).
"""
from __future__ import annotations

import torch

from meta_mapg.algos.corrections import (
    AgentRollout,
    compute_meta_corrections,
    compute_pg_grads,
)
from meta_mapg.policies import init_paired_actors


def _toy_rollout(obs_dim: int, T: int = 8, n_actions: int = 5, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(T, obs_dim, generator=g)
    actions = torch.randint(0, n_actions, (T,), generator=g)
    advantages = torch.randn(T, generator=g)
    log_probs_old = torch.zeros(T)
    return AgentRollout(obs=obs, actions=actions, advantages=advantages,
                        log_probs_old=log_probs_old)


def test_zero_lambdas_zero_correction():
    """λ_peer = λ_own = 0  ⇒  correction = 0."""
    actors, _ = init_paired_actors(seed=42, n_agents=2, obs_dim=4, n_actions=5)
    ros = [_toy_rollout(4, seed=k) for k in range(2)]
    peers = [(actors[1], ros[1])]
    corr = compute_meta_corrections(actors[0], peers, ros[0],
                                     lam_peer=0.0, lam_own=0.0, eta_inner=0.1)
    assert all((c == 0).all() for c in corr), "correction should be zero"


def test_peer_correction_is_nontrivial():
    """λ_peer > 0 should produce a non-zero peer-aware correction."""
    actors, _ = init_paired_actors(seed=42, n_agents=2, obs_dim=4, n_actions=5)
    ros = [_toy_rollout(4, seed=k) for k in range(2)]
    peers = [(actors[1], ros[1])]
    corr = compute_meta_corrections(actors[0], peers, ros[0],
                                     lam_peer=1.0, lam_own=0.0, eta_inner=0.1)
    norm = sum((c.detach() ** 2).sum().item() for c in corr) ** 0.5
    assert norm > 1e-6, f"peer correction is suspiciously small: {norm}"


def test_peer_term_is_NOT_g_times_theta():
    """Reproduce the previous bug check: peer term ≠ g_peer · θ_peer.

    The old broken code computed `(g_peer * θ_peer).sum()` for the peer term,
    which makes the peer correction independent of own actor's parameters.
    Here we verify our new correction *changes* when actor_i's params change."""
    actors, _ = init_paired_actors(seed=42, n_agents=2, obs_dim=4, n_actions=5)
    ros = [_toy_rollout(4, seed=k) for k in range(2)]
    peers = [(actors[1], ros[1])]
    corr_a = compute_meta_corrections(actors[0], peers, ros[0],
                                       lam_peer=1.0, lam_own=0.0, eta_inner=0.1)
    # Perturb actor_0 parameters; correction must change.
    with torch.no_grad():
        for p in actors[0].parameters():
            p.add_(torch.randn_like(p) * 0.5)
    corr_b = compute_meta_corrections(actors[0], peers, ros[0],
                                       lam_peer=1.0, lam_own=0.0, eta_inner=0.1)
    diffs = sum(((a - b) ** 2).sum().item() for a, b in zip(corr_a, corr_b))
    assert diffs > 1e-4, "peer correction does not depend on actor_i — bug!"


def test_own_correction_is_nontrivial():
    actors, _ = init_paired_actors(seed=42, n_agents=2, obs_dim=4, n_actions=5)
    ros = [_toy_rollout(4, seed=k) for k in range(2)]
    peers = [(actors[1], ros[1])]
    corr = compute_meta_corrections(actors[0], peers, ros[0],
                                     lam_peer=0.0, lam_own=0.5, eta_inner=0.1)
    norm = sum((c.detach() ** 2).sum().item() for c in corr) ** 0.5
    assert norm > 1e-7, f"own correction is suspiciously small: {norm}"


def test_pg_grads_finite():
    actors, _ = init_paired_actors(seed=42, n_agents=2, obs_dim=4, n_actions=5)
    ros = [_toy_rollout(4, seed=k) for k in range(2)]
    grads = compute_pg_grads(actors[0], ros[0])
    assert all(torch.isfinite(g).all() for g in grads), "PG grads contain non-finite values"


def test_paired_init_is_identical():
    """Two construction calls with the same seed must produce identical weights."""
    a1, _ = init_paired_actors(seed=7, n_agents=2, obs_dim=4, n_actions=5)
    a2, _ = init_paired_actors(seed=7, n_agents=2, obs_dim=4, n_actions=5)
    for p1, p2 in zip(a1[0].parameters(), a2[0].parameters()):
        assert torch.allclose(p1, p2), "paired init not deterministic"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(f"  {name} ... ", end="")
            fn()
            print("ok")
    print("all correction tests passed")
