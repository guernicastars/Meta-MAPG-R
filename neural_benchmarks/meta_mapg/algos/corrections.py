"""LOLA-PG / Meta-MAPG opponent-aware corrections — *correctly* implemented.

The surrogate gradient for agent i is

    g_i = g_i^PG  +  λ_peer * Δ_peer^i  +  λ_own * Δ_own^i

with

    Δ_peer^i = η_inner * Σ_{j ≠ i} ∇_{θ_i} [ (∇_{θ_j} J_j)^⊤ (∇_{θ_j} J_i^DiCE) ]
    Δ_own^i  = η_inner * ∇_{θ_i} ‖ ∇_{θ_i} J_i ‖²
              = 2 η_inner · H_{ii} · g_i^PG    (Hessian-gradient product)

J_i^DiCE uses the "magic box" / DiCE operator (Foerster et al., 2018):

    𝕄(x) = exp(x - stop_grad(x))             # value 1, gradient = ∇x
    L̃    = Σ_t  A_i^t · 𝕄(Σ_t' (log π_i(a_i^{t'}) + Σ_{j≠i} log π_j(a_j^{t'})))

Why DiCE and not a plain sum of log-probs?  For *separate* actor networks
the cross-Hessian of  E[A_i · (log π_i + log π_j)] w.r.t. (θ_i, θ_j) is
**zero** (the only term involving both is the cross-derivative of
log π_i + log π_j, which vanishes because the two networks share no params).
The DiCE outer-product expansion

    ∇_{θ_i} ∇_{θ_j}  𝕄(log π_i + log π_j) = 𝕄 · ∇_{θ_i} log π_i · ∇_{θ_j} log π_j

picks up exactly the joint-trajectory cross-Hessian that LOLA-PG needs.

Plan compliance:
    §4 — "Apply correction to actor loss only" ✓ (we never touch critic)
    §4 — "Stop-grad critic/value estimates"    ✓ (`advantages.detach()`)
    §4 — "Clip ‖Δ_corr‖ ≤ c · ‖Δ_PG‖"          ✓ (in train.py)
    §4 — "Conservative clipping"               ✓ (default c = 1.0)
    §4 — "One or a few differentiable inner steps" ✓ (one-step LOLA)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from ..policies.actor_critic import MLPActor


# ---------------------------------------------------------------------------
# Rollout container (per-agent, batched along time)
# ---------------------------------------------------------------------------

@dataclass
class AgentRollout:
    obs: torch.Tensor          # [T, obs_dim]
    actions: torch.Tensor      # [T] (long)
    advantages: torch.Tensor   # [T]
    log_probs_old: torch.Tensor  # [T] (from rollout collection time, detached)


# ---------------------------------------------------------------------------
# Surrogates
# ---------------------------------------------------------------------------

def _pg_return(actor: MLPActor, ro: AgentRollout) -> torch.Tensor:
    """Vanilla policy-gradient *return* (positive sign).

        J_i^PG = E[A_i · log π_i(a_i)]

    Used for the meta-correction inner gradients only.  The outer training
    step uses a clipped PPO surrogate (see ippo.py)."""
    logp = actor.log_prob(ro.obs, ro.actions)
    return (ro.advantages.detach() * logp).mean()


def _dice_return(actor_i: MLPActor, peers: Iterable[tuple[MLPActor, AgentRollout]],
                 ro_i: AgentRollout) -> torch.Tensor:
    """DiCE / magic-box surrogate for agent i.

        joint = log π_i(a_i) + Σ_{j≠i} log π_j(a_j)
        𝕄    = exp(joint - stop_grad(joint))                  # value = 1
        L̃    = E[ A_i · 𝕄 ]                                   # value = E[A_i]

    Differentiating 𝕄 once gives the score-function cross-policy-gradient;
    differentiating twice gives the outer-product joint-trajectory
    cross-Hessian — exactly the term LOLA-PG needs.  Note that for *separate*
    networks the plain ``A_i · (log π_i + log π_j)`` form has zero
    cross-Hessian because ∂² (log π_i + log π_j) / ∂θ_i ∂θ_j = 0.  DiCE
    fixes this with the exponential expansion.
    """
    logp_i = actor_i.log_prob(ro_i.obs, ro_i.actions)
    joint = logp_i
    for actor_j, ro_j in peers:
        joint = joint + actor_j.log_prob(ro_j.obs, ro_j.actions)
    magic = torch.exp(joint - joint.detach())   # value 1, gradient = ∇joint
    return (ro_i.advantages.detach() * magic).mean()


# ---------------------------------------------------------------------------
# Correction computation
# ---------------------------------------------------------------------------

def compute_pg_grads(actor: MLPActor, ro: AgentRollout) -> list[torch.Tensor]:
    """Return ∇_θ (-J_i^PG), i.e. the *loss* gradient (so subtracting moves us up J).

    These are the standard policy-gradient direction the optimiser will follow.
    We return *loss* gradients (i.e. -J) so that downstream code can simply
    add them, matching standard PyTorch optimiser conventions."""
    J_i = _pg_return(actor, ro)
    grads = torch.autograd.grad(-J_i, list(actor.parameters()), create_graph=False,
                                allow_unused=True)
    return [_zero_or(g, p) for g, p in zip(grads, actor.parameters())]


def compute_meta_corrections(
    actor_i: MLPActor,
    peers: list[tuple[MLPActor, AgentRollout]],
    ro_i: AgentRollout,
    *,
    lam_peer: float,
    lam_own: float,
    eta_inner: float,
) -> list[torch.Tensor]:
    """Return the additive *loss* gradient correction for agent i.

    Sign convention: returned tensors are added to the PG-loss gradient,
    so they push the optimiser *away* from {peer-,own-}lookahead return.
    Equivalently, the *parameter* update is

        θ_i ← θ_i - α · (∇_θ_i loss^PG + correction)

    which is the same as

        θ_i ← θ_i + α · (∇_θ_i J_i^PG + λ_peer Δ_peer + λ_own Δ_own)

    with Δ_peer/Δ_own as defined in the module docstring.
    """
    if lam_peer == 0.0 and lam_own == 0.0:
        return [torch.zeros_like(p) for p in actor_i.parameters()]

    correction = [torch.zeros_like(p) for p in actor_i.parameters()]

    # --- peer term ----------------------------------------------------------
    if lam_peer > 0.0 and peers:
        # 1. Detached vector  g_j = ∇_{θ_j} J_j  for every peer j.
        g_j_detach: list[list[torch.Tensor]] = []
        for actor_j, ro_j in peers:
            J_j = _pg_return(actor_j, ro_j)
            g_j = torch.autograd.grad(J_j, list(actor_j.parameters()),
                                      create_graph=False, allow_unused=True)
            g_j_detach.append([_zero_or(g, p).detach()
                               for g, p in zip(g_j, actor_j.parameters())])

        # 2. DiCE surrogate J_i with grads enabled on θ_i and *all* θ_j.
        J_i_dice = _dice_return(actor_i, peers, ro_i)

        # 3. For every peer j: u_j = ∇_{θ_j} J_i^DiCE, with create_graph=True so
        #    we can take ∇_{θ_i} of the inner product.
        scalar = J_i_dice.new_zeros(())
        for (actor_j, _), gd_j in zip(peers, g_j_detach):
            u_j = torch.autograd.grad(J_i_dice, list(actor_j.parameters()),
                                       create_graph=True, retain_graph=True,
                                       allow_unused=True)
            for gd, u in zip(gd_j, u_j):
                if u is None:
                    continue
                scalar = scalar + (gd * u).sum()

        # 4. ∇_{θ_i} of the scalar = -correction (we want to *add* J, hence sign).
        peer_loss_grad = torch.autograd.grad(-eta_inner * scalar,
                                             list(actor_i.parameters()),
                                             allow_unused=True)
        for c, g in zip(correction, peer_loss_grad):
            if g is not None:
                c.add_(lam_peer * g)

    # --- own term -----------------------------------------------------------
    if lam_own > 0.0:
        J_i = _pg_return(actor_i, ro_i)
        g_i = torch.autograd.grad(J_i, list(actor_i.parameters()),
                                   create_graph=True, allow_unused=True)
        # ∇_{θ_i} ‖g_i‖² = 2 H_ii g_i
        norm_sq = sum((g * g).sum() for g in g_i if g is not None)
        own_loss_grad = torch.autograd.grad(-eta_inner * norm_sq,
                                             list(actor_i.parameters()),
                                             allow_unused=True)
        for c, g in zip(correction, own_loss_grad):
            if g is not None:
                c.add_(lam_own * g)

    return correction


def _zero_or(g: torch.Tensor | None, p: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(p) if g is None else g
