"""IPPO trainer: independent PPO per agent + Meta-MAPG actor corrections.

Design notes
------------
* One **actor** and one **critic** per agent (no parameter sharing — see
  ``policies.actor_critic``).
* Critic loss is plain MSE; actor loss is the standard clipped-PPO surrogate.
* Meta-MAPG corrections (``corrections.compute_meta_corrections``) are added
  as **additive gradients** on top of the actor PG-loss gradient — *not*
  baked into the loss tensor.  This keeps PPO's clip + entropy bonus untouched
  and matches the engineering plan §4 ("apply correction to actor loss only").
* GAE-λ advantages, normalised per-rollout.
* Mini-batch updates, ``n_epochs`` of mini-batch SGD per rollout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..policies.actor_critic import MLPActor, MLPCritic
from ..utils import clip_correction, cosine, grad_norm
from .arms import get_arm_coefficients
from .corrections import AgentRollout, compute_meta_corrections, compute_pg_grads


# ---------------------------------------------------------------------------
# Buffers + GAE
# ---------------------------------------------------------------------------

@dataclass
class RolloutBuffer:
    obs:         list[np.ndarray] = field(default_factory=list)
    actions:     list[int]        = field(default_factory=list)
    rewards:     list[float]      = field(default_factory=list)
    values:      list[float]      = field(default_factory=list)
    log_probs:   list[float]      = field(default_factory=list)
    dones:       list[bool]       = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.obs)

    def add(self, obs, action, reward, value, log_prob, done):
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))
        self.dones.append(bool(done))

    def reset(self):
        self.obs.clear(); self.actions.clear(); self.rewards.clear()
        self.values.clear(); self.log_probs.clear(); self.dones.clear()


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                last_value: float, *, gamma: float = 0.99, lam: float = 0.95
                ) -> tuple[np.ndarray, np.ndarray]:
    T = rewards.size
    advantages = np.zeros_like(rewards)
    last_adv = 0.0
    for t in reversed(range(T)):
        next_val = last_value if t == T - 1 else values[t + 1]
        next_nonterm = 1.0 - (1.0 if t == T - 1 else float(dones[t]))
        delta = rewards[t] + gamma * next_val * next_nonterm - values[t]
        last_adv = delta + gamma * lam * next_nonterm * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@dataclass
class IPPOConfig:
    # Optimisation
    actor_lr:        float = 3e-4
    critic_lr:       float = 1e-3
    n_epochs:        int   = 4
    minibatch_size:  int   = 256
    clip_eps:        float = 0.2
    entropy_coef:    float = 0.01
    grad_clip:       float = 0.5
    # GAE
    gamma:           float = 0.99
    gae_lambda:      float = 0.95
    # Meta-MAPG
    eta_inner:       float = 0.1   # inner-loop LR scale used by corrections
    correction_clip: float = 1.0   # ‖Δ_corr‖ ≤ c · ‖Δ_PG‖   (plan §4)


class IPPOTrainer:
    """Multi-agent IPPO with optional Meta-MAPG actor corrections.

    Agents are trained jointly from a shared rollout (one rollout per agent).
    Each call to :meth:`update` performs one PPO update for every agent and
    returns a dict of diagnostic metrics.
    """

    def __init__(
        self,
        actors: list[MLPActor],
        critics: list[MLPCritic],
        *,
        cfg: IPPOConfig,
        arm: str,
        T_warm: int,
        device: str = "cpu",
    ):
        self.actors  = actors
        self.critics = critics
        self.cfg     = cfg
        self.arm     = arm
        self.T_warm  = T_warm
        self.device  = device

        self.actor_opts  = [torch.optim.Adam(a.parameters(), lr=cfg.actor_lr)
                            for a in actors]
        self.critic_opts = [torch.optim.Adam(c.parameters(), lr=cfg.critic_lr)
                            for c in critics]

    # ---- inference --------------------------------------------------------

    @torch.no_grad()
    def act(self, obs_list: Sequence[np.ndarray]) -> list[tuple[int, float, float]]:
        """Sample one action per agent.  Returns list of (action, logprob, value)."""
        out = []
        for actor, critic, obs in zip(self.actors, self.critics, obs_list):
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            a, lp = actor.sample(o)
            v = critic(o)
            out.append((int(a.item()), float(lp.item()), float(v.item())))
        return out

    # ---- update -----------------------------------------------------------

    def update(self, buffers: list[RolloutBuffer], last_values: list[float],
               current_step: int) -> dict:
        """Run one IPPO update for every agent + apply Meta-MAPG correction."""
        cfg = self.cfg
        lam_peer, lam_own = get_arm_coefficients(
            self.arm, current_step=current_step, T_warm=self.T_warm)

        # Build per-agent rollouts (tensors live on device).
        rollouts: list[AgentRollout] = []
        returns_list: list[torch.Tensor] = []
        for buf, last_v in zip(buffers, last_values):
            obs       = torch.as_tensor(np.array(buf.obs),     dtype=torch.float32, device=self.device)
            actions   = torch.as_tensor(buf.actions,           dtype=torch.long,    device=self.device)
            rewards   = np.array(buf.rewards,   dtype=np.float32)
            values    = np.array(buf.values,    dtype=np.float32)
            dones     = np.array(buf.dones,     dtype=np.float32)
            adv_np, ret_np = compute_gae(rewards, values, dones, last_v,
                                          gamma=cfg.gamma, lam=cfg.gae_lambda)
            adv = torch.as_tensor(adv_np, dtype=torch.float32, device=self.device)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ret = torch.as_tensor(ret_np, dtype=torch.float32, device=self.device)
            log_probs_old = torch.as_tensor(buf.log_probs, dtype=torch.float32, device=self.device)
            rollouts.append(AgentRollout(obs=obs, actions=actions,
                                         advantages=adv, log_probs_old=log_probs_old))
            returns_list.append(ret)

        n_agents = len(self.actors)
        T = rollouts[0].obs.shape[0]
        diag: dict[str, list[float]] = {
            "pg_norm": [], "corr_norm": [], "corr_scale": [], "cosine_corr_pg": [],
            "policy_loss": [], "value_loss": [], "entropy": [],
            "lam_peer_eff": [lam_peer]*n_agents, "lam_own_eff": [lam_own]*n_agents,
        }

        for i in range(n_agents):
            # ---------------------------------------------------------------
            # 1. Meta-MAPG correction (computed once per update, off the FULL rollout).
            # ---------------------------------------------------------------
            peers = [(self.actors[j], rollouts[j]) for j in range(n_agents) if j != i]
            pg_grads = compute_pg_grads(self.actors[i], rollouts[i])
            corr_grads = compute_meta_corrections(
                self.actors[i], peers, rollouts[i],
                lam_peer=lam_peer, lam_own=lam_own, eta_inner=cfg.eta_inner)
            corr_grads_clipped, scale = clip_correction(
                pg_grads, corr_grads, c=cfg.correction_clip)
            cos = cosine(pg_grads, corr_grads_clipped) if (lam_peer > 0 or lam_own > 0) else 0.0

            diag["pg_norm"].append(float(grad_norm(pg_grads)))
            diag["corr_norm"].append(float(grad_norm(corr_grads_clipped)))
            diag["corr_scale"].append(scale)
            diag["cosine_corr_pg"].append(cos)

            # ---------------------------------------------------------------
            # 2. PPO mini-batch updates for actor + critic.
            # ---------------------------------------------------------------
            ro = rollouts[i]
            ret = returns_list[i]
            critic_i = self.critics[i]
            actor_i = self.actors[i]

            mean_pl, mean_vl, mean_ent = [], [], []
            idx = np.arange(T)
            mb = cfg.minibatch_size
            for _ in range(cfg.n_epochs):
                np.random.shuffle(idx)
                for start in range(0, T, mb):
                    sl = idx[start:start + mb]
                    obs_mb = ro.obs[sl]
                    act_mb = ro.actions[sl]
                    adv_mb = ro.advantages[sl]
                    ret_mb = ret[sl]
                    lp_old_mb = ro.log_probs_old[sl]

                    # ---- actor ----
                    dist = actor_i.dist(obs_mb)
                    lp_new = dist.log_prob(act_mb)
                    ratio = torch.exp(lp_new - lp_old_mb)
                    surr1 = ratio * adv_mb
                    surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_mb
                    pol_loss = -torch.min(surr1, surr2).mean()
                    ent = dist.entropy().mean()
                    actor_loss = pol_loss - cfg.entropy_coef * ent

                    self.actor_opts[i].zero_grad(set_to_none=True)
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor_i.parameters(), cfg.grad_clip)
                    self.actor_opts[i].step()

                    # ---- critic ----
                    v = critic_i(obs_mb)
                    val_loss = F.mse_loss(v, ret_mb)
                    self.critic_opts[i].zero_grad(set_to_none=True)
                    val_loss.backward()
                    nn.utils.clip_grad_norm_(critic_i.parameters(), cfg.grad_clip)
                    self.critic_opts[i].step()

                    mean_pl.append(float(pol_loss.item()))
                    mean_vl.append(float(val_loss.item()))
                    mean_ent.append(float(ent.item()))

            diag["policy_loss"].append(float(np.mean(mean_pl)))
            diag["value_loss"].append(float(np.mean(mean_vl)))
            diag["entropy"].append(float(np.mean(mean_ent)))

            # ---------------------------------------------------------------
            # 3. Apply Meta-MAPG correction as one *extra* gradient step on
            #    the actor.  We use the cached, clipped correction rather
            #    than recomputing — see plan §4 (correction lives off PG).
            # ---------------------------------------------------------------
            if (lam_peer > 0 or lam_own > 0) and any(g.abs().sum() > 0 for g in corr_grads_clipped):
                with torch.no_grad():
                    for p, g in zip(actor_i.parameters(), corr_grads_clipped):
                        p.add_(g, alpha=-cfg.actor_lr)

        return diag
