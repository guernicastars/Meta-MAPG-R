"""Single-(arm, seed) training loop for one MARL benchmark run.

Used by both pilot (one (arm, seed)) and orchestrated full sweeps.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .algos.arms import ARMS, get_arm_coefficients
from .algos.ippo import IPPOConfig, IPPOTrainer, RolloutBuffer
from .envs import make_env
from .policies import init_paired_actors
from .utils import JsonlLogger, paired_seed_init


@dataclass
class TrainConfig:
    benchmark:    str
    env_id:       str
    arm:          str
    seed:         int
    total_steps:  int           = 500_000
    rollout_len:  int           = 256
    eval_every:   int           = 50_000
    eval_episodes:int           = 50
    T_warm:       int           = 100_000   # for handoff arm
    threshold:    float         = -25.0     # high-return basin threshold
    output_dir:   str           = "artifacts/runs"
    device:       str           = "cuda"
    ippo:         IPPOConfig    = field(default_factory=IPPOConfig)


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def _collect(env, trainer: IPPOTrainer, rollout_len: int):
    """Collect a fixed-length on-policy rollout across all agents.

    Returns (buffers, last_values, episode_returns_completed,
             cooperation_rates_completed, soup_rates_completed).
    """
    n = trainer.actors.__len__()
    buffers = [RolloutBuffer() for _ in range(n)]
    completed_returns: list[list[float]] = []
    completed_coop:    list[float]       = []
    completed_soups:   list[float]       = []

    obs = env._last_obs if hasattr(env, "_last_obs") and env._last_obs else env.reset(seed=None)
    if isinstance(obs, dict):                              # PettingZoo dict
        obs = [np.array(obs[a], dtype=np.float32) for a in env.agents]

    for _ in range(rollout_len):
        actions = []
        log_probs = []
        values = []
        for i, (a_t, lp, v) in enumerate(trainer.act(obs)):
            actions.append(a_t); log_probs.append(lp); values.append(v)

        next_obs, rewards, dones, info = env.step(actions)
        for i in range(n):
            buffers[i].add(obs[i], actions[i], rewards[i], values[i], log_probs[i], dones[i])

        if all(dones):
            completed_returns.append(list(info.get("episode_returns",
                                                    {a: 0.0 for a in env.agents}).values())
                                      if isinstance(info.get("episode_returns"), dict)
                                      else list(info.get("episode_returns", [])))
            if "cooperation_rate" in info:
                completed_coop.append(float(info["cooperation_rate"]))
            if "soups_delivered" in info:
                completed_soups.append(float(info["soups_delivered"]))
            next_obs = env.reset(seed=None)
        obs = next_obs

    # Bootstrap values for GAE.
    with torch.no_grad():
        last_values = []
        for critic, ob in zip(trainer.critics, obs):
            t = torch.as_tensor(ob, dtype=torch.float32, device=trainer.device).unsqueeze(0)
            last_values.append(float(critic(t).item()))

    return buffers, last_values, completed_returns, completed_coop, completed_soups


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(env_factory, actors, *, n_episodes: int, seed: int,
             device: str) -> dict[str, float]:
    """Run greedy(-ish) evaluation: argmax actions for n_episodes."""
    returns_per_episode: list[float] = []
    coop_per_episode:    list[float] = []
    soup_per_episode:    list[float] = []

    env = env_factory(seed=seed * 7 + 13)
    for ep in range(n_episodes):
        obs = env.reset(seed=seed * 7 + 13 + ep)
        done = False
        ep_ret = 0.0
        ep_coop = 0.0
        coop_steps = 0
        soup_total = 0.0
        steps = 0
        while not done and steps < env.spec.max_cycles + 5:
            actions = []
            for i, actor in enumerate(actors):
                t = torch.as_tensor(obs[i], dtype=torch.float32, device=device).unsqueeze(0)
                logits = actor.logits(t)
                actions.append(int(logits.argmax(dim=-1).item()))
            obs, rewards, dones, info = env.step(actions)
            ep_ret += float(np.mean(rewards))
            steps += 1
            done = all(dones)
            if "cooperation_rate" in info:
                ep_coop += float(info["cooperation_rate"])
                coop_steps += 1
            if "soups_delivered" in info:
                soup_total = max(soup_total, float(info["soups_delivered"]))
        returns_per_episode.append(ep_ret)
        if coop_steps > 0:
            coop_per_episode.append(ep_coop / coop_steps)
        if soup_total > 0 or "soups_delivered" in info:
            soup_per_episode.append(soup_total)
    env.close()

    out = {
        "eval_return_mean":   float(np.mean(returns_per_episode)),
        "eval_return_median": float(np.median(returns_per_episode)),
        "eval_return_std":    float(np.std(returns_per_episode)),
        "eval_n":             int(n_episodes),
    }
    if coop_per_episode:
        out["cooperation_rate"] = float(np.mean(coop_per_episode))
    if soup_per_episode:
        out["soups_delivered"]  = float(np.mean(soup_per_episode))
    return out


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> dict[str, Any]:
    """Train one (arm, seed) for `total_steps` and write JSONL logs."""
    if cfg.arm not in ARMS:
        raise ValueError(f"unknown arm: {cfg.arm}")
    paired_seed_init(cfg.seed)

    out_dir = Path(cfg.output_dir) / cfg.benchmark / cfg.env_id / cfg.arm / f"seed_{cfg.seed:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))

    device = cfg.device if torch.cuda.is_available() else "cpu"
    env = make_env(cfg.benchmark, cfg.env_id, seed=cfg.seed)
    spec = env.spec

    # Heterogeneous obs (e.g. simple_speaker_listener: speaker=3, listener=11)
    # gets per-agent dims; homogeneous envs (simple_spread) collapse to a scalar.
    if hasattr(spec, "obs_dims"):
        per_agent_dims = list(spec.obs_dims)
        obs_dim = per_agent_dims if len(set(per_agent_dims)) > 1 else per_agent_dims[0]
    else:
        obs_dim = spec.obs_dim
    actors, critics = init_paired_actors(
        seed=cfg.seed, n_agents=spec.n_agents,
        obs_dim=obs_dim, n_actions=spec.n_actions,
        hidden=(64, 64), device=device,
    )
    trainer = IPPOTrainer(actors, critics, cfg=cfg.ippo, arm=cfg.arm,
                          T_warm=cfg.T_warm, device=device)

    train_log = JsonlLogger(out_dir / "train.jsonl")
    eval_log  = JsonlLogger(out_dir / "eval.jsonl")
    summary: dict[str, Any] = {
        "config": asdict(cfg),
        "first_hit_step": -1,
        "evals": [],
    }

    steps = 0
    last_eval_step = 0
    next_eval = 0
    t_start = time.time()
    while steps < cfg.total_steps:
        buffers, last_v, ep_returns, ep_coop, ep_soup = _collect(env, trainer, cfg.rollout_len)
        diag = trainer.update(buffers, last_v, current_step=steps)
        steps += cfg.rollout_len * spec.n_agents

        # Periodic logging of training-side stats.
        train_log.log(step=steps, **{k: float(np.mean(v)) for k, v in diag.items()
                                     if v and isinstance(v, list)},
                      ep_returns_train=float(np.mean([np.mean(r) for r in ep_returns]))
                          if ep_returns else float("nan"),
                      n_completed_eps=len(ep_returns))

        # Eval at fixed budgets.
        if steps >= next_eval:
            metrics = evaluate(
                env_factory=lambda seed: make_env(cfg.benchmark, cfg.env_id, seed=seed),
                actors=actors, n_episodes=cfg.eval_episodes, seed=cfg.seed,
                device=device,
            )
            metrics["step"] = steps
            metrics["wall_seconds"] = time.time() - t_start
            metrics["success"] = bool(metrics["eval_return_mean"] >= cfg.threshold)
            eval_log.log(**metrics)
            summary["evals"].append(metrics)
            if metrics["success"] and summary["first_hit_step"] < 0:
                summary["first_hit_step"] = steps
            next_eval = steps + cfg.eval_every

    # Final evaluation.
    metrics = evaluate(
        env_factory=lambda seed: make_env(cfg.benchmark, cfg.env_id, seed=seed),
        actors=actors, n_episodes=max(cfg.eval_episodes, 100), seed=cfg.seed,
        device=device,
    )
    metrics["step"] = steps
    metrics["wall_seconds"] = time.time() - t_start
    metrics["success"] = bool(metrics["eval_return_mean"] >= cfg.threshold)
    metrics["final"] = True
    eval_log.log(**metrics)
    summary["final_eval"] = metrics
    summary["total_steps"] = steps

    train_log.close(); eval_log.close()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    env.close()
    return summary
