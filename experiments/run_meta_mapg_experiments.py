from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ["standard_pg", "meta_pg", "lola_style", "meta_mapg"]
METHOD_LABELS = {
    "standard_pg": "PG",
    "meta_pg": "Meta-PG",
    "lola_style": "Peer only",
    "meta_mapg": "Meta-MAPG",
}
GAME_LABELS = {
    "ipd": "IPD",
    "stag_hunt": "Stag Hunt",
}


@dataclass(frozen=True)
class Game:
    name: str
    payoff_p1: np.ndarray
    payoff_p2: np.ndarray
    horizon: int = 1
    discount: float = 0.96

    @property
    def n_states(self) -> int:
        return 1 if self.horizon == 1 else 5


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def logit(p: float) -> float:
    p = min(max(p, 1e-5), 1.0 - 1e-5)
    return math.log(p / (1.0 - p))


def stag_hunt() -> Game:
    # Action index 0 is cooperate, action index 1 is defect.
    p1 = np.array([[4.0, 0.0], [3.0, 2.0]], dtype=float)
    return Game("stag_hunt", p1, p1.T.copy())


def prisoners_dilemma() -> Game:
    p1 = np.array([[3.0, 0.0], [5.0, 1.0]], dtype=float)
    return Game("ipd", p1, p1.T.copy(), horizon=12, discount=0.96)


def games() -> dict[str, Game]:
    return {"stag_hunt": stag_hunt(), "ipd": prisoners_dilemma()}


def action_index(cooperate: np.ndarray) -> np.ndarray:
    return np.where(cooperate, 0, 1)


def sample_batch(
    theta: np.ndarray,
    game: Game,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample trajectories and return returns, trajectory scores, and log-Hessian diagonals.

    Policies are Bernoulli in every state, with action 1 in the Bernoulli variable
    representing cooperation. The matrix payoffs use action index 0 for cooperation,
    so the sampled actions are converted before rewards are read.
    """

    n_states = game.n_states
    returns = np.zeros((batch_size, 2), dtype=float)
    scores = np.zeros((batch_size, 2, n_states), dtype=float)
    hess_diag = np.zeros((batch_size, 2, n_states), dtype=float)

    if game.horizon == 1:
        p = sigmoid(theta[:, 0])
        coop = rng.random((batch_size, 2)) < p[None, :]
        idx = action_index(coop)
        returns[:, 0] = game.payoff_p1[idx[:, 0], idx[:, 1]]
        returns[:, 1] = game.payoff_p2[idx[:, 0], idx[:, 1]]
        scores[:, 0, 0] = coop[:, 0].astype(float) - p[0]
        scores[:, 1, 0] = coop[:, 1].astype(float) - p[1]
        hess_diag[:, 0, 0] = -p[0] * (1.0 - p[0])
        hess_diag[:, 1, 0] = -p[1] * (1.0 - p[1])
        return returns, scores, hess_diag

    states = np.zeros(batch_size, dtype=int)
    batch_idx = np.arange(batch_size)
    for t in range(game.horizon):
        p0 = sigmoid(theta[0, states])
        p1 = sigmoid(theta[1, states])
        coop0 = rng.random(batch_size) < p0
        coop1 = rng.random(batch_size) < p1
        idx0 = action_index(coop0)
        idx1 = action_index(coop1)
        disc = game.discount**t
        returns[:, 0] += disc * game.payoff_p1[idx0, idx1]
        returns[:, 1] += disc * game.payoff_p2[idx0, idx1]
        np.add.at(scores[:, 0, :], (batch_idx, states), coop0.astype(float) - p0)
        np.add.at(scores[:, 1, :], (batch_idx, states), coop1.astype(float) - p1)
        np.add.at(hess_diag[:, 0, :], (batch_idx, states), -p0 * (1.0 - p0))
        np.add.at(hess_diag[:, 1, :], (batch_idx, states), -p1 * (1.0 - p1))
        states = 1 + 2 * idx0 + idx1

    return returns, scores, hess_diag


@dataclass
class GradientComponents:
    base: np.ndarray
    own: np.ndarray
    peer: np.ndarray
    reward_estimate: np.ndarray


def estimate_components(
    theta: np.ndarray,
    game: Game,
    batch_size: int,
    rng: np.random.Generator,
    inner_lr: float,
) -> GradientComponents:
    returns, scores, hess_diag = sample_batch(theta, game, batch_size, rng)
    dim = game.n_states
    base = np.zeros((2, dim), dtype=float)
    own = np.zeros((2, dim), dtype=float)
    peer = np.zeros((2, dim), dtype=float)

    for player in range(2):
        opp = 1 - player
        r_self = returns[:, player]
        r_opp = returns[:, opp]
        score_self = scores[:, player, :]
        score_opp = scores[:, opp, :]

        g_self = np.mean(r_self[:, None] * score_self, axis=0)
        q_self_wrt_opp = np.mean(r_self[:, None] * score_opp, axis=0)

        h_self = np.einsum("b,bi,bj->ij", r_self, score_self, score_self) / float(batch_size)
        h_self += np.diag(np.mean(r_self[:, None] * hess_diag[:, player, :], axis=0))
        cross_opp_self = np.einsum("b,bi,bj->ij", r_opp, score_opp, score_self) / float(batch_size)

        base[player] = g_self
        own[player] = inner_lr * h_self.T @ g_self
        peer[player] = inner_lr * cross_opp_self.T @ q_self_wrt_opp

    return GradientComponents(
        base=base,
        own=own,
        peer=peer,
        reward_estimate=np.mean(returns, axis=0),
    )


def expected_return(theta: np.ndarray, game: Game) -> np.ndarray:
    if game.horizon == 1:
        p = sigmoid(theta[:, 0])
        probs = np.array([p[0], 1.0 - p[0]])[:, None] * np.array([p[1], 1.0 - p[1]])[None, :]
        return np.array(
            [np.sum(probs * game.payoff_p1), np.sum(probs * game.payoff_p2)],
            dtype=float,
        )

    dist = np.zeros(game.n_states, dtype=float)
    dist[0] = 1.0
    total = np.zeros(2, dtype=float)
    for t in range(game.horizon):
        next_dist = np.zeros_like(dist)
        for state in range(game.n_states):
            if dist[state] == 0.0:
                continue
            p = sigmoid(theta[:, state])
            for c1 in (0, 1):
                for c2 in (0, 1):
                    prob = (p[0] if c1 else 1.0 - p[0]) * (p[1] if c2 else 1.0 - p[1])
                    idx1, idx2 = action_index(np.array([c1, c2], dtype=bool))
                    total[0] += (game.discount**t) * dist[state] * prob * game.payoff_p1[idx1, idx2]
                    total[1] += (game.discount**t) * dist[state] * prob * game.payoff_p2[idx1, idx2]
                    next_state = 1 + 2 * int(idx1) + int(idx2)
                    next_dist[next_state] += dist[state] * prob
        dist = next_dist
    return total


def cooperation_probs(theta: np.ndarray, game: Game) -> np.ndarray:
    probs = sigmoid(theta)
    if game.horizon == 1:
        return probs[:, 0]
    # The initial state is the cleanest finite-horizon proxy for equilibrium selection.
    return probs[:, 0]


def update_from_components(
    comps: GradientComponents,
    method: str,
    peer_coef: float,
    own_coef: float,
) -> np.ndarray:
    update = comps.base.copy()
    if method in {"meta_pg", "meta_mapg"}:
        update += own_coef * comps.own
    if method in {"lola_style", "meta_mapg"}:
        update += peer_coef * comps.peer
    return update


def run_rollout(
    game: Game,
    method: str,
    seed: int,
    steps: int,
    batch_size: int,
    lr: float,
    inner_lr: float,
    peer_coef: float,
    own_coef: float,
    init_theta: np.ndarray | None = None,
    lr_power: float = 0.25,
    lambda_power: float = 0.0,
    log_every: int = 10,
) -> tuple[np.ndarray, list[dict[str, float | int | str]]]:
    rng = np.random.default_rng(seed)
    theta = (
        init_theta.astype(float).copy()
        if init_theta is not None
        else rng.normal(loc=0.0, scale=1.35, size=(2, game.n_states))
    )
    rows: list[dict[str, float | int | str]] = []

    for step in range(steps):
        comps = estimate_components(theta, game, batch_size, rng, inner_lr)
        lr_step = lr / ((step + 10.0) ** lr_power)
        lambda_step = peer_coef / ((step + 1.0) ** lambda_power)
        update = update_from_components(comps, method, lambda_step, own_coef)
        theta = np.clip(theta + lr_step * update, -8.0, 8.0)

        if step % log_every == 0 or step == steps - 1:
            ret = expected_return(theta, game)
            coop = cooperation_probs(theta, game)
            rows.append(
                {
                    "game": game.name,
                    "method": method,
                    "seed": seed,
                    "step": step,
                    "reward_p1": float(ret[0]),
                    "reward_p2": float(ret[1]),
                    "coop_p1": float(coop[0]),
                    "coop_p2": float(coop[1]),
                    "base_norm": float(np.linalg.norm(comps.base)),
                    "own_norm": float(np.linalg.norm(comps.own)),
                    "peer_norm": float(np.linalg.norm(comps.peer)),
                    "lr_step": float(lr_step),
                    "lambda_step": float(lambda_step),
                }
            )
    return theta, rows


def is_success(theta: np.ndarray, game: Game, threshold: float = 0.82) -> bool:
    coop = cooperation_probs(theta, game)
    return bool(np.min(coop) >= threshold)


def run_ablation(args: argparse.Namespace, outdir: Path) -> pd.DataFrame:
    trace_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    for game in games().values():
        for method in METHODS:
            for seed in range(args.seeds):
                theta, rows = run_rollout(
                    game=game,
                    method=method,
                    seed=1000 + 37 * seed,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    inner_lr=args.inner_lr,
                    peer_coef=args.peer_coef,
                    own_coef=args.own_coef,
                    lr_power=args.lr_power,
                    lambda_power=args.lambda_power,
                )
                trace_rows.extend(rows)
                ret = expected_return(theta, game)
                coop = cooperation_probs(theta, game)
                summary_rows.append(
                    {
                        "experiment": "ablation",
                        "game": game.name,
                        "method": method,
                        "seed": seed,
                        "success": int(is_success(theta, game)),
                        "final_reward_mean": float(np.mean(ret)),
                        "final_coop_min": float(np.min(coop)),
                        "final_coop_mean": float(np.mean(coop)),
                    }
                )

    trace = pd.DataFrame(trace_rows)
    summary = pd.DataFrame(summary_rows)
    trace.to_csv(outdir / "ablation_trace.csv", index=False)
    summary.to_csv(outdir / "ablation_summary.csv", index=False)
    return summary


def run_restart(args: argparse.Namespace, outdir: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for game in games().values():
        for method in METHODS:
            for seed in range(args.seeds):
                # Same seed gives paired restart initialisations and rollout RNG
                # streams across methods, reducing between-initialisation noise.
                rng = np.random.default_rng(9000 + 41 * seed)
                success = False
                final_theta = None
                restart_count = args.max_restarts
                for restart in range(args.max_restarts + 1):
                    init_theta = rng.uniform(low=-3.0, high=3.0, size=(2, game.n_states))
                    theta, _ = run_rollout(
                        game=game,
                        method=method,
                        seed=int(rng.integers(0, 2**31 - 1)),
                        steps=args.restart_steps,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        inner_lr=args.inner_lr,
                        peer_coef=args.peer_coef,
                        own_coef=args.own_coef,
                        init_theta=init_theta,
                        lr_power=args.lr_power,
                        lambda_power=0.0,
                    )
                    final_theta = theta
                    if is_success(theta, game, threshold=args.success_threshold):
                        success = True
                        restart_count = restart
                        break
                ret = expected_return(final_theta, game) if final_theta is not None else np.array([np.nan, np.nan])
                rows.append(
                    {
                        "experiment": "restart",
                        "game": game.name,
                        "method": method,
                        "seed": seed,
                        "success": int(success),
                        "restarts": int(restart_count),
                        "final_reward_mean": float(np.mean(ret)),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "restart_summary.csv", index=False)
    return df


def run_restart_selection(args: argparse.Namespace, outdir: Path) -> pd.DataFrame:
    game = stag_hunt()
    rows: list[dict[str, float | int | str]] = []
    for method in ["standard_pg", "meta_mapg"]:
        for seed in range(args.selection_seeds):
            # Paired design: for a fixed seed, PG and Meta-MAPG see the same
            # restart initialisations and rollout RNG seeds.
            rng = np.random.default_rng(18000 + 53 * seed)
            best_welfare = -np.inf
            found_payoff_dominant = False
            first_hit_budget = args.selection_budget + 1
            cumulative_gap = 0.0
            for budget in range(1, args.selection_budget + 1):
                init_theta = rng.uniform(low=-3.0, high=3.0, size=(2, game.n_states))
                theta, _ = run_rollout(
                    game=game,
                    method=method,
                    seed=int(rng.integers(0, 2**31 - 1)),
                    steps=args.selection_steps,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    inner_lr=args.inner_lr,
                    peer_coef=args.selection_peer_coef,
                    own_coef=args.own_coef,
                    init_theta=init_theta,
                    lr_power=args.lr_power,
                    lambda_power=0.0,
                )
                welfare = float(np.sum(expected_return(theta, game)))
                best_welfare = max(best_welfare, welfare)
                found_payoff_dominant = found_payoff_dominant or is_success(
                    theta,
                    game,
                    threshold=args.success_threshold,
                )
                if found_payoff_dominant and first_hit_budget == args.selection_budget + 1:
                    first_hit_budget = budget
                welfare_gap = max(0.0, 8.0 - best_welfare)
                cumulative_gap += welfare_gap
                rows.append(
                    {
                        "experiment": "restart_selection",
                        "game": game.name,
                        "method": method,
                        "seed": seed,
                        "budget": budget,
                        "best_welfare": float(best_welfare),
                        "welfare_gap": float(welfare_gap),
                        "cumulative_welfare_gap": float(cumulative_gap),
                        "payoff_dominant_found": int(found_payoff_dominant),
                        "first_hit_budget": int(first_hit_budget),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "restart_selection.csv", index=False)
    return df


def run_trajectories(args: argparse.Namespace, outdir: Path) -> pd.DataFrame:
    game = stag_hunt()
    rows: list[dict[str, float | int | str]] = []
    grid = np.linspace(args.trajectory_min_init, args.trajectory_max_init, args.trajectory_grid_size)
    trajectory_id = 0
    for method in ["standard_pg", "meta_mapg"]:
        for i, p1 in enumerate(grid):
            for j, p2 in enumerate(grid):
                init_theta = np.zeros((2, game.n_states), dtype=float)
                init_theta[0, 0] = logit(float(p1))
                init_theta[1, 0] = logit(float(p2))
                theta, trace = run_rollout(
                    game=game,
                    method=method,
                    seed=21000 + 409 * trajectory_id + (0 if method == "standard_pg" else 1),
                    steps=args.trajectory_steps,
                    batch_size=args.trajectory_batch_size,
                    lr=args.lr,
                    inner_lr=args.inner_lr,
                    peer_coef=args.trajectory_peer_coef,
                    own_coef=args.own_coef,
                    init_theta=init_theta,
                    lr_power=args.lr_power,
                    lambda_power=0.0,
                    log_every=args.trajectory_log_every,
                )
                success = is_success(theta, game, threshold=args.success_threshold)
                for row in trace:
                    rows.append(
                        {
                            "experiment": "trajectory",
                            "game": game.name,
                            "method": method,
                            "trajectory_id": trajectory_id,
                            "init_p1": float(p1),
                            "init_p2": float(p2),
                            "step": int(row["step"]),
                            "coop_p1": float(row["coop_p1"]),
                            "coop_p2": float(row["coop_p2"]),
                            "success": int(success),
                        }
                    )
                trajectory_id += 1
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "trajectory_trace.csv", index=False)
    return df


def run_basin(args: argparse.Namespace, outdir: Path) -> pd.DataFrame:
    game = stag_hunt()
    rows: list[dict[str, float | int | str]] = []
    grid = np.linspace(0.05, 0.95, args.grid_size)
    for method in ["standard_pg", "meta_mapg"]:
        for i, p1 in enumerate(grid):
            for j, p2 in enumerate(grid):
                init_theta = np.zeros((2, game.n_states), dtype=float)
                init_theta[0, 0] = logit(float(p1))
                init_theta[1, 0] = logit(float(p2))
                theta, _ = run_rollout(
                    game=game,
                    method=method,
                    seed=12000 + 101 * i + 13 * j + (0 if method == "standard_pg" else 1),
                    steps=args.basin_steps,
                    batch_size=args.basin_batch_size,
                    lr=args.lr,
                    inner_lr=args.inner_lr,
                    peer_coef=args.basin_peer_coef,
                    own_coef=args.own_coef,
                    init_theta=init_theta,
                    lr_power=args.lr_power,
                    lambda_power=0.0,
                    log_every=args.basin_steps + 1,
                )
                coop = cooperation_probs(theta, game)
                rows.append(
                    {
                        "experiment": "basin",
                        "game": game.name,
                        "method": method,
                        "init_p1": float(p1),
                        "init_p2": float(p2),
                        "success": int(is_success(theta, game, threshold=args.success_threshold)),
                        "final_coop_min": float(np.min(coop)),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "basin_map.csv", index=False)
    return df


def run_peer_sweep(args: argparse.Namespace, outdir: Path) -> pd.DataFrame:
    """Sweep the peer-shaping coefficient and measure basin success + local drift.

    Operationalises Prop. 4.3 / Cor. 4.5: as the shaping coefficient lambda
    grows, the certified local drift constant mu_lambda >= mu + lambda mu_M
    should be reflected in (i) larger measured cooperative basin volume and
    (ii) faster one-restart hit rate on Stag Hunt.
    """
    game = stag_hunt()
    lambdas = np.array(args.sweep_lambdas, dtype=float)
    grid = np.linspace(0.05, 0.95, args.sweep_grid_size)

    rows: list[dict[str, float | int | str]] = []

    # Baseline PG once (lambda-independent) to serve as a reference line.
    pg_successes = []
    for i, p1 in enumerate(grid):
        for j, p2 in enumerate(grid):
            init_theta = np.zeros((2, game.n_states), dtype=float)
            init_theta[0, 0] = logit(float(p1))
            init_theta[1, 0] = logit(float(p2))
            theta, _ = run_rollout(
                game=game,
                method="standard_pg",
                seed=30000 + 101 * i + 13 * j,
                steps=args.sweep_steps,
                batch_size=args.sweep_batch_size,
                lr=args.lr,
                inner_lr=args.inner_lr,
                peer_coef=0.0,
                own_coef=args.own_coef,
                init_theta=init_theta,
                lr_power=args.lr_power,
                lambda_power=0.0,
                log_every=args.sweep_steps + 1,
            )
            pg_successes.append(int(is_success(theta, game, threshold=args.success_threshold)))
    pg_rate = float(np.mean(pg_successes))

    for lam in lambdas:
        successes: list[int] = []
        hit_budgets: list[int] = []
        for i, p1 in enumerate(grid):
            for j, p2 in enumerate(grid):
                init_theta = np.zeros((2, game.n_states), dtype=float)
                init_theta[0, 0] = logit(float(p1))
                init_theta[1, 0] = logit(float(p2))
                theta, trace = run_rollout(
                    game=game,
                    method="meta_mapg",
                    seed=31000 + 101 * i + 13 * j + int(round(100.0 * float(lam))),
                    steps=args.sweep_steps,
                    batch_size=args.sweep_batch_size,
                    lr=args.lr,
                    inner_lr=args.inner_lr,
                    peer_coef=float(lam),
                    own_coef=args.own_coef,
                    init_theta=init_theta,
                    lr_power=args.lr_power,
                    lambda_power=0.0,
                    log_every=1,
                )
                hit_step: int | None = None
                for row in trace:
                    if float(row["coop_p1"]) >= args.success_threshold and float(row["coop_p2"]) >= args.success_threshold:
                        hit_step = int(row["step"])
                        break
                success_flag = int(is_success(theta, game, threshold=args.success_threshold))
                successes.append(success_flag)
                if hit_step is not None and success_flag:
                    hit_budgets.append(hit_step)
        mean_hit = float(np.mean(hit_budgets)) if hit_budgets else float("nan")
        n_hits = int(len(hit_budgets))
        rows.append(
            {
                "lambda": float(lam),
                "meta_mapg_basin_rate": float(np.mean(successes)),
                "mean_first_hit_step_given_success": mean_hit,
                "n_success": n_hits,
                "n_total": int(len(successes)),
                "pg_basin_rate": pg_rate,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "peer_sweep.csv", index=False)
    return df


def _empirical_drift_constant(
    game: Game,
    peer_coef: float,
    own_coef: float,
    inner_lr: float,
    batch_size: int,
    seed: int,
) -> float:
    """Numerical symmetric-part eigenvalue of DF_lambda at the cooperate fixed point.

    Returns mu_hat = -lambda_max((J + J^T)/2) as an empirical proxy for the
    local drift margin. Uses finite-difference Jacobian of the sample-mean
    opponent-aware update.
    """
    rng = np.random.default_rng(seed)
    # Anchor near the payoff-dominant equilibrium.
    theta0 = np.array([[logit(0.9)], [logit(0.9)]], dtype=float)

    def f(theta: np.ndarray) -> np.ndarray:
        comps = estimate_components(theta, game, batch_size, rng, inner_lr)
        return update_from_components(comps, "meta_mapg", peer_coef, own_coef).reshape(-1)

    base = f(theta0.copy())
    eps = 1e-3
    dim = base.size
    jac = np.zeros((dim, dim), dtype=float)
    flat = theta0.reshape(-1)
    for k in range(dim):
        perturbed = flat.copy()
        perturbed[k] += eps
        jac[:, k] = (f(perturbed.reshape(theta0.shape)) - base) / eps
    sym = 0.5 * (jac + jac.T)
    eigvals = np.linalg.eigvalsh(sym)
    return float(-np.max(eigvals))


def run_annealing_ablation(args: argparse.Namespace, outdir: Path) -> pd.DataFrame:
    """Compare constant-lambda Meta-MAPG against the two-phase annealed schedule.

    Phase 1 runs constant lambda_c for N_0 steps, phase 2 decays as
    lambda_n = lambda_c * (1 + (n - N_0) / scale)^{-q}. This is the actual
    algorithm underlying Thm. 4.4; without annealing, all experiments only
    test phase-1 basin entry.
    """
    game = stag_hunt()
    n0 = args.anneal_phase1_steps
    total_steps = args.anneal_total_steps
    scale = max(1, args.anneal_scale)
    q = args.anneal_power
    lam_c = args.peer_coef
    rows: list[dict[str, float | int | str]] = []
    configs = [
        ("pg", "standard_pg", "constant", 0.0),
        ("meta_mapg_constant", "meta_mapg", "constant", lam_c),
        ("meta_mapg_two_phase", "meta_mapg", "two_phase", lam_c),
    ]
    rng_master = np.random.default_rng(50000)
    init_thetas = [
        rng_master.normal(loc=0.0, scale=1.35, size=(2, game.n_states))
        for _ in range(args.anneal_seeds)
    ]

    for label, method, schedule, lam_const in configs:
        for seed in range(args.anneal_seeds):
            theta = init_thetas[seed].copy()
            rng = np.random.default_rng(51000 + 71 * seed + (0 if label == "pg" else (1 if "constant" in label else 2)))
            lambda_trace: list[float] = []
            coop_trace: list[float] = []
            for step in range(total_steps):
                comps = estimate_components(theta, game, args.anneal_batch_size, rng, args.inner_lr)
                lr_step = args.lr / ((step + 10.0) ** args.lr_power)
                if schedule == "constant":
                    lam_step = lam_const
                else:
                    if step < n0:
                        lam_step = lam_c
                    else:
                        lam_step = lam_c / ((1.0 + (step - n0) / float(scale)) ** q)
                update = update_from_components(comps, method, lam_step, args.own_coef)
                theta = np.clip(theta + lr_step * update, -8.0, 8.0)
                lambda_trace.append(float(lam_step))
                coop_trace.append(float(np.min(cooperation_probs(theta, game))))
            # Second-half stability: how much does the policy wobble once it should be settling?
            second_half = np.array(coop_trace[total_steps // 2 :])
            rows.append(
                {
                    "label": label,
                    "method": method,
                    "schedule": schedule,
                    "seed": seed,
                    "final_coop_min": float(coop_trace[-1]),
                    "second_half_coop_mean": float(np.mean(second_half)),
                    "second_half_coop_std": float(np.std(second_half, ddof=1)) if len(second_half) > 1 else 0.0,
                    "success": int(np.min(cooperation_probs(theta, game)) >= args.success_threshold),
                    "final_lambda": float(lambda_trace[-1]),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "annealing_ablation.csv", index=False)
    return df


def plot_peer_sweep(sweep: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    ax = axes[0]
    ax.plot(sweep["lambda"], sweep["meta_mapg_basin_rate"], marker="o", color="#b279a2", linewidth=1.9, label="Meta-MAPG")
    ax.axhline(float(sweep["pg_basin_rate"].iloc[0]), color="#4c78a8", linestyle="--", linewidth=1.2, label="PG baseline")
    ax.set_xlabel(r"Peer-shaping coefficient $\lambda$")
    ax.set_ylabel("Basin success rate (grid)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    finite = sweep.dropna(subset=["mean_first_hit_step_given_success"])
    ax.plot(
        finite["lambda"],
        finite["mean_first_hit_step_given_success"],
        marker="s",
        color="#e45756",
        linewidth=1.9,
    )
    ax.set_xlabel(r"Peer-shaping coefficient $\lambda$")
    ax.set_ylabel("Mean first-hit step | success")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(outdir / "peer_sweep.pdf")
    fig.savefig(outdir / "peer_sweep.png", dpi=180)
    plt.close(fig)


def plot_annealing_ablation(anneal: pd.DataFrame, outdir: Path, args: argparse.Namespace | None = None) -> None:
    labels = ["pg", "meta_mapg_constant", "meta_mapg_two_phase"]
    pretty = {
        "pg": "PG",
        "meta_mapg_constant": "Meta-MAPG (constant $\\lambda$)",
        "meta_mapg_two_phase": "Meta-MAPG (two-phase)",
    }
    colors = {
        "pg": "#4c78a8",
        "meta_mapg_constant": "#b279a2",
        "meta_mapg_two_phase": "#2fbf71",
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))

    ax = axes[0]
    for lbl in labels:
        sub = anneal[anneal["label"] == lbl]
        rate = float(sub["success"].mean())
        sem = float(sub["success"].std(ddof=1) / np.sqrt(len(sub)))
        ax.bar(pretty[lbl], rate, yerr=1.96 * sem, color=colors[lbl], alpha=0.85)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Cooperative success rate")
    ax.tick_params(axis="x", labelrotation=15)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Outcome", fontsize=9)

    ax = axes[1]
    if args is not None:
        steps = np.arange(args.anneal_total_steps)
        lam_const = float(args.peer_coef) * np.ones_like(steps, dtype=float)
        lam_two = np.where(
            steps < args.anneal_phase1_steps,
            float(args.peer_coef),
            float(args.peer_coef)
            / np.maximum(1.0, 1.0 + (steps - args.anneal_phase1_steps) / float(args.anneal_scale)) ** args.anneal_power,
        )
        ax.plot(steps, lam_const, color=colors["meta_mapg_constant"], linewidth=1.8, label="constant $\\lambda$")
        ax.plot(steps, lam_two, color=colors["meta_mapg_two_phase"], linewidth=1.8, label="two-phase")
        ax.axvline(args.anneal_phase1_steps, color="grey", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Step")
        ax.set_ylabel(r"Shaping $\lambda_n$")
        ax.set_title("Schedule", fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(outdir / "annealing_ablation.pdf")
    fig.savefig(outdir / "annealing_ablation.png", dpi=180)
    plt.close(fig)


def run_estimator_sanity(args: argparse.Namespace, outdir: Path) -> pd.DataFrame:
    game = stag_hunt()
    theta = np.array([[logit(0.58)], [logit(0.58)]], dtype=float)
    ref_rng = np.random.default_rng(777)
    ref = estimate_components(theta, game, args.reference_batch_size, ref_rng, args.inner_lr)
    ref_update = update_from_components(ref, "meta_mapg", args.peer_coef, args.own_coef).reshape(-1)
    rows: list[dict[str, float | int | str]] = []
    for batch_size in args.sanity_batches:
        estimates = []
        for rep in range(args.sanity_reps):
            rng = np.random.default_rng(16000 + 97 * rep + batch_size)
            comps = estimate_components(theta, game, batch_size, rng, args.inner_lr)
            update = update_from_components(comps, "meta_mapg", args.peer_coef, args.own_coef).reshape(-1)
            estimates.append(update)
        arr = np.stack(estimates, axis=0)
        mean = np.mean(arr, axis=0)
        rows.append(
            {
                "experiment": "estimator_sanity",
                "batch_size": int(batch_size),
                "bias_norm_to_reference": float(np.linalg.norm(mean - ref_update)),
                "variance_trace": float(np.trace(np.cov(arr.T))),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "estimator_sanity.csv", index=False)
    return df


def save_summary_table(ablation: pd.DataFrame, restart: pd.DataFrame, outdir: Path) -> None:
    summary = (
        ablation.groupby(["game", "method"])
        .agg(success_rate=("success", "mean"), reward=("final_reward_mean", "mean"), reward_std=("final_reward_mean", "std"))
        .reset_index()
    )
    restart_summary = (
        restart.groupby(["game", "method"])
        .agg(success_rate=("success", "mean"), restarts=("restarts", "mean"), restarts_std=("restarts", "std"))
        .reset_index()
    )
    with (outdir / "main_results_table.tex").open("w", newline="") as handle:
        handle.write("\\begin{tabular}{llccc}\\toprule\n")
        handle.write("Game & Method & Ablation success & Restart success & Restarts \\\\\\midrule\n")
        ordered_rows = []
        for game_name in ["stag_hunt", "ipd"]:
            for method_name in METHODS:
                match = summary[(summary["game"] == game_name) & (summary["method"] == method_name)]
                if not match.empty:
                    ordered_rows.append(match.iloc[0])
        for row in ordered_rows:
            game = row["game"].replace("_", " ")
            method = METHOD_LABELS[row["method"]]
            restart_row = restart_summary[
                (restart_summary["game"] == row["game"]) & (restart_summary["method"] == row["method"])
            ]
            if restart_row.empty:
                restart_success = "--"
                restarts = "--"
            else:
                rr = restart_row.iloc[0]
                restart_success = f"{100.0 * rr['success_rate']:.0f}\\%"
                restarts = f"{rr['restarts']:.2f}"
            handle.write(
                f"{game} & {method} & {100.0 * row['success_rate']:.0f}\\% & {restart_success} & {restarts} \\\\\n"
            )
        handle.write("\\bottomrule\n\\end{tabular}\n")


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def plot_ablation(ablation: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), sharey=True)
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    for ax, (game_name, game_df) in zip(axes, ablation.groupby("game")):
        means: list[float] = []
        err_low: list[float] = []
        err_high: list[float] = []
        for method in METHODS:
            sub = game_df[game_df["method"] == method]
            k = int(sub["success"].sum())
            n = int(len(sub))
            p, lo, hi = _wilson_ci(k, n)
            means.append(p)
            err_low.append(p - lo)
            err_high.append(hi - p)
        ax.bar(
            range(len(METHODS)),
            means,
            yerr=[err_low, err_high],
            capsize=3.5,
            color=colors,
            error_kw={"elinewidth": 1.2, "ecolor": "#222"},
        )
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=25, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(GAME_LABELS.get(game_name, game_name.replace("_", " ").title()))
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Cooperative convergence rate")
    fig.tight_layout()
    fig.savefig(outdir / "ablation_success.pdf")
    fig.savefig(outdir / "ablation_success.png", dpi=180)
    plt.close(fig)


def plot_restart(restart: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    for ax, (game_name, game_df) in zip(axes, restart.groupby("game")):
        grouped = game_df.groupby("method")["restarts"].mean().reindex(METHODS)
        err = game_df.groupby("method")["restarts"].std().reindex(METHODS).fillna(0.0)
        ax.bar(range(len(METHODS)), grouped.values, yerr=err.values, capsize=4, color=colors)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=25, ha="right")
        ax.set_title(GAME_LABELS.get(game_name, game_name.replace("_", " ").title()))
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Restarts until success")
    fig.tight_layout()
    fig.savefig(outdir / "restart_efficiency.pdf")
    fig.savefig(outdir / "restart_efficiency.png", dpi=180)
    plt.close(fig)


def plot_restart_selection(selection: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.25))
    colors = {"standard_pg": "#4c78a8", "meta_mapg": "#b279a2"}
    for method in ["standard_pg", "meta_mapg"]:
        sub = selection[selection["method"] == method]
        grouped = (
            sub.groupby("budget")
            .agg(
                success=("payoff_dominant_found", "mean"),
                sem_success=("payoff_dominant_found", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x)))),
                area_gap=("cumulative_welfare_gap", "mean"),
            )
            .reset_index()
        )
        x = grouped["budget"].to_numpy(dtype=float)
        y = grouped["success"].to_numpy(dtype=float)
        sem = grouped["sem_success"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=2.0, color=colors[method], label=METHOD_LABELS[method])
        ax.fill_between(
            x,
            np.clip(y - 1.96 * sem, 0.0, 1.0),
            np.clip(y + 1.96 * sem, 0.0, 1.0),
            color=colors[method],
            alpha=0.18,
            linewidth=0,
        )

    ax.set_xlabel("Restart budget K")
    ax.set_ylabel("P(first hit by K)")
    ax.set_xlim(1, float(selection["budget"].max()))
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)

    inset = ax.inset_axes([0.54, 0.17, 0.41, 0.34])
    for method in ["standard_pg", "meta_mapg"]:
        sub = selection[selection["method"] == method]
        area = (
            sub.groupby("budget")
            .agg(
                mean_gap=("cumulative_welfare_gap", "mean"),
                sem_gap=(
                    "cumulative_welfare_gap",
                    lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0,
                ),
            )
            .reset_index()
        )
        x = area["budget"].to_numpy(dtype=float)
        y = area["mean_gap"].to_numpy(dtype=float)
        sem = area["sem_gap"].to_numpy(dtype=float)
        inset.plot(x, y, color=colors[method], linewidth=1.5)
        inset.fill_between(
            x,
            y - 1.96 * sem,
            y + 1.96 * sem,
            color=colors[method],
            alpha=0.18,
            linewidth=0,
        )
    inset.set_title("Area gap (lower better)", fontsize=7)
    inset.tick_params(labelsize=7)
    inset.grid(alpha=0.2)

    ax.legend(loc="lower left", fontsize=7, frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / "restart_selection.pdf")
    fig.savefig(outdir / "restart_selection.png", dpi=180)
    plt.close(fig)


def plot_trajectories(trajectories: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.15), sharex=True, sharey=True)
    for ax, method in zip(axes, ["standard_pg", "meta_mapg"]):
        sub = trajectories[trajectories["method"] == method]
        for _, traj in sub.groupby("trajectory_id"):
            success = bool(traj["success"].iloc[0])
            color = "#2fbf71" if success else "#e74c3c"
            ax.plot(
                traj["coop_p1"],
                traj["coop_p2"],
                color=color,
                alpha=0.34,
                linewidth=0.9,
            )
            ax.scatter(
                traj["coop_p1"].iloc[0],
                traj["coop_p2"].iloc[0],
                color=color,
                alpha=0.85,
                s=8,
                linewidths=0,
            )
        ax.scatter([0.97], [0.97], marker="*", s=95, color="black", label="Payoff-dominant")
        ax.scatter([0.03], [0.03], marker="x", s=45, color="#555555", label="Risk-dominant")
        ax.set_title(METHOD_LABELS[method])
        ax.set_xlabel("Player 1 cooperation")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Player 2 cooperation")
    axes[1].legend(loc="lower right", fontsize=7, frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / "trajectory_visualization.pdf")
    fig.savefig(outdir / "trajectory_visualization.png", dpi=180)
    plt.close(fig)


def plot_basin_with_trajectories(
    basin: pd.DataFrame,
    trajectories: pd.DataFrame,
    outdir: Path,
) -> None:
    """Combined figure: basin map background + separatrix + overlaid trajectories.

    The 0.5-success contour is the empirical basin frontier; trajectories
    confirm which side flows to which equilibrium.
    """
    methods = ["standard_pg", "meta_mapg"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharex=True, sharey=True)
    # Palette: light cream for failure region, soft lavender for success region.
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "basin_bg", ["#fdf6e3", "#cfd8e8"]
    )
    for ax, method in zip(axes, methods):
        sub_basin = basin[basin["method"] == method]
        pivot = (
            sub_basin.pivot(index="init_p2", columns="init_p1", values="success")
            .sort_index(ascending=True)
        )
        xs = np.sort(sub_basin["init_p1"].unique())
        ys = np.sort(sub_basin["init_p2"].unique())
        Z = pivot.values.astype(float)

        ax.pcolormesh(
            xs,
            ys,
            Z,
            shading="auto",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            rasterized=True,
        )
        # Empirical basin frontier: 0.5 contour of the success indicator.
        # Smooth lightly to get a continuous separatrix from the binary grid.
        from scipy.ndimage import gaussian_filter  # type: ignore

        Z_smooth = gaussian_filter(Z, sigma=0.8)
        ax.contour(
            xs,
            ys,
            Z_smooth,
            levels=[0.5],
            colors=["#2a2a2a"],
            linewidths=1.4,
            linestyles="--",
        )

        sub_traj = trajectories[trajectories["method"] == method]
        for _, traj in sub_traj.groupby("trajectory_id"):
            success = bool(traj["success"].iloc[0])
            color = "#2fbf71" if success else "#e74c3c"
            ax.plot(
                traj["coop_p1"],
                traj["coop_p2"],
                color=color,
                alpha=0.55,
                linewidth=0.95,
            )
            ax.scatter(
                traj["coop_p1"].iloc[0],
                traj["coop_p2"].iloc[0],
                color=color,
                alpha=0.9,
                s=10,
                linewidths=0,
            )

        ax.scatter([0.97], [0.97], marker="*", s=100, color="black", label="Payoff-dominant")
        ax.scatter([0.03], [0.03], marker="x", s=50, color="#222222", label="Risk-dominant")
        rate = float(sub_basin["success"].mean())
        ax.set_title(f"{METHOD_LABELS[method]} (basin {100.0 * rate:.1f}\\%)")
        ax.set_xlabel("Player 1 cooperation")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.18)

    axes[0].set_ylabel("Player 2 cooperation")

    # Single legend with separatrix entry, placed on the second panel.
    from matplotlib.lines import Line2D

    separatrix_handle = Line2D([0], [0], color="#2a2a2a", linestyle="--", linewidth=1.4, label="Basin frontier")
    payoff_handle = Line2D(
        [0], [0], marker="*", color="w", markerfacecolor="black", markersize=9, label="Payoff-dominant", linewidth=0
    )
    risk_handle = Line2D(
        [0], [0], marker="x", color="#222222", markersize=7, label="Risk-dominant", linewidth=0
    )
    axes[1].legend(
        handles=[separatrix_handle, payoff_handle, risk_handle],
        loc="lower right",
        fontsize=6.5,
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(outdir / "basin_trajectories.pdf")
    fig.savefig(outdir / "basin_trajectories.png", dpi=180)
    plt.close(fig)


def plot_basin(basin: pd.DataFrame, outdir: Path) -> None:
    methods = ["standard_pg", "meta_mapg"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharex=True, sharey=True)
    for ax, method in zip(axes, methods):
        sub = basin[basin["method"] == method]
        pivot = sub.pivot(index="init_p2", columns="init_p1", values="success").sort_index(ascending=True)
        ax.imshow(
            pivot.values,
            origin="lower",
            extent=[sub["init_p1"].min(), sub["init_p1"].max(), sub["init_p2"].min(), sub["init_p2"].max()],
            vmin=0,
            vmax=1,
            cmap="YlGnBu",
            aspect="equal",
        )
        rate = sub["success"].mean()
        ax.set_title(f"{METHOD_LABELS[method]} ({100.0 * rate:.1f}%)")
        ax.set_xlabel("Initial p1(C)")
        ax.grid(color="white", alpha=0.15)
    axes[0].set_ylabel("Initial p2(C)")
    fig.tight_layout()
    fig.savefig(outdir / "basin_maps.pdf")
    fig.savefig(outdir / "basin_maps.png", dpi=180)
    plt.close(fig)


def plot_sanity(sanity: pd.DataFrame, outdir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(4.8, 3.0))
    ax1.plot(sanity["batch_size"], sanity["bias_norm_to_reference"], marker="o", color="#4c78a8")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("Bias to high-batch reference", color="#4c78a8")
    ax1.tick_params(axis="y", labelcolor="#4c78a8")
    ax2 = ax1.twinx()
    ax2.plot(sanity["batch_size"], sanity["variance_trace"], marker="s", color="#f58518")
    ax2.set_ylabel("Estimator variance trace", color="#f58518")
    ax2.tick_params(axis="y", labelcolor="#f58518")
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "estimator_sanity.pdf")
    fig.savefig(outdir / "estimator_sanity.png", dpi=180)
    plt.close(fig)


def write_manifest(args: argparse.Namespace, outdir: Path) -> None:
    with (outdir / "run_config.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "value"])
        for key, value in sorted(vars(args).items()):
            writer.writerow([key, value])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sample-based Meta-MAPG restart experiments.")
    parser.add_argument("--outdir", type=Path, default=Path("artifacts/main"))
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--restart-steps", type=int, default=120)
    parser.add_argument("--max-restarts", type=int, default=12)
    parser.add_argument("--selection-budget", type=int, default=12)
    parser.add_argument("--selection-seeds", type=int, default=100)
    parser.add_argument("--selection-steps", type=int, default=120)
    parser.add_argument("--trajectory-steps", type=int, default=140)
    parser.add_argument("--trajectory-batch-size", type=int, default=384)
    parser.add_argument("--trajectory-grid-size", type=int, default=5)
    parser.add_argument("--trajectory-log-every", type=int, default=2)
    parser.add_argument("--trajectory-min-init", type=float, default=0.08)
    parser.add_argument("--trajectory-max-init", type=float, default=0.92)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--basin-batch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=0.9)
    parser.add_argument("--lr-power", type=float, default=0.24)
    parser.add_argument("--inner-lr", type=float, default=0.55)
    parser.add_argument("--peer-coef", type=float, default=1.5)
    parser.add_argument("--basin-peer-coef", type=float, default=None)
    parser.add_argument("--selection-peer-coef", type=float, default=None)
    parser.add_argument("--trajectory-peer-coef", type=float, default=None)
    parser.add_argument("--own-coef", type=float, default=0.35)
    parser.add_argument("--lambda-power", type=float, default=0.0)
    parser.add_argument("--success-threshold", type=float, default=0.82)
    parser.add_argument("--grid-size", type=int, default=21)
    parser.add_argument("--basin-steps", type=int, default=140)
    parser.add_argument("--reference-batch-size", type=int, default=120000)
    parser.add_argument("--sanity-reps", type=int, default=80)
    parser.add_argument("--sanity-batches", type=int, nargs="+", default=[48, 96, 192, 384, 768])
    parser.add_argument("--skip-basin", action="store_true")
    parser.add_argument("--skip-sanity", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--skip-annealing", action="store_true")
    parser.add_argument(
        "--sweep-lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
    )
    parser.add_argument("--sweep-grid-size", type=int, default=11)
    parser.add_argument("--sweep-steps", type=int, default=140)
    parser.add_argument("--sweep-batch-size", type=int, default=192)
    parser.add_argument("--sweep-jacobian-batch", type=int, default=8192)
    parser.add_argument("--anneal-seeds", type=int, default=80)
    parser.add_argument("--anneal-phase1-steps", type=int, default=100)
    parser.add_argument("--anneal-total-steps", type=int, default=260)
    parser.add_argument("--anneal-scale", type=int, default=30)
    parser.add_argument("--anneal-power", type=float, default=0.7)
    parser.add_argument("--anneal-batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.basin_peer_coef is None:
        args.basin_peer_coef = args.peer_coef
    if args.selection_peer_coef is None:
        args.selection_peer_coef = args.peer_coef
    if args.trajectory_peer_coef is None:
        args.trajectory_peer_coef = args.peer_coef
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    write_manifest(args, outdir)

    ablation = run_ablation(args, outdir)
    restart = run_restart(args, outdir)
    selection = run_restart_selection(args, outdir)
    trajectories = run_trajectories(args, outdir)
    plot_ablation(ablation, outdir)
    plot_restart(restart, outdir)
    plot_restart_selection(selection, outdir)
    plot_trajectories(trajectories, outdir)

    if not args.skip_basin:
        basin = run_basin(args, outdir)
        plot_basin(basin, outdir)
        plot_basin_with_trajectories(basin, trajectories, outdir)
    if not args.skip_sanity:
        sanity = run_estimator_sanity(args, outdir)
        plot_sanity(sanity, outdir)
    if not args.skip_sweep:
        sweep = run_peer_sweep(args, outdir)
        plot_peer_sweep(sweep, outdir)
    if not args.skip_annealing:
        anneal = run_annealing_ablation(args, outdir)
        plot_annealing_ablation(anneal, outdir, args=args)

    save_summary_table(ablation, restart, outdir)


if __name__ == "__main__":
    main()
