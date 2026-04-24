#!/usr/bin/env python3
"""
Non-tabular IPD with MLP policy - proof-of-transport experiment.

Repeats the IPD ablation (PG vs Meta-PG vs LOLA-style vs Meta-MAPG)
with a 2-layer MLP policy (16 hidden units, tanh) instead of tabular Bernoulli.

Uses DiCE-style surrogate + autograd for correct higher-order gradients.
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn

R_MATRIX = torch.tensor([[3.0, 0.0],
                          [5.0, 1.0]])

HORIZON = 12
GAMMA = 0.96
N_STATES = 5  # CC=0, CD=1, DC=2, DD=3, START=4
SUCCESS_THRESHOLD = 0.82


class MLPPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_STATES, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.zero_()

    def forward(self, x):
        return self.net(x)


def rollout_batch(policy0, policy1, batch_size, horizon=HORIZON, gamma=GAMMA):
    device = next(policy0.parameters()).device
    B, H = batch_size, horizon

    all_states = torch.zeros(B, H, N_STATES, device=device)
    all_actions = torch.zeros(B, H, 2, dtype=torch.long, device=device)
    all_rewards = torch.zeros(B, H, 2, device=device)

    state = torch.zeros(B, N_STATES, device=device)
    state[:, 4] = 1.0  # START state

    for t in range(H):
        all_states[:, t] = state

        logit0 = policy0(state).squeeze(-1)
        logit1 = policy1(state).squeeze(-1)

        prob_coop0 = torch.sigmoid(logit0)
        prob_coop1 = torch.sigmoid(logit1)

        a0 = torch.bernoulli(1.0 - prob_coop0).long()
        a1 = torch.bernoulli(1.0 - prob_coop1).long()

        all_actions[:, t, 0] = a0
        all_actions[:, t, 1] = a1
        all_rewards[:, t, 0] = R_MATRIX[a0, a1]
        all_rewards[:, t, 1] = R_MATRIX[a1, a0]

        next_state_idx = a0 * 2 + a1
        state = torch.zeros(B, N_STATES, device=device)
        state.scatter_(1, next_state_idx.unsqueeze(1), 1.0)

    returns = torch.zeros(B, 2, device=device)
    for t in range(H):
        returns += (gamma ** t) * all_rewards[:, t]

    return {
        "states": all_states,
        "actions": all_actions,
        "rewards": all_rewards,
        "returns": returns,
    }


def dice_operator(log_probs):
    return (log_probs - log_probs.detach()).exp()


def compute_losses(policy0, policy1, trajs, method,
                   peer_coef=1.5, own_coef=0.35, inner_lr=0.55):
    states = trajs["states"]
    actions = trajs["actions"]
    rewards = trajs["rewards"]
    B, H = states.shape[0], states.shape[1]

    logit0 = policy0(states).squeeze(-1)
    logit1 = policy1(states).squeeze(-1)

    prob_coop0 = torch.sigmoid(logit0)
    prob_coop1 = torch.sigmoid(logit1)

    a0, a1 = actions[:, :, 0], actions[:, :, 1]
    lp0 = torch.where(a0 == 0,
                       torch.log(prob_coop0 + 1e-8),
                       torch.log(1.0 - prob_coop0 + 1e-8))
    lp1 = torch.where(a1 == 0,
                       torch.log(prob_coop1 + 1e-8),
                       torch.log(1.0 - prob_coop1 + 1e-8))

    cum_lp0 = torch.cumsum(lp0, dim=1)
    cum_lp1 = torch.cumsum(lp1, dim=1)

    discounts = GAMMA ** torch.arange(H, dtype=torch.float32, device=states.device).unsqueeze(0)
    disc_r0 = rewards[:, :, 0] * discounts
    disc_r1 = rewards[:, :, 1] * discounts

    dice0 = dice_operator(cum_lp0)
    dice1 = dice_operator(cum_lp1)

    # Joint DiCE surrogates - include both agents' dice for cross-term extraction.
    v0_surr = (disc_r0.detach() * dice0 * dice1).sum(dim=1).mean()
    v1_surr = (disc_r1.detach() * dice0 * dice1).sum(dim=1).mean()

    params0 = list(policy0.parameters())
    params1 = list(policy1.parameters())

    # First-order gradients through the DiCE surrogate
    g0_0 = torch.autograd.grad(v0_surr, params0, create_graph=True, retain_graph=True)
    g1_1 = torch.autograd.grad(v1_surr, params1, create_graph=True, retain_graph=True)

    # Base PG: apply detached first-order gradient as a linear loss
    l0 = -sum((g.detach() * p).sum() for g, p in zip(g0_0, params0))
    l1 = -sum((g.detach() * p).sum() for g, p in zip(g1_1, params1))

    if method == "standard_pg":
        return l0, l1

    # Own-learning: minimize -0.5 * alpha * ||g||^2.
    # The gradient is -alpha * H^T * g, the own-learning correction.
    own_l0 = -0.5 * own_coef * inner_lr * sum((g ** 2).sum() for g in g0_0)
    own_l1 = -0.5 * own_coef * inner_lr * sum((g ** 2).sum() for g in g1_1)

    if method == "meta_pg":
        return l0 + own_l0, l1 + own_l1

    # Peer-learning: minimize -alpha * <grad_theta1 V0, grad_theta1 V1.detach()>.
    # The .detach() on opponent's gradient is critical:
    # we differentiate through how OUR presence affects opponent's value landscape,
    # NOT through the opponent's own learning dynamics.
    g0_1 = torch.autograd.grad(v0_surr, params1, create_graph=True, retain_graph=True)
    g1_0 = torch.autograd.grad(v1_surr, params0, create_graph=True, retain_graph=True)

    peer_l0 = -peer_coef * inner_lr * sum(
        (g01 * g11.detach()).sum() for g01, g11 in zip(g0_1, g1_1))
    peer_l1 = -peer_coef * inner_lr * sum(
        (g10 * g00.detach()).sum() for g10, g00 in zip(g1_0, g0_0))

    if method == "lola_style":
        return l0 + peer_l0, l1 + peer_l1

    if method == "meta_mapg":
        return l0 + own_l0 + peer_l0, l1 + own_l1 + peer_l1

    raise ValueError(f"Unknown method: {method}")


def train_one_seed(method, seed, n_steps=260, batch_size=384,
                   peer_coef=1.5, own_coef=0.35, lr=0.9, lr_power=0.24,
                   inner_lr=0.55, log_every=10,
                   schedule="constant", phase1_steps=100,
                   anneal_scale=30, anneal_power=0.7):
    actual_seed = 1000 + 37 * seed
    torch.manual_seed(actual_seed)
    np.random.seed(actual_seed)

    policy0 = MLPPolicy()
    policy1 = MLPPolicy()

    coop_history = []

    for step in range(1, n_steps + 1):
        # Match tabular schedule: lr / (step + 10)^lr_power
        current_lr = lr / ((step + 10.0) ** lr_power)
        if schedule == "constant":
            current_peer_coef = peer_coef
        elif schedule == "two_phase":
            if step <= phase1_steps:
                current_peer_coef = peer_coef
            else:
                current_peer_coef = peer_coef / (
                    1.0 + ((step - 1) - phase1_steps) / float(max(1, anneal_scale))
                ) ** anneal_power
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        with torch.no_grad():
            trajs = rollout_batch(policy0, policy1, batch_size)

        loss0, loss1 = compute_losses(
            policy0, policy1, trajs, method,
            peer_coef=current_peer_coef, own_coef=own_coef, inner_lr=inner_lr
        )

        policy0.zero_grad()
        policy1.zero_grad()

        loss0.backward(retain_graph=True)
        grads0 = {n: p.grad.clone() for n, p in policy0.named_parameters() if p.grad is not None}
        policy0.zero_grad()
        policy1.zero_grad()

        loss1.backward()
        grads1 = {n: p.grad.clone() for n, p in policy1.named_parameters() if p.grad is not None}

        with torch.no_grad():
            for n, p in policy0.named_parameters():
                if n in grads0:
                    p -= current_lr * grads0[n]
            for n, p in policy1.named_parameters():
                if n in grads1:
                    p -= current_lr * grads1[n]

        if step % log_every == 0 or step == n_steps:
            with torch.no_grad():
                start = torch.zeros(1, N_STATES)
                start[0, 4] = 1.0
                c0 = torch.sigmoid(policy0(start)).item()
                c1 = torch.sigmoid(policy1(start)).item()
                coop_history.append((step, c0, c1))

    final_coop0 = coop_history[-1][1]
    final_coop1 = coop_history[-1][2]
    final_coop = (final_coop0 + final_coop1) / 2.0
    success = final_coop >= SUCCESS_THRESHOLD

    with torch.no_grad():
        eval_trajs = rollout_batch(policy0, policy1, 1000)
        final_return = eval_trajs["returns"].mean(dim=0)

    time_to_conv = n_steps
    for i, (s, c0, c1) in enumerate(coop_history):
        if (c0 + c1) / 2.0 >= SUCCESS_THRESHOLD:
            stays = all((coop_history[j][1] + coop_history[j][2]) / 2.0 >= SUCCESS_THRESHOLD * 0.95
                        for j in range(i, len(coop_history)))
            if stays:
                time_to_conv = s
                break

    return {
        "method": method, "seed": seed, "actual_seed": actual_seed,
        "schedule": schedule,
        "final_coop": final_coop, "final_coop0": final_coop0, "final_coop1": final_coop1,
        "final_return0": final_return[0].item(), "final_return1": final_return[1].item(),
        "success": success, "time_to_conv": time_to_conv,
        "coop_history": coop_history,
    }


def wilson_interval(successes, n):
    if n == 0:
        return 0.0, 0.0
    rate = successes / n
    z = 1.96
    denom = 1 + z**2 / n
    center = (rate + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((rate * (1 - rate) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def run_annealing_compare(args):
    configs = [
        ("meta_mapg_constant", "constant"),
        ("meta_mapg_two_phase", "two_phase"),
    ]
    results = []
    print("\n--- MLP Meta-MAPG annealing comparison ---")
    print(f"  seeds={args.anneal_seeds} steps={args.anneal_steps} "
          f"phase1={args.anneal_phase1_steps} scale={args.anneal_scale} power={args.anneal_power}")

    for label, schedule in configs:
        label_results = []
        for seed in range(args.anneal_seeds):
            result = train_one_seed(
                method="meta_mapg",
                seed=seed,
                n_steps=args.anneal_steps,
                batch_size=args.batch_size,
                peer_coef=args.peer_coef,
                own_coef=args.own_coef,
                lr=args.lr,
                lr_power=args.lr_power,
                inner_lr=args.inner_lr,
                log_every=args.anneal_log_every,
                schedule=schedule,
                phase1_steps=args.anneal_phase1_steps,
                anneal_scale=args.anneal_scale,
                anneal_power=args.anneal_power,
            )
            result["label"] = label
            label_results.append(result)
            results.append(result)
        succ = sum(1 for r in label_results if r["success"])
        mean_coop = np.mean([r["final_coop"] for r in label_results])
        print(f"  {label}: {succ}/{len(label_results)} success, mean_coop={mean_coop:.3f}")
    return results


def save_annealing_compare(results, out_dir, args):
    csv_path = os.path.join(out_dir, "mlp_ipd_annealing_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "label", "schedule", "seed", "final_coop", "final_return", "success", "time_to_conv"
        ])
        w.writeheader()
        for r in results:
            w.writerow({
                "label": r["label"],
                "schedule": r["schedule"],
                "seed": r["seed"],
                "final_coop": f"{r['final_coop']:.4f}",
                "final_return": f"{(r['final_return0'] + r['final_return1']) / 2:.4f}",
                "success": int(r["success"]),
                "time_to_conv": r["time_to_conv"],
            })
    print(f"Saved: {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
        labels = ["meta_mapg_constant", "meta_mapg_two_phase"]
        pretty = {
            "meta_mapg_constant": "Constant shaping",
            "meta_mapg_two_phase": "Two-phase",
        }
        colors = {
            "meta_mapg_constant": "#b279a2",
            "meta_mapg_two_phase": "#2fbf71",
        }

        ax = axes[0]
        for i, lbl in enumerate(labels):
            sub = [r for r in results if r["label"] == lbl]
            n = len(sub)
            succ = sum(1 for r in sub if r["success"])
            rate = succ / n
            ci_lo, ci_hi = wilson_interval(succ, n)
            ax.bar(i, 100 * rate, color=colors[lbl], edgecolor="black",
                   linewidth=0.5, width=0.6, alpha=0.9)
            ax.errorbar(i, 100 * rate,
                        yerr=[[100 * (rate - ci_lo)], [100 * (ci_hi - rate)]],
                        fmt="none", ecolor="black", capsize=4, linewidth=1.2)
            ax.text(i, 100 * ci_hi + 2, f"{100 * rate:.0f}%",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([pretty[lbl] for lbl in labels], fontsize=10)
        ax.set_ylabel("Cooperative success rate (%)", fontsize=10)
        ax.set_ylim(0, 105)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2)

        ax = axes[1]
        for lbl in labels:
            sub = [r for r in results if r["label"] == lbl]
            steps = np.array([h[0] for h in sub[0]["coop_history"]], dtype=float)
            traces = np.array([[min(h[1], h[2]) for h in r["coop_history"]] for r in sub], dtype=float)
            mean = traces.mean(axis=0)
            sem = traces.std(axis=0, ddof=1) / np.sqrt(traces.shape[0]) if traces.shape[0] > 1 else np.zeros_like(mean)
            ax.plot(steps, mean, color=colors[lbl], linewidth=1.9, label=pretty[lbl])
            ax.fill_between(steps, np.clip(mean - 1.96 * sem, 0.0, 1.0),
                            np.clip(mean + 1.96 * sem, 0.0, 1.0),
                            color=colors[lbl], alpha=0.18)
        ax.axvline(args.anneal_phase1_steps, color="grey", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Training step", fontsize=10)
        ax.set_ylabel("Min start-state cooperation", fontsize=10)
        ax.set_ylim(0.0, 1.02)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right", fontsize=8)

        fig.suptitle(
            f"MLP IPD constant vs two-phase Meta-MAPG ({args.anneal_seeds} seeds, {args.anneal_steps} steps)",
            fontsize=11,
            y=1.02,
        )
        fig.tight_layout()
        fig_path = os.path.join(out_dir, "mlp_annealing.pdf")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        fig.savefig(fig_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        print(f"Saved: {fig_path}")
        plt.close(fig)
    except ImportError:
        print("matplotlib not available for annealing comparison")


def main():
    parser = argparse.ArgumentParser(description="MLP IPD ablation experiment")
    parser.add_argument("--skip_ablation", action="store_true")
    parser.add_argument("--run_annealing_compare", action="store_true")
    parser.add_argument("--n_seeds", type=int, default=30)
    parser.add_argument("--n_steps", type=int, default=260)
    parser.add_argument("--batch_size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=0.9)
    parser.add_argument("--lr_power", type=float, default=0.24)
    parser.add_argument("--peer_coef", type=float, default=1.5)
    parser.add_argument("--own_coef", type=float, default=0.35)
    parser.add_argument("--inner_lr", type=float, default=0.55)
    parser.add_argument("--out_dir", type=str, default="artifacts/mlp")
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["standard_pg", "meta_pg", "lola_style", "meta_mapg"])
    parser.add_argument("--anneal_seeds", type=int, default=20)
    parser.add_argument("--anneal_steps", type=int, default=2000)
    parser.add_argument("--anneal_phase1_steps", type=int, default=100)
    parser.add_argument("--anneal_scale", type=int, default=30)
    parser.add_argument("--anneal_power", type=float, default=0.7)
    parser.add_argument("--anneal_log_every", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    if not args.skip_ablation:
        print(f"MLP IPD: {args.n_seeds} seeds x {len(args.methods)} methods")
        print(f"  steps={args.n_steps} batch={args.batch_size} lr={args.lr}/(t+10)^{args.lr_power}")
        print(f"  inner_lr={args.inner_lr} peer_coef={args.peer_coef} own_coef={args.own_coef}")

        all_results = []
        for method in args.methods:
            print(f"\n--- {method} ---")
            method_results = []
            for seed in range(args.n_seeds):
                result = train_one_seed(
                    method=method, seed=seed,
                    n_steps=args.n_steps, batch_size=args.batch_size,
                    peer_coef=args.peer_coef, own_coef=args.own_coef,
                    lr=args.lr, lr_power=args.lr_power, inner_lr=args.inner_lr,
                )
                method_results.append(result)
                all_results.append(result)
                if (seed + 1) % 10 == 0:
                    successes = sum(1 for r in method_results if r["success"])
                    coops = [r["final_coop"] for r in method_results]
                    print(f"  {seed+1}/{args.n_seeds}: {successes}/{len(method_results)} "
                          f"success, mean_coop={np.mean(coops):.3f}")

            successes = sum(1 for r in method_results if r["success"])
            coops = [r["final_coop"] for r in method_results]
            print(f"  FINAL: {successes}/{args.n_seeds} ({100*successes/args.n_seeds:.0f}%) "
                  f"mean_coop={np.mean(coops):.3f} +/- {np.std(coops):.3f}")

        csv_path = os.path.join(args.out_dir, "mlp_ipd_summary.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "method", "seed", "final_coop", "final_return", "success", "time_to_conv"])
            w.writeheader()
            for r in all_results:
                w.writerow({
                    "method": r["method"], "seed": r["seed"],
                    "final_coop": f"{r['final_coop']:.4f}",
                    "final_return": f"{(r['final_return0']+r['final_return1'])/2:.4f}",
                    "success": int(r["success"]),
                    "time_to_conv": r["time_to_conv"],
                })
        print(f"Saved: {csv_path}")

        print("\n" + "=" * 72)
        print(f"{'Method':>15s} | {'Success':>8s} | {'95% CI':>14s} | {'Mean Coop':>10s} | {'Mean Ret':>10s}")
        print("-" * 72)
        for method in args.methods:
            mrs = [r for r in all_results if r["method"] == method]
            n = len(mrs)
            succ = sum(1 for r in mrs if r["success"])
            rate = succ / n
            ci_lo, ci_hi = wilson_interval(succ, n)
            mean_coop = np.mean([r["final_coop"] for r in mrs])
            mean_ret = np.mean([(r["final_return0"]+r["final_return1"])/2 for r in mrs])
            label = {"standard_pg": "PG", "meta_pg": "Meta-PG",
                     "lola_style": "Peer only", "meta_mapg": "Meta-MAPG"}[method]
            print(f"{label:>15s} | {100*rate:5.0f}%   | [{100*ci_lo:.1f}%, {100*ci_hi:.1f}%] | "
                  f"{mean_coop:9.3f}  | {mean_ret:9.2f}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6.5, 3.5))
            method_labels = {"standard_pg": "PG", "meta_pg": "Meta-PG",
                             "lola_style": "Peer only", "meta_mapg": "Meta-MAPG"}
            colors = ["#4c78a8", "#72b7b2", "#b279a2", "#e45756"]

            x_pos = range(len(args.methods))
            for i, method in enumerate(args.methods):
                mrs = [r for r in all_results if r["method"] == method]
                n = len(mrs)
                succ = sum(1 for r in mrs if r["success"])
                rate = succ / n
                ci_lo, ci_hi = wilson_interval(succ, n)

                ax.bar(i, 100*rate, color=colors[i], edgecolor="black", linewidth=0.5,
                       width=0.6, alpha=0.85)
                ax.errorbar(i, 100*rate, yerr=[[100*(rate-ci_lo)], [100*(ci_hi-rate)]],
                           fmt="none", ecolor="black", capsize=4, linewidth=1.2)
                ax.text(i, 100*ci_hi + 2, f"{100*rate:.0f}%",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

            ax.set_xticks(list(x_pos))
            ax.set_xticklabels([method_labels[m] for m in args.methods], fontsize=10)
            ax.set_ylabel("Cooperative success rate (%)", fontsize=10)
            ax.set_title(f"Non-tabular IPD: 2-layer MLP (16 hidden, tanh), {args.n_seeds} seeds",
                        fontsize=11)
            ax.set_ylim(0, 105)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.2)

            fig.tight_layout()
            fig_path = os.path.join(args.out_dir, "mlp_ipd.pdf")
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            fig.savefig(fig_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
            print(f"Saved: {fig_path}")
            plt.close()
        except ImportError:
            print("matplotlib not available")

    if args.run_annealing_compare:
        anneal_results = run_annealing_compare(args)
        save_annealing_compare(anneal_results, args.out_dir, args)

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
