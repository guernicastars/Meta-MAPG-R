# Coauthor Handoff — Non-Tabular IPD with MLP Policy

**Purpose.** Workshop reviewers will flag the paper for being tabular-only after the Discussion gestures at deep MARL. One small non-tabular experiment closes this gap. It is a **proof-of-transport**, not a scaling study — its only job is to show that the Meta-MAPG ablation (PG vs Meta-PG vs peer-only vs full) survives a modestly parameterized policy.

Target slot in the paper: a new subsection at the end of §5 (Experiments), or a new appendix section. One figure + one short paragraph is enough.

---

## 1. Scope — what this experiment is and is not

**Is:**
- A controlled repeat of the IPD ablation already in the paper, but with an MLP instead of a tabular Bernoulli policy.
- Same four methods: `standard_pg`, `meta_pg`, `lola_style` (peer-only), `meta_mapg`.
- ≥30 seeds for headline numbers. 100 if compute allows.
- Stag Hunt stays tabular; only IPD goes non-tabular.

**Is not:**
- A deep MARL benchmark. No CNNs, no Atari, no PPO baseline.
- A scaling study. Do not sweep hidden-dim, depth, batch size.
- A claim that Meta-MAPG wins in deep MARL. Language must stay "qualitative effect survives parameterization."

If the qualitative ordering (peer methods > non-peer methods) holds with overlapping CIs, that is success.

---

## 2. Environment

Use the same IPD game as in the existing tabular experiment:
- 2 agents, 12-step horizon, discount 0.96.
- Payoffs `[[3,0],[5,1]]` for row player, transpose for column.
- State: one-hot encoding of the previous joint action, 4 dims (CC, CD, DC, DD) plus a "start" flag → 5 dims total. Match `Game.n_states=5` in `run_meta_mapg_experiments.py:40`.

**Rationale for state encoding.** Keep it comparable to the tabular experiment, where each of the 5 "states" was a Bernoulli scalar. The MLP sees the same information; only the policy parameterization changes.

---

## 3. Policy architecture

- **Input.** 5-dim one-hot previous joint action (cf. tabular 5-state table).
- **Hidden.** 2 fully connected layers, 16 units each, `tanh` activations.
- **Output.** Single logit → Bernoulli probability of cooperation.
- **Parameter count.** 5·16 + 16 + 16·16 + 16 + 16·1 + 1 ≈ 385 per agent.

Do not use layer norm, dropout, or residuals. Keep the network plain so the only thing that differs from the tabular run is the parameterization.

Initialization: `nn.Linear` default (Kaiming uniform) with final layer scaled by 0.1 so initial cooperation probability is close to 0.5.

---

## 4. Gradient estimators — the actual work

The tabular code in `run_meta_mapg_experiments.py:130-166` (`estimate_components`) builds three pieces by hand using explicit score / Hessian formulas:
- `base[player] = E[R_i · s_i]`
- `own[player] = inner_lr · H_ii^T · g_i` (own-learning, Meta-PG)
- `peer[player] = inner_lr · C_{j,i}^T · q_{j wrt i}` (peer-learning, LOLA-style)

You cannot port those closed forms directly — for an MLP you need autodiff. Two implementation paths:

**Path A: DiCE / stop-gradient (preferred).**
Use DiCE-style magic boxes [Foerster et al. 2018] to get correct higher-order gradients for the policy-gradient and LOLA-like terms. PyTorch snippet:
```python
def dice(log_probs):
    # Exp-of-diff trick: dice(x) has value 1 and gradient exp(x - x.detach()) * grad(x)
    return (log_probs - log_probs.detach()).exp()
```
Then for agent `i`:
- `base_loss_i = -(returns_i.detach() * dice(sum_log_probs_i)).sum() / B`
- `own_loss_i = -inner_lr · inner_product(g_i, H_ii · g_i)` — computed via a single autograd pass through a DiCE surrogate that includes both policies.
- `peer_loss_i` analogous, using the cross-term.

This keeps the math aligned with the tabular code. The resulting `(base, own, peer)` arrays have the same shape as the tabular version, just flattened over MLP parameters.

**Path B: finite-difference baseline.**
If DiCE is too fiddly, use numerical directional derivatives for the own/peer terms. Slower and noisier but dependency-free. Only use as a sanity check against Path A.

**Critical correctness check.** Once you have the three components, verify on **the tabular IPD** that your MLP-aware code reproduces the existing tabular numbers to within seed noise. If yes, port is correct. If no, stop and fix before running any MLP experiments. Do this by running the MLP code with `n_states=5`, a linear "MLP" of one layer (no hidden), and Bernoulli output — it should match `run_rollout` in the current code.

---

## 5. Training loop

```python
for seed in range(n_seeds):
    torch.manual_seed(seed)
    policies = [MLPPolicy(...), MLPPolicy(...)]
    optimizers = [custom_sgd(p.parameters(), lr=lr_schedule) for p in policies]
    for step in range(n_steps):
        # 1. Sample B trajectories of length 12 under current joint policy
        trajs = rollout(policies, B=batch_size, horizon=12)
        # 2. Build DiCE surrogate losses per agent
        losses = compute_ablation_losses(policies, trajs, method=method,
                                         peer_coef=peer_coef, own_coef=own_coef)
        # 3. Apply one joint SGD step
        for i in range(2):
            optimizers[i].zero_grad()
            losses[i].backward()
            # manually scale gradient if your custom scheduler needs it
            optimizers[i].step()
        # 4. Log cooperation rate at start-state every log_every
        ...
    record(seed, method, final_coop_rate)
```

Step-size schedule: match the existing code (`lr=0.9`, `lr_power=0.24`). Constant peer coefficient `peer_coef=1.5`, own coefficient `own_coef=0.35`. **Do not tune.** If the ablation ordering fails to show up cleanly at these values, that's a real finding — report it; don't re-tune until the answer is pretty.

Seeds: use the same seed schedule as the tabular ablation (`1000 + 37 * seed`) for comparability.

---

## 6. Compute budget

Per seed: 260 steps × batch 384 × 12 horizon × 2 agents × forward+backward through MLP. On CPU roughly 1 min/seed; on GPU negligible.

For 30 seeds × 4 methods = 120 runs, budget ~2 h on CPU, ~15 min on GPU. Start with 30 seeds; expand to 100 only if results are promising.

---

## 7. Metrics and success criteria

Primary: **cooperative success rate** at the end of training, same threshold as tabular (final cooperation rate ≥ 0.82 at start state).

Secondary: **final mean return** per method, **time-to-convergence** (first step where cooperation ≥ 0.82 and stays there).

**Success criterion (what we need for the paper to be improved):**
- Peer methods (`lola_style`, `meta_mapg`) must be noticeably above non-peer methods (`standard_pg`, `meta_pg`) with non-overlapping 95% CIs, **or** we must honestly report that the effect weakens under MLP parameterization and discuss why.

Either outcome is publishable. The trap to avoid is reporting only the case where it works.

---

## 8. Integration with existing pipeline

- Put the new code in `experiments/run_mlp_ipd.py` (a separate file, not bolted onto the NumPy script). The existing experiment script is NumPy-based; pulling PyTorch into it will slow the other runs.
- Write outputs to `artifacts/mlp/` so they don't collide with the tabular artifacts.
- Produce `mlp_ipd_summary.csv` with columns `method, seed, final_coop, final_return, success` and `mlp_ipd.pdf` (a bar plot analogous to `ablation_success.pdf`).
- Add a `run_mlp_ipd.sh` wrapper that sets the seed schedule and peer/own coefs.
- Add ONE paragraph + ONE figure to `main.tex`. New subsection "Non-tabular check" inside §5, with placement right before "Discussion".

Suggested figure caption template:
> **Non-tabular IPD check.** Ablation success rate on IPD when each agent's policy is a 2-layer MLP with 16 hidden units per layer. Peer-aware methods (peer-only, Meta-MAPG) remain [above / comparable to] non-peer methods (PG, Meta-PG) at the same peer coefficient used in the tabular experiments; the qualitative effect survives modest parameterization. This is a proof-of-transport, not a deep-MARL benchmark.

---

## 9. Gotchas and things to watch

1. **Autodiff through LOLA term.** Standard PyTorch `backward()` does not produce the LOLA gradient correctly; you need DiCE or manual double-backward. Easy to get wrong silently. Verify against tabular numbers before running full experiments.

2. **Batch-size sensitivity.** LOLA-style estimators have high variance. If you see wild seed-to-seed variance, raise batch size to 512 or 768 before adding regularization.

3. **Policy saturation.** Tanh MLPs can saturate early with large step sizes. If you see all agents collapse to always-defect within the first 20 steps, scale the final linear layer's init variance down further (×0.05 instead of ×0.1).

4. **State encoding.** If you accidentally feed the current state instead of the previous-action one-hot, the IPD becomes memoryless and cooperation collapses. Verify with a single rollout print.

5. **Do not add a critic.** This is REINFORCE-level; an actor-critic would change the estimator and invalidate comparison to the tabular version.

6. **Do not tune `peer_coef`.** The whole paper uses 1.5 everywhere. Keep it.

---

## 10. Deliverables for integration

Push to the paper branch:
- `experiments/run_mlp_ipd.py` — implementation.
- `artifacts/mlp/mlp_ipd_summary.csv` — raw per-seed results.
- `figures/mlp_ipd.pdf` — bar plot.
- One new paragraph + `\begin{figure}` block in `main.tex` in §5, plus one line in the reproducibility appendix.
- A short note in the commit message stating seeds count, result ordering, and whether the ablation ordering held.

If it doesn't hold, open a PR with the negative result and ping the lead before modifying captions in the main experiment section — we may need to recalibrate the "peer-learning matters" claim.

---

## 11. Time estimate

- Environment + MLP + DiCE: 3–4 hours.
- Correctness check against tabular code: 1 hour.
- Full 30-seed runs: 2 hours CPU or 15 min GPU.
- Plot + paper integration: 30 min.

Total: one focused working day.

---

## 12. What to skip if time-limited

If compute or calendar is tight, the minimum viable version is:
- Drop `meta_pg` (redundant under MLP — the own-learning term empirically matters less than peer).
- 20 seeds instead of 30 (enough for a directional claim, not for tight CIs).
- Report only success rate, skip the time-to-convergence column.

The absolute floor is: PG vs Meta-MAPG, 20 seeds, MLP policy, single-panel success-rate bar. Anything less isn't worth adding to the paper.
