# Coauthor Memo — Inner-Unroll Sweep L ∈ {1, 3, 5} for IPD

## Why

A reviewer concern: peer-only and full Meta-MAPG are within CIs in both Stag Hunt
(42% vs 42%, 1.24 vs 1.24 restarts) and IPD (32% vs 37% within Wilson CI). The
own-learning term is therefore empirically indistinguishable from zero at L=1,
which is what the current code implements in both `run_meta_mapg_experiments.py`
and `run_mlp_ipd.py`.

Meta-MAPG's own-term is supposed to compound across L inner updates
(`own_i = η · H_i · v_i`, iterated). At L=1 it is structurally small compared
to the peer term; larger L should amplify it. The sweep tests this.

Target outcome: either the own-only ablation lifts with L, validating the
name, or it stays within CIs and we reframe as "opponent-aware PG" in the
next revision.

## What to run

Scope is IPD only (Stag Hunt is one-shot, L>1 is meaningless there).

For each L ∈ {1, 3, 5}, repeat the four-method ablation
(standard_pg, meta_pg, lola_style, meta_mapg) with ≥50 seeds. Single plot:
Wilson 95% CI bars grouped by L on the x-axis, method as the hue.

Target slot in the paper: one subfigure in `app:mlp`, two sentences of text.

## What needs to change in code

Both scripts currently do a single analytic inner step. To get an honest L,
run `L` explicit sample-based inner adaptations on a "virtual" joint policy
copy, then differentiate the outer loss through them.

### Tabular (`experiments/run_meta_mapg_experiments.py`)

The cleanest change is a new helper that replaces `estimate_components` for
IPD only:

```python
def estimate_components_L(theta, game, batch_size, rng, inner_lr, L):
    # L=1: existing code. L>1: unroll inner updates.
    theta_inner = theta.copy()
    for _ in range(L):
        comps_inner = estimate_components(theta_inner, game, batch_size, rng, inner_lr)
        theta_inner = theta_inner + inner_lr * (comps_inner["current"] +
                                                comps_inner["own"] +
                                                comps_inner["peer"])
    # Final outer components evaluated at the L-unrolled policy
    return estimate_components(theta_inner, game, batch_size, rng, inner_lr)
```

Then add `--inner-unroll` CLI flag and plumb through the IPD branches only.

*Caveat*: the derivation in `app:decomp` assumed a single inner step. For
L>1, the own-term is `η · sum_{k=0..L-1} H_i(φ^k) · v_i(φ^k)` and the peer
term is its cross-agent analogue. The truncation bias `β_n` grows with L
rather than shrinking, so at constant L this is a phase-1 statement. For
phase 2 use `L_n = ⌈r log n / log(1/γ)⌉` as in Assumption 3b.

### MLP (`experiments/run_mlp_ipd.py`)

DiCE-based. Same structure: replace the single analytic inner-step surrogate
(`own_l0`, `peer_l0` at line 146/159) with an L-step unroll using
`torch.func.functional_call` or `higher`. Higher-order autograd works fine;
memory scales linearly in L at this policy size.

If torch.func is awkward, just do L analytic inner steps: compute the one-step
meta-loss on a fresh virtual copy of each policy, create `policy_L` via
`L` successive `.step()` calls on a fresh optimizer, then compute the outer
REINFORCE gradient on that `policy_L`. Retain the DiCE surrogate for the
single step that contributes the higher-order term.

## Expected compute

Tabular IPD: current 100 seeds × 260 steps ≈ 30s on laptop. L=3 ≈ 90s, L=5
≈ 150s. Total sweep ≈ 5 min.

MLP IPD: current 100 seeds × 260 steps ≈ 30–60 min on CPU.
L=3 ≈ 1.5–3h, L=5 ≈ 3–5h. Reduce to 50 seeds for the sweep if compute-bound.

## Outputs to send back

Please drop CSVs under `artifacts/main/L_sweep_ipd.csv` and
`artifacts/mlp/L_sweep_ipd.csv`. A figure named `figures/l_sweep_ipd.pdf`
with the four methods for each L. I'll integrate into the appendix and
update the two-sentence text in `sec:experiments` currently flagging this
as follow-up.

## Decision rule

- If own-only success rate rises monotonically with L and the gap between
  Meta-MAPG and peer-only becomes significant (non-overlapping Wilson CIs)
  at L=3 or L=5 → keep the Meta-MAPG framing, add the figure.
- Otherwise → we rename in the v2 submission to
  "Restart-globalised opponent-aware PG" and demote the own-term to an
  appendix remark.
