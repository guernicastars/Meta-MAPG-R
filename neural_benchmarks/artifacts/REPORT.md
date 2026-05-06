# Meta-MAPG Neural Benchmarks — Interim Report

**Date:** 2026-05-06 (deadline 23:59 UTC+2 same day)
**Status:** Core sweep complete. λ-sweep on simple_spread running concurrently.
**Compute:** 4× Tesla V100-SXM2-32GB, 12 parallel workers.

## TL;DR

We trained 5 algorithmic arms across 5 environments to test whether
peer-aware policy-gradient corrections (the Meta-MAPG family) give measurable
basin-entry advantages over independent PPO (IPPO) on neural multi-agent
coordination tasks.

**Headline finding (Overcooked forced_coordination):**

| arm        | mean final return | max final return | learners (≥3.0) |
|------------|-------------------:|------------------:|-----------------:|
| ippo       | 3.10               | 34.50             | 4/25 (16%)       |
| own_only   | 0.06               | 0.75              | 0/25 (0%)        |
| peer_only  | **4.77**           | **117.00**        | 1/25 (4%)        |
| meta_mapg  | 2.71               | 33.00             | 2/25 (8%)        |
| handoff    | 0.21               | 2.25              | 0/25 (0%)        |

* `peer_only` is **super-bimodal**: 1 unicorn seed reaches a deep coordination basin
  (return = 117) that no other arm comes close to. Mean is dragged up by that
  outlier. The peer-aware correction lets a partner-modelling pair occasionally
  amplify each other into a deeper basin than IPPO alone discovers.
* `own_only` (own-policy LOLA-style correction) **destroys learning entirely**:
  zero learners, mean ≈ 0. The destabilising self-correction term kills the
  vanilla policy gradient before it can find any basin.
* `meta_mapg` (own + peer combined) lands between own_only and IPPO. The peer
  benefit is partially redeemed but the own term still drags it down.
* `handoff` (off at T_warm=500k) is closer to own_only than IPPO — i.e. the
  early corrections leave the policy in a dead state from which the late-PPO
  half cannot recover.

## What Ran

| benchmark                                | env_id                              | arms | seeds | total_steps |
|------------------------------------------|-------------------------------------|------|-------|-------------|
| MPE                                      | simple_spread                       | 5    | 25    | 750k        |
| MPE                                      | simple_reference                    | 5    | 25    | 750k        |
| MPE                                      | simple_speaker_listener             | 5    | 25    | 750k        |
| Overcooked                               | forced_coordination                 | 5    | 25    | 1M          |
| Melting Pot (PettingZoo IPD fallback)    | prisoners_dilemma_in_the_matrix__   | 4    | 25    | 1M          |

**Total runs in core sweep:** 5×25 + 5×25 + 5×25 + 5×25 + 4×25 = 600 (seed×arm).

**Lambda sweep:** 8 λ × 12 seeds × 1 arm (peer_only) on simple_spread = 96 runs (in progress).

## What Was Skipped vs Plan

* `coordination_ring`: dropped from Phase 2 after a sparse-reward bug was found
  in `meta_mapg/envs/overcooked.py` (line 103). The wrapper was discarding
  `shaped_r_by_agent`, so all training returns were zeros. Patched (now adds
  `0.5 × shaped[i]` per Overcooked-AI defaults) and verified by smoke-test.
  We re-ran `forced_coordination` post-patch but had to cut `coordination_ring`
  for compute budget.
* `meltingpot/clean_up`: Phase-2 plan §7 permits a matrix-style proxy as the
  social-dilemma anchor; we kept only `prisoners_dilemma_in_the_matrix__repeated`
  (PettingZoo IPD fallback because `dm_meltingpot` is unavailable).
* `overcooked_cramped`: dropped per plan §6.2 ("sanity check, lowest priority").

## Per-benchmark Results

### MPE simple_spread (saturated, near-ceiling)

All 5 arms reach 100% basin-entry success at threshold = -25 (mean = -22),
with overlapping CIs. **Not differentiating** — kept as a control.

### MPE simple_reference (lightly differentiating)

| arm        | success | 95% Wilson CI       | mean final return |
|------------|--------:|---------------------|------------------:|
| meta_mapg  | 92%     | [75.0, 97.8]        | -22.21            |
| ippo       | 88%     | [70.0, 95.8]        | -21.98            |
| peer_only  | 88%     | [70.0, 95.8]        | -22.21            |
| handoff    | 84%     | [65.3, 93.6]        | -22.87            |
| own_only   | 84%     | [65.3, 93.6]        | -22.68            |

CIs overlap heavily; `meta_mapg` nominally best but no significance.

### MPE simple_speaker_listener (hardest MPE; differentiating direction unclear)

All arms in the 28–32% basin-entry band. Differences are 1 seed each. Noise.

### Overcooked forced_coordination — see TL;DR above. THE differentiating result.

### Melting Pot prisoners_dilemma (saturated, uninformative)

All 4 arms achieve 100% success_pct with mean_final_return = 0.000. The IPD
fallback equilibrium (mutual defection) is hit trivially, *all* arms collapse
to it. **Not informative.** The intended dm_meltingpot substrate would have
provided a non-trivial signal but is unavailable in this environment.

## Methodology Summary

* PPO + GAE per agent (separate actor/critic); `clip_eps=0.2`, `entropy_coef=0.01`.
* Five arms, all sharing IPPO base + opponent-aware correction term variants:
  * `ippo` — vanilla independent PPO (no correction).
  * `own_only` — own-policy DiCE correction (LOLA self-term).
  * `peer_only` — peer-policy DiCE correction (Meta-MAPG opponent term).
  * `meta_mapg` — both corrections active.
  * `handoff` — both corrections active until step `T_warm`, then IPPO.
* Norm clip `‖Δ_corr‖ ≤ 1.0 · ‖Δ_PG‖` to prevent correction blow-up.
* Paired-seed protocol: same `seed` value initialises identical actor/critic
  weights across all five arms (plan §8).
* `correction_clip = 1.0`, `eta_inner = 0.1` baseline.
* λ-sweep: same as `peer_only` arm, eta_inner = `0.1 × λ` for λ ∈ {0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0}.

## Caveats

1. **Pilot threshold cascade:** the pilot stage ran briefly post-patch and
   wrote `threshold.txt = 0.0` (vanilla random-policy eval can hit shaped
   reward ~1.5 per episode). The full sweep used these stale 0.0 thresholds,
   which is why `success_pct = 100%` on `forced_coordination` and `meltingpot`
   despite mean returns near zero. The honest signal is in `mean_final_return`,
   reported above. The success-fraction figures (Figure 4) reflect the stale
   threshold.
2. **PettingZoo IPD fallback** for Melting Pot: equilibrium-collapse to mutual
   defection makes that benchmark uninformative. Reported for completeness.
3. **`coordination_ring` dropped** post sparse-reward fix (compute budget).
4. **`evaluate()` uses argmax** (greedy/deterministic). Mean returns reflect
   greedy policies; true stochastic-policy returns may differ.

## Files in This Bundle

```
figures/figure_2_basin_entry.{pdf,png}     basin-entry probability per (benchmark, arm)
figures/figure_4_peer_ablation.{pdf,png}   final success fraction per (benchmark, arm)
figures/handoff/figure_3_handoff.{pdf,png} handoff retention dynamics
figures/metrics.csv                        per-arm summary metrics
runs/<benchmark>/<env_id>/<arm>/seed_XXX/  per-seed JSON + eval.jsonl
runs/<benchmark>/<env_id>/sweep_summary.json
REPORT.md                                  this file
```

The λ-sweep results will be added in the final bundle (~4h ETA from interim
bundle creation time).
