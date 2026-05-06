# Neural Benchmarks for Meta-MAPG

This directory contains the **neural-policy benchmark suite** for the Meta-MAPG opponent-aware corrections study, complementing the tabular and MLP-IPD experiments at the repo root.

**Compute:** 4×Tesla V100-SXM2-32GB, 12-way data-parallel orchestration via `multiprocessing.spawn`. Approximately 350 GPU-hours for the core sweep + pilot + λ-sweep.

## Headline Result (Overcooked `forced_coordination`, n=25 paired seeds)

| arm | mean | max | learners (≥3.0) |
|---|---:|---:|---:|
| ippo | 3.10 | 34.5 | 4/25 (16%) |
| own_only | 0.06 | 0.75 | **0/25 (0%)** |
| peer_only | **4.77** | **117.00** | 1/25 (4%) — unicorn |
| meta_mapg | 2.71 | 33.0 | 2/25 (8%) |
| handoff | 0.21 | 2.25 | 0/25 (0%) |

**Headline finding:** `peer_only` produces a uniquely deep coordination basin (max 117) that no other arm comes close to.
`own_only` (LOLA self-term) catastrophically destroys learning. The full Meta-MAPG algorithm (`meta_mapg`) is **strictly dominated** by its `peer_only` ablation: the destabilising self-correction term cancels the peer benefit.

See `meta_mapg_neural.pdf` (11-page report) for the full discussion, methodology, and per-environment tables.

## Layout

```
neural_benchmarks/
├── README_NEURAL.md                          this file
├── meta_mapg_neural.pdf                      11-page interim report
├── meta_mapg_neural.tex                      LaTeX source
├── figure_2_basin_entry.pdf                  basin-entry trajectories per (env, arm)
├── figure_3_handoff.pdf                      handoff retention dynamics
├── figure_4_peer_ablation.pdf                final success fraction per (env, arm)
├── meta_mapg_neural_interim_*.zip            self-contained bundle (figures+report+summaries)
│
├── meta_mapg/                                core library (algorithms, envs, plotting)
│   ├── algos/ippo.py                         IPPO + opponent-aware corrections (5 arm impls)
│   ├── envs/{mpe,overcooked,meltingpot}.py   benchmark wrappers
│   ├── orchestrator.py                       multi-GPU sweep dispatcher
│   ├── train.py                              per-run PPO + GAE + corrections training loop
│   ├── plotting/figures.py                   figure_2 / figure_3 / figure_4 generators
│   └── utils.py                              Wilson CI, bootstrap CI, norm helpers
│
├── scripts/
│   ├── pilot.py                              short pilot run → threshold.txt
│   ├── run_sweep.py                          full sweep launcher
│   ├── lambda_sweep.py                       λ ∈ {0…5} on simple_spread
│   └── make_figures.py                       regenerate figures from artifacts/runs/
│
├── configs/                                  YAML per-env (5 environments)
│   ├── mpe_simple_spread.yaml
│   ├── mpe_simple_reference.yaml
│   ├── mpe_speaker_listener.yaml
│   ├── overcooked_forced.yaml
│   ├── overcooked_ring.yaml                  not run (compute budget)
│   └── meltingpot_pd_arena.yaml
│
├── tests/                                    smoke tests
├── pyproject.toml
├── run_all.sh                                end-to-end pilot+sweep+λ+figures+bundle script
├── setup.sh                                  uv venv + dependencies
│
└── artifacts/
    ├── REPORT.md                             markdown version of the report
    │
    ├── figures/
    │   ├── figure_2_basin_entry.{pdf,png}
    │   ├── figure_3_handoff.{pdf,png}        (under handoff/)
    │   ├── figure_4_peer_ablation.{pdf,png}
    │   └── metrics.csv                       per (env, arm) summary table
    │
    ├── pilot/
    │   └── <env>/<arm>/seed_XXX/             pilot-stage runs (250k steps each)
    │       ├── config.json
    │       └── summary.json
    │
    └── runs/                                 main 600-run sweep
        ├── mpe/{simple_spread,simple_reference,simple_speaker_listener}/
        │   ├── sweep_summary.json
        │   ├── lam_*/                        (λ-sweep, simple_spread only)
        │   └── <arm>/seed_XXX/
        │       ├── config.json               full TrainConfig used for this run
        │       ├── eval.jsonl                per-checkpoint eval-time series (KEY DATA)
        │       └── summary.json              per-seed final aggregated metrics
        ├── overcooked/forced_coordination/<arm>/seed_XXX/{config,eval,summary}.json[l]
        └── meltingpot/prisoners_dilemma_in_the_matrix__repeated/<arm>/seed_XXX/{...}
```

**Note:** the per-seed `train.jsonl` files (per-rollout training-time metrics; gradient norms,
correction magnitudes, etc.; ~400 KB each, ~250 MB total across 624 seeds) are **NOT included**
in this repo. They are preserved on the training cluster and available on request.
For plot reconstruction the `eval.jsonl` + `summary.json` + `config.json` are sufficient.

## What was run

Total: **600 core runs + 48 pilot runs + 96 λ-sweep runs = 744 paired training runs.**

| Benchmark | Env | Arms | Seeds | Steps | Status |
|---|---|---:|---:|---:|---|
| MPE | `simple_spread` | 5 | 25 | 750k | ✅ done |
| MPE | `simple_reference` | 5 | 25 | 750k | ✅ done |
| MPE | `simple_speaker_listener` | 5 | 25 | 750k | ✅ done |
| Overcooked-AI | `forced_coordination` | 5 | 25 | 1M | ✅ done |
| Melting Pot | `prisoners_dilemma_in_the_matrix__repeated`* | 4 | 25 | 1M | ✅ done |
| MPE λ-sweep | `simple_spread` × 8 λ | 1 | 12 | 750k | 🔄 in progress |

*PettingZoo IPD fallback (dm_meltingpot unavailable). All arms collapse to defection equilibrium → uninformative.

## Algorithmic arms

All five arms share the IPPO backbone (PPO + GAE per agent, separate actor/critic, paired-seed init).
Differences:

- **`ippo`** — vanilla; no correction. Baseline.
- **`own_only`** — adds DiCE self-correction (LOLA self-term).
- **`peer_only`** — adds DiCE peer-correction (Meta-MAPG opponent term). **The interesting arm.**
- **`meta_mapg`** — both corrections active. The published full algorithm.
- **`handoff`** — `meta_mapg` until step `T_warm`, then revert to `ippo`.

All corrections norm-clipped: `‖Δ_corr‖ ≤ 1.0 · ‖Δ_PG‖`. `eta_inner = 0.1` baseline.

## Plot regeneration

```bash
cd neural_benchmarks
python scripts/make_figures.py \
    --runs-root artifacts/runs \
    --out-dir   artifacts/figures \
    --config-dir configs
```

This reads the per-seed `eval.jsonl` files, computes Wilson CIs / bootstrap CIs, and re-renders all 4 figures + `metrics.csv`. To explore your own derived plots from the raw eval-time data:

```python
import json
from pathlib import Path
import pandas as pd

# Per-seed eval trajectory for one (env, arm)
arm_dir = Path("artifacts/runs/overcooked/forced_coordination/peer_only")
trajectories = []
for sd in sorted(arm_dir.iterdir()):
    rows = [json.loads(l) for l in (sd / "eval.jsonl").read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df["seed"] = sd.name
    trajectories.append(df)
all_seeds = pd.concat(trajectories)
# all_seeds has columns: step, eval_return_mean, eval_return_std, eval_episodes, seed
```

Each `eval.jsonl` row records the result of evaluating the policy on 50 episodes
at a given training step (greedy / argmax actions).

## Caveats (see report §6)

- **Pilot threshold cascade**: `figure_4` reads `100%` success on `forced_coordination` because
  the auto-pilot wrote `threshold = 0.0`. The honest signal is in **mean final return** in `metrics.csv`
  and `tables/forced_coordination` of the report.
- **dm_meltingpot fallback**: PettingZoo IPD is uninformative. All 4 arms → 0.0.
- **Greedy eval**: argmax actions; stochastic-policy returns may be higher.
- **Single seed family**: seeds 0–24, paired across arms.
- **`coordination_ring` dropped** post-sparse-reward fix (compute budget).

## Citation context

This neural-benchmark sweep is a companion to the tabular Meta-MAPG analysis in the parent repo's
`main.tex` / `main.pdf`. The peer-only-dominance finding here ($\arm{meta\_mapg}$ strictly dominated by
$\arm{peer\_only}$ on coordination tasks) extends the tabular two-phase convergence narrative
to the deep-RL regime.
