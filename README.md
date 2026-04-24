# Meta-MAPG Restart Paper

This repository contains a fresh ICML-style workshop draft for the Meta-MAPG two-phase convergence and restart-globalisation thesis.

Canonical files:

- `main.tex`: LaTeX source for the paper.
- `main.pdf`: compiled PDF.
- `references.bib`: bibliography.
- `experiments/run_meta_mapg_experiments.py`: sample-based tabular experiments.
- `experiments/run_mlp_ipd.py`: optional PyTorch MLP-policy IPD check.
- `artifacts/main/`: experiment outputs used in the paper.
- `artifacts/mlp/`: aggregate MLP-policy IPD check outputs.
- `figures/` and `tables/`: paper-ready copies of the generated figures and summary table.

Regenerate the experiments from this directory with:

```bash
python3 experiments/run_meta_mapg_experiments.py \
  --outdir artifacts/main \
  --seeds 100 \
  --steps 260 \
  --restart-steps 120 \
  --max-restarts 12 \
  --selection-budget 12 \
  --selection-seeds 100 \
  --selection-steps 120 \
  --trajectory-steps 140 \
  --trajectory-batch-size 384 \
  --trajectory-grid-size 5 \
  --batch-size 384 \
  --basin-batch-size 192 \
  --grid-size 21 \
  --basin-steps 140 \
  --reference-batch-size 120000 \
  --sanity-reps 80 \
  --own-coef 0.35 \
  --peer-coef 1.5
```

Regenerate the optional MLP-policy IPD appendix check with PyTorch installed:

```bash
python3 experiments/run_mlp_ipd.py \
  --n_seeds 100 \
  --n_steps 260 \
  --batch_size 384 \
  --peer_coef 1.5 \
  --own_coef 0.35 \
  --inner_lr 0.55 \
  --out_dir artifacts/mlp
```

Regenerate the appendix constant-vs-two-phase MLP comparison with:

```bash
python3 experiments/run_mlp_ipd.py \
  --skip_ablation \
  --run_annealing_compare \
  --anneal_seeds 20 \
  --anneal_steps 2000 \
  --anneal_phase1_steps 100 \
  --anneal_scale 30 \
  --anneal_power 0.7 \
  --anneal_log_every 20 \
  --batch_size 384 \
  --peer_coef 1.5 \
  --own_coef 0.35 \
  --inner_lr 0.55 \
  --out_dir artifacts/mlp
```

Recompile the paper with:

```bash
latexmk -pdf -interaction=nonstopmode main.tex
```
