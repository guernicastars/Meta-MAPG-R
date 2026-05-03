#!/usr/bin/env bash
# Autonomous validation pipeline.
# Runs validation phases sequentially, writes a tex fragment per phase, then
# rebuilds tex/results_validation.pdf at the end.

set -e
set -o pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/vlad/development/ICML26/.venv/bin/python}"
LOGFILE="artifacts/validation/pipeline.log"
mkdir -p artifacts/validation figures/validation tex

stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

log() {
    echo "[$(stamp)] $*" | tee -a "$LOGFILE"
}

run_phase() {
    local phase=$1
    shift
    log "PHASE $phase START"
    if "$PYTHON" experiments/run_validation_suite.py --phase "$phase" "$@" >>"$LOGFILE" 2>&1; then
        log "PHASE $phase DONE"
        "$PYTHON" experiments/write_phase_tex.py --phases "${phase,,}" >>"$LOGFILE" 2>&1 || true
    else
        log "PHASE $phase FAILED (continuing)"
    fi
}

run_phase A --phase-a-grid 15 --phase-a-batch 16384 --phase-a-reps 6
run_phase B --phase-b-grid 51 --phase-b-steps 140 --phase-b-batch 192
run_phase D --phase-d-seeds 80 --phase-d-n0 100 --phase-d-total 260 --phase-d-scale 30 --phase-d-q 0.7 --phase-d-batch 256
run_phase F
run_phase E --phase-e-grid 21 --phase-e-steps 140 --phase-e-batch 192
run_phase C --phase-c-grid 21 --phase-c-steps 140 --phase-c-batch 128 --phase-c-seeds 10

# New phases (depend on A and C data above)
run_phase A2
run_phase P
run_phase R
run_phase L
run_phase D2 --phase-d2-seeds 80 --phase-d2-n0 100 --phase-d2-total 260 --phase-d2-scale 30 --phase-d2-q 0.7 --phase-d2-batch 256
run_phase G --phase-g-seeds 5 --phase-g-grid 21 --phase-g-steps 140 --phase-g-batch 192
run_phase Q --phase-q-seeds 5 --phase-q-grid 21 --phase-q-steps 140 --phase-q-batch 192
run_phase H --phase-h-grids 11 21 51 --phase-h-steps 140 --phase-h-batch 192
run_phase I --phase-i-grid 51 --phase-i-steps 140 --phase-i-batch 192
run_phase M --phase-m-grid 21 --phase-m-steps 140 --phase-m-batch 192
run_phase N --phase-n-grid 21 --phase-n-steps 140 --phase-n-batch 192
run_phase O
run_phase T --phase-t-grid 21 --phase-t-total 140 --phase-t-warm-steps 5 10 25 50 100
run_phase U --phase-u-grid 21 --phase-u-total 200
run_phase V
run_phase W
run_phase X --phase-x-k-values 1 5 10 25 50
run_phase Y --phase-y-max-steps 100
run_phase Z

# Supplementary phases AA-FF (reviewer-gap experiments)
run_phase AA --phase-aa-grid 21 --phase-aa-steps 1000 --phase-aa-batch 192
run_phase BB --phase-bb-lambdas 0.0 0.5 1.0 2.0 5.0 --phase-bb-grid 11 --phase-bb-steps 1000 --phase-bb-batch 192
run_phase DD --phase-dd-qs 0.0 0.25 0.5 1.0 1.5 2.0 --phase-dd-seeds 80 --phase-dd-n0 100 --phase-dd-total 2000 --phase-dd-scale 30 --phase-dd-batch 256
run_phase EE --phase-ee-seeds 80 --phase-ee-warm-steps 1000 --phase-ee-cool-steps 500 --phase-ee-batch 192 --phase-ee-checkpoint-every 50
run_phase FF --phase-ff-seeds 80 --phase-ff-steps 500 --phase-ff-K 8 --phase-ff-batch 192
# CC last: MLP IPD unroll dominates wall-clock (cap at L in {1,3} if >2h per run)
run_phase CC --phase-cc-Ls 1 3 5 --phase-cc-tabular-seeds 40 --phase-cc-tabular-steps 500 --phase-cc-mlp-seeds 30 --phase-cc-mlp-steps 500

log "BUILD pdf"
cd tex
if command -v latexmk >/dev/null 2>&1; then
    latexmk -pdf -interaction=nonstopmode -halt-on-error results_validation.tex >>"../$LOGFILE" 2>&1 || \
        log "latex compile failed — see $LOGFILE"
else
    pdflatex -interaction=nonstopmode -halt-on-error results_validation.tex >>"../$LOGFILE" 2>&1 || true
    pdflatex -interaction=nonstopmode -halt-on-error results_validation.tex >>"../$LOGFILE" 2>&1 || \
        log "latex compile failed — see $LOGFILE"
fi
cd ..

log "PIPELINE END"
