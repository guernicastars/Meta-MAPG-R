#!/usr/bin/env bash
# Autonomous validation pipeline.
# Runs phases A..F sequentially, writes a tex fragment per phase, then
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
run_phase L
run_phase D2 --phase-d2-seeds 80 --phase-d2-n0 100 --phase-d2-total 260 --phase-d2-scale 30 --phase-d2-q 0.7 --phase-d2-batch 256
run_phase G --phase-g-seeds 5 --phase-g-grid 21 --phase-g-steps 140 --phase-g-batch 192
run_phase Q --phase-q-seeds 5 --phase-q-grid 21 --phase-q-steps 140 --phase-q-batch 192
run_phase H --phase-h-grids 11 21 51 --phase-h-steps 140 --phase-h-batch 192
run_phase I --phase-i-grid 51 --phase-i-steps 140 --phase-i-batch 192
run_phase M --phase-m-grid 21 --phase-m-steps 140 --phase-m-batch 192
run_phase N --phase-n-grid 21 --phase-n-steps 140 --phase-n-batch 192
run_phase O

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
