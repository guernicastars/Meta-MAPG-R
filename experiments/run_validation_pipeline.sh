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
