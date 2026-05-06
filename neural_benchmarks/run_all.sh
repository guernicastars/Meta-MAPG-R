#!/usr/bin/env bash
# Master orchestrator.  Runs the FULL plan:
#   pilot → propagate-threshold → real-sweep × {3 MPE × 3 Overcooked × 2 Melting Pot}
#         → λ-sweep → figures → bundle.
#
# Designed for nohup-style detach:
#     nohup bash run_all.sh > run_all.log 2>&1 &
#     tail -f run_all.log
#
# Subset env vars (all optional):
#     PARALLEL=4               # how many concurrent runs (defaults to # GPUs)
#     SEEDS_OVERRIDE=10        # smoke test with fewer seeds
#     SKIP_PILOT=1             # skip the pilot stage  (uses YAML threshold)
#     SKIP_OVERCOOKED=1        # skip all 3 Overcooked layouts
#     SKIP_MELTINGPOT=1        # skip all Melting Pot substrates
#     SKIP_LAMBDA=1            # skip the λ-sweep on simple_spread
#     SKIP_FIGURES=1           # don't render figures (e.g. for headless smoke)
#     PRE_FLIGHT=1             # 5-minute smoke: 2 seeds, MPE only, no pilot, no λ

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# Headless rendering — required for Overcooked-AI's pygame import.
export SDL_VIDEODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1

# shellcheck disable=SC1091
source .venv/bin/activate

# ---------- helpers ----------
echo_step() { echo; echo "==================================================================="; echo "[run_all] $1"; echo "==================================================================="; }

PARALLEL="${PARALLEL:-12}"
SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-}"
TOTAL_STEPS_OVERRIDE="${TOTAL_STEPS_OVERRIDE:-}"
EVAL_EVERY_OVERRIDE="${EVAL_EVERY_OVERRIDE:-}"
PILOT_DIR="artifacts/pilot"

# Pre-flight smoke convenience: 2 seeds, 50k steps each, MPE only, no pilot.
if [[ "${PRE_FLIGHT:-0}" == "1" ]]; then
    SEEDS_OVERRIDE="${SEEDS_OVERRIDE:-2}"
    TOTAL_STEPS_OVERRIDE="${TOTAL_STEPS_OVERRIDE:-50000}"
    EVAL_EVERY_OVERRIDE="${EVAL_EVERY_OVERRIDE:-25000}"
    SKIP_OVERCOOKED=1
    SKIP_MELTINGPOT=1
    SKIP_PILOT=1
    SKIP_LAMBDA=1
    PARALLEL="${PARALLEL:-4}"
    echo "[run_all] PRE_FLIGHT=1 — 2 seeds × 50k steps × MPE only × no pilot/λ.  ETA ~5 min."
fi

CONFIGS_MPE=(
    configs/mpe_simple_spread.yaml
    configs/mpe_simple_reference.yaml
    configs/mpe_speaker_listener.yaml
)
CONFIGS_OVERCOOKED=(
    # v3.5: dropped overcooked_cramped (plan §6.2 "sanity check", lowest priority)
    configs/overcooked_ring.yaml
    configs/overcooked_forced.yaml
)
CONFIGS_MELTINGPOT=(
    # v3.5: dropped meltingpot_clean_up (RGB substrate; MLP-on-flat-pixels weak;
    # plan §7 permits using matrix-style proxy as the social-dilemma anchor).
    configs/meltingpot_pd_arena.yaml
)

run_pilot() {
    local cfg="$1"
    echo_step "PILOT $cfg"
    python scripts/pilot.py "$cfg" \
        --pilot-seeds 4 \
        --output-dir "$PILOT_DIR" \
        --max-parallel "$PARALLEL"
}

run_full() {
    local cfg="$1"
    echo_step "FULL  $cfg"
    extra=()
    if [[ -n "$SEEDS_OVERRIDE" ]];        then extra+=(--seeds "$SEEDS_OVERRIDE"); fi
    if [[ -n "$TOTAL_STEPS_OVERRIDE" ]];  then extra+=(--total-steps "$TOTAL_STEPS_OVERRIDE"); fi
    if [[ -n "$EVAL_EVERY_OVERRIDE" ]];   then extra+=(--eval-every "$EVAL_EVERY_OVERRIDE"); fi
    if [[ -d "$PILOT_DIR" ]];             then extra+=(--pilot-dir "$PILOT_DIR"); fi
    python scripts/run_sweep.py "$cfg" \
        --output-dir artifacts/runs \
        --max-parallel "$PARALLEL" "${extra[@]}"
}

# ---------- 1. Pilot all benchmarks ----------
if [[ "${SKIP_PILOT:-0}" != "1" ]]; then
    for cfg in "${CONFIGS_MPE[@]}";        do run_pilot "$cfg"; done
    if [[ "${SKIP_OVERCOOKED:-0}" != "1" ]]; then
        for cfg in "${CONFIGS_OVERCOOKED[@]}"; do run_pilot "$cfg"; done
    fi
    if [[ "${SKIP_MELTINGPOT:-0}" != "1" ]]; then
        for cfg in "${CONFIGS_MELTINGPOT[@]}"; do run_pilot "$cfg"; done
    fi
    echo_step "PILOT THRESHOLDS PERSISTED"
    find "$PILOT_DIR" -name threshold.txt -exec sh -c 'echo "  $1: $(cat "$1")"' _ {} \;
fi

# ---------- 2. Full sweeps (use pilot thresholds via PILOT_DIR) ----------
for cfg in "${CONFIGS_MPE[@]}";        do run_full "$cfg"; done
if [[ "${SKIP_OVERCOOKED:-0}" != "1" ]]; then
    for cfg in "${CONFIGS_OVERCOOKED[@]}"; do run_full "$cfg"; done
fi
if [[ "${SKIP_MELTINGPOT:-0}" != "1" ]]; then
    for cfg in "${CONFIGS_MELTINGPOT[@]}"; do run_full "$cfg"; done
fi

# ---------- 3. Lambda sweep on one MPE task (plan §5.5) ----------
if [[ "${SKIP_LAMBDA:-0}" != "1" ]]; then
    echo_step "LAMBDA sweep on MPE simple_spread"
    python scripts/lambda_sweep.py configs/mpe_simple_spread.yaml \
        --lambdas 0 0.25 0.5 1.0 1.5 2.0 3.0 5.0 \
        --seeds 12 \
        --output-dir artifacts/runs \
        --max-parallel "$PARALLEL"
fi

# ---------- 4. Figures ----------
if [[ "${SKIP_FIGURES:-0}" != "1" ]]; then
    echo_step "FIGURES"
    python scripts/make_figures.py \
        --runs-root artifacts/runs \
        --out-dir   artifacts/figures \
        --config-dir configs
fi

# ---------- 5. Bundle ----------
echo_step "BUNDLE"
ts="$(date +%Y%m%d_%H%M%S)"
out_zip="meta_mapg_neural_results_${ts}.zip"
( cd artifacts && zip -qr "../${out_zip}" figures runs pilot 2>/dev/null || true )
echo "[run_all] artefacts bundled into ${out_zip}"

echo_step "DONE"
