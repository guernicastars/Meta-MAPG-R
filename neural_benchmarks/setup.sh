#!/usr/bin/env bash
# One-shot setup.  After unzip:
#     bash setup.sh
# creates `.venv` via uv, installs all deps, verifies CUDA + GPU count.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# Headless rendering for Overcooked-AI's pygame import on a server.
export SDL_VIDEODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1

# ---------- 1. uv check ----------
if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi
echo "[setup] uv: $(uv --version)"

# ---------- 2. venv ----------
if [[ ! -d .venv ]]; then
    uv venv .venv --python 3.10
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# ---------- 3. core deps ----------
echo "[setup] installing core deps (this can take 2-5 min) ..."
uv pip install -e .

# ---------- 4. optional extras ----------
echo "[setup] installing Overcooked-AI ..."
uv pip install -e ".[overcooked]" || echo "[setup][warn] overcooked install failed; benchmark will be skipped."

echo "[setup] attempting Melting Pot install (heavy native deps) ..."
uv pip install dm-meltingpot 2>/dev/null || {
    echo "[setup][warn] dm-meltingpot install failed (this is common — needs dmlab2d/bazel)."
    echo "[setup][info] We will fall back to PettingZoo prisoners-dilemma social-dilemma proxy."
    uv pip install -e ".[meltingpot]"
}

# ---------- 5. verify ----------
python - <<'PY'
import importlib, importlib.util, sys
print("\n=== verification ===")
for m in ["torch","numpy","pettingzoo","gymnasium","supersuit","matplotlib","seaborn","pandas","yaml","tqdm","rich"]:
    if importlib.util.find_spec(m):
        x = importlib.import_module(m)
        print(f"  {m:20s} {getattr(x,'__version__','?')}")
    else:
        print(f"  {m:20s} MISSING")
        sys.exit(1)

import torch
print(f"\n  torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"  device count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu[{i}]: {p.name}  {p.total_memory/1e9:.1f} GB")

# Optional benchmarks
for m in ["overcooked_ai_py","meltingpot"]:
    print(f"  optional {m:20s} {'OK' if importlib.util.find_spec(m) else 'NOT INSTALLED'}")
PY

echo "[setup] done.  Activate with: source .venv/bin/activate"
