#!/usr/bin/env bash
# activate_lp_sft.sh — set LP-SFT path variables.
#
# Usage:
#   conda create -n lpsft python=3.10 -y
#   conda activate lpsft          # or: python -m venv .venv && source .venv/bin/activate
#   pip install -r requirements.txt
#   source ./activate_lp_sft.sh
#
# Optional overrides (export before sourcing):
#   MODELS_DIR  — base models (default: $LP_SFT_ROOT/models)
#   DATA_DIR    — tokenized data + R cache (default: $LP_SFT_ROOT/data)
#   CKPT_DIR    — training outputs (default: $LP_SFT_ROOT/ckpts)

(return 0 2>/dev/null) || {
    echo "[activate_lp_sft] ERROR: run with: source $0"
    exit 1
}

if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
    _LP_SFT_SCRIPT="${BASH_SOURCE[0]}"
else
    _LP_SFT_SCRIPT="$0"
fi
export LP_SFT_ROOT="$(cd "$(dirname "$_LP_SFT_SCRIPT")" && pwd)"
export LP_SFT_DIR="$LP_SFT_ROOT"

export PYTHONPATH="$LP_SFT_ROOT:${PYTHONPATH:-}"

export MODELS_DIR="${MODELS_DIR:-$LP_SFT_ROOT/models}"
export DATA_DIR="${DATA_DIR:-$LP_SFT_ROOT/data}"
export CKPT_DIR="${CKPT_DIR:-$LP_SFT_ROOT/ckpts}"

mkdir -p "$MODELS_DIR" "$DATA_DIR" "$CKPT_DIR"

if ! command -v python >/dev/null 2>&1; then
    echo "[activate_lp_sft] ERROR: python not found — activate your venv/conda first" >&2
    return 1 2>/dev/null || exit 1
fi

echo "[activate_lp_sft] LP_SFT_ROOT   = $LP_SFT_ROOT"
echo "[activate_lp_sft] python     = $(command -v python)"
echo "[activate_lp_sft] MODELS_DIR = $MODELS_DIR"
echo "[activate_lp_sft] DATA_DIR   = $DATA_DIR"
echo "[activate_lp_sft] CKPT_DIR   = $CKPT_DIR"

python - <<'PY' || { echo "[activate_lp_sft] ERROR: install deps with: pip install -r requirements.txt" >&2; return 1 2>/dev/null || exit 1; }
import importlib
missing = []
for pkg in ("torch", "transformers", "datasets", "deepspeed"):
    try:
        importlib.import_module(pkg)
    except Exception:
        missing.append(pkg)
if missing:
    raise SystemExit(f"missing packages: {missing}")
import torch, transformers
print(f"[activate_lp_sft] OK: torch={torch.__version__} transformers={transformers.__version__}")
PY

echo "[activate_lp_sft] DONE."
