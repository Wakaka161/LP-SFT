#!/bin/bash
# Smoke-test training losses on a GPU worker (Qwen3-4B × magicoder).
#
#   cd LP-SFT && source activate_lp_sft.sh
#   bash scripts/smoke_test_all.sh
#
# Env:
#   SMOKE_SAMPLES=200   — samples per run (default 200)
#   LOSSES="ce gem"     — subset (default: ce dft eaft gem lp_sft; asft excluded — needs ~2× VRAM)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v nvidia-smi >/dev/null 2>&1 || [ "$(nvidia-smi -L 2>/dev/null | wc -l)" -lt 1 ]; then
    echo "ERROR: no GPU detected — run on a GPU worker." >&2
    exit 1
fi

# shellcheck disable=SC1091
source ./activate_lp_sft.sh

SMOKE_SAMPLES="${SMOKE_SAMPLES:-200}"
LOSSES="${LOSSES:-ce dft eaft gem lp_sft}"

echo "========== [smoke_all] prepare model =========="
[ -f "$MODELS_DIR/Qwen3-4B-Base/config.json" ] || {
    echo "ERROR: missing $MODELS_DIR/Qwen3-4B-Base — download or place the base model first." >&2
    exit 1
}

echo "========== [smoke_all] prepare data (${SMOKE_SAMPLES} lines) =========="
mkdir -p "$DATA_DIR/magicoder"
DATA_FILE="$DATA_DIR/magicoder/magicoder_sft_train_qwen3-4b_tokenized.jsonl"
if [ ! -s "$DATA_FILE" ] || [ "$(wc -l < "$DATA_FILE")" -lt "$SMOKE_SAMPLES" ]; then
    echo "ERROR: need at least ${SMOKE_SAMPLES} lines in $DATA_FILE" >&2
    exit 1
fi
wc -l "$DATA_FILE"

CACHE_FILE="$DATA_DIR/magicoder/magicoder_R_qwen3_4b_base_n${SMOKE_SAMPLES}_plateau_K10_reflg_setN1.jsonl"
if [[ " $LOSSES " == *" lp_sft "* ]] && [ ! -f "$CACHE_FILE" ]; then
    if [ -f "$DATA_DIR/magicoder/magicoder_R_qwen3_4b_base_n1000_plateau_K10_reflg_setN1.jsonl" ]; then
        echo "[smoke_all] reusing existing 1k lp_sft cache"
    else
        echo "========== [smoke_all] precompute lp_sft cache =========="
        SMOKE=1 SMOKE_SAMPLES="$SMOKE_SAMPLES" SAVE_TOPK=1 SAVE_REF_LOGITS=1 K_SAVE=10 SET_METHOD=N1 N_GPU=1 \
            bash scripts/qwen3-4b/precompute.sh magicoder
    fi
fi

PASSED=()
FAILED=()
for loss in $LOSSES; do
    echo ""
    echo "========== [smoke_all] train loss=$loss =========="
    if SMOKE=1 SMOKE_SAMPLES="$SMOKE_SAMPLES" N_GPU=1 PER_DEVICE_BS=1 EFF_BS=1 REPORT_TO=none \
        bash scripts/qwen3-4b/train.sh magicoder "$loss"; then
        PASSED+=("$loss")
        echo "[smoke_all] OK: $loss"
    else
        FAILED+=("$loss")
        echo "[smoke_all] FAIL: $loss" >&2
    fi
done

echo ""
echo "========== [smoke_all] summary =========="
echo "  passed: ${PASSED[*]:-(none)}"
echo "  failed: ${FAILED[*]:-(none)}"
[ "${#FAILED[@]}" -eq 0 ]
