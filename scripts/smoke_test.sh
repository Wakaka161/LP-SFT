#!/bin/bash
# End-to-end smoke test on a GPU worker (Qwen3-4B × magicoder, 1k samples).
#
# Prereq: activate a Python env with torch/transformers/deepspeed.
#
#   cd LP-SFT && source activate_lp_sft.sh
#   bash scripts/smoke_test.sh
#
# Optional: LOSS=lp_sft bash scripts/smoke_test.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v nvidia-smi >/dev/null 2>&1 || [ "$(nvidia-smi -L 2>/dev/null | wc -l)" -lt 1 ]; then
    echo "ERROR: no GPU detected — run this script on a GPU worker." >&2
    exit 1
fi

# shellcheck disable=SC1091
source ./activate_lp_sft.sh

LOSS="${LOSS:-ce}"

echo "========== [smoke] 1/4 Prepare model =========="
[ -f "$MODELS_DIR/Qwen3-4B-Base/config.json" ] || {
    echo "ERROR: missing $MODELS_DIR/Qwen3-4B-Base — download or place the base model first." >&2
    exit 1
}

echo "========== [smoke] 2/4 Prepare data (1k lines) =========="
mkdir -p "$DATA_DIR/magicoder"
DATA_FILE="$DATA_DIR/magicoder/magicoder_sft_train_qwen3-4b_tokenized.jsonl"
if [ ! -s "$DATA_FILE" ]; then
    echo "ERROR: missing $DATA_FILE — place a tokenized jsonl slice first." >&2
    exit 1
fi
wc -l "$DATA_FILE"

if [ "$LOSS" = "lp_sft" ]; then
    echo "========== [smoke] 3/4 Precompute R cache =========="
    SMOKE=1 SAVE_TOPK=1 SAVE_REF_LOGITS=1 K_SAVE=10 SET_METHOD=N1 N_GPU=1 \
        bash scripts/qwen3-4b/precompute.sh magicoder
    STEP=4
else
    echo "========== [smoke] 3/4 Skip precompute (LOSS=$LOSS) =========="
    STEP=4
fi

echo "========== [smoke] ${STEP}/4 Train ($LOSS) =========="
SMOKE=1 N_GPU=1 PER_DEVICE_BS=1 EFF_BS=1 REPORT_TO=none \
    bash scripts/qwen3-4b/train.sh magicoder "$LOSS"

echo "========== [smoke] PASSED =========="
