#!/bin/bash
# Qwen3-4B precompute wrapper → scripts/precompute_R.sh <dataset>
#
# Examples:
#   bash scripts/qwen3-4b/precompute.sh magicoder
#   SAVE_TOPK=1 SAVE_REF_LOGITS=1 K_SAVE=10 SET_METHOD=N1 \
#       bash scripts/qwen3-4b/precompute.sh magicoder
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LP_SFT_DIR="$(cd "$DIR/../.." && pwd)"
cd "$LP_SFT_DIR"
# shellcheck disable=SC1091
source ./activate_lp_sft.sh > /dev/null 2>&1

export TOK_TAG="${TOK_TAG:-qwen3-4b}"
export BASE_MODEL="${BASE_MODEL:-qwen3_4b_base}"
export MODEL_PATH="${MODEL_PATH:-$MODELS_DIR/Qwen3-4B-Base}"

exec bash "$DIR/../precompute_R.sh" "$@"
