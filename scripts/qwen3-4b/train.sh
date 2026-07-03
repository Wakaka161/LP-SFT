#!/bin/bash
# Qwen3-4B training wrapper → scripts/train.sh <dataset> <loss>
#
# Examples:
#   bash scripts/qwen3-4b/train.sh magicoder ce
#   bash scripts/qwen3-4b/train.sh magicoder asft
#   bash scripts/qwen3-4b/train.sh numinamath lp_sft
#   LP_SFT_MU=1.0 LP_SFT_R_WEIGHT=R \
#       bash scripts/qwen3-4b/train.sh magicoder lp_sft
#   SMOKE=1 bash scripts/qwen3-4b/train.sh magicoder ce
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LP_SFT_DIR="$(cd "$DIR/../.." && pwd)"
cd "$LP_SFT_DIR"
# shellcheck disable=SC1091
source ./activate_lp_sft.sh > /dev/null 2>&1

export MODEL_TAG="${MODEL_TAG:-qwen3-4b}"
export R_BASE_MODEL="${R_BASE_MODEL:-qwen3_4b_base}"

exec bash "$DIR/../train.sh" "$@"
