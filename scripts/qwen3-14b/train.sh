#!/bin/bash
# Qwen3-14B training wrapper → scripts/train.sh <dataset> <loss>
#
# Examples:
#   bash scripts/qwen3-14b/train.sh magicoder ce
#   LP_SFT_MU=1.0 bash scripts/qwen3-14b/train.sh magicoder lp_sft
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LP_SFT_DIR="$(cd "$DIR/../.." && pwd)"
cd "$LP_SFT_DIR"
# shellcheck disable=SC1091
source ./activate_lp_sft.sh > /dev/null 2>&1

export MODEL_TAG="${MODEL_TAG:-qwen3-14b}"
export R_BASE_MODEL="${R_BASE_MODEL:-qwen3_14b_base}"

exec bash "$DIR/../train.sh" "$@"
