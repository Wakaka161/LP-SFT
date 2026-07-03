#!/bin/bash
# Llama-3.1-8B training wrapper → scripts/train.sh <dataset> <loss>
#
# Examples:
#   bash scripts/llama3.1/train.sh magicoder ce
#   LP_SFT_MU=1.0 bash scripts/llama3.1/train.sh magicoder lp_sft
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LP_SFT_DIR="$(cd "$DIR/../.." && pwd)"
cd "$LP_SFT_DIR"
# shellcheck disable=SC1091
source ./activate_lp_sft.sh > /dev/null 2>&1

export MODEL_TAG="${MODEL_TAG:-llama3.1-8b}"
export R_BASE_MODEL="${R_BASE_MODEL:-llama3_1_8b_base}"

exec bash "$DIR/../train.sh" "$@"
