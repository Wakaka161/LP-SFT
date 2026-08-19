#!/bin/bash
# 一键 precompute R cache (模型无关): 推荐用 scripts/<model>/precompute.sh
#
# 用法:
#   source activate_lp_sft.sh
#   bash scripts/qwen3-4b/precompute.sh magicoder
#   SMOKE=1 bash scripts/precompute_R.sh magicoder
#
# 可选环境变量:
#   SMOKE=1                  → 1k sample, 1 卡, ~1 min
#   MAX_SAMPLES=N            → 限制样本数 (默认 numinamath=100K, 其他全量)
#   N_GPU=8                  → 几卡并行 (默认自动检测)
#   TOK_TAG=qwen3-4b         → 读哪个 tokenizer 的 tokenized jsonl (qwen3-4b / llama3.1-8b)
#   DATA_FILE=...            → 直接指定 tokenized jsonl (覆盖 TOK_TAG 推导)
#   BASE_MODEL=qwen3_4b_base → 模型名 (默认), 决定输出文件名
#   MODEL_PATH=...           → 覆盖模型路径 (默认 $MODELS_DIR/Qwen3-4B-Base)
#   TOPK=30                  → top-K 用于近似熵 (默认 30, K0)
#   SAVE_TOPK=1              → 同时保存 ref top-K_save ids 和 k_t (用于 plateau loss)
#   K_SAVE=10                → 保存的 ref top-K 数量, 同时 = |S_t'| 上限 (默认 10)
#   SAVE_REF_LOGITS=1        → 还保存 ref top-K logits 与 label_ref_logit (用于 lp_sft).
#                              要求 SAVE_TOPK=1. 输出文件名后缀变为 _plateau_K${K_SAVE}_reflg.
#   SET_METHOD=N2|N1         → k_t 用哪个 effective size: N2=collision (默认, 与现有 cache 兼容),
#                              N1=Shannon (set 普遍更大). N1 时输出文件名再加 _setN1 后缀.

set -e
set -o pipefail

DATASET="${1:?usage: bash precompute_R.sh <numinamath|magicoder|ultrafeedback>}"

# TOK_TAG: 决定读哪个 tokenizer 的 tokenized jsonl (token id 随 tokenizer 变, cache 必须和训练用同一份).
#   qwen3-4b (默认) / llama3.1-8b / ...
# 必须和 BASE_MODEL / MODEL_PATH 配套: 例如 llama 要 TOK_TAG=llama3.1-8b + MODEL_PATH=$MODELS_DIR/Llama-3.1-8B.
TOK_TAG="${TOK_TAG:-qwen3-4b}"

# ---------- 数据路径 ----------
case "$DATASET" in
    numinamath)       DEFAULT_MAX_SAMPLES=100000 ;;  # NuminaMath-CoT ~860K → 抽 100K
    magicoder)        DEFAULT_MAX_SAMPLES="" ;;       # 全量 75K
    code_mix)         DEFAULT_MAX_SAMPLES="" ;;       # 全量 107K (70% magic + 25% numin + 5% uf)
    all_mix)          DEFAULT_MAX_SAMPLES="" ;;       # ~236K (75K magicoder + 100K numinamath + 61K ultrafeedback)
    opencodeinstruct) DEFAULT_MAX_SAMPLES="" ;;       # 已离线筛成 100K
    ultrafeedback)    DEFAULT_MAX_SAMPLES="" ;;       # 全量 64K
    flan)             DEFAULT_MAX_SAMPLES=100000 ;;   # Plan C 过滤后 ~131K → 训练取 100K
    *)
        echo "ERROR: unknown dataset $DATASET (numinamath|magicoder|code_mix|all_mix|opencodeinstruct|ultrafeedback|flan)" >&2; exit 1 ;;
esac
# DATA_FILE 默认按 TOK_TAG 推导, 可被环境变量覆盖
DATA_FILE="${DATA_FILE:-$DATA_DIR/$DATASET/${DATASET}_sft_train_${TOK_TAG}_tokenized.jsonl}"

# ---------- 检查环境变量 ----------
: "${MODELS_DIR:?ERROR: please 'source activate_lp_sft.sh' first}"
: "${DATA_DIR:?ERROR: please 'source activate_lp_sft.sh' first}"

BASE_MODEL="${BASE_MODEL:-qwen3_4b_base}"
MODEL_PATH="${MODEL_PATH:-$MODELS_DIR/Qwen3-4B-Base}"
TOPK="${TOPK:-30}"
SAVE_TOPK="${SAVE_TOPK:-0}"
K_SAVE="${K_SAVE:-10}"
SAVE_REF_LOGITS="${SAVE_REF_LOGITS:-0}"
if [ "$SAVE_REF_LOGITS" = "1" ] && [ "$SAVE_TOPK" != "1" ]; then
    echo "ERROR: SAVE_REF_LOGITS=1 requires SAVE_TOPK=1 (S_t comes from the same top-K_save)" >&2
    exit 1
fi
SET_METHOD="${SET_METHOD:-N2}"
if [ "$SET_METHOD" != "N2" ] && [ "$SET_METHOD" != "N1" ]; then
    echo "ERROR: SET_METHOD must be N1 or N2, got $SET_METHOD" >&2
    exit 1
fi
# MAX_SEQ_LEN: 截断长度 (默认 2048，与训练 max_seq_length 一致).
# 若显式传入 0 则表示不截断.
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"

[ -f "$MODEL_PATH/config.json" ] || { echo "ERROR: model not found at $MODEL_PATH" >&2; exit 1; }
[ -f "$DATA_FILE" ] || { echo "ERROR: data file not found at $DATA_FILE" >&2; exit 1; }

# ---------- max_samples / smoke ----------
if [ "${SMOKE:-0}" = "1" ]; then
    MAX_SAMPLES="${SMOKE_SAMPLES:-1000}"
    RUN_TAG="smoke"
else
    MAX_SAMPLES="${MAX_SAMPLES:-$DEFAULT_MAX_SAMPLES}"
    RUN_TAG="full"
fi

# ---------- 输出路径 ----------
# 跟数据集放一起, 方便训练时找
OUT_DIR="$DATA_DIR/$DATASET"
mkdir -p "$OUT_DIR"
# plateau cache (with topk) 用 _plateau 后缀, 区分于 plain R cache
# lp_sft cache (with topk + ref_logits) 再加 _reflg 后缀
TAG=""
if [ "$SAVE_TOPK" = "1" ]; then
    TAG="_plateau_K${K_SAVE}"
    if [ "$SAVE_REF_LOGITS" = "1" ]; then
        TAG="${TAG}_reflg"
    fi
    # N2 是默认, 不加后缀以保持现有 cache 文件名兼容; N1 加 _setN1 后缀
    if [ "$SET_METHOD" = "N1" ]; then
        TAG="${TAG}_setN1"
    fi
fi
if [ -n "$MAX_SAMPLES" ]; then
    OUT_FILE="$OUT_DIR/${DATASET}_R_${BASE_MODEL}_n${MAX_SAMPLES}${TAG}.jsonl"
else
    OUT_FILE="$OUT_DIR/${DATASET}_R_${BASE_MODEL}_full${TAG}.jsonl"
fi

# ---------- N_GPU ----------
if [ -z "$N_GPU" ]; then
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        N_GPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
    else
        N_GPU=$(nvidia-smi -L 2>/dev/null | wc -l)
    fi
    [ "$N_GPU" -lt 1 ] && N_GPU=1
fi

# SMOKE 强制单卡 (避免 NCCL 初始化开销)
if [ "${SMOKE:-0}" = "1" ]; then
    N_GPU=1
fi

echo "============================================================"
echo "[precompute_R] dataset      = $DATASET"
echo "[precompute_R] base_model   = $BASE_MODEL ($MODEL_PATH)"
echo "[precompute_R] data         = $DATA_FILE"
echo "[precompute_R] max_samples  = ${MAX_SAMPLES:-(all)}"
echo "[precompute_R] topk (K0)    = $TOPK"
echo "[precompute_R] save_topk        = $SAVE_TOPK"
if [ "$SAVE_TOPK" = "1" ]; then
    echo "[precompute_R] K_save           = $K_SAVE"
    echo "[precompute_R] set_method       = $SET_METHOD  (k_t = clamp(ceil($SET_METHOD), 1, K_save))"
fi
echo "[precompute_R] save_ref_logits  = $SAVE_REF_LOGITS"
echo "[precompute_R] max_seq_len  = $MAX_SEQ_LEN"
echo "[precompute_R] n_gpu        = $N_GPU"
echo "[precompute_R] tag          = $RUN_TAG"
echo "[precompute_R] out          = $OUT_FILE"
echo "============================================================"

# ---------- 组装可选参数 ----------
OPT_ARGS=()
if [ -n "$MAX_SAMPLES" ]; then
    OPT_ARGS+=(--max_samples "$MAX_SAMPLES")
fi
if [ "$SAVE_TOPK" = "1" ]; then
    OPT_ARGS+=(--save_topk --topk_save_ids "$K_SAVE")
fi
if [ "$SAVE_REF_LOGITS" = "1" ]; then
    OPT_ARGS+=(--save_ref_logits)
fi
if [ "$SAVE_TOPK" = "1" ]; then
    OPT_ARGS+=(--set_method "$SET_METHOD")
fi
if [ -n "$MAX_SEQ_LEN" ] && [ "$MAX_SEQ_LEN" != "0" ]; then
    OPT_ARGS+=(--max_seq_len "$MAX_SEQ_LEN")
fi

# ---------- 启动 ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOSS_DIR="$(cd "$SCRIPT_DIR/../losses" && pwd)"
PRECOMPUTE_PY="$LOSS_DIR/precompute_R.py"
PY="${PYTHON:-python}"
echo "[precompute_R] python       = $(command -v "$PY" || echo "$PY")"
if [ "$N_GPU" -gt 1 ]; then
    # master_port: 跟 train_qwen3_4b.sh 对齐 — 用 [20000,29999] (IANA registered, 避开 ephemeral)
    # 并带 EADDRINUSE 重试. 49152+ 段会撞 Linux ip_local_port_range → 秒退 EADDRINUSE.
    USER_FIXED_PORT=0
    if [ -n "${MASTER_PORT:-}" ]; then
        USER_FIXED_PORT=1
    fi
    MAX_LAUNCH_RETRY="${MAX_LAUNCH_RETRY:-3}"
    DS_RC=0
    LAST_TRY=0
    for launch_try in $(seq 1 "$MAX_LAUNCH_RETRY"); do
        LAST_TRY=$launch_try
        if [ "$USER_FIXED_PORT" = "1" ]; then
            if ss -tln 2>/dev/null | awk '{print $4}' | grep -q ":$MASTER_PORT\$"; then
                echo "[precompute_R] ERROR: MASTER_PORT=$MASTER_PORT 已被占用." >&2
                ss -tlnp 2>/dev/null | grep ":$MASTER_PORT" >&2
                exit 1
            fi
        else
            PORT_OK=0
            for _ptry in 1 2 3 4 5; do
                CAND=$(( 20000 + RANDOM % 10000 ))
                if ! ss -tln 2>/dev/null | awk '{print $4}' | grep -q ":$CAND\$"; then
                    MASTER_PORT=$CAND
                    PORT_OK=1
                    break
                fi
            done
            [ "$PORT_OK" = "1" ] || { echo "[precompute_R] ERROR: 5 次随机选 master_port 均被占用." >&2; exit 1; }
        fi
        export MASTER_PORT
        export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
        echo "[precompute_R] launching torchrun ($N_GPU GPUs, port=$MASTER_PORT, attempt $launch_try / $MAX_LAUNCH_RETRY)"
        set +e
        set +o pipefail
        "$PY" -m torch.distributed.run \
            --nproc_per_node="$N_GPU" --master_port "$MASTER_PORT" \
            "$PRECOMPUTE_PY" \
            --model_path "$MODEL_PATH" \
            --data_file  "$DATA_FILE" \
            --out_file   "$OUT_FILE" \
            --topk       "$TOPK" \
            "${OPT_ARGS[@]}" \
            2>&1 | tee "${OUT_FILE}.log"
        DS_RC=${PIPESTATUS[0]}
        set -e
        set -o pipefail
        if [ "$DS_RC" = "0" ]; then
            break
        fi
        if [ "$USER_FIXED_PORT" = "1" ]; then
            echo "[precompute_R] torchrun rc=$DS_RC (用户固定 MASTER_PORT, 不自动重试)." >&2
            break
        fi
        if tail -n 300 "${OUT_FILE}.log" 2>/dev/null \
                | grep -qiE "address already in use|EADDRINUSE|address in use"; then
            echo "[precompute_R] WARN: attempt $launch_try hit EADDRINUSE on $MASTER_PORT, 换 port 重试." >&2
            unset MASTER_PORT
            continue
        fi
        echo "[precompute_R] torchrun rc=$DS_RC 但 log 无 EADDRINUSE, 视为真实失败." >&2
        break
    done
    if [ "$DS_RC" != "0" ]; then
        echo "[precompute_R] FAILED rc=$DS_RC after $LAST_TRY attempt(s)" >&2
        exit "$DS_RC"
    fi
else
    echo "[precompute_R] launching single-GPU"
    "$PY" "$PRECOMPUTE_PY" \
        --model_path "$MODEL_PATH" \
        --data_file  "$DATA_FILE" \
        --out_file   "$OUT_FILE" \
        --topk       "$TOPK" \
        "${OPT_ARGS[@]}" \
        2>&1 | tee "${OUT_FILE}.log"
fi

echo "============================================================"
echo "[precompute_R] DONE -> $OUT_FILE"
echo "============================================================"
