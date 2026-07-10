#!/bin/bash
# 统一 SFT 训练脚本 (模型无关): 推荐用 scripts/<model>/train.sh 设置 MODEL_TAG
# 数据集: {numinamath, magicoder, opencodeinstruct, sciinstruct, ultrafeedback, ...}
# 方法:   {ce, dft, eaft, gem, asft, lp_sft}
# 用法:
#   source activate_lp_sft.sh
#   bash scripts/qwen3-4b/train.sh magicoder ce
#   bash scripts/qwen3-4b/train.sh magicoder lp_sft
#   SMOKE=1 bash scripts/train.sh magicoder ce   # 也可直接调用本脚本
#
# 可选环境变量:
#   SMOKE=1                       → 1k 样本 + 1 epoch + 频繁 log/save, 用于 pipeline 验证
#   MAX_TRAIN_SAMPLES=100000      → 限制训练样本数 (默认: numinamath=100k, 其他全量)
#   NUM_EPOCHS=3                  → 训练 epoch 数 (默认 3)
#   PER_DEVICE_BS=4               → 每卡 batch (默认 4)
#   EFF_BS=128                    → 目标有效 batch (默认 128, 自动算 grad_acc)
#   N_GPU=8                       → GPU 数, 默认自动检测
#   GEM_BETA=0.7                  → GEM 超参 (仅 loss=gem 用)
#   PLATEAU_K_SAVE=10             → lp_sft cache |S_t'| 上限 (默认 10, 必须等于 cache K_save)
#   LP_SFT_MU=0.03           → lp_sft 第二项权重 (默认 0.03, 推荐扫 {0.01,0.03,0.05,0.1})
#   LP_SFT_TAU=1.0           → lp_sft reference 温度 (默认 1.0, 第一阶段固定; 后续可扫 {1.0,1.5,2.0})
#   LP_SFT_R_WEIGHT=none|R|R2|R3 → lp_sft 第二项权重: mu, mu*R_t, mu*R_t^2, 或 mu*R_t^3 (默认 none)
#   LP_SFT_SET_METHOD=N2|N1        → 训练时从 cache 读 k_n2 (默认) 或 k_n1
#                                   precompute 请用 SET_METHOD=N1 (文件名带 _setN1)
#   LP_SFT_K_ROUND_MODE=precomputed|round|ceil|floor
#                                 → 如何从 cache 导出 k_t:
#                                   precomputed (默认) 直接用 cache 里存的 k_n1/k_n2/k;
#                                   round/ceil/floor  从 cache 里的 n1_vals/n2_vals 原生值重新计算,
#                                   无需重跑 precompute_R.
#   LP_SFT_K_THRESHOLD=1.2         → 阈值 T (>1.0 时启用): n<T → k=1, else k=ceil(n).
#                                   覆盖 LP_SFT_K_ROUND_MODE. 推荐扫 {1.1, 1.2, 1.3}.
#   LP_SFT_MODE=additive|r_interp → lp_sft combination mode
#                                         additive: CE + mu*L_set (default)
#                                         r_interp: (1-R)*CE + R*L_set (convex interp, ignores mu/r_weight)
#   R_CACHE_PATH=...              → cache 路径 (覆盖默认自动推导)
#   ASFT_KL_WEIGHT=0.05           → ASFT KL anchor λ (default 0.05)
#   ZERO_STAGE=2|3                → DeepSpeed ZeRO (asft auto-selects 3)
#   R_BASE_MODEL=qwen3_4b_base    → cache 是用哪个 base 模型算的 (默认从 MODEL_TAG 推导)

set -e
set -x
set -o pipefail   # 让 `cmd | tee log` 中 cmd 的非零退出码能被 set -e 捕获 (否则会 silent pass)

# ---------- 参数解析 ----------
DATASET="${1:?usage: bash scripts/train.sh <numinamath|magicoder|opencodeinstruct|sciinstruct|ultrafeedback> <ce|gem|...>}"
LOSS="${2:?usage: bash scripts/train.sh <dataset> <ce|gem|lp_sft|eaft|...>}"

# TOK_FILE_TAG 决定 tokenized 数据文件名中的模型标识部分.
# Qwen3 系列共用同一 tokenizer (4B/14B tokenized 文件通用), Llama 使用独立的.
TOK_FILE_TAG="${TOK_FILE_TAG:-${MODEL_TAG:-qwen3-4b}}"
# Qwen3-14B 复用 Qwen3-4B 的 tokenized 数据 (tokenizer 完全一致)
if [ "$TOK_FILE_TAG" = "qwen3-14b" ]; then
    TOK_FILE_TAG="qwen3-4b"
fi

case "$DATASET" in
    numinamath)
        TRAIN_FILE="$DATA_DIR/numinamath/numinamath_sft_train_${TOK_FILE_TAG}_tokenized.jsonl"
        TEST_FILE=""
        DEFAULT_MAX_SAMPLES=100000   # NuminaMath-CoT ~860K → 抽 100K
        ;;
    magicoder)
        TRAIN_FILE="$DATA_DIR/magicoder/magicoder_sft_train_${TOK_FILE_TAG}_tokenized.jsonl"
        TEST_FILE=""
        DEFAULT_MAX_SAMPLES=""       # 75K 全量
        ;;
    opencodeinstruct)
        TRAIN_FILE="$DATA_DIR/opencodeinstruct/opencodeinstruct_sft_train_${TOK_FILE_TAG}_tokenized.jsonl"
        TEST_FILE=""
        DEFAULT_MAX_SAMPLES=""       # 已经离线筛成 100K
        ;;
    sciinstruct)
        TRAIN_FILE="$DATA_DIR/sciinstruct/sciinstruct_sft_train_${TOK_FILE_TAG}_tokenized.jsonl"
        TEST_FILE=""
        DEFAULT_MAX_SAMPLES=""       # HF mirror 当前可用 91,750 条
        ;;
    ultrafeedback)
        TRAIN_FILE="$DATA_DIR/ultrafeedback/ultrafeedback_sft_train_${TOK_FILE_TAG}_tokenized.jsonl"
        TEST_FILE="$DATA_DIR/ultrafeedback/ultrafeedback_sft_test_${TOK_FILE_TAG}_tokenized.jsonl"
        DEFAULT_MAX_SAMPLES=""       # 64K 全量
        ;;
    flan)
        TRAIN_FILE="$DATA_DIR/flan/flan_sft_train_${TOK_FILE_TAG}_tokenized.jsonl"
        TEST_FILE=""
        DEFAULT_MAX_SAMPLES=100000   # Plan C 过滤后 ~650K，默认取 100K 训练
        ;;
    code_mix)
        TRAIN_FILE="$DATA_DIR/code_mix/code_mix_sft_train_${TOK_FILE_TAG}_tokenized.jsonl"
        TEST_FILE=""
        DEFAULT_MAX_SAMPLES=""       # 107K 全量 (70% magic + 25% numin + 5% uf)
        ;;
    all_mix)
        TRAIN_FILE="$DATA_DIR/all_mix/all_mix_sft_train_${TOK_FILE_TAG}_tokenized.jsonl"
        TEST_FILE=""
        DEFAULT_MAX_SAMPLES=""       # ~236K 全量 (75K magicoder + 100K numinamath + 61K ultrafeedback)
        ;;
    *)
        echo "ERROR: unknown dataset $DATASET (must be numinamath|magicoder|code_mix|all_mix|opencodeinstruct|sciinstruct|ultrafeedback|flan)" >&2
        exit 1 ;;
esac

case "$LOSS" in
    ce|dft|eaft|gem|asft|lp_sft) ;;
    *) echo "ERROR: unknown loss $LOSS (must be ce|dft|eaft|gem|asft|lp_sft)" >&2; exit 1 ;;
esac

# lp_sft 需要含 ref logits 的 precompute cache (SAVE_REF_LOGITS=1).
USE_LP_SFT_CACHE=0
if [ "$LOSS" = "lp_sft" ]; then
    USE_LP_SFT_CACHE=1
fi

# ---------- 环境检查 ----------
export FLASH_ATTENTION_DETERMINISTIC="${FLASH_ATTENTION_DETERMINISTIC:-1}"

: "${MODELS_DIR:?ERROR: please 'source activate_lp_sft.sh' first}"
: "${DATA_DIR:?ERROR: please 'source activate_lp_sft.sh' first}"
: "${CKPT_DIR:?ERROR: please 'source activate_lp_sft.sh' first}"

MODEL_TAG="${MODEL_TAG:-qwen3-4b}"
case "$MODEL_TAG" in
    qwen3-4b)    MODEL_NAME_OR_PATH="$MODELS_DIR/Qwen3-4B-Base" ;;
    qwen3-14b)   MODEL_NAME_OR_PATH="$MODELS_DIR/Qwen3-14B-Base" ;;
    llama3.1-8b) MODEL_NAME_OR_PATH="$MODELS_DIR/Llama-3.1-8B" ;;
    *) MODEL_NAME_OR_PATH="$MODELS_DIR/$MODEL_TAG" ;;
esac
# Warm-start override: MODEL_NAME_OR_PATH_OVERRIDE 可指向任意 checkpoint 目录 (如 1ep SFT ckpt).
# RUN_WARMTAG: 非空时附加到 RUN_NAME, 用于区分 warm/cold start. 例: "warm1ep".
if [ -n "${MODEL_NAME_OR_PATH_OVERRIDE:-}" ]; then
    echo "[train] MODEL_NAME_OR_PATH_OVERRIDE=$MODEL_NAME_OR_PATH_OVERRIDE  (overrides default $MODEL_NAME_OR_PATH)"
    MODEL_NAME_OR_PATH="$MODEL_NAME_OR_PATH_OVERRIDE"
fi

USE_FLASH_ATTN="${USE_FLASH_ATTN:-true}"
[ -f "$MODEL_NAME_OR_PATH/config.json" ] || { echo "ERROR: model not found at $MODEL_NAME_OR_PATH" >&2; exit 1; }
[ -f "$TRAIN_FILE" ] || { echo "ERROR: train file not found at $TRAIN_FILE" >&2; exit 1; }

# ASFT: frozen base model for live forward KL anchor (no precompute cache).
if [ -z "${ASFT_REF_MODEL_PATH:-}" ]; then
    case "$MODEL_TAG" in
        qwen3-4b)    ASFT_REF_MODEL_PATH="$MODELS_DIR/Qwen3-4B-Base" ;;
        qwen3-14b)   ASFT_REF_MODEL_PATH="$MODELS_DIR/Qwen3-14B-Base" ;;
        llama3.1-8b) ASFT_REF_MODEL_PATH="$MODELS_DIR/Llama-3.1-8B" ;;
    esac
fi
if [ "$LOSS" = "asft" ]; then
    [ -f "$ASFT_REF_MODEL_PATH/config.json" ] || {
        echo "ERROR: ASFT ref model not found at $ASFT_REF_MODEL_PATH" >&2
        exit 1
    }
fi
ASFT_REF_TOPK="${ASFT_REF_TOPK:-0}"
case "$ASFT_REF_TOPK" in
    ''|*[!0-9]*) echo "ERROR: ASFT_REF_TOPK must be a non-negative int, got '$ASFT_REF_TOPK'" >&2; exit 1 ;;
esac

# ---------- 自动算 GPU 数和 grad_acc ----------
if [ -z "$N_GPU" ]; then
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        N_GPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
    else
        N_GPU=$(nvidia-smi -L | wc -l)
    fi
fi

PER_DEVICE_BS="${PER_DEVICE_BS:-4}"
EFF_BS="${EFF_BS:-128}"
GRAD_ACC=$(( EFF_BS / (PER_DEVICE_BS * N_GPU) ))
[ "$GRAD_ACC" -lt 1 ] && GRAD_ACC=1
ACTUAL_EFF_BS=$(( PER_DEVICE_BS * N_GPU * GRAD_ACC ))

# ---------- smoke / 全量切换 ----------
if [ "${SMOKE:-0}" = "1" ]; then
    MAX_SAMPLES="${SMOKE_SAMPLES:-1000}"
    NUM_EPOCHS=1
    LOGGING_STEPS=1
    SAVE_STRATEGY="no"
    EVAL_STRATEGY="no"
    RUN_TAG="smoke"
else
    MAX_SAMPLES="${MAX_TRAIN_SAMPLES:-$DEFAULT_MAX_SAMPLES}"
    NUM_EPOCHS="${NUM_EPOCHS:-3}"
    LOGGING_STEPS=10
    # 默认只存最终 ckpt (train.py 末尾的 trainer.save_model() 无条件存一次), 不存中间
    # epoch ckpt, 节省 ~2x 磁盘. 如需保留中间 epoch ckpt: SAVE_STRATEGY=epoch bash ...
    SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
    EVAL_STRATEGY="no"
    RUN_TAG="full"
fi

# ---------- GEM 超参 ----------
GEM_BETA="${GEM_BETA:-0.7}"

# ---------- ASFT 超参 (Zhu et al. ICLR 2026) ----------
ASFT_KL_WEIGHT="${ASFT_KL_WEIGHT:-0.05}"

# ---------- lp_sft cache K_save ----------
PLATEAU_K_SAVE="${PLATEAU_K_SAVE:-10}"

# ---------- LP-SFT loss CE 超参 ----------
# mu:  weight on H(q_ref^S, p_theta^S) (CE + mu * set_loss). Default 0.03; sweep {0.01,0.03,0.05,0.1}.
# tau: temperature applied to ref logits before in-set softmax. Default 1.0; later sweep {1.0,1.5,2.0}.
LP_SFT_MU="${LP_SFT_MU:-0.03}"
LP_SFT_TAU="${LP_SFT_TAU:-1.0}"
LP_SFT_R_WEIGHT="${LP_SFT_R_WEIGHT:-none}"
case "$(echo "$LP_SFT_R_WEIGHT" | tr '[:upper:]' '[:lower:]')" in
    none) LP_SFT_R_WEIGHT="none" ;;
    r)    LP_SFT_R_WEIGHT="R" ;;
    r2)   LP_SFT_R_WEIGHT="R2" ;;
    r3)   LP_SFT_R_WEIGHT="R3" ;;
    *) echo "ERROR: LP_SFT_R_WEIGHT must be none, R, R2, or R3, got $LP_SFT_R_WEIGHT" >&2; exit 1 ;;
esac
LP_SFT_SET_METHOD="${LP_SFT_SET_METHOD:-N2}"
if [ "$LP_SFT_SET_METHOD" != "N2" ] && [ "$LP_SFT_SET_METHOD" != "N1" ]; then
    echo "ERROR: LP_SFT_SET_METHOD must be N1 or N2, got $LP_SFT_SET_METHOD" >&2
    exit 1
fi
LP_SFT_K_ROUND_MODE="${LP_SFT_K_ROUND_MODE:-precomputed}"
case "$LP_SFT_K_ROUND_MODE" in
    precomputed|round|ceil|floor) ;;
    *) echo "ERROR: LP_SFT_K_ROUND_MODE must be precomputed|round|ceil|floor, got $LP_SFT_K_ROUND_MODE" >&2; exit 1 ;;
esac
LP_SFT_K_THRESHOLD="${LP_SFT_K_THRESHOLD:-0}"
LP_SFT_MODE="${LP_SFT_MODE:-additive}"
case "$LP_SFT_MODE" in
    additive|r_interp) ;;
    *) echo "ERROR: LP_SFT_MODE must be additive or r_interp, got $LP_SFT_MODE" >&2; exit 1 ;;
esac

# ---------- EAFT baseline 超参 ----------
# k:     top-K for entropy approx (上游硬编码 20). 无需 cache / ref model.
EAFT_ALPHA="${EAFT_ALPHA:-1.0}"
EAFT_K="${EAFT_K:-20}"

# R_BASE_MODEL: 自动从 MODEL_TAG 推导默认值 (若未显式设置)
if [ -z "${R_BASE_MODEL:-}" ]; then
    case "$MODEL_TAG" in
        qwen3-4b)    R_BASE_MODEL="qwen3_4b_base" ;;
        qwen3-14b)   R_BASE_MODEL="qwen3_14b_base" ;;
        llama3.1-8b) R_BASE_MODEL="llama3_1_8b_base" ;;
        *) R_BASE_MODEL="${MODEL_TAG//-/_}_base" ;;
    esac
fi
if [ "$USE_LP_SFT_CACHE" = "1" ]; then
    if [ -z "$R_CACHE_PATH" ]; then
        # Precompute with SET_METHOD=N1 writes *_setN1.jsonl (contains both k_n1 and k_n2).
        TAG="_plateau_K${PLATEAU_K_SAVE}_reflg_setN1"
        if [ -n "$MAX_SAMPLES" ]; then
            CANDS=("$DATA_DIR/$DATASET/${DATASET}_R_${R_BASE_MODEL}_n${MAX_SAMPLES}${TAG}.jsonl")
        elif [ -n "$MAX_TRAIN_SAMPLES" ]; then
            CANDS=("$DATA_DIR/$DATASET/${DATASET}_R_${R_BASE_MODEL}_n${MAX_TRAIN_SAMPLES}${TAG}.jsonl")
        elif [ -n "$DEFAULT_MAX_SAMPLES" ]; then
            CANDS=("$DATA_DIR/$DATASET/${DATASET}_R_${R_BASE_MODEL}_n${DEFAULT_MAX_SAMPLES}${TAG}.jsonl")
        else
            CANDS=("$DATA_DIR/$DATASET/${DATASET}_R_${R_BASE_MODEL}_full${TAG}.jsonl")
        fi
        if [ ! -f "${CANDS[0]}" ]; then
            FB=$(ls -t "$DATA_DIR/$DATASET/${DATASET}_R_${R_BASE_MODEL}"_*"_plateau_K${PLATEAU_K_SAVE}_reflg"*.jsonl 2>/dev/null | head -1)
            if [ -n "$FB" ]; then
                R_CACHE_PATH="$FB"
                echo "[train] exact lp_sft cache not found; falling back to: $R_CACHE_PATH"
            fi
        else
            R_CACHE_PATH="${CANDS[0]}"
        fi
    fi
    [ -f "$R_CACHE_PATH" ] || {
        echo "ERROR: --loss $LOSS 需要 ref-align cache, 但找不到. 试过: ${CANDS[*]}" >&2
        echo "       请先跑 'SAVE_TOPK=1 SAVE_REF_LOGITS=1 K_SAVE=$PLATEAU_K_SAVE bash scripts/precompute_R.sh $DATASET' 生成." >&2
        exit 1
    }
fi

# ---------- 输出目录 ----------
SEED=1234
TIME_STEP=$(date "+%Y-%m-%d-%H-%M-%S")
# loss-specific tag (区分同 loss 不同超参的 ablation)
LOSS_TAG=""
if [ "$LOSS" = "gem" ]; then
    LOSS_TAG="_beta${GEM_BETA}"
elif [ "$LOSS" = "lp_sft" ]; then
    LOSS_TAG="_mu${LP_SFT_MU}_tau${LP_SFT_TAU}"
    if [ "$LP_SFT_R_WEIGHT" != "none" ]; then
        LOSS_TAG="${LOSS_TAG}_${LP_SFT_R_WEIGHT}"
    fi
    if [ "$LP_SFT_MODE" = "r_interp" ]; then
        LOSS_TAG="${LOSS_TAG}_rinterp"
    fi
    if [ "$LP_SFT_SET_METHOD" = "N1" ]; then
        LOSS_TAG="${LOSS_TAG}_setN1"
    fi
    if [ -n "$LP_SFT_K_THRESHOLD" ] && [ "$LP_SFT_K_THRESHOLD" != "0" ]; then
        _thr_gt=$(python3 -c "print(1 if float('$LP_SFT_K_THRESHOLD')>1.0 else 0)" 2>/dev/null || echo "$(echo "$LP_SFT_K_THRESHOLD > 1.0" | bc -l 2>/dev/null || echo 0)")
        if [ "$_thr_gt" = "1" ]; then
            LOSS_TAG="${LOSS_TAG}_thr${LP_SFT_K_THRESHOLD}"
        fi
    fi
elif [ "$LOSS" = "asft" ]; then
    LOSS_TAG="_kl${ASFT_KL_WEIGHT}"
    if [ "$ASFT_REF_TOPK" != "0" ]; then
        LOSS_TAG="${LOSS_TAG}_reftopk${ASFT_REF_TOPK}"
    fi
elif [ "$LOSS" = "eaft" ]; then
    LOSS_TAG="_a${EAFT_ALPHA}"
    if [ "$EAFT_K" != "20" ]; then
        LOSS_TAG="${LOSS_TAG}_k${EAFT_K}"
    fi
fi
# dft: no extra hyperparams, LOSS_TAG stays empty
RUN_NAME="sft_${LOSS}${LOSS_TAG}-${MODEL_TAG}-${DATASET}-${RUN_TAG}${RUN_WARMTAG:+-${RUN_WARMTAG}}-${TIME_STEP}-s${SEED}${RUN_NAME_SUFFIX:-}"
OUTPUT_DIR="$CKPT_DIR/$MODEL_TAG/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "[train] dataset       = $DATASET ($TRAIN_FILE)"
echo "[train] loss          = $LOSS"
echo "[train] model         = $MODEL_NAME_OR_PATH"
echo "[train] n_gpu         = $N_GPU"
echo "[train] per_device_bs = $PER_DEVICE_BS  grad_acc = $GRAD_ACC  eff_bs = $ACTUAL_EFF_BS"
echo "[train] max_samples   = ${MAX_SAMPLES:-(all)}"
echo "[train] num_epochs    = $NUM_EPOCHS"
echo "[train] save_strategy = $SAVE_STRATEGY  smoke=${SMOKE:-0}"
echo "[train] output_dir    = $OUTPUT_DIR"
if [ "$LOSS" = "gem" ]; then
    echo "[train] gem_beta      = $GEM_BETA"
fi
if [ "$LOSS" = "asft" ]; then
    echo "[train] asft_kl_weight = $ASFT_KL_WEIGHT"
    echo "[train] asft_ref_model = $ASFT_REF_MODEL_PATH"
    echo "[train] asft_ref_topk  = $ASFT_REF_TOPK (0 = full vocab)"
    echo "[train] note           = L = DFT + lambda * KL(pi_ref || pi_theta); live ref forward, no cache."
fi
if [ "$LOSS" = "lp_sft" ]; then
    echo "[train] lp_sft_mu      = $LP_SFT_MU  (weight on H(q_ref^S, p_theta^S))"
    echo "[train] lp_sft_tau     = $LP_SFT_TAU (ref temperature for q_ref^S)"
    echo "[train] lp_sft_r_weight= $LP_SFT_R_WEIGHT (none: mu; R: mu*R_t; R2: mu*R_t^2; R3: mu*R_t^3)"
    echo "[train] lp_sft_mode     = $LP_SFT_MODE (additive: CE+mu*L_set; r_interp: (1-R)*CE+R*L_set)"
    echo "[train] lp_sft_set        = S_t \\ {y_t}; |S'|==1 → add top-1 ref alt from cache"
    echo "[train] lp_sft_set_method    = $LP_SFT_SET_METHOD (k_t = clamp($LP_SFT_K_ROUND_MODE($LP_SFT_SET_METHOD), 1, K))"
    echo "[train] lp_sft_k_round_mode  = $LP_SFT_K_ROUND_MODE"
    echo "[train] lp_sft_k_threshold   = $LP_SFT_K_THRESHOLD  (>1.0: n<T→k=1, else ceil)"
    echo "[train] plateau_K_save      = $PLATEAU_K_SAVE  (cache size, |S_t| upper bound)"
    echo "[train] cache               = $R_CACHE_PATH"
    echo "[train] note                = L = CE(y_t) + mu * H(q_ref^S, p_theta^S) on S_t' = S_t \\ {y_t}."
    echo "                              q_ref^S is the ref model's softmax restricted to S_t' (with tau)."
fi
if [ "$LOSS" = "eaft" ]; then
    echo "[train] eaft_alpha        = $EAFT_ALPHA  (w_t=(H~_t)^alpha; 1=EAFT, 2=EAFT2, 3=EAFT3)"
    echo "[train] eaft_k            = $EAFT_K      (top-K entropy approx; 上游默认 20)"
    echo "[train] note              = L = sum_t w_t * CE_t, H~_t = 学生 top-K 熵 / 3.0 (detached). 无需 cache."
fi
echo "============================================================"

# ---------- 组装可选参数 ----------
OPT_ARGS=()
if [ -n "$MAX_SAMPLES" ]; then
    OPT_ARGS+=(--max_train_samples "$MAX_SAMPLES")
fi
if [ -n "$TEST_FILE" ] && [ "$EVAL_STRATEGY" != "no" ]; then
    OPT_ARGS+=(--test_tokenized_file "$TEST_FILE" --evaluation_strategy "$EVAL_STRATEGY")
fi
if [ "$LOSS" = "gem" ]; then
    OPT_ARGS+=(--gem_beta "$GEM_BETA" --gem_h "linear")
fi
if [ "$LOSS" = "asft" ]; then
    OPT_ARGS+=(--asft_kl_weight "$ASFT_KL_WEIGHT"
               --asft_ref_topk "$ASFT_REF_TOPK"
               --asft_ref_model_path "$ASFT_REF_MODEL_PATH")
fi
if [ "$LOSS" = "lp_sft" ]; then
    # LP-SFT (lp_sft) 复用 plateau cache + ref logits 字段:
    # 需要 K_save (cache 维度) + cache 路径 + mu + tau + loss_mode + set_method.
    OPT_ARGS+=(--lp_sft_mu "$LP_SFT_MU"
               --lp_sft_tau "$LP_SFT_TAU"
               --lp_sft_r_weight "$LP_SFT_R_WEIGHT"
               --lp_sft_mode "$LP_SFT_MODE"
               --lp_sft_set_method "$LP_SFT_SET_METHOD"
               --lp_sft_k_round_mode "$LP_SFT_K_ROUND_MODE"
               --lp_sft_k_threshold "$LP_SFT_K_THRESHOLD"
               --plateau_K_save "$PLATEAU_K_SAVE"
               --renyi_R_cache_path "$R_CACHE_PATH")
fi
if [ "$LOSS" = "eaft" ]; then
    # EAFT: 不需要 cache; 只透传 alpha / k.
    OPT_ARGS+=(--eaft_alpha "$EAFT_ALPHA"
               --eaft_k "$EAFT_K")
fi

# ---------- cd 到 repo 根目录, train.py 期望相对路径 ----------
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
cd "$SCRIPT_DIR/.."
pwd

# ---------- choose deepspeed zero stage ----------
if [ -z "${ZERO_STAGE:-}" ]; then
    if [ "$LOSS" = "asft" ]; then
        ZERO_STAGE=3
        echo "[train] auto-selected ZERO_STAGE=3 because --loss=asft holds a frozen ref model per rank."
    else
        ZERO_STAGE=2
    fi
fi
case "$ZERO_STAGE" in
    2) DS_CONFIG="scripts/zero2.json" ;;
    3) DS_CONFIG="scripts/zero3.json" ;;
    *) echo "ERROR: unknown ZERO_STAGE=$ZERO_STAGE (must be 2 or 3)" >&2; exit 1 ;;
esac
echo "[train] zero_stage    = $ZERO_STAGE  ds_config = $DS_CONFIG"
ls -la train.py "$DS_CONFIG"

# ---------- 选 deepspeed ----------
DEEPSPEED_CMD="${DEEPSPEED_CMD:-deepspeed}"
echo "[train] deepspeed     = $DEEPSPEED_CMD"

REPORT_TO="${REPORT_TO:-none}"

# ---------- 跑训练 (带 EADDRINUSE 重试) ----------
# master_port: 29500 是 deepspeed/torch 默认, 这台 worker 一直有别的东西占着.
# 之前用 49152-65151 撞了 ephemeral race: Linux 默认 ip_local_port_range = 32768..60999,
# 这是内核给 outgoing connections 分配的临时源端口池. 我们 ss -tln 检查过端口空闲,
# 但 deepspeed launch 后 8 个子进程做 NCCL discovery / 读模型等大量外发 TCP 连接,
# 内核可能把我们选中的端口临时拨给某个外发连接 → rank 0 bind TCPStore 时 EADDRINUSE.
#
# 治本方案: 把 deepspeed launch 包在一个 N 次重试循环里. 每轮:
#   1) 在 [20000, 29999] (IANA registered 段, ephemeral 外) 随机选 port + ss -tln 预检.
#   2) 起 deepspeed; 失败时 grep 最近 log 找 "address already in use" / EADDRINUSE.
#      - 命中 ⇒ port race, 换 port 重试.
#      - 未命中 (OOM / CUDA error / python error) ⇒ 真实失败, 立即退出 (不重试).
#   3) 用户固定了 MASTER_PORT 时不重试 (尊重用户意愿; 只做一次, 失败就报错).
#
# 想固定 port: MASTER_PORT=29501 bash train_qwen3_4b.sh ... (会跳过重试逻辑).
# 想调重试次数: MAX_LAUNCH_RETRY=5 bash train_qwen3_4b.sh ...
USER_FIXED_PORT=0
if [ -n "${MASTER_PORT:-}" ]; then
    USER_FIXED_PORT=1
fi

MAX_LAUNCH_RETRY="${MAX_LAUNCH_RETRY:-3}"
DS_RC=0
LAST_TRY=0
for launch_try in $(seq 1 "$MAX_LAUNCH_RETRY"); do
    LAST_TRY=$launch_try
    # ---- 选 port (用户固定 → 仅一次预检; 自动 → 重选) ----
    if [ "$USER_FIXED_PORT" = "1" ]; then
        if ss -tln 2>/dev/null | awk '{print $4}' | grep -q ":$MASTER_PORT\$"; then
            echo "[train] ERROR: 用户指定的 MASTER_PORT=$MASTER_PORT 已被占用." >&2
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
        [ "$PORT_OK" = "1" ] || { echo "[train] ERROR: 5 次随机选 master_port 均被占用." >&2; exit 1; }
    fi
    export MASTER_PORT
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    echo "[train] master_addr   = $MASTER_ADDR"
    echo "[train] master_port   = $MASTER_PORT  (attempt $launch_try / $MAX_LAUNCH_RETRY)"

    {
        echo ""
        echo "==================== launch attempt $launch_try  port=$MASTER_PORT ===================="
        date
    } >> "$OUTPUT_DIR/training.log"

    # ---- 起 deepspeed; 暂关 errexit/pipefail, 自己处理退出码 ----
    set +e
    set +o pipefail
    "$DEEPSPEED_CMD" --num_gpus="$N_GPU" --master_port "$MASTER_PORT" train.py \
        --deepspeed "$DS_CONFIG" \
        --seed $SEED \
        --model_name_or_path "$MODEL_NAME_OR_PATH" \
        --train_tokenized_file "$TRAIN_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --per_device_train_batch_size "$PER_DEVICE_BS" \
        --gradient_accumulation_steps "$GRAD_ACC" \
        --save_strategy "$SAVE_STRATEGY" \
        --loss "$LOSS" \
        --learning_rate 2e-5 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.03 \
        --num_train_epochs "$NUM_EPOCHS" \
        --logging_steps "$LOGGING_STEPS" \
        --report_to "$REPORT_TO" \
        --gradient_checkpointing True \
        --overwrite_output_dir \
        --bf16 True \
        --max_seq_length 2048 \
        --use_flash_attn "$USE_FLASH_ATTN" \
        "${OPT_ARGS[@]}" \
        2>&1 | tee -a "$OUTPUT_DIR/training.log"
    DS_RC=${PIPESTATUS[0]}
    set -e
    set -o pipefail

    if [ "$DS_RC" = "0" ]; then
        # 训练成功即写 marker; worker 不依赖 train.sh 末尾 echo / exit code.
        date -Iseconds > "$OUTPUT_DIR/.train_success"
        break
    fi

    # ---- 是 port race 吗? 真实失败别瞎重试 ----
    if [ "$USER_FIXED_PORT" = "1" ]; then
        echo "[train] deepspeed exited with rc=$DS_RC — user fixed MASTER_PORT, no auto-retry." >&2
        break
    fi

    if tail -n 300 "$OUTPUT_DIR/training.log" 2>/dev/null \
            | grep -qiE "address already in use|EADDRINUSE|address in use"; then
        echo "[train] WARN: attempt $launch_try hit port race (EADDRINUSE on $MASTER_PORT). 换 port 重试." >&2
        unset MASTER_PORT
        continue
    else
        echo "[train] deepspeed rc=$DS_RC — not EADDRINUSE, likely OOM / CUDA / Python error." >&2
        echo "[train]   详见 $OUTPUT_DIR/training.log" >&2
        break
    fi
done

if [ "$DS_RC" != "0" ]; then
    echo "[train] FAILED rc=$DS_RC after $LAST_TRY attempt(s) -> $OUTPUT_DIR" >&2
    exit "$DS_RC"
fi

date -Iseconds > "$OUTPUT_DIR/.train_success"
echo "============================================================"
echo "[train] DONE -> $OUTPUT_DIR  after $LAST_TRY attempt(s)"
echo "============================================================"
