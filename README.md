# LP-SFT

Official **training** code for **LP-SFT**.

LP-SFT augments standard supervised fine-tuning with a local KL anchor against the frozen base model, preserving the base model's relative preferences over an adaptively chosen set of non-target alternatives.

## Quick Start

**Prerequisites:** Linux, NVIDIA GPU(s), Python 3.10, CUDA. `flash-attn` is recommended but optional (see [Installation](#installation)).

```bash
# 1. Environment
conda create -n lpsft python=3.10 -y && conda activate lpsft
pip install -r requirements.txt
cd LP-SFT && source activate_lp_sft.sh

# 2. Base model → models/   (see models/README.md)
huggingface-cli download Qwen/Qwen3-4B-Base --local-dir models/Qwen3-4B-Base

# 3. Tokenized data → data/<dataset>/   (see data/README.md)

# 4a. Baselines — train directly
bash scripts/qwen3-4b/train.sh magicoder ce
ASFT_KL_WEIGHT=0.05 bash scripts/qwen3-4b/train.sh magicoder asft

# 4b. LP-SFT — precompute, then train
SAVE_TOPK=1 SAVE_REF_LOGITS=1 K_SAVE=10 SET_METHOD=N1 \
    bash scripts/qwen3-4b/precompute.sh magicoder
bash scripts/qwen3-4b/train.sh magicoder lp_sft          # default LP_SFT_MU=0.03
```

**Smoke test** (one GPU; Qwen3-4B × magicoder):

```bash
LOSS=lp_sft bash scripts/smoke_test.sh
```

Full list: env vars at the top of [`scripts/train.sh`](scripts/train.sh) and [`scripts/precompute_R.sh`](scripts/precompute_R.sh).

To smoke-test all losses (baselines + `lp_sft`; `asft` excluded — needs ~2× VRAM):

```bash
bash scripts/smoke_test_all.sh
```

## Installation

```bash
pip install -r requirements.txt
```

If `flash-attn` fails to build, install the rest first, then either skip it (`USE_FLASH_ATTN=false` in `scripts/train.sh`) or install a wheel matching your CUDA/PyTorch version separately.

## Losses

| Loss | Precompute? | Ref model at train time? | Default DeepSpeed |
|------|-------------|--------------------------|-------------------|
| `ce` | No | No | ZeRO-2 |
| `dft` | No | No | ZeRO-2 |
| `eaft` | No | No | ZeRO-2 |
| `gem` | No | No | ZeRO-2 |
| `asft` | No | **Yes** (live forward) | ZeRO-3 |
| `lp_sft` | **Yes** | No (cached ref logits) | ZeRO-2 |

```bash
bash scripts/<model>/train.sh <dataset> <loss>
```

Other backbones: `qwen3-14b`, `llama3.1` — same pattern under `scripts/`.

Key env vars: `LP_SFT_MU` (default `0.03`), `ASFT_KL_WEIGHT` (default `0.05`).

## Data & Models

| Folder | Purpose |
|--------|---------|
| [`models/`](models/README.md) | HuggingFace base checkpoints |
| [`data/`](data/README.md) | Tokenized `.jsonl` + R cache (`lp_sft` only) |
| [`ckpts/`](ckpts/README.md) | Training outputs |

Override paths: `export MODELS_DIR=... DATA_DIR=... CKPT_DIR=...` before `source activate_lp_sft.sh`.

## Tests

```bash
python losses/test_asft_loss.py
```

## Acknowledgements

Training code adapted from [GEM](https://github.com/liziniu/GEM) (Li et al., *Preserving Diversity in Supervised Fine-Tuning of Large Language Models*, ICLR 2025).

ASFT baseline from [zhuchichi56/ASFT](https://github.com/zhuchichi56/ASFT) (Zhu et al., *Anchored Supervised Fine-Tuning*, ICLR 2026).

## License

Apache 2.0 — see [LICENSE](./LICENSE).
