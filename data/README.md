# Data

Place **tokenized training jsonl** and **precomputed R caches** here. `source activate_lp_sft.sh` sets `DATA_DIR` to this folder by default.

## Directory layout

Each dataset lives in its own subdirectory:

```
data/
├── magicoder/
│   ├── magicoder_sft_train_qwen3-4b_tokenized.jsonl   # Qwen3 tokenizer
│   ├── magicoder_sft_train_llama3.1-8b_tokenized.jsonl
│   └── magicoder_R_qwen3_4b_base_full_plateau_K10_reflg_setN1.jsonl  # lp_sft cache
├── numinamath/
│   └── ...
└── ultrafeedback/
    └── ...
```

### Tokenized JSONL format

One JSON object per line (HF `datasets` json loader):

```json
{"input_ids": [151644, 8948, ...], "labels": [-100, -100, 77091, ...]}
```

- `labels[i] == -100` → no loss at position `i` (typically the prompt)
- `labels[i] >= 0` → supervised response token

The training script asserts both fields exist. Optional keys in the jsonl are ignored unless needed by the collator.

### Tokenized train file naming

```
{dataset}/{dataset}_sft_train_{tok_tag}_tokenized.jsonl
```

- `tok_tag=qwen3-4b` for Qwen3-4B / 14B (shared tokenizer)
- `tok_tag=llama3.1-8b` for Llama-3.1

Supported dataset names (CLI argument to `train.sh` / `precompute.sh`):

`magicoder`, `numinamath`, `ultrafeedback`, `opencodeinstruct`, `sciinstruct`, `flan`, `code_mix`, `all_mix`

### R cache (`lp_sft` only)

Produced by `scripts/<model>/precompute.sh`. Filenames use the legacy `_plateau_K{K}` suffix from the upstream GEM pipeline — this is the **lp_sft** precompute cache (Rényi geometry + ref top-K logits).

Typical filename:

```
{dataset}/{dataset}_R_{base_model}_full_plateau_K10_reflg_setN1.jsonl
```

Run precompute with `SAVE_TOPK=1 SAVE_REF_LOGITS=1 K_SAVE=10 SET_METHOD=N1`.

## Custom location

```bash
export DATA_DIR=/path/to/your/data
source activate_lp_sft.sh
```
