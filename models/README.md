# Models

Place **HuggingFace-format base model checkpoints** here. `source activate_lp_sft.sh` sets `MODELS_DIR` to this folder by default.

Download with `huggingface-cli` or `git lfs clone`, then ensure each directory contains `config.json` and weight files.

## Script wrappers (recommended)

| Script wrapper | Directory name |
|----------------|----------------|
| `scripts/qwen3-4b/` | `Qwen3-4B-Base/` |
| `scripts/qwen3-14b/` | `Qwen3-14B-Base/` |
| `scripts/llama3.1/` | `Llama-3.1-8B/` |

Example:

```bash
huggingface-cli download Qwen/Qwen3-4B-Base --local-dir models/Qwen3-4B-Base
```

`scripts/train.sh` also accepts other `MODEL_TAG` values (e.g. `gemma3-12b`) via env override, but only the three wrappers above are maintained in this repo.

## Custom location

```bash
export MODELS_DIR=/path/to/your/models
source activate_lp_sft.sh
```
