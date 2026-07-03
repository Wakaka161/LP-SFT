# Checkpoints

Training outputs are written here by default (`CKPT_DIR`).

Layout:

```
ckpts/
└── {model_tag}/
    └── sft_{loss}-.../
        ├── config.json
        ├── model*.safetensors
        └── ...
```

Example: `ckpts/qwen3-4b/sft_lp_sft_mu1.0_tau1.0-.../`

## Custom location

```bash
export CKPT_DIR=/path/to/your/ckpts
source activate_lp_sft.sh
```
