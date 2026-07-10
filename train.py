#!/usr/bin/env python
# coding=utf-8
"""
This file is modified from the huggingface example for finetuning language models
[run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py)
"""

import logging
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import sys
from typing import Optional
from functools import partial
import datasets
import torch
import torch.distributed as dist
import deepspeed
from datasets import load_dataset
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Optional, List, Union

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from packaging import version

from sft_trainer_v2 import SFTTrainer
if version.parse(transformers.__version__) < version.parse("4.46.0"):
    raise RuntimeError(
        "LP-SFT training requires transformers>=4.46.0 "
        f"(got {transformers.__version__}). pip install 'transformers>=4.46.0'"
    )

# losses/ holds LP-SFT loss implementations and data collators.
_LOSSES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "losses")
if _LOSSES_DIR not in sys.path:
    sys.path.insert(0, _LOSSES_DIR)
from data_utils import (  # noqa: E402
    load_R_cache,
    build_ref_align_full,
    select_k_for_set_method,
    cache_has_ref_logits,
    DataCollatorForSFTWithRefAlign,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    adam_beta2: float = field(default=0.95, metadata={"help": "Beta2 for AdamW"})
    loss: str = field(
        default="ce",
        metadata={
            "help": "Loss name",
            "choices": ["ce", "dft", "eaft", "gem", "asft", "lp_sft"],
        },
    )
    gem_beta: float = field(default=0.7, metadata={"help": "Hyper-parameter in GEM. A value between 0 and 1. A value close to 1.0 makes GEM behave more like CE, while a value close to 0.0 preserves more diversity."})
    gem_h: str = field(
        default="linear", metadata={"help": "Function $h$ in GEM. The 'logsigmoid' function is more adaptive, but the difference between 'logsigmoid' and 'linear' is usually negligible.", "choices": ["logsigmoid", "linear"]}
    )
    print_entropy: bool = field(
        default=False, metadata={"help": "Print entropy during training"}
    )
    # ------ LP-SFT (lp_sft) ------
    plateau_K_save: int = field(
        default=10,
        metadata={"help": "Ref top-K cap |S_t'| <= K_save. Must match precompute cache K_save."}
    )
    lp_sft_mu: float = field(
        default=0.03,
        metadata={"help": "Weight on the in-set ref-soft-CE term: "
                          "L = CE + mu * H(q_ref^S, p_theta^S). "
                          "Recommended sweep (tau=1.0): {0.01, 0.03, 0.05, 0.1}."}
    )
    lp_sft_tau: float = field(
        default=1.0,
        metadata={"help": "Temperature applied to reference logits before in-set softmax: "
                          "q_ref^S(v) = softmax(z_ref(v)/tau) over S_t'. "
                          "tau=1.0 (default) replicates the natural ref distribution; "
                          "tau>1 flattens the soft labels."}
    )
    lp_sft_r_weight: str = field(
        default="none",
        metadata={"help": "Optional per-token R weighting for the lp_sft alignment term: "
                          "'none' => mu, 'R' => mu*R_t, 'R2' => mu*R_t^2, "
                          "'R3' => mu*R_t^3."}
    )
    lp_sft_mode: str = field(
        default="additive",
        metadata={"help": "Loss combination mode for lp_sft: "
                          "'additive' => CE + mu_t * L_set (original); "
                          "'r_interp' => (1-R_t)*CE + R_t*L_set (convex interpolation). "
                          "r_interp ignores mu/r_weight."}
    )
    lp_sft_set_method: str = field(
        default="N2",
        metadata={"help": "Which effective size to use for k_t when loading from cache: "
                          "'N2' (collision, default) or 'N1' (Shannon, larger sets). "
                          "Selects k_n2 or k_n1 from cache (falls back to legacy 'k' if missing)."}
    )
    lp_sft_k_round_mode: str = field(
        default="precomputed",
        metadata={"help": "How to derive k from the cache: "
                          "'precomputed' (default) – use stored k_n1/k_n2/k; "
                          "'round' – recompute from raw n1_vals/n2_vals using mathematical rounding; "
                          "'ceil'  – recompute using ceiling (legacy behavior); "
                          "'floor' – recompute using floor. "
                          "Requires cache produced with --save_topk (n1_vals/n2_vals fields). "
                          "Falls back to stored k if raw vals are absent."}
    )
    lp_sft_k_threshold: float = field(
        default=0.0,
        metadata={"help": "Threshold T for k rounding (0 = disabled). "
                          "When T > 1.0: tokens with n_value < T are set to k=1; "
                          "remaining tokens use ceil(n_value). "
                          "Overrides lp_sft_k_round_mode when active. "
                          "Recommended sweep: {1.1, 1.2, 1.3}."}
    )
    # ------ EAFT baseline ------
    eaft_alpha: float = field(
        default=1.0,
        metadata={"help": "Power of the EAFT adaptive weight w_t = (H~_t)^alpha. "
                          "1.0 => EAFT (paper default), 2.0 => EAFT2, 3.0 => EAFT3."}
    )
    eaft_k: int = field(
        default=20,
        metadata={"help": "Top-K used for the EAFT entropy approximation. "
                          "Upstream uses K=20 (Pearson corr 0.999 vs exact entropy)."}
    )
    # ------ ASFT baseline (Zhu et al. ICLR 2026) ------
    asft_kl_weight: float = field(
        default=0.05,
        metadata={"help": "ASFT KL-anchor strength lambda. Paper uses 0.05; "
                          "0.03 recommended under bf16 mixed precision."}
    )
    asft_ref_model_path: str = field(
        default=None,
        metadata={"help": "Frozen base model for ASFT KL anchor. "
                          "Defaults to model_name_or_path if unset."}
    )
    asft_ref_topk: int = field(
        default=0,
        metadata={"help": "If > 0, truncate ref distribution to top-K per token "
                          "(0 = full-vocab ASFT)."}
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )


@dataclass
class DataArguments:
    train_tokenized_file: str = field(
        default=None, metadata={"help": "huggingface dataset name or local data path"}
    )
    test_tokenized_file: str = field(
        default=None, metadata={"help": "huggingface dataset name or local data path"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    renyi_R_cache_path: Optional[str] = field(
        default=None,
        metadata={"help": (
            "Path to precomputed cache jsonl (from scripts/precompute_R.sh). "
            "Each line: {sample_idx, R: [floats], n_tokens, k: [ints], topk_ids: [[int]]}. "
            "Required when --loss lp_sft (must be produced with --save_topk --save_ref_logits)."
        )}
    )


class CustomDataset(Dataset):
    def __init__(
        self,
        training_args,
        data_args,
        model_args,
        train_tokenized_file,
        R_cache_path=None,
        plateau_K_save: int = 10,
        attach_ref_align: bool = False,
        set_method: str = "N2",
        k_round_mode: str = "precomputed",
        k_threshold: float = 0.0,
    ):
        self.training_args = training_args
        self.data_args = data_args
        self.model_args = model_args
        self.plateau_K_save = int(plateau_K_save)
        self.attach_ref_align = bool(attach_ref_align)
        self.set_method = str(set_method)
        self.k_round_mode = str(k_round_mode)
        self.k_threshold = float(k_threshold)

        raw_datasets = load_dataset(
            "json",
            data_files=[train_tokenized_file],
            cache_dir=self.model_args.cache_dir,
        )
        self.data = raw_datasets["train"]

        if self.data_args.max_train_samples is not None:
            max_samples = min(len(self.data), self.data_args.max_train_samples)
            self.data = self.data.select(range(max_samples))

        # ---- cache (optional, only when --loss plateau) ----
        # 注意: sample 在 cache 中按 jsonl 原始行号索引. CustomDataset 用 select(range(N))
        # 保留前 N 个 sample, 所以 __getitem__ 的 item 索引 0..N-1 == jsonl 行号 0..N-1, 对齐.
        self.R_cache = None
        if R_cache_path is not None:
            logger.info(f"Loading R cache from {R_cache_path} ...")
            self.R_cache = load_R_cache(R_cache_path)
            logger.info(f"  loaded for {len(self.R_cache):,} samples")
            if self.attach_ref_align and not cache_has_ref_logits(self.R_cache):
                raise ValueError(
                    f"--loss lp_sft requires cache with 'topk_logits' and "
                    f"'label_ref_logit' fields, but {R_cache_path} doesn't have them. "
                    f"Please re-run precompute_R.py with --save_topk --save_ref_logits."
                )
            # warn if cache is smaller than training set
            n_need = len(self.data)
            n_missing = sum(1 for i in range(n_need) if i not in self.R_cache)
            if n_missing > 0:
                logger.warning(
                    f"R cache missing for {n_missing}/{n_need} samples. "
                    f"Those will be assigned R=0 / k=0 placeholders (loss falls back to plain CE there)."
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        example = self.data[item]
        assert "input_ids" in example
        assert "labels" in example

        out = {k: torch.tensor(v, dtype=torch.long) for k, v in example.items()}

        # Attach renyi_R / plateau_k / plateau_topk_ids aligned with labels (length = len(labels)).
        # 留作 list, 由 DataCollatorForSFTWithPlateau 在 collate 时统一 pad + 转 tensor.
        if self.R_cache is not None:
            entry = self.R_cache.get(item, None)
            L = len(example["labels"])
            K_save = self.plateau_K_save
            if entry is None:
                out["renyi_R"] = [0.0] * L
                if self.attach_ref_align:
                    out["ref_topk_logits"] = [[0.0] * K_save for _ in range(L)]
                    out["ref_label_logit"] = [0.0] * L
                    out["plateau_k"] = [0] * L
                    out["plateau_topk_ids"] = [[0] * K_save for _ in range(L)]
            else:
                if self.attach_ref_align:
                    try:
                        k_for_method = select_k_for_set_method(
                            entry, self.set_method,
                            k_round_mode=self.k_round_mode, k_save=K_save,
                            k_threshold=self.k_threshold,
                        )
                        R_full, k_full, topk_full, topk_lg_full, lab_lg_full = build_ref_align_full(
                            example["labels"],
                            entry["R"], k_for_method, entry["topk_ids"],
                            entry["topk_logits"], entry["label_ref_logit"],
                            K_save=K_save,
                        )
                    except (KeyError, ValueError) as e:
                        logger.warning(
                            f"sample {item}: ref-align cache mismatch ({e}); "
                            f"falling back to all-zero placeholders (plain CE for this sample)."
                        )
                        R_full          = [0.0] * L
                        k_full          = [0] * L
                        topk_full       = [[0] * K_save for _ in range(L)]
                        topk_lg_full    = [[0.0] * K_save for _ in range(L)]
                        lab_lg_full     = [0.0] * L
                    out["renyi_R"]          = R_full
                    out["plateau_k"]        = k_full
                    out["plateau_topk_ids"] = topk_full
                    out["ref_topk_logits"]  = topk_lg_full
                    out["ref_label_logit"]  = lab_lg_full
        return out


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    global_rank = dist.get_rank()
    logger.warning(
        f"Process rank: {global_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    )
    logger.info(f"Training parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        if "llama-3" in tokenizer.name_or_path.lower():
            tokenizer.pad_token_id = len(tokenizer) - 1
            tokenizer.pad_token = tokenizer.decode(tokenizer.pad_token_id)
        elif tokenizer.eos_token is not None:
            # 部分 base 模型无 pad_token, 用 eos 即可
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype="auto",
        attn_implementation=(
            "flash_attention_2" if model_args.use_flash_attn else "eager"
        ),
    )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # gather deepspeed to get "real" embedding size
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]
    # resize does its own gather
    if len(tokenizer) > embedding_size:
        # pad to multiple for tensor cores.
        logging.warning(f"len(tokenizer) > embedding_size!!! we are resizing...")
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    # set up datasets
    # lp_sft consumes precomputed ref-align cache (k, topk_ids, ref logits).
    # DataCollatorForSFTWithRefAlign pads all cache fields.
    use_lp_sft_cache = training_args.loss == "lp_sft"
    if use_lp_sft_cache and not data_args.renyi_R_cache_path:
        raise ValueError(
            f"--loss {training_args.loss} requires --renyi_R_cache_path to be set."
        )
    if use_lp_sft_cache:
        if training_args.remove_unused_columns:
            logger.info(
                f"[{training_args.loss}] forcing remove_unused_columns=False so "
                f"cache fields reach the collator."
            )
            training_args.remove_unused_columns = False

    train_dataset = CustomDataset(
        training_args, data_args, model_args, data_args.train_tokenized_file,
        R_cache_path=data_args.renyi_R_cache_path if use_lp_sft_cache else None,
        plateau_K_save=training_args.plateau_K_save,
        attach_ref_align=use_lp_sft_cache,
        set_method=getattr(training_args, "lp_sft_set_method", "N2"),
        k_round_mode=getattr(training_args, "lp_sft_k_round_mode", "precomputed"),
        k_threshold=getattr(training_args, "lp_sft_k_threshold", 0.0),
    )
    if data_args.test_tokenized_file:
        test_dataset = CustomDataset(training_args, data_args, model_args, data_args.test_tokenized_file)
    else:
        test_dataset = None

    if use_lp_sft_cache:
        data_collator = DataCollatorForSFTWithRefAlign(
            tokenizer=tokenizer, model=model, padding="longest",
            K_save=training_args.plateau_K_save,
        )
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding="longest"
        )

    # initalize a trainer
    # here we use a custom trainer that moves the model to CPU when saving the checkpoint in FSDP mode
    # we can switch to the default trainer after moving to deepspeed (let's don't change too much for now)

    ref_model = None
    if training_args.loss == "asft":
        ref_path = training_args.asft_ref_model_path or model_args.model_name_or_path
        logger.info(f"[asft] loading frozen reference model from {ref_path}")
        import transformers.integrations.deepspeed as _hf_ds_int
        _saved_z3_ref = _hf_ds_int._hf_deepspeed_config_weak_ref
        try:
            _hf_ds_int._hf_deepspeed_config_weak_ref = None
            ref_model = AutoModelForCausalLM.from_pretrained(
                ref_path,
                torch_dtype="auto",
                attn_implementation=(
                    "flash_attention_2" if model_args.use_flash_attn else "eager"
                ),
            )
        finally:
            _hf_ds_int._hf_deepspeed_config_weak_ref = _saved_z3_ref
        for p in ref_model.parameters():
            p.requires_grad = False
        ref_model.eval()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        preprocess_logits_for_metrics=None,
        compute_metrics=None,
        ref_model=ref_model,
    )

    # Training
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    if "llama-3" in model.config.name_or_path.lower() and isinstance(model.generation_config.eos_token_id, int):
        model.generation_config.eos_token_id = [128001, 128009]
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)


if __name__ == "__main__":
    main()
