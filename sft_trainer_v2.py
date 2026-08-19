import os
import sys
import torch
import torch.nn.functional as F

from transformers import Trainer
from transformers.trainer import (
    ###
    _is_peft_model,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    is_torch_xla_available,
    SaveStrategy
)
from typing import List, Optional, Dict

# losses/ holds LP-SFT loss implementations.
_LOSSES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "losses")
if _LOSSES_DIR not in sys.path:
    sys.path.insert(0, _LOSSES_DIR)
from lp_sft_loss import (
    lp_sft_loss as _lp_sft_loss_fn,
    lp_sft_loss_diagnostics as _lp_sft_loss_diag_fn,
)
from eaft_loss import eaft_loss as _eaft_loss_fn, eaft_diagnostics as _eaft_diag_fn
from asft_loss import asft_loss as _asft_loss_fn


class SFTTrainer(Trainer):
    def __init__(self, *args, ref_model=None, **kwargs):
        """SFTTrainer.

        ref_model: frozen base model for ASFT's full-vocab KL anchor.
        """
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self._ref_model_placed = False

    @torch.no_grad
    def compute_training_logs(self, logits, labels):
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]

        mask = shift_labels != -100
        shift_logits = shift_logits[mask]
        shift_labels = shift_labels[mask]

        training_logs = {}
        if self.args.print_entropy:
            entropy = chunked_entropy_from_logits(
                shift_logits,
                batch_size=max(1, shift_logits.size(0) // 4),
            ).mean()
            training_logs["entropy"] = round(entropy.item(), 2)

        return training_logs

    def dft_loss(self, logits, labels, num_items_in_batch, ignore_index=-100):
        """Dynamic Fine-Tuning (DFT) loss — Wu et al. 2025 (ICLR 2026).

        L_DFT = -E[ sg(π(y*_t)) · log π(y*_t) ]

        Equivalent to standard per-token CE scaled by the model's current probability
        of the gold token (stop-gradient). One-line change over CE.
        Reference: https://arxiv.org/abs/2508.05629
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        mask = shift_labels != ignore_index
        shift_logits = shift_logits[mask]   # [N_valid, V]
        shift_labels = shift_labels[mask]   # [N_valid]

        # Per-token CE, no reduction
        ce_per_token = F.cross_entropy(shift_logits, shift_labels, reduction="none")  # [N_valid]

        # π(y*_t) — stop-gradient so no gradient flows through the weight
        with torch.no_grad():
            label_probs = F.softmax(shift_logits, dim=-1).gather(
                -1, shift_labels.unsqueeze(-1)
            ).squeeze(-1)  # [N_valid]

        per_token_loss = ce_per_token * label_probs  # [N_valid]

        if num_items_in_batch is not None:
            return per_token_loss.sum() / num_items_in_batch
        else:
            return per_token_loss.mean()

    def gem_loss(self, logits, labels, num_items_in_batch, beta=0.7, ignore_index=-100, h="logsigmoid"):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        mask = shift_labels != -100
        shift_logits = shift_logits[mask]
        shift_labels = shift_labels[mask]

        with torch.no_grad():
            logits_on_labels = torch.gather(
                shift_logits, dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            logits_diff = shift_logits - logits_on_labels.unsqueeze(-1)
            if h == "linear":
                weights = torch.ones_like(logits_diff)
            elif h == "logsigmoid":
                weights = F.sigmoid(0.01 * logits_diff)
            else:
                raise ValueError(h)

        gene_log_probs = F.log_softmax(shift_logits, dim=-1)
        q_probs = torch.exp(F.log_softmax(shift_logits / beta, dim=-1)).detach()

        real_log_probs = torch.gather(
            gene_log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        )

        if num_items_in_batch is not None:
            loss = -torch.sum(
                q_probs * weights * (real_log_probs - gene_log_probs), dim=-1
            ).sum() / num_items_in_batch
        else:
            loss = -torch.sum(
                q_probs * weights * (real_log_probs - gene_log_probs), dim=-1
            ).mean()

        return loss

    def lp_sft_loss(self, logits, labels, plateau_k, plateau_topk_ids,
                                ref_topk_logits, ref_label_logit,
                                num_items_in_batch, renyi_R=None,
                                mu=1.0, tau=1.0, r_weight="none",
                                K_save=10, ignore_index=-100,
                                loss_mode="additive"):
        """Precomputed LP-SFT (lp_sft). See losses/lp_sft_loss.py.

        S_t' = S_t \\ {y_t}; when |S'|==1, add the single highest-ref-prob alt.
        Loss = CE(student, y_t) + mu * H(q_ref^S, p_theta^S)  (additive)
           or (1-R_t)*CE + R_t*H(q_ref^S, p_theta^S)          (r_interp)
        """
        return _lp_sft_loss_fn(
            logits=logits, labels=labels,
            plateau_k=plateau_k,
            plateau_topk_ids=plateau_topk_ids,
            ref_topk_logits=ref_topk_logits,
            ref_label_logit=ref_label_logit,
            renyi_R=renyi_R,
            num_items_in_batch=num_items_in_batch,
            mu=mu, tau=tau, r_weight=r_weight,
            K_save=K_save,
            ignore_index=ignore_index,
            loss_mode=loss_mode,
        )

    def eaft_loss(self, logits, labels, num_items_in_batch,
                  alpha=1.0, k=20, ignore_index=-100):
        """EAFT (Entropy-Adaptive Fine-Tuning) loss. See losses/eaft_loss.py.

        L_EAFT = sum_t w_t * CE_t, w_t = (H~_t)^alpha, where H~_t is the top-K
        (renormalized) entropy of the student's own distribution / 3.0. Verbatim
        port of LLaMA-Factory `use_eaft_loss`; no cache / reference model needed.
        """
        return _eaft_loss_fn(
            logits=logits, labels=labels,
            num_items_in_batch=num_items_in_batch,
            alpha=alpha, k=k,
            ignore_index=ignore_index,
        )

    def asft_loss(self, logits, labels, ref_logits, num_items_in_batch,
                  kl_weight=0.05, ignore_index=-100,
                  return_diagnostics=False, ref_topk=None):
        """ASFT (Anchored SFT) — Zhu et al. ICLR 2026. See losses/asft_loss.py."""
        return _asft_loss_fn(
            logits=logits, labels=labels, ref_logits=ref_logits,
            num_items_in_batch=num_items_in_batch,
            kl_weight=kl_weight,
            ignore_index=ignore_index,
            return_diagnostics=return_diagnostics,
            ref_topk=ref_topk,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # plateau loss 需要从 inputs 里拿这三个字段, 但模型 forward 不接受, 先 pop 出来.
        renyi_R          = inputs.pop("renyi_R", None)
        plateau_k        = inputs.pop("plateau_k", None)
        plateau_topk_ids = inputs.pop("plateau_topk_ids", None)
        # lp_sft 还需要这两个 ref logit 字段.
        ref_topk_logits  = inputs.pop("ref_topk_logits", None)
        ref_label_logit  = inputs.pop("ref_label_logit", None)

        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            if self.args.loss == "ce" or self.control.should_evaluate:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            elif self.args.loss == "gem":
                loss = self.gem_loss(
                    outputs.logits,
                    inputs["labels"],
                    num_items_in_batch=num_items_in_batch,
                    beta=self.args.gem_beta,
                    h=self.args.gem_h
                )
            elif self.args.loss == "lp_sft":
                # LP-SFT loss CE: full CE on y_t + mu * H(q_ref^S, p_theta^S)
                # within S_t' = S_t ∪ {y_t}. Reuses plateau cache (k, topk_ids) plus the
                # extra ref-logits fields produced by precompute_R.py --save_ref_logits.
                if (plateau_k is None or plateau_topk_ids is None
                        or ref_topk_logits is None or ref_label_logit is None):
                    raise ValueError(
                        "loss=lp_sft requires per-token `plateau_k`, "
                        "`plateau_topk_ids`, `ref_topk_logits`, and `ref_label_logit` "
                        "in batch. Did you forget to set --renyi_R_cache_path "
                        "(with cache produced from --save_topk --save_ref_logits) "
                        "and use DataCollatorForSFTWithRefAlign?"
                    )
                _r_weight = str(getattr(self.args, "lp_sft_r_weight", "none")).lower()
                if _r_weight != "none" and renyi_R is None:
                    raise ValueError(
                        "loss=lp_sft with --lp_sft_r_weight != none "
                        "requires per-token `renyi_R` in batch."
                    )
                _loss_mode = str(
                    getattr(self.args, "lp_sft_mode", "additive")
                ).lower()
                loss = self.lp_sft_loss(
                    outputs.logits,
                    inputs["labels"],
                    plateau_k=plateau_k,
                    plateau_topk_ids=plateau_topk_ids,
                    ref_topk_logits=ref_topk_logits,
                    ref_label_logit=ref_label_logit,
                    num_items_in_batch=num_items_in_batch,
                    renyi_R=renyi_R,
                    mu=self.args.lp_sft_mu,
                    tau=self.args.lp_sft_tau,
                    r_weight=_r_weight,
                    K_save=self.args.plateau_K_save,
                    loss_mode=_loss_mode,
                )
                if _lp_sft_loss_diag_fn is not None and not self.control.should_evaluate:
                    diag = _lp_sft_loss_diag_fn(
                        logits=outputs.logits.detach(),
                        labels=inputs["labels"],
                        plateau_k=plateau_k,
                        plateau_topk_ids=plateau_topk_ids,
                        ref_topk_logits=ref_topk_logits,
                        ref_label_logit=ref_label_logit,
                        renyi_R=renyi_R,
                        mu=self.args.lp_sft_mu,
                        tau=self.args.lp_sft_tau,
                        r_weight=_r_weight,
                        K_save=self.args.plateau_K_save,
                        loss_mode=_loss_mode,
                    )
                    self._pending_plateau_diag = {k: round(v, 4) for k, v in diag.items()}
                else:
                    self._pending_plateau_diag = {}
            elif self.args.loss == "eaft":
                # EAFT (Entropy-Adaptive Fine-Tuning): per-token CE reweighted by the
                # student's own normalized top-K entropy. No cache / reference model.
                loss = self.eaft_loss(
                    outputs.logits,
                    inputs["labels"],
                    num_items_in_batch=num_items_in_batch,
                    alpha=self.args.eaft_alpha,
                    k=self.args.eaft_k,
                )
                if _eaft_diag_fn is not None and not self.control.should_evaluate:
                    diag = _eaft_diag_fn(
                        logits=outputs.logits.detach(),
                        labels=inputs["labels"],
                        alpha=self.args.eaft_alpha,
                        k=self.args.eaft_k,
                    )
                    self._pending_plateau_diag = {k: round(v, 4) for k, v in diag.items()}
                else:
                    self._pending_plateau_diag = {}
            elif self.args.loss == "asft":
                if self.ref_model is None:
                    raise RuntimeError(
                        "loss=asft requires SFTTrainer(ref_model=<frozen base model>). "
                        "Did you forget --asft_ref_model_path?"
                    )
                if not self._ref_model_placed:
                    target_device = outputs.logits.device
                    target_dtype = outputs.logits.dtype
                    self.ref_model = self.ref_model.to(device=target_device, dtype=target_dtype)
                    self.ref_model.eval()
                    self._ref_model_placed = True

                with torch.no_grad():
                    ref_outputs = self.ref_model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask", None),
                        use_cache=False,
                    )
                    ref_logits = ref_outputs.logits.detach()

                _ls = max(int(getattr(self.args, "logging_steps", 1)), 1)
                _next_opt_step = self.state.global_step + 1
                _want_diag = (_next_opt_step % _ls == 0) and not self.control.should_evaluate
                _asft_ref_topk = getattr(self.args, "asft_ref_topk", 0) or None

                if _want_diag:
                    loss, diag = self.asft_loss(
                        outputs.logits,
                        inputs["labels"],
                        ref_logits=ref_logits,
                        num_items_in_batch=num_items_in_batch,
                        kl_weight=self.args.asft_kl_weight,
                        return_diagnostics=True,
                        ref_topk=_asft_ref_topk,
                    )
                    self._pending_plateau_diag = {k: round(v, 4) for k, v in diag.items()}
                else:
                    loss = self.asft_loss(
                        outputs.logits,
                        inputs["labels"],
                        ref_logits=ref_logits,
                        num_items_in_batch=num_items_in_batch,
                        kl_weight=self.args.asft_kl_weight,
                        ref_topk=_asft_ref_topk,
                    )
                    self._pending_plateau_diag = {}
            elif self.args.loss == "dft":
                # DFT (Dynamic Fine-Tuning): per-token CE scaled by sg(π(y*_t)).
                # Wu et al. 2025 (ICLR 2026): https://arxiv.org/abs/2508.05629
                loss = self.dft_loss(
                    outputs.logits,
                    inputs["labels"],
                    num_items_in_batch=num_items_in_batch,
                )

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        # ziniu add logs
        if not self.control.should_evaluate:
            self.training_logs = self.compute_training_logs(
                outputs.logits, inputs["labels"]
            )
            self.training_logs["ce_loss"] = (
                outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            )
            self.training_logs["ce_loss"] = round(self.training_logs["ce_loss"].item(), 4)
            # plateau diagnostics (set by the plateau branch above; empty otherwise)
            pending = getattr(self, "_pending_plateau_diag", None)
            if pending:
                self.training_logs.update(pending)
                self._pending_plateau_diag = {}

        return (loss, outputs) if return_outputs else loss

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, **kwargs):
        # 兼容性: transformers 4.51+ 调用时会传 `learning_rate=`  kwarg, 老版本没有.
        # 用 **kwargs 接住所有新参数, 内部自己用 self._get_learning_rate() 读 lr 即可.
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()
            if getattr(self, "training_logs", None):
                logs.update(self.training_logs)

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


def chunked_entropy_from_logits(chunk_logits, batch_size=None):
    """
    Compute entropy from logits in a memory-efficient manner by introducing a batch_size parameter.

    Args:
        chunk_logits (torch.Tensor): Logits tensor of shape (total_samples, num_classes).
        batch_size (int): Number of samples to process per batch.

    Returns:
        torch.Tensor: Entropy tensor of shape (total_samples,).
    """
    total_samples, num_classes = chunk_logits.shape
    entropy_list = []
    if batch_size is None:
        batch_size = total_samples

    # Process logits in batches
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        logits_batch = chunk_logits[start_idx:end_idx]  # Get a batch of logits

        # Compute logsumexp for the current batch
        logsumexp_batch = torch.logsumexp(logits_batch, dim=-1, keepdim=False)  # Shape: (batch_size,)
        # Compute probabilities in log-space without computing softmax
        normalized_logits = logits_batch - logsumexp_batch.unsqueeze(-1)       # Shape: (batch_size, num_classes)
        exp_normalized_logits = torch.exp(normalized_logits)                   # Shape: (batch_size, num_classes)
        # Compute entropy for the batch
        entropy_batch = logsumexp_batch - (logits_batch * exp_normalized_logits).sum(dim=-1)  # Shape: (batch_size,)

        entropy_list.append(entropy_batch)  # Store entropy for the current batch

    # Concatenate results from all batches
    if len(entropy_list) > 0:
        return torch.cat(entropy_list, dim=0)
    else:
        return torch.tensor(0.0)