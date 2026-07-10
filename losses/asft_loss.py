"""
ASFT -- Anchored Supervised Fine-Tuning loss.

Paper:    "Anchored Supervised Fine-Tuning"
          Zhu, Su, Lai, Ma, Zhang, Yang, Chen (Peking U. / SUSTech / Shanghai AI Lab).
          arXiv:2509.23753 ; ICLR 2026 Poster.
          https://arxiv.org/abs/2509.23753 ; https://openreview.net/forum?id=PORko7QT64
Ref impl: zhuchichi56/ASFT  --  train_v2.py::EnhancedTrainer.compute_loss
          mode == "asft" branch (https://github.com/zhuchichi56/ASFT).

================================================================================
OBJECTIVE
================================================================================

ASFT augments DFT (Wu et al. 2025) with a forward-KL anchor to the *frozen base
model* on the full vocabulary:

    L_ASFT(theta) = L_DFT(theta)  +  lambda * E_s [ KL( pi_ref(.|s) || pi_theta(.|s) ) ]

where:

  * L_DFT(theta) = -E[ sg(pi_theta(y_t | s)) * log pi_theta(y_t | s) ]    (Wu+'25)
                 = sg(pi_theta(y_t | s)) * CE_t      (per token CE re-weighted by
                                                     stop-grad student prob of label)

  * The KL term is *forward* KL (P=ref, Q=student), mode-covering, computed over
    the full vocabulary V at every supervised token position.

  * pi_ref is fixed (the pretrained / base checkpoint). The reference logits
    must be supplied by the caller (no forward inside this function).

This is a verbatim port of the `mode == "asft"` branch of
zhuchichi56/ASFT/train_v2.py. Batch reduction follows the HF Trainer
convention used by all other losses in this repo: sum over valid tokens,
divide by `num_items_in_batch` (global supervised-token count).

================================================================================
SHIFT CONVENTION
================================================================================

Same as the rest of this repo: shift_logits = logits[:, :-1, :],
shift_labels = labels[:, 1:]. The reference is shifted identically.

================================================================================
FIDELITY NOTES
================================================================================

* DFT weight: probs.gather(label).detach() over the *student's softmax of
  shift_logits*. We replicate this exactly (no temperature, no float32 cast,
  same gather indexing) -- see ASFT train_v2.py lines 96-110.

* KL: torch.nn.functional.kl_div( log_softmax(student), softmax(ref),
  reduction="none" ).sum(-1) == sum_v p_ref * (log p_ref - log p_student)
  == KL(p_ref || p_student).  Reference does *not* go through log_softmax with
  log_target=True, matching ASFT's exact call.

* Combination: weighted_losses = dft_loss + kl_weight * kl_div (per-token),
  then reduce.

* No KL clipping / chunking is done here; if memory is a concern, the trainer
  should chunk the inputs themselves.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Main loss
# --------------------------------------------------------------------------------------
def asft_loss(
    logits: torch.Tensor,                          # [B, L, V]  -- student
    labels: torch.Tensor,                          # [B, L]
    ref_logits: torch.Tensor,                      # [B, L, V]  -- frozen base (no grad expected)
    num_items_in_batch: Optional[torch.Tensor] = None,
    kl_weight: float = 0.03,
    ignore_index: int = -100,
    return_diagnostics: bool = False,
    ref_topk: Optional[int] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    """Compute the ASFT loss = DFT + lambda * forward_KL(ref || student).

    Args:
        logits:       student logits [B, L, V].
        labels:       gold labels    [B, L] (with `ignore_index` on prompt/padding).
        ref_logits:   frozen-base logits [B, L, V] (same shape as `logits`).
                      Caller is responsible for computing these under torch.no_grad().
        num_items_in_batch: HF Trainer's global count of supervised tokens.
        kl_weight:    anchoring strength lambda. Paper uses 0.05 (medical) / 0.1
                      (math); bf16-mixed-precision recommended setting per the
                      ASFT GitHub README is 0.03.
        ignore_index: label id to ignore (default -100).
        return_diagnostics: if True, also return a dict of `asft/...` scalars.
        ref_topk:     ablation knob (None = full vocabulary ASFT, default). When
                      set to an int K, the reference distribution is TRUNCATED to
                      its top-K positions per token: ref_probs are zeroed
                      everywhere except the K positions with highest ref logit,
                      and NOT renormalized.  Inside `F.kl_div(log_input, target)`
                      the zeroed positions contribute `0 * (log 0 - log_input)`,
                      which PyTorch evaluates as 0 by convention -- so the anchor
                      only fires on the top-K positions of ref. This matches the
                      `inset_ref_align_ce + normalize_mode='none' + ref_logsumexp`
                      anchor formula exactly (when set_label_mode treats y_t the
                      same way -- ASFT-truncated keeps y_t implicitly only if
                      y_t is already in top-K).  Used to isolate the "top-K vs V"
                      effect from other inset/ASFT pipeline differences (FA2 vs
                      eager, ref_logits cache, etc.).

    Returns:
        scalar loss tensor (or (loss, diag) if return_diagnostics).
    """
    if logits.shape != ref_logits.shape:
        raise ValueError(
            f"ref_logits shape {tuple(ref_logits.shape)} must match logits "
            f"shape {tuple(logits.shape)}"
        )
    V = logits.size(-1)

    # ---- shift for next-token prediction (logits[:, t] predicts labels[:, t+1]) ----
    shift_logits = logits[..., :-1, :].contiguous().view(-1, V)      # [N_all, V]
    shift_labels = labels[..., 1:].contiguous().view(-1)             # [N_all]
    shift_labels = shift_labels.to(shift_logits.device)
    # Reference shifted identically. Detach in case caller didn't.
    shift_ref_logits = ref_logits[..., :-1, :].contiguous().view(-1, V).detach()  # [N_all, V]

    valid_mask = shift_labels != ignore_index                        # [N_all]
    if not valid_mask.any():
        zero = torch.tensor(
            0.0, device=logits.device, dtype=logits.dtype, requires_grad=True
        )
        if return_diagnostics:
            return zero, {}
        return zero

    # === Per-token CE (no reduction); for ignored positions clamp label first
    # so cross_entropy doesn't error -- we mask them out below anyway.
    # We follow ASFT ref impl: clamp labels then weight by stopgrad student prob.
    safe_labels = shift_labels.clamp(min=0, max=V - 1)
    ce_per_token = F.cross_entropy(
        shift_logits, safe_labels, reduction="none"
    )                                                                 # [N_all]

    # === DFT weight: stop-grad student prob of the gold token ===
    with torch.no_grad():
        student_probs = F.softmax(shift_logits, dim=-1)               # [N_all, V]
        dft_weight = student_probs.gather(
            -1, safe_labels.unsqueeze(-1)
        ).squeeze(-1)                                                 # [N_all]
    dft_per_token = ce_per_token * dft_weight                         # [N_all]

    # === Forward-KL anchor: KL(p_ref || p_student) per token ===
    # F.kl_div(log_input, target, reduction="none") computes
    #   target * (log(target) - log_input)  elementwise,
    # then we sum over vocab to get KL(target || input).
    student_log_probs = F.log_softmax(shift_logits, dim=-1)           # [N_all, V]
    ref_probs = F.softmax(shift_ref_logits, dim=-1)                   # [N_all, V]

    # Optional top-K truncation of the reference distribution (ablation): keep
    # only the K largest ref_probs entries per token, zero the rest. Note we do
    # NOT renormalize -- this is intentional (matches the inset
    # `normalize_mode='none' + ref_logsumexp` formula where p_ref values are the
    # true full-vocab probabilities but only contribute on top-K positions).
    # With target=0 at masked positions, `F.kl_div` returns 0 there (0*log0=0
    # convention), so KL reduces to a sum over the top-K positions only.
    ref_in_set_mass_mean = None
    if ref_topk is not None:
        K = int(ref_topk)
        if K <= 0:
            raise ValueError(f"ref_topk must be > 0, got {K}")
        V_eff = ref_probs.size(-1)
        if K < V_eff:
            topk_vals, topk_idx = ref_probs.topk(K, dim=-1)           # [N_all, K]
            mask_topk = torch.zeros_like(ref_probs)
            mask_topk.scatter_(-1, topk_idx, 1.0)
            ref_probs = ref_probs * mask_topk                          # zero the rest
            ref_in_set_mass_mean = topk_vals.sum(dim=-1)               # [N_all]
        # else: K >= V_eff -> no truncation needed

    kl_per_token = F.kl_div(
        student_log_probs, ref_probs, reduction="none"
    ).sum(dim=-1)                                                     # [N_all]

    # === Combine + reduce ===
    per_token_loss = dft_per_token + kl_weight * kl_per_token         # [N_all]
    masked_loss = per_token_loss[valid_mask]                          # [N_valid]

    if num_items_in_batch is not None:
        total_loss = masked_loss.sum()
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(total_loss.device)
        loss = total_loss / num_items_in_batch
    else:
        loss = masked_loss.mean()
    loss = loss.to(logits.dtype) if logits.dtype != torch.float32 else loss

    if not return_diagnostics:
        return loss

    with torch.no_grad():
        dft_valid = dft_per_token[valid_mask]
        kl_valid = kl_per_token[valid_mask]
        ce_valid = ce_per_token[valid_mask]
        diag: Dict[str, float] = {
            "asft/loss":          masked_loss.mean().item(),
            "asft/dft_loss":      dft_valid.mean().item(),
            "asft/kl":            kl_valid.mean().item(),
            "asft/kl_weighted":   (kl_weight * kl_valid).mean().item(),
            "asft/ce_loss":       ce_valid.mean().item(),
            "asft/dft_weight_mean": dft_weight[valid_mask].mean().item(),
            "asft/dft_weight_min":  dft_weight[valid_mask].min().item(),
            "asft/dft_weight_max":  dft_weight[valid_mask].max().item(),
            "asft/kl_weight":     float(kl_weight),
        }
        if ref_topk is not None:
            diag["asft/ref_topk"] = float(ref_topk)
            if ref_in_set_mass_mean is not None:
                diag["asft/ref_in_set_mass_mean"] = (
                    ref_in_set_mass_mean[valid_mask].mean().item()
                )
    return loss, diag


# --------------------------------------------------------------------------------------
# Standalone diagnostics (no grad)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def asft_diagnostics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ref_logits: torch.Tensor,
    kl_weight: float = 0.03,
    ignore_index: int = -100,
) -> Dict[str, float]:
    """Standalone diagnostics path (no grad). Mirrors the loss math."""
    if logits.shape != ref_logits.shape:
        raise ValueError(
            f"ref_logits shape {tuple(ref_logits.shape)} must match logits "
            f"shape {tuple(logits.shape)}"
        )
    V = logits.size(-1)
    shift_logits = logits[..., :-1, :].contiguous().view(-1, V)
    shift_labels = labels[..., 1:].contiguous().view(-1).to(shift_logits.device)
    shift_ref_logits = ref_logits[..., :-1, :].contiguous().view(-1, V)

    valid_mask = shift_labels != ignore_index
    if not valid_mask.any():
        return {}
    safe_labels = shift_labels.clamp(min=0, max=V - 1)

    ce_per_token = F.cross_entropy(shift_logits, safe_labels, reduction="none")

    student_probs = F.softmax(shift_logits, dim=-1)
    dft_weight = student_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    dft_per_token = ce_per_token * dft_weight

    student_log_probs = F.log_softmax(shift_logits, dim=-1)
    ref_probs = F.softmax(shift_ref_logits, dim=-1)
    kl_per_token = F.kl_div(student_log_probs, ref_probs, reduction="none").sum(-1)

    per_token_loss = dft_per_token + kl_weight * kl_per_token
    masked = per_token_loss[valid_mask]

    return {
        "asft/loss":          masked.mean().item(),
        "asft/dft_loss":      dft_per_token[valid_mask].mean().item(),
        "asft/kl":            kl_per_token[valid_mask].mean().item(),
        "asft/kl_weighted":   (kl_weight * kl_per_token)[valid_mask].mean().item(),
        "asft/ce_loss":       ce_per_token[valid_mask].mean().item(),
        "asft/dft_weight_mean": dft_weight[valid_mask].mean().item(),
        "asft/dft_weight_min":  dft_weight[valid_mask].min().item(),
        "asft/dft_weight_max":  dft_weight[valid_mask].max().item(),
        "asft/kl_weight":     float(kl_weight),
    }
