"""
EAFT -- Entropy-Adaptive Fine-Tuning loss.

Paper:  "Entropy-Adaptive Fine-Tuning: Resolving Confident Conflicts to Mitigate
         Forgetting" (Diao et al., 2026, arXiv:2601.02151).
Ref impl: LLaMA-Factory `use_eaft_loss` --
          src/llamafactory/train/trainer_utils.py::_eaft_cross_entropy
          (https://github.com/ymxyll/LlamaFactory-EAFT)

================================================================================
OBJECTIVE
================================================================================

EAFT is a per-token re-weighting of the standard CE loss by the (normalized)
token-level entropy of the *model's own* predictive distribution:

    L_EAFT = - sum_t  H~_t * log P_theta(y_t | x, y_<t)
           =   sum_t  w_t * CE_t,        w_t = (H~_t) ** alpha

where H~_t is the top-K-approximated, normalized Shannon entropy at token t.

Intuition: low-entropy tokens where the model is confident but the ground truth
is low-probability ("Confident Conflicts") are the main driver of catastrophic
forgetting -- EAFT down-weights them; high-entropy (genuinely uncertain) tokens
keep full supervision.

================================================================================
FIDELITY TO THE REFERENCE IMPLEMENTATION
================================================================================

This module is a verbatim port of LLaMA-Factory's `_eaft_cross_entropy`. The
following details are reproduced EXACTLY (a numerical-equivalence unit test in
test_eaft_loss.py asserts torch.allclose against a copy of the upstream code):

  * logits cast to fp32 before any computation.
  * per-token CE via F.cross_entropy(..., reduction="none", ignore_index).
  * entropy computed under torch.no_grad() => the adaptive weight is DETACHED
    (no gradient flows through the gating term).
  * top-K (default K=20) logits are RE-NORMALIZED among themselves
    (logsumexp over the top-K) before the entropy is computed, i.e. the entropy
    is that of the truncated-and-renormalized top-K distribution.
  * the entropy is normalized by the CONSTANT 3.0 (upstream hard-codes this;
    it approximates log(20) ~= 2.996). We keep 3.0 regardless of K to stay
    bit-identical to the reference; `entropy_norm` is exposed only for ablation.
  * adaptive_weight = entropy_term ** alpha  (alpha=1 => EAFT, 2 => EAFT2, ...).
  * batch reduction: sum(w_t * CE_t) / num_items_in_batch  (HF Trainer
    convention), or .mean() when num_items_in_batch is None.

================================================================================
SHIFT CONVENTION
================================================================================

The reference flattens all L logit positions and pads labels by one at the end
before shifting. We use the equivalent in-repo convention
(shift_logits = logits[:, :-1], shift_labels = labels[:, 1:]); for every valid
(non-ignored) token the two produce the IDENTICAL set of (logit_t, y_{t+1})
pairs, hence identical per-token CE / entropy / loss.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Main loss
# --------------------------------------------------------------------------------------
def eaft_loss(
    logits: torch.Tensor,                          # [B, L, V]
    labels: torch.Tensor,                          # [B, L]
    num_items_in_batch: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    k: int = 20,
    entropy_norm: float = 3.0,
    ignore_index: int = -100,
    return_diagnostics: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    """Compute the EAFT loss (entropy-adaptive token-reweighted CE).

    Args:
        logits:  student logits [B, L, V].
        labels:  target labels  [B, L] (with `ignore_index` on prompt/padding).
        num_items_in_batch:  HF Trainer's global count of supervised tokens. If
            given, loss = sum(w*CE) / num_items_in_batch; else .mean().
        alpha:   power of the adaptive weight w_t = H~_t ** alpha
                 (1 => EAFT, 2 => EAFT2, 3 => EAFT3).
        k:       top-K used for the entropy approximation (upstream default 20).
        entropy_norm:  constant entropy normalizer (upstream hard-codes 3.0).
        ignore_index:  label id to ignore (default -100).
        return_diagnostics:  if True, also return a dict of `eaft/...` scalars.

    Returns:
        scalar loss tensor (or (loss, diag) if return_diagnostics).
    """
    if alpha < 0:
        raise ValueError(f"eaft alpha must be >= 0, got {alpha}")
    if k < 1:
        raise ValueError(f"eaft k must be >= 1, got {k}")

    # fp32 for numerical parity with the reference (and entropy stability).
    logits = logits.float()
    V = logits.size(-1)
    K = int(min(k, V))

    # ---- shift for next-token prediction (logits[:, t] predicts labels[:, t+1]) ----
    shift_logits = logits[..., :-1, :].contiguous().view(-1, V)   # [B*(L-1), V]
    shift_labels = labels[..., 1:].contiguous().view(-1)          # [B*(L-1)]
    shift_labels = shift_labels.to(shift_logits.device)

    per_token_loss = F.cross_entropy(
        shift_logits, shift_labels, ignore_index=ignore_index, reduction="none"
    )
    valid_mask = shift_labels != ignore_index
    if not valid_mask.any():
        zero = torch.tensor(
            0.0, device=logits.device, dtype=logits.dtype, requires_grad=True
        )
        if return_diagnostics:
            return zero, {}
        return zero

    valid_losses = per_token_loss[valid_mask]                     # [N]  (grad flows)

    # ---- entropy-adaptive weight (DETACHED; no grad through the gate) ----
    with torch.no_grad():
        source_detached = shift_logits[valid_mask].detach()       # [N, V]
        topk_val, _ = torch.topk(source_detached, k=K, dim=-1)    # [N, K]
        logsumexp_topk = torch.logsumexp(topk_val, dim=-1, keepdim=True)
        log_probs_topk = topk_val - logsumexp_topk                # renormalize over top-K
        probs_topk = torch.exp(log_probs_topk)
        entropy_approx = -(probs_topk * log_probs_topk).sum(dim=-1)   # [N]
        entropy_term = entropy_approx / float(entropy_norm)
        adaptive_weight = torch.pow(entropy_term, alpha)              # [N]

    weighted_losses = valid_losses * adaptive_weight              # [N]

    if num_items_in_batch is not None:
        total_loss = weighted_losses.sum()
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(total_loss.device)
        loss = total_loss / num_items_in_batch
    else:
        loss = weighted_losses.mean()
    loss = loss.to(logits.dtype) if logits.dtype != torch.float32 else loss

    if not return_diagnostics:
        return loss

    with torch.no_grad():
        w = adaptive_weight
        diag: Dict[str, float] = {
            "eaft/loss":            weighted_losses.mean().item(),
            "eaft/ce_loss":         valid_losses.mean().item(),
            "eaft/weight_mean":     w.mean().item(),
            "eaft/weight_min":      w.min().item(),
            "eaft/weight_max":      w.max().item(),
            "eaft/entropy_mean":    entropy_approx.mean().item(),
            "eaft/entropy_norm_mean": entropy_term.mean().item(),
            # fraction of "confident" tokens that get heavily down-weighted
            "eaft/weight_lt_0.2_ratio": (w < 0.2).float().mean().item(),
            "eaft/weight_gt_0.8_ratio": (w > 0.8).float().mean().item(),
            "eaft/alpha":           float(alpha),
            "eaft/K":               float(K),
        }
    return loss, diag


# --------------------------------------------------------------------------------------
# Standalone diagnostics (no grad)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def eaft_diagnostics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
    k: int = 20,
    entropy_norm: float = 3.0,
    ignore_index: int = -100,
) -> Dict[str, float]:
    """Standalone diagnostics path (no grad). Mirrors the loss math; dict only."""
    logits = logits.float()
    V = logits.size(-1)
    K = int(min(k, V))
    shift_logits = logits[..., :-1, :].contiguous().view(-1, V)
    shift_labels = labels[..., 1:].contiguous().view(-1).to(shift_logits.device)

    per_token_loss = F.cross_entropy(
        shift_logits, shift_labels, ignore_index=ignore_index, reduction="none"
    )
    valid_mask = shift_labels != ignore_index
    if not valid_mask.any():
        return {}
    valid_losses = per_token_loss[valid_mask]

    source = shift_logits[valid_mask]
    topk_val, _ = torch.topk(source, k=K, dim=-1)
    logsumexp_topk = torch.logsumexp(topk_val, dim=-1, keepdim=True)
    log_probs_topk = topk_val - logsumexp_topk
    probs_topk = torch.exp(log_probs_topk)
    entropy_approx = -(probs_topk * log_probs_topk).sum(dim=-1)
    entropy_term = entropy_approx / float(entropy_norm)
    w = torch.pow(entropy_term, alpha)
    weighted_losses = valid_losses * w

    return {
        "eaft/loss":            weighted_losses.mean().item(),
        "eaft/ce_loss":         valid_losses.mean().item(),
        "eaft/weight_mean":     w.mean().item(),
        "eaft/weight_min":      w.min().item(),
        "eaft/weight_max":      w.max().item(),
        "eaft/entropy_mean":    entropy_approx.mean().item(),
        "eaft/entropy_norm_mean": entropy_term.mean().item(),
        "eaft/weight_lt_0.2_ratio": (w < 0.2).float().mean().item(),
        "eaft/weight_gt_0.8_ratio": (w > 0.8).float().mean().item(),
        "eaft/alpha":           float(alpha),
        "eaft/K":               float(K),
    }
