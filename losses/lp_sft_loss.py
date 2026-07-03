"""
LP-SFT (lp_sft) — precomputed LP-SFT loss.

Per-token loss:

    L_t = L_CE_t  +  mu_t * L_set_t

    L_CE_t  = -log p_theta(y_t)                                    # standard CE on full vocab
    L_set_t = H(q_ref^S, p_theta^S)
            = - sum_{v in S_t'} q_ref^S(v) * log p_theta^S(v)      # "soft-label CE" inside S_t'

    mu_t        = mu, mu * R_t, mu * R_t^2, or mu * R_t^3 depending on r_weight
    S_t'        = S_t union {y_t}  (set_label_mode=union_label, default)
                or S_t \\ {y_t}    (set_label_mode=minus_label)
    p_theta^S(v) = softmax over student logits restricted to S_t'
    q_ref^S(v)  = softmax over reference logits restricted to S_t', with temperature tau:
                  q_ref^S(v) = exp(z_ref(v)/tau) / sum_{u in S_t'} exp(z_ref(u)/tau)

This loss is NOT vanilla label smoothing: the soft target is *the reference model's local
distribution*, not a fixed (1-eps, eps/(K-1), ...) tuple.

================================================================================
RATIONALE FOR LOCAL NORMALISATION (i.e. softmax restricted to S_t')
================================================================================

If we instead used the reference's *full-vocab* probabilities as soft labels
(- sum_v q_ref(v) log p_theta(v)), the second term would actively pull mass
toward S_t' tokens *globally*, fighting CE for y_t mass and entangling
"alternative-aware regularisation" with "vocab-level recalibration".

Local normalisation isolates the *relative shape* inside S_t' from the absolute
mass on S_t'. The first term (full CE) is responsible for absolute y_t learning;
the second term only adjusts the *relative ratio* of y_t vs alternatives within
the local set, copying the reference's local geometry.

================================================================================
S_t' CONSTRUCTION (vectorised, no per-sample Python loop)
================================================================================

For each valid token t we have precomputed:
    S_ids[t]:           top-K_save ref ids (sorted desc by ref prob), padded with 0 beyond k_t
    S_ref_logits[t]:    ref logits at S_ids[t] (sorted desc by ref logit)
    k[t]:               actual set size in [1, K_save]
    label_ref_logit[t]: ref logit at the true label y_t

We construct an *extended* buffer of width K_save + 1:
    ext_ids[t]         = [S_ids[t, :K_save],         label_id]                 # [K_save+1]
    ext_ref_logits[t]  = [S_ref_logits[t, :K_save],  label_ref_logit]          # [K_save+1]

with a per-row valid mask:
    valid[t, :K_save]  = (arange(K_save) < k[t])                               # first k slots are real
    valid[t, K_save]   = NOT label_in_set[t]                                   # last slot only if label∉S_t

union_label (default):
    - if y_t already in S_t (label_in_set=True):  effective set = S_t,            |S_t'| = k[t]
    - if y_t NOT  in S_t (label_in_set=False):    effective set = S_t ∪ {y_t},   |S_t'| = k[t] + 1

minus_label (alternative-only alignment):
    - S_t' = S_t \\ {y_t}; label is never included in the in-set term
    - if y_t in S_t:  |S_t'| = k[t] - 1; else |S_t'| = k[t]

minus_anchor (alternative alignment, y_t kept as a detached anchor):
    - U_t = S_t ∪ {y_t} (same buffer as union_label) but the *student* logit at
      the y_t slot is stop-grad, so the in-set term shapes only the alternatives
      while y_t stays a fixed reference point. Avoids the |S_t'|==1 degeneracy of
      minus_label without any extra hyper-parameter. See DEGENERATE CASES below.

================================================================================
DEGENERATE CASES
================================================================================

  |S_t'| <= 1:
      The local distribution has no other entry to compare with y_t. We set
      L_set_t = 0 explicitly. (Math also gives 0 because log_softmax over a
      single live slot is 0 and ref_probs at that slot is 1, so 1*0=0.)

  minus_anchor (anchored minus_label -- avoids the |S_t'|==1 degeneracy):
      Pure minus_label drops y_t entirely, so a token whose only alternative is a
      single v != y_t collapses to {v} and contributes 0. minus_anchor instead
      KEEPS y_t in the local set (denominator) exactly like union_label, but
      DETACHES the student logit at y_t so the in-set term never trains y_t:

          U_t = S_t ∪ {y_t}
          p_theta^U(v) = softmax_U([..., stop_grad(z_theta(y_t))])
          q_ref^U      = softmax_U(z_ref/tau)
          L_set_t      = - sum_{u in U_t} q_ref^U(u) log p_theta^U(u)

      Because z_theta(y_t) is stop-grad, the gradient on each alternative a is the
      clean CE gradient p_theta^U(a) - q_ref^U(a) and *nothing* flows into y_t (CE
      term 1 alone owns y_t). A lone alternative now compares against the real,
      fixed y_t anchor instead of degenerating. No extra hyper-parameter. |S_t'|==1
      can still only happen here when S_t = {y_t} (no alternative at all), where 0
      is the correct answer.

  ref_probs at invalid slots is 0 (masked softmax), and student log_probs at
  invalid slots is -inf, so we mask the contribution to 0 explicitly to avoid
  0 * -inf = NaN.

================================================================================
GRADIENT FLOW
================================================================================

q_ref^S is detached -- gradients only flow into student logits (both via the
full-vocab CE and via the in-set log_softmax). Reference logits are pure data.

================================================================================
CACHE CONTRACT
================================================================================

We extend the plateau cache with two new fields (produced by precompute_R.py
with --save_ref_logits + --save_topk):

    ref_topk_logits  [B, L, K_save]  float, S_ref_logits projected to label-aligned positions
    ref_label_logit  [B, L]          float, label_ref_logit at each label-aligned position

Both are zero (or any padding value) at -100 positions; they are never read
because the corresponding label is masked out.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Main loss
# --------------------------------------------------------------------------------------
def lp_sft_loss(
    logits: torch.Tensor,                          # [B, L, V]
    labels: torch.Tensor,                          # [B, L]
    plateau_k: torch.Tensor,                       # [B, L]               long, in [0, K_save]
    plateau_topk_ids: torch.Tensor,                # [B, L, K_save]       long
    ref_topk_logits: torch.Tensor,                 # [B, L, K_save]       float
    ref_label_logit: torch.Tensor,                 # [B, L]               float
    renyi_R: Optional[torch.Tensor] = None,         # [B, L]               float, optional R_t
    num_items_in_batch: Optional[torch.Tensor] = None,
    mu: float = 0.03,
    tau: float = 1.0,
    r_weight: str = "none",                         # none | R | R2 | R3
    set_label_mode: str = "union_label",            # union_label | minus_label | minus_anchor
    K_save: int = 10,
    ignore_index: int = -100,
    return_diagnostics: bool = False,
    loss_mode: str = "additive",                    # additive | r_interp
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    """Compute LP-SFT (lp_sft) per-token loss.

    Returns a scalar tensor (mean of per-token L_t, or sum / num_items_in_batch
    if provided -- HF Trainer convention).

    If `return_diagnostics=True`, also returns a dict of scalar metrics
    prefixed `lp_sft/...`.
    """
    if plateau_topk_ids.size(-1) != K_save:
        raise ValueError(
            f"plateau_topk_ids last dim ({plateau_topk_ids.size(-1)}) "
            f"!= K_save ({K_save}); check collator and CLI args."
        )
    if ref_topk_logits.size(-1) != K_save:
        raise ValueError(
            f"ref_topk_logits last dim ({ref_topk_logits.size(-1)}) "
            f"!= K_save ({K_save}); check collator and CLI args."
        )
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    r_weight = str(r_weight).lower()
    if r_weight not in {"none", "r", "r2", "r3"}:
        raise ValueError(f"r_weight must be one of none|R|R2|R3, got {r_weight}")
    if r_weight != "none" and renyi_R is None:
        raise ValueError("r_weight != none requires renyi_R to be provided")
    set_label_mode = str(set_label_mode).lower()
    _VALID_SET_LABEL_MODES = {
        "union_label", "minus_label", "minus_anchor",
        "minusy_min2", "minusy_min2_strict", "minusy_union_fallback",
    }
    if set_label_mode not in _VALID_SET_LABEL_MODES:
        raise ValueError(
            f"set_label_mode must be one of {sorted(_VALID_SET_LABEL_MODES)}, "
            f"got {set_label_mode}"
        )
    loss_mode = str(loss_mode).lower()
    if loss_mode not in {"additive", "r_interp"}:
        raise ValueError(f"loss_mode must be additive or r_interp, got {loss_mode}")
    if loss_mode == "r_interp":
        if set_label_mode != "union_label":
            raise ValueError("loss_mode='r_interp' requires set_label_mode='union_label'")
        if renyi_R is None:
            raise ValueError("loss_mode='r_interp' requires renyi_R")

    # ---- shift for next-token prediction (logits[:, t] predicts labels[:, t+1]) ----
    shift_logits        = logits[..., :-1, :].contiguous()              # [B, L-1, V]
    shift_labels        = labels[..., 1:].contiguous()                  # [B, L-1]
    shift_k             = plateau_k[..., 1:].contiguous()               # [B, L-1]
    shift_topk_ids      = plateau_topk_ids[..., 1:, :].contiguous()     # [B, L-1, K_save]
    shift_ref_topk_lg   = ref_topk_logits[..., 1:, :].contiguous()      # [B, L-1, K_save]
    shift_ref_label_lg  = ref_label_logit[..., 1:].contiguous()         # [B, L-1]
    shift_R             = renyi_R[..., 1:].contiguous() if renyi_R is not None else None

    mask = shift_labels != ignore_index                                  # [B, L-1]
    if mask.sum() == 0:
        zero = torch.tensor(
            0.0, device=logits.device, dtype=logits.dtype, requires_grad=True
        )
        if return_diagnostics:
            return zero, {}
        return zero

    # ---- gather valid positions: [N, ...] ----
    logits_v        = shift_logits[mask]                                 # [N, V]
    labels_v        = shift_labels[mask]                                 # [N]
    k_v             = shift_k[mask]                                      # [N]
    topk_ids_v      = shift_topk_ids[mask]                               # [N, K_save]
    ref_topk_lg_v   = shift_ref_topk_lg[mask]                            # [N, K_save]
    ref_label_lg_v  = shift_ref_label_lg[mask]                           # [N]
    R_v             = shift_R[mask].float() if shift_R is not None else None

    N = labels_v.size(0)
    device = logits.device

    # ---- standard full-vocab CE (term 1) ----
    logits_v_f = logits_v.float()
    log_probs_full = F.log_softmax(logits_v_f, dim=-1)                   # [N, V]
    real_logp = log_probs_full.gather(
        -1, labels_v.unsqueeze(-1)
    ).squeeze(-1)                                                         # [N]
    ce_per_token = -real_logp                                             # [N]

    # ---- build S_t' mask ----
    pos = torch.arange(K_save, device=device).unsqueeze(0)               # [1, K_save]
    valid_top = pos < k_v.clamp(min=0).unsqueeze(-1)                     # [N, K_save]
    match = (topk_ids_v == labels_v.unsqueeze(-1)) & valid_top           # [N, K_save]
    label_in_set = match.any(dim=-1)                                     # [N]

    # For the new K_save-wide modes we track mode-specific extras for diagnostics.
    _min2_needs_expansion: Optional[torch.Tensor] = None
    _min2_extra_valid: Optional[torch.Tensor] = None
    _unionfb_is_fallback: Optional[torch.Tensor] = None

    if set_label_mode == "minus_label":
        # S_t' = S_t \ {y_t}
        ext_ids = topk_ids_v                                             # [N, K_save]
        ext_ref_logits = ref_topk_lg_v.float()                           # [N, K_save]
        valid_ext = valid_top & (topk_ids_v != labels_v.unsqueeze(-1))   # [N, K_save]

    elif set_label_mode == "minusy_min2":
        # S_t' = S_t \ {y_t}, but if |S_t'| < 2, extend beyond k_t using the saved
        # K_save buffer (still excluding y_t) until we have at least 2 non-label tokens.
        # NOTE: this also expands positions where S_t = {y_t} (k=1, label-only set),
        # borrowing rank-2/3 tokens from K_save even when the reference assigns them
        # near-zero probability. For a stricter version that skips those positions,
        # use set_label_mode='minusY_min2_strict'.
        valid_nonlabel = topk_ids_v != labels_v.unsqueeze(-1)            # [N, K_save] – no k_v gate
        valid_minus = valid_top & valid_nonlabel                         # [N, K_save]
        set_size_minus = valid_minus.sum(-1, keepdim=True)               # [N, 1]
        needs_expansion = set_size_minus < 2                             # [N, 1]
        # Slots beyond k_v (within K_save) that are non-label: used for fallback
        beyond_kv = ~valid_top                                           # [N, K_save]
        extra_valid = needs_expansion & beyond_kv & valid_nonlabel       # [N, K_save]

        ext_ids = topk_ids_v                                             # [N, K_save]
        ext_ref_logits = ref_topk_lg_v.float()                           # [N, K_save]
        valid_ext = valid_minus | extra_valid                            # [N, K_save]

        _min2_needs_expansion = needs_expansion.squeeze(-1)              # [N]
        _min2_extra_valid = extra_valid                                  # [N, K_save]

    elif set_label_mode == "minusy_min2_strict":
        # S_t' = S_t \ {y_t}, extended to >= 2 non-label tokens ONLY when there is
        # already exactly 1 genuine non-label token in S_t (i.e. set_size_minus == 1).
        # When S_t = {y_t} (set_size_minus == 0), L_lp_sft = 0, same as minus_label.
        # This avoids borrowing near-zero-probability rank-2/3 tokens for label-only sets.
        valid_nonlabel = topk_ids_v != labels_v.unsqueeze(-1)            # [N, K_save]
        valid_minus = valid_top & valid_nonlabel                         # [N, K_save]
        set_size_minus = valid_minus.sum(-1, keepdim=True)               # [N, 1]
        needs_expansion = set_size_minus == 1                            # [N, 1] — only when 1 real alt
        beyond_kv = ~valid_top                                           # [N, K_save]
        extra_valid = needs_expansion & beyond_kv & valid_nonlabel       # [N, K_save]

        ext_ids = topk_ids_v                                             # [N, K_save]
        ext_ref_logits = ref_topk_lg_v.float()                           # [N, K_save]
        valid_ext = valid_minus | extra_valid                            # [N, K_save]

        _min2_needs_expansion = needs_expansion.squeeze(-1)              # [N]
        _min2_extra_valid = extra_valid                                  # [N, K_save]

    elif set_label_mode == "minusy_union_fallback":
        # Default: S_t' = S_t \ {y_t}  (same as minus_label)
        # Fallback: when |S_t| == 2 AND y_t ∈ S_t, use S_t' = S_t (keep y_t in set)
        # to avoid the common {y_t, v} → degenerate single-token situation.
        is_fallback = (k_v == 2) & label_in_set                         # [N]
        valid_minus = valid_top & (topk_ids_v != labels_v.unsqueeze(-1)) # [N, K_save]
        valid_union = valid_top                                          # [N, K_save]
        valid_ext = torch.where(
            is_fallback.unsqueeze(-1), valid_union, valid_minus
        )                                                                # [N, K_save]

        ext_ids = topk_ids_v                                             # [N, K_save]
        ext_ref_logits = ref_topk_lg_v.float()                           # [N, K_save]
        _unionfb_is_fallback = is_fallback                               # [N]

    else:
        # union_label and minus_anchor share the same buffer (label kept in S_t').
        label_col_ids = labels_v.unsqueeze(-1)                           # [N, 1]
        label_col_logit = ref_label_lg_v.float().unsqueeze(-1)           # [N, 1]
        ext_ids = torch.cat([topk_ids_v, label_col_ids], dim=-1)       # [N, K_save+1]
        ext_ref_logits = torch.cat(
            [ref_topk_lg_v.float(), label_col_logit], dim=-1
        )                                                                # [N, K_save+1]
        valid_last = (~label_in_set).unsqueeze(-1)                       # [N, 1]
        valid_ext = torch.cat([valid_top, valid_last], dim=-1)           # [N, K_save+1]

    set_size = valid_ext.sum(dim=-1).long()                              # [N]   |S_t'|

    # ---- gather student logits at ext_ids ----
    student_set_logits = logits_v_f.gather(-1, ext_ids)                   # [N, K_save+1]

    NEG_INF = float("-inf")

    # ---- minus_anchor only: stop-grad y_t logit so CE term owns y_t exclusively.
    if set_label_mode == "minus_anchor":
        is_label_pos = (ext_ids == labels_v.unsqueeze(-1)) & valid_ext   # [N, W]
        student_set_logits = torch.where(
            is_label_pos, student_set_logits.detach(), student_set_logits
        )

    # ---- masked log_softmax for student over S_t' ----
    student_set_logits_masked = student_set_logits.masked_fill(~valid_ext, NEG_INF)
    student_log_probs_S = F.log_softmax(student_set_logits_masked, dim=-1)  # [N, K_save+1]

    # ---- masked softmax for ref over S_t' (with temperature) ----
    ref_set_scaled        = ext_ref_logits / float(tau)
    ref_set_scaled_masked = ref_set_scaled.masked_fill(~valid_ext, NEG_INF)
    ref_probs_S           = F.softmax(ref_set_scaled_masked, dim=-1).detach()  # [N, K_save+1]

    # ---- in-set cross entropy: H(q_ref^S, p_theta^S) ----
    # Avoid 0 * -inf NaN at masked-out slots by zeroing student log_probs there.
    student_log_probs_S_safe = torch.where(
        valid_ext, student_log_probs_S, torch.zeros_like(student_log_probs_S)
    )
    set_per_token = -(ref_probs_S * student_log_probs_S_safe).sum(dim=-1)  # [N]

    # ---- |S_t'| <= 1: degenerate, force exactly 0 (math gives 0 anyway, this is for safety) ----
    set_per_token = torch.where(
        set_size > 1, set_per_token, torch.zeros_like(set_per_token)
    )                                                                      # [N]

    # ---- combine ----
    if loss_mode == "r_interp":
        R_safe = R_v.to(set_per_token.dtype).clamp(0.0, 1.0)
        mu_eff = R_safe  # stored for diagnostics consistency
        loss_per_token = (1.0 - R_safe) * ce_per_token + R_safe * set_per_token
    elif r_weight == "none":
        mu_eff = torch.full_like(set_per_token, float(mu))
        loss_per_token = ce_per_token + mu_eff * set_per_token
    else:
        R_safe = R_v.to(set_per_token.dtype).clamp(0.0, 1.0)
        if r_weight == "r":
            mu_eff = float(mu) * R_safe
        elif r_weight == "r2":
            mu_eff = float(mu) * R_safe.pow(2)
        else:
            mu_eff = float(mu) * R_safe.pow(3)
        loss_per_token = ce_per_token + mu_eff * set_per_token             # [N], fp32

    # ---- batch reduction (HF Trainer convention) ----
    if num_items_in_batch is not None:
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss_per_token.device)
        loss = loss_per_token.sum() / num_items_in_batch
    else:
        loss = loss_per_token.mean()
    loss = loss.to(logits.dtype)

    if not return_diagnostics:
        return loss

    # ---- diagnostics ----
    with torch.no_grad():
        # locate y_t inside the extended buffer (always at slot K_save when label_in_set=False;
        # else somewhere in [0, k_v) where match is True).
        # We compute q_ref^S(y_t) and p_theta^S(y_t) via gather on ext_ids == labels_v.
        is_label_slot = (ext_ids == labels_v.unsqueeze(-1)) & valid_ext         # [N, K_save+1]
        q_ref_at_label    = (ref_probs_S * is_label_slot.float()).sum(-1)       # [N]
        # student probs in S' at label: convert log_probs (at valid only) to probs
        student_probs_S = student_log_probs_S_safe.exp() * valid_ext.float()
        p_theta_at_label  = (student_probs_S * is_label_slot.float()).sum(-1)   # [N]

        # KL(q_ref^S || p_theta^S) = sum q * (log q - log p), restricted to valid
        log_q = torch.where(
            valid_ext & (ref_probs_S > 0),
            ref_probs_S.clamp_min(1e-30).log(),
            torch.zeros_like(ref_probs_S),
        )
        kl_terms = ref_probs_S * (log_q - student_log_probs_S_safe)              # [N, K_save+1]
        kl_per_token = (kl_terms * valid_ext.float()).sum(-1)                    # [N]

        set_size_f = set_size.float()
        _MODE_ENC = {
            "union_label": 0.0, "minus_label": 1.0, "minus_anchor": 2.0,
            "minusy_min2": 3.0, "minusy_union_fallback": 4.0,
            "minusy_min2_strict": 5.0,
        }
        diag: Dict[str, float] = {
            "lp_sft/loss":                 loss_per_token.mean().item(),
            "lp_sft/ce_loss":              ce_per_token.mean().item(),
            "lp_sft/set_loss":             set_per_token.mean().item(),
            "lp_sft/weighted_set_loss":    (mu_eff * set_per_token).mean().item(),
            "lp_sft/mu_eff_mean":          mu_eff.mean().item(),
            "lp_sft/mu_eff_max":           mu_eff.max().item(),
            "lp_sft/set_size_mean":        set_size_f.mean().item(),
            "lp_sft/label_in_set_ratio":   label_in_set.float().mean().item(),
            "lp_sft/set_size_eq_1_ratio":  (set_size == 1).float().mean().item(),
            "lp_sft/set_size_lt2_ratio":   (set_size < 2).float().mean().item(),
            "lp_sft/active_lp_sft_ratio":   (set_size > 1).float().mean().item(),
            "lp_sft/set_size_2to6_ratio":  ((set_size >= 2) & (set_size <= 6)).float().mean().item(),
            "lp_sft/set_size_gt6_ratio":   (set_size > 6).float().mean().item(),
            "lp_sft/q_ref_at_label_mean":  q_ref_at_label.mean().item(),
            "lp_sft/p_theta_at_label_mean": p_theta_at_label.mean().item(),
            "lp_sft/kl_q_ref_to_p_mean":   kl_per_token.mean().item(),
            "lp_sft/k_mean":               k_v.float().mean().item(),
            "lp_sft/mu":                   float(mu),
            "lp_sft/tau":                  float(tau),
            "lp_sft/r_weight":             {"none": 0.0, "r": 1.0, "r2": 2.0, "r3": 3.0}[r_weight],
            "lp_sft/set_label_mode":       _MODE_ENC[set_label_mode],
        }
        # Mode-specific diagnostics
        if _min2_needs_expansion is not None:
            added = _min2_extra_valid.sum(-1).float()               # [N]
            fallback_mask = _min2_needs_expansion                   # [N] bool
            diag["lp_sft/min2_fallback_ratio"]  = fallback_mask.float().mean().item()
            diag["lp_sft/min2_success_ratio"]   = (
                (set_size >= 2) & fallback_mask
            ).float().sum().item() / (fallback_mask.float().sum().item() + 1e-9)
            diag["lp_sft/mean_added_tokens"]    = (
                added[fallback_mask].mean().item() if fallback_mask.any() else 0.0
            )
        if _unionfb_is_fallback is not None:
            diag["lp_sft/union_fallback_ratio"] = _unionfb_is_fallback.float().mean().item()
        if R_v is not None:
            diag["lp_sft/R_mean"] = R_v.float().mean().item()
            diag["lp_sft/R2_mean"] = R_v.float().pow(2).mean().item()
            diag["lp_sft/R3_mean"] = R_v.float().pow(3).mean().item()
        if loss_mode == "r_interp":
            diag["lp_sft/loss_mode"] = 1.0  # 1.0 = r_interp
            diag["lp_sft/mean_ce_weight"] = (1.0 - R_v.float()).mean().item()
            diag["lp_sft/mean_set_weight"] = R_v.float().mean().item()
        else:
            diag["lp_sft/loss_mode"] = 0.0  # 0.0 = additive
    return loss, diag


# --------------------------------------------------------------------------------------
# Standalone diagnostics (no grad) -- mirrors loss math
# --------------------------------------------------------------------------------------
@torch.no_grad()
def lp_sft_loss_diagnostics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    plateau_k: torch.Tensor,
    plateau_topk_ids: torch.Tensor,
    ref_topk_logits: torch.Tensor,
    ref_label_logit: torch.Tensor,
    renyi_R: Optional[torch.Tensor] = None,
    mu: float = 0.03,
    tau: float = 1.0,
    r_weight: str = "none",
    set_label_mode: str = "union_label",
    K_save: int = 10,
    ignore_index: int = -100,
    loss_mode: str = "additive",
) -> Dict[str, float]:
    """Standalone diagnostics path (no grad). Mirrors the loss math; returns dict only.

    Keys are prefixed `lp_sft/` to distinguish from other loss diagnostics.
    """
    shift_logits        = logits[..., :-1, :].contiguous()
    shift_labels        = labels[..., 1:].contiguous()
    shift_k             = plateau_k[..., 1:].contiguous()
    shift_topk_ids      = plateau_topk_ids[..., 1:, :].contiguous()
    shift_ref_topk_lg   = ref_topk_logits[..., 1:, :].contiguous()
    shift_ref_label_lg  = ref_label_logit[..., 1:].contiguous()
    shift_R             = renyi_R[..., 1:].contiguous() if renyi_R is not None else None

    mask = shift_labels != ignore_index
    if mask.sum() == 0:
        return {}

    logits_v        = shift_logits[mask]
    labels_v        = shift_labels[mask]
    k_v             = shift_k[mask]
    topk_ids_v      = shift_topk_ids[mask]
    ref_topk_lg_v   = shift_ref_topk_lg[mask]
    ref_label_lg_v  = shift_ref_label_lg[mask]
    R_v             = shift_R[mask].float() if shift_R is not None else None

    N = labels_v.size(0)
    device = labels_v.device
    logits_v_f = logits_v.float()
    log_probs_full = F.log_softmax(logits_v_f, dim=-1)
    real_logp = log_probs_full.gather(-1, labels_v.unsqueeze(-1)).squeeze(-1)
    ce_per_token = -real_logp

    set_label_mode = str(set_label_mode).lower()
    _VALID_SET_LABEL_MODES = {
        "union_label", "minus_label", "minus_anchor",
        "minusy_min2", "minusy_min2_strict", "minusy_union_fallback",
    }
    if set_label_mode not in _VALID_SET_LABEL_MODES:
        raise ValueError(
            f"set_label_mode must be one of {sorted(_VALID_SET_LABEL_MODES)}, "
            f"got {set_label_mode}"
        )

    pos = torch.arange(K_save, device=device).unsqueeze(0)
    valid_top = pos < k_v.clamp(min=0).unsqueeze(-1)
    match = (topk_ids_v == labels_v.unsqueeze(-1)) & valid_top
    label_in_set = match.any(dim=-1)

    _min2_needs_expansion: Optional[torch.Tensor] = None
    _min2_extra_valid: Optional[torch.Tensor] = None
    _unionfb_is_fallback: Optional[torch.Tensor] = None

    if set_label_mode == "minus_label":
        ext_ids = topk_ids_v
        ext_ref_logits = ref_topk_lg_v.float()
        valid_ext = valid_top & (topk_ids_v != labels_v.unsqueeze(-1))

    elif set_label_mode in ("minusy_min2", "minusy_min2_strict"):
        valid_nonlabel = topk_ids_v != labels_v.unsqueeze(-1)
        valid_minus = valid_top & valid_nonlabel
        set_size_minus = valid_minus.sum(-1, keepdim=True)
        if set_label_mode == "minusy_min2":
            needs_expansion = set_size_minus < 2       # expand when 0 or 1 non-label tokens
        else:
            needs_expansion = set_size_minus == 1      # strict: expand only when 1 real alt
        beyond_kv = ~valid_top
        extra_valid = needs_expansion & beyond_kv & valid_nonlabel
        ext_ids = topk_ids_v
        ext_ref_logits = ref_topk_lg_v.float()
        valid_ext = valid_minus | extra_valid
        _min2_needs_expansion = needs_expansion.squeeze(-1)
        _min2_extra_valid = extra_valid

    elif set_label_mode == "minusy_union_fallback":
        is_fallback = (k_v == 2) & label_in_set
        valid_minus = valid_top & (topk_ids_v != labels_v.unsqueeze(-1))
        valid_union = valid_top
        valid_ext = torch.where(is_fallback.unsqueeze(-1), valid_union, valid_minus)
        ext_ids = topk_ids_v
        ext_ref_logits = ref_topk_lg_v.float()
        _unionfb_is_fallback = is_fallback

    else:
        label_col_ids = labels_v.unsqueeze(-1)
        label_col_logit = ref_label_lg_v.float().unsqueeze(-1)
        ext_ids = torch.cat([topk_ids_v, label_col_ids], dim=-1)
        ext_ref_logits = torch.cat([ref_topk_lg_v.float(), label_col_logit], dim=-1)
        valid_last = (~label_in_set).unsqueeze(-1)
        valid_ext = torch.cat([valid_top, valid_last], dim=-1)
    set_size = valid_ext.sum(dim=-1).long()

    student_set_logits = logits_v_f.gather(-1, ext_ids)
    NEG_INF = float("-inf")

    if set_label_mode == "minus_anchor":
        is_label_pos = (ext_ids == labels_v.unsqueeze(-1)) & valid_ext
        student_set_logits = torch.where(
            is_label_pos, student_set_logits.detach(), student_set_logits
        )

    student_log_probs_S = F.log_softmax(
        student_set_logits.masked_fill(~valid_ext, NEG_INF), dim=-1
    )
    ref_probs_S = F.softmax(
        (ext_ref_logits / float(tau)).masked_fill(~valid_ext, NEG_INF), dim=-1
    )
    student_log_probs_S_safe = torch.where(
        valid_ext, student_log_probs_S, torch.zeros_like(student_log_probs_S)
    )
    set_per_token = -(ref_probs_S * student_log_probs_S_safe).sum(-1)
    set_per_token = torch.where(
        set_size > 1, set_per_token, torch.zeros_like(set_per_token)
    )
    r_weight = str(r_weight).lower()
    if r_weight not in {"none", "r", "r2", "r3"}:
        raise ValueError(f"r_weight must be one of none|R|R2|R3, got {r_weight}")
    loss_mode = str(loss_mode).lower()
    if loss_mode == "r_interp":
        if R_v is None:
            raise ValueError("loss_mode='r_interp' requires renyi_R")
        R_safe = R_v.to(set_per_token.dtype).clamp(0.0, 1.0)
        mu_eff = R_safe
        total_per_token = (1.0 - R_safe) * ce_per_token + R_safe * set_per_token
    elif r_weight == "none":
        mu_eff = torch.full_like(set_per_token, float(mu))
        total_per_token = ce_per_token + mu_eff * set_per_token
    else:
        if R_v is None:
            raise ValueError("r_weight != none requires renyi_R to be provided")
        R_safe = R_v.to(set_per_token.dtype).clamp(0.0, 1.0)
        if r_weight == "r":
            mu_eff = float(mu) * R_safe
        elif r_weight == "r2":
            mu_eff = float(mu) * R_safe.pow(2)
        else:
            mu_eff = float(mu) * R_safe.pow(3)
        total_per_token = ce_per_token + mu_eff * set_per_token

    is_label_slot = (ext_ids == labels_v.unsqueeze(-1)) & valid_ext
    q_ref_at_label = (ref_probs_S * is_label_slot.float()).sum(-1)
    student_probs_S = student_log_probs_S_safe.exp() * valid_ext.float()
    p_theta_at_label = (student_probs_S * is_label_slot.float()).sum(-1)

    log_q = torch.where(
        valid_ext & (ref_probs_S > 0),
        ref_probs_S.clamp_min(1e-30).log(),
        torch.zeros_like(ref_probs_S),
    )
    kl_terms = ref_probs_S * (log_q - student_log_probs_S_safe)
    kl_per_token = (kl_terms * valid_ext.float()).sum(-1)

    set_size_f = set_size.float()
    _MODE_ENC = {
        "union_label": 0.0, "minus_label": 1.0, "minus_anchor": 2.0,
        "minusy_min2": 3.0, "minusy_union_fallback": 4.0,
        "minusy_min2_strict": 5.0,
    }
    out = {
        "lp_sft/loss":                 total_per_token.mean().item(),
        "lp_sft/ce_loss":              ce_per_token.mean().item(),
        "lp_sft/set_loss":             set_per_token.mean().item(),
        "lp_sft/weighted_set_loss":    (mu_eff * set_per_token).mean().item(),
        "lp_sft/mu_eff_mean":          mu_eff.mean().item(),
        "lp_sft/mu_eff_max":           mu_eff.max().item(),
        "lp_sft/set_size_mean":        set_size_f.mean().item(),
        "lp_sft/label_in_set_ratio":   label_in_set.float().mean().item(),
        "lp_sft/set_size_eq_1_ratio":  (set_size == 1).float().mean().item(),
        "lp_sft/set_size_lt2_ratio":   (set_size < 2).float().mean().item(),
        "lp_sft/active_lp_sft_ratio":   (set_size > 1).float().mean().item(),
        "lp_sft/set_size_2to6_ratio":  ((set_size >= 2) & (set_size <= 6)).float().mean().item(),
        "lp_sft/set_size_gt6_ratio":   (set_size > 6).float().mean().item(),
        "lp_sft/q_ref_at_label_mean":  q_ref_at_label.mean().item(),
        "lp_sft/p_theta_at_label_mean": p_theta_at_label.mean().item(),
        "lp_sft/kl_q_ref_to_p_mean":   kl_per_token.mean().item(),
        "lp_sft/k_mean":               k_v.float().mean().item(),
        "lp_sft/mu":                   float(mu),
        "lp_sft/tau":                  float(tau),
        "lp_sft/r_weight":             {"none": 0.0, "r": 1.0, "r2": 2.0, "r3": 3.0}[r_weight],
        "lp_sft/set_label_mode":       _MODE_ENC[set_label_mode],
    }
    if _min2_needs_expansion is not None:
        added = _min2_extra_valid.sum(-1).float()
        fallback_mask = _min2_needs_expansion
        out["lp_sft/min2_fallback_ratio"]  = fallback_mask.float().mean().item()
        out["lp_sft/min2_success_ratio"]   = (
            (set_size >= 2) & fallback_mask
        ).float().sum().item() / (fallback_mask.float().sum().item() + 1e-9)
        out["lp_sft/mean_added_tokens"]    = (
            added[fallback_mask].mean().item() if fallback_mask.any() else 0.0
        )
    if _unionfb_is_fallback is not None:
        out["lp_sft/union_fallback_ratio"] = _unionfb_is_fallback.float().mean().item()
    if R_v is not None:
        out["lp_sft/R_mean"] = R_v.float().mean().item()
        out["lp_sft/R2_mean"] = R_v.float().pow(2).mean().item()
        out["lp_sft/R3_mean"] = R_v.float().pow(3).mean().item()
    if loss_mode == "r_interp":
        out["lp_sft/loss_mode"] = 1.0
        out["lp_sft/mean_ce_weight"] = (1.0 - R_v.float()).mean().item()
        out["lp_sft/mean_set_weight"] = R_v.float().mean().item()
    else:
        out["lp_sft/loss_mode"] = 0.0
    return out
