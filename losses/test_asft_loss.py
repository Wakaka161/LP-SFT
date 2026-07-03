"""
Unit tests for asft_loss.py.

The first test asserts numerical equivalence to a verbatim copy of the
ASFT (mode=='asft') branch from zhuchichi56/ASFT/train_v2.py
(EnhancedTrainer.compute_loss). Both implementations are run on the same random
inputs and the resulting scalar losses are required to be torch.allclose.

Run:
    python losses/test_asft_loss.py
or:
    python -m pytest losses/test_asft_loss.py -v
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

# allow `python test_asft_loss.py` from RenyiSFT/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from asft_loss import asft_loss, asft_diagnostics  # noqa: E402

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Verbatim copy of the ASFT-official asft branch (only the loss math; the
# trainer plumbing around it is stripped).  Source:
#   https://github.com/zhuchichi56/ASFT/blob/main/train_v2.py#L96-L122
# ---------------------------------------------------------------------------
def _asft_official(
    logits: torch.Tensor,      # [B, L, V]
    labels: torch.Tensor,      # [B, L]
    ref_logits: torch.Tensor,  # [B, L, V]
    kl_weight: float,
    ignore_index: int = -100,
) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_ref_logits = ref_logits[..., :-1, :].contiguous()
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    shift_ref_logits = shift_ref_logits.view(-1, shift_ref_logits.size(-1))

    valid_mask = shift_labels != ignore_index
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=shift_logits.device, requires_grad=True)

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    safe_labels = shift_labels.clamp(min=0, max=shift_logits.size(-1) - 1)
    token_losses = loss_fct(shift_logits, safe_labels)

    probs = torch.softmax(shift_logits, dim=-1)
    valid_labels = safe_labels
    weights = probs.gather(1, valid_labels.unsqueeze(-1)).squeeze(-1).detach()
    dft_losses = token_losses * weights

    with torch.no_grad():
        ref = shift_ref_logits
    kl_div = F.kl_div(
        F.log_softmax(shift_logits, dim=-1),
        F.softmax(ref, dim=-1),
        reduction="none",
    ).sum(dim=-1)

    weighted_losses = dft_losses + kl_weight * kl_div
    return weighted_losses[valid_mask].sum() / valid_mask.sum()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def _make_random_batch(B=2, L=8, V=37, ignore_some=True, seed=0):
    torch.manual_seed(seed)
    logits = torch.randn(B, L, V, requires_grad=True)
    ref_logits = torch.randn(B, L, V)  # frozen, no grad
    labels = torch.randint(0, V, (B, L))
    if ignore_some:
        # mark the first 2 positions of every row as ignore (prompt)
        labels[:, :2] = IGNORE_INDEX
    return logits, labels, ref_logits


def test_numerical_equivalence_to_official():
    logits, labels, ref_logits = _make_random_batch(seed=42)
    kl_weight = 0.03

    loss_ours = asft_loss(
        logits=logits,
        labels=labels,
        ref_logits=ref_logits,
        kl_weight=kl_weight,
        reduction="mean_over_valid",
    )
    loss_off = _asft_official(
        logits=logits, labels=labels, ref_logits=ref_logits, kl_weight=kl_weight
    )
    assert torch.allclose(loss_ours, loss_off, atol=1e-6, rtol=1e-6), (
        f"ASFT mismatch vs official: ours={loss_ours.item()} off={loss_off.item()}"
    )


def test_numerical_equivalence_sweep_kl_weight():
    logits, labels, ref_logits = _make_random_batch(seed=1)
    for kl in [0.0, 0.01, 0.03, 0.1, 0.5, 1.0]:
        a = asft_loss(
            logits=logits, labels=labels, ref_logits=ref_logits,
            kl_weight=kl, reduction="mean_over_valid",
        )
        b = _asft_official(logits, labels, ref_logits, kl_weight=kl)
        assert torch.allclose(a, b, atol=1e-6), f"mismatch at kl={kl}: {a} vs {b}"


def test_gradient_flows_through_student_only():
    logits, labels, ref_logits = _make_random_batch(seed=7)
    # ref must not require grad (we assume caller computes it under no_grad)
    assert not ref_logits.requires_grad
    loss = asft_loss(
        logits=logits, labels=labels, ref_logits=ref_logits,
        kl_weight=0.05, reduction="mean_over_valid",
    )
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    # sanity: there is some non-zero gradient on supervised positions
    assert logits.grad.abs().sum() > 0


def test_kl_weight_zero_reduces_to_dft():
    """When kl_weight=0, ASFT reduces to DFT (and is independent of ref_logits)."""
    logits, labels, ref_logits = _make_random_batch(seed=3)
    loss_asft_no_kl = asft_loss(
        logits=logits, labels=labels, ref_logits=ref_logits,
        kl_weight=0.0, reduction="mean_over_valid",
    )

    shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
    shift_labels = labels[..., 1:].contiguous().view(-1)
    mask = shift_labels != IGNORE_INDEX
    safe = shift_labels.clamp(min=0, max=logits.size(-1) - 1)
    ce = F.cross_entropy(shift_logits, safe, reduction="none")
    with torch.no_grad():
        p = F.softmax(shift_logits, dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    dft = (ce * p)[mask].mean()
    assert torch.allclose(loss_asft_no_kl, dft, atol=1e-6)


def test_diagnostics_dict_keys():
    logits, labels, ref_logits = _make_random_batch(seed=99)
    loss, diag = asft_loss(
        logits=logits, labels=labels, ref_logits=ref_logits,
        kl_weight=0.03, reduction="mean_over_valid", return_diagnostics=True,
    )
    expected = {
        "asft/loss", "asft/dft_loss", "asft/kl", "asft/kl_weighted",
        "asft/ce_loss", "asft/dft_weight_mean", "asft/dft_weight_min",
        "asft/dft_weight_max", "asft/kl_weight",
    }
    assert expected.issubset(set(diag.keys()))
    # standalone diagnostics path returns the same scalars within fp tolerance
    diag2 = asft_diagnostics(
        logits.detach(), labels, ref_logits, kl_weight=0.03
    )
    assert math.isclose(diag["asft/dft_loss"], diag2["asft/dft_loss"], rel_tol=1e-5)
    assert math.isclose(diag["asft/kl"], diag2["asft/kl"], rel_tol=1e-5)


def test_all_ignore_index_returns_zero_grad_safe():
    """Edge case: no valid tokens in a microbatch should return a 0 scalar with
    requires_grad=True so HF Trainer's .backward() doesn't crash."""
    logits = torch.randn(1, 4, 17, requires_grad=True)
    ref_logits = torch.randn(1, 4, 17)
    labels = torch.full((1, 4), IGNORE_INDEX, dtype=torch.long)
    loss = asft_loss(
        logits=logits, labels=labels, ref_logits=ref_logits, kl_weight=0.03,
    )
    assert loss.item() == 0.0
    assert loss.requires_grad


def test_sum_div_num_items_matches_mean_when_num_items_equals_valid():
    """The two reductions agree when num_items_in_batch == #valid tokens."""
    logits, labels, ref_logits = _make_random_batch(seed=5)
    n_valid = (labels[..., 1:] != IGNORE_INDEX).sum().to(logits.dtype)
    loss_mean = asft_loss(
        logits=logits, labels=labels, ref_logits=ref_logits,
        kl_weight=0.03, reduction="mean_over_valid",
    )
    loss_sum = asft_loss(
        logits=logits, labels=labels, ref_logits=ref_logits,
        kl_weight=0.03, reduction="sum_div_num_items",
        num_items_in_batch=n_valid,
    )
    assert torch.allclose(loss_mean, loss_sum, atol=1e-6)


if __name__ == "__main__":
    # Manual driver if pytest is not available
    tests = [
        test_numerical_equivalence_to_official,
        test_numerical_equivalence_sweep_kl_weight,
        test_gradient_flows_through_student_only,
        test_kl_weight_zero_reduces_to_dft,
        test_diagnostics_dict_keys,
        test_all_ignore_index_returns_zero_grad_safe,
        test_sum_div_num_items_matches_mean_when_num_items_equals_valid,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  OK   {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERR  {t.__name__}: {type(e).__name__}: {e}")
    if failed:
        sys.exit(1)
    print(f"\nAll {len(tests)} asft_loss tests passed.")
