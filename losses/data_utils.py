"""
Data utilities for Rényi-based SFT (R cache + lp_sft):

  - load_R_cache:                       load precomputed cache jsonl
  - build_R_full:                       project R onto full-length array aligned with labels
  - build_plateau_full:                 project (R, k, topk_ids) onto full-length
  - build_ref_align_full:               project (R, k, topk_ids, topk_logits, label_ref_logit)
  - DataCollatorForSFTWithR:            collator that pads `renyi_R`
  - DataCollatorForSFTWithPlateau:      pads `renyi_R` + `plateau_k` + `plateau_topk_ids`
  - DataCollatorForSFTWithRefAlign:     also pads `ref_topk_logits` + `ref_label_logit`

Storage convention (cache jsonl, one line per sample):

  Plain R cache (legacy, used by older Rényi-CE training):
    {"sample_idx": int, "R": [float, ...], "n_tokens": int}

  Plateau cache (with --save_topk in precompute_R.py):
    {"sample_idx": int,
     "R":        [float, ...],
     "n_tokens": int,
     "k":        [int, ...],          # k_t = clamp(ceil(N2), 1, K_save)
     "topk_ids": [[int x K_save],...]}

  LP-SFT cache (additionally with --save_ref_logits):
    + "topk_logits":     [[float x K_save], ...]   # ref logits at topk_ids (sorted desc)
    + "label_ref_logit": [float, ...]              # ref logit at the true label y_t

  R / k / topk_ids / topk_logits / label_ref_logit lengths all
  == sum(labels[1:] != ignore_index) for that sample.

At runtime, each sample is augmented with placeholders aligned to len(labels):
    renyi_R[i]:                R value at i-th label position; 0.0 elsewhere
    plateau_k[i]:              k_t at i-th label position;     0   elsewhere
    plateau_topk_ids[i, :K]:   ref top-K ids;                  0   elsewhere
    ref_topk_logits[i, :K]:    ref logits at top-K ids;        0.0 elsewhere
    ref_label_logit[i]:        ref logit at y_t;               0.0 elsewhere

Padding values are never read by the loss (corresponding label is -100 → masked out).
"""
import json
from typing import Dict, List, Optional, Tuple

import torch
from transformers import DataCollatorForSeq2Seq


# --------------------------------------------------------------------------------------
# Cache loading
# --------------------------------------------------------------------------------------
def load_R_cache(path: str) -> Dict[int, Dict[str, list]]:
    """Load cache jsonl into dict[sample_idx -> dict with 'R' and optional extras].

    Always returns a dict-of-dict. Keys present:
      - 'R':                List[float]         (always)
      - 'k':                List[int]           (if cache was produced with --save_topk)
      - 'k_n1':             List[int]           (if cache has per-method k; Shannon)
      - 'k_n2':             List[int]           (if cache has per-method k; Collision)
      - 'topk_ids':         List[List[int]]     (if cache was produced with --save_topk)
      - 'topk_logits':      List[List[float]]   (if cache was produced with --save_ref_logits)
      - 'label_ref_logit':  List[float]         (if cache was produced with --save_ref_logits)
    """
    cache: Dict[int, Dict[str, list]] = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            entry: Dict[str, list] = {"R": list(d["R"])}
            if "k" in d:
                entry["k"] = list(d["k"])
            if "k_n1" in d:
                entry["k_n1"] = list(d["k_n1"])
            if "k_n2" in d:
                entry["k_n2"] = list(d["k_n2"])
            if "topk_ids" in d:
                entry["topk_ids"] = [list(row) for row in d["topk_ids"]]
            if "topk_logits" in d:
                entry["topk_logits"] = [list(row) for row in d["topk_logits"]]
            if "label_ref_logit" in d:
                entry["label_ref_logit"] = list(d["label_ref_logit"])
            if "n1_vals" in d:
                entry["n1_vals"] = list(d["n1_vals"])
            if "n2_vals" in d:
                entry["n2_vals"] = list(d["n2_vals"])
            cache[int(d["sample_idx"])] = entry
    return cache


def select_k_for_set_method(
    entry: Dict[str, list],
    set_method: str = "N2",
    k_round_mode: str = "max",
    k_save: int = 10,
    k_threshold: float = 0.0,
) -> list:
    """Return the k list from *entry* that matches *set_method*.

    k_round_mode controls how k is derived:
      'max'          – force k_t = k_save for EVERY token (paper default: fixed top-K).
                       Independent of N1/N2; only needs entry length (via 'R' or topk rows).
      'precomputed'  – use the stored k_n1/k_n2/k field (backward-compat / N-adaptive ablation).
      'ceil'         – recompute from raw n1_vals/n2_vals using ceil.
      'round'        – recompute from raw n1_vals/n2_vals using round (half-to-even).
      'floor'        – recompute from raw n1_vals/n2_vals using floor (≥1).

    k_threshold (float, default 0 = disabled):
      When > 1.0, tokens with n_value < k_threshold are forced to k=1
      (treated as "almost certain"), remaining tokens use ceil.
      Overrides k_round_mode when active.

    When k_round_mode in {'ceil', 'round', 'floor'}, n1_vals/n2_vals must be present
    in entry; falls back silently to stored k if the raw vals are missing.
    """
    import math

    def _apply(vals: list, mode: str) -> list:
        if mode == "ceil":
            return [max(1, min(k_save, math.ceil(v))) for v in vals]
        if mode == "round":
            return [max(1, min(k_save, round(v))) for v in vals]
        if mode == "floor":
            return [max(1, min(k_save, math.floor(v))) for v in vals]
        raise ValueError(f"Unknown k_round_mode: {mode!r}")

    def _apply_threshold(vals: list, threshold: float) -> list:
        return [1 if v < threshold else max(1, min(k_save, math.ceil(v)))
                for v in vals]

    def _seq_len() -> int:
        if "R" in entry:
            return len(entry["R"])
        if "topk_ids" in entry:
            return len(entry["topk_ids"])
        if "k" in entry:
            return len(entry["k"])
        if "k_n2" in entry:
            return len(entry["k_n2"])
        if "k_n1" in entry:
            return len(entry["k_n1"])
        raise ValueError(
            "k_round_mode='max' needs a length field in cache "
            "('R', 'topk_ids', or 'k'/'k_n1'/'k_n2')."
        )

    if k_threshold > 1.0:
        n_key = "n1_vals" if set_method == "N1" else "n2_vals"
        if n_key in entry:
            return _apply_threshold(entry[n_key], k_threshold)

    # Paper default: fixed top-K_max for every token.
    if k_round_mode == "max":
        return [int(k_save)] * _seq_len()

    if k_round_mode != "precomputed":
        n_key = "n1_vals" if set_method == "N1" else "n2_vals"
        if n_key in entry:
            return _apply(entry[n_key], k_round_mode)
        # fallback: raw vals not in old cache → use stored k
    if set_method == "N2" and "k_n2" in entry:
        return entry["k_n2"]
    if set_method == "N1" and "k_n1" in entry:
        return entry["k_n1"]
    return entry["k"]


def cache_has_topk(cache: Dict[int, Dict[str, list]]) -> bool:
    """Return True if cache (any entry) contains 'k' and 'topk_ids' fields (plateau-ready)."""
    if not cache:
        return False
    sample = next(iter(cache.values()))
    return ("k" in sample) and ("topk_ids" in sample)


def cache_has_ref_logits(cache: Dict[int, Dict[str, list]]) -> bool:
    """Return True if cache (any entry) contains 'topk_logits' and 'label_ref_logit'
    fields (ba-loss-ce-ready)."""
    if not cache:
        return False
    sample = next(iter(cache.values()))
    return ("topk_logits" in sample) and ("label_ref_logit" in sample)


# --------------------------------------------------------------------------------------
# Per-sample projection: (R, k, topk_ids) → length-of-labels arrays
# --------------------------------------------------------------------------------------
def build_R_full(
    labels: List[int],
    R_values: List[float],
    ignore_index: int = -100,
    fill: float = 0.0,
) -> List[float]:
    """Project R values (length=n_valid) onto length=len(labels), aligned with labels.

    For SFT with next-token shift:
        shift_labels = labels[1:],  shift_R = renyi_R[1:]
    so at the k-th valid position (labels[pos] != ignore_index for pos in 1..L-1),
    we set renyi_R[pos] = R_values[k].
    Position 0 is always set to `fill` (never used as a training target).
    """
    L = len(labels)
    R_full = [fill] * L
    valid_idx = [i for i in range(1, L) if labels[i] != ignore_index]
    if len(R_values) != len(valid_idx):
        raise ValueError(
            f"R length mismatch: got {len(R_values)} R values, "
            f"but found {len(valid_idx)} valid (label != {ignore_index}) positions at idx>=1"
        )
    for k, pos in enumerate(valid_idx):
        R_full[pos] = float(R_values[k])
    return R_full


def build_plateau_full(
    labels: List[int],
    R_values: List[float],
    k_values: List[int],
    topk_ids_values: List[List[int]],
    K_save: int,
    ignore_index: int = -100,
) -> Tuple[List[float], List[int], List[List[int]]]:
    """Project (R, k, topk_ids) onto length=len(labels), aligned with labels.

    Returns:
        R_full:        [L]                length-of-labels list of float, 0.0 at non-valid positions
        k_full:        [L]                length-of-labels list of int,   0   at non-valid positions
        topk_ids_full: [L, K_save]        length-of-labels list of K_save-int rows; rows of zeros at
                                          non-valid positions (never read because label==-100 there).
    """
    L = len(labels)
    valid_idx = [i for i in range(1, L) if labels[i] != ignore_index]
    n_valid = len(valid_idx)
    if not (len(R_values) == len(k_values) == len(topk_ids_values) == n_valid):
        raise ValueError(
            f"length mismatch: R={len(R_values)}, k={len(k_values)}, "
            f"topk_ids={len(topk_ids_values)}, valid={n_valid}"
        )

    R_full: List[float]       = [0.0] * L
    k_full: List[int]         = [0] * L
    topk_ids_full: List[List[int]] = [[0] * K_save for _ in range(L)]

    for j, pos in enumerate(valid_idx):
        R_full[pos] = float(R_values[j])
        k_full[pos] = int(k_values[j])
        row = list(topk_ids_values[j])
        if len(row) != K_save:
            raise ValueError(
                f"topk_ids row length mismatch at valid idx {j}: got {len(row)}, expected {K_save}"
            )
        topk_ids_full[pos] = [int(x) for x in row]

    return R_full, k_full, topk_ids_full


def build_ref_align_full(
    labels: List[int],
    R_values: List[float],
    k_values: List[int],
    topk_ids_values: List[List[int]],
    topk_logits_values: List[List[float]],
    label_ref_logit_values: List[float],
    K_save: int,
    ignore_index: int = -100,
) -> Tuple[List[float], List[int], List[List[int]], List[List[float]], List[float]]:
    """Project (R, k, topk_ids, topk_logits, label_ref_logit) onto length=len(labels).

    Returns:
        R_full:               [L]                length-of-labels list of float
        k_full:               [L]                length-of-labels list of int
        topk_ids_full:        [L, K_save]
        topk_logits_full:     [L, K_save]        ref logits at top-K_save ids
        label_ref_logit_full: [L]                ref logit at the true label y_t

    Padding (positions where label==ignore_index): zeros throughout, never read.
    """
    L = len(labels)
    valid_idx = [i for i in range(1, L) if labels[i] != ignore_index]
    n_valid = len(valid_idx)
    if not (
        len(R_values) == len(k_values) == len(topk_ids_values)
        == len(topk_logits_values) == len(label_ref_logit_values) == n_valid
    ):
        raise ValueError(
            f"length mismatch: R={len(R_values)}, k={len(k_values)}, "
            f"topk_ids={len(topk_ids_values)}, topk_logits={len(topk_logits_values)}, "
            f"label_ref_logit={len(label_ref_logit_values)}, valid={n_valid}"
        )

    R_full: List[float]                    = [0.0] * L
    k_full: List[int]                      = [0] * L
    topk_ids_full: List[List[int]]         = [[0] * K_save for _ in range(L)]
    topk_logits_full: List[List[float]]    = [[0.0] * K_save for _ in range(L)]
    label_ref_logit_full: List[float]      = [0.0] * L

    for j, pos in enumerate(valid_idx):
        R_full[pos] = float(R_values[j])
        k_full[pos] = int(k_values[j])
        row_id  = list(topk_ids_values[j])
        row_lg  = list(topk_logits_values[j])
        if len(row_id) != K_save:
            raise ValueError(
                f"topk_ids row length mismatch at valid idx {j}: got {len(row_id)}, expected {K_save}"
            )
        if len(row_lg) != K_save:
            raise ValueError(
                f"topk_logits row length mismatch at valid idx {j}: got {len(row_lg)}, expected {K_save}"
            )
        topk_ids_full[pos]    = [int(x) for x in row_id]
        topk_logits_full[pos] = [float(x) for x in row_lg]
        label_ref_logit_full[pos] = float(label_ref_logit_values[j])

    return R_full, k_full, topk_ids_full, topk_logits_full, label_ref_logit_full


# --------------------------------------------------------------------------------------
# Collators
# --------------------------------------------------------------------------------------
class DataCollatorForSFTWithR(DataCollatorForSeq2Seq):
    """Like DataCollatorForSeq2Seq, but also pads `renyi_R` field.

    The padding side follows `tokenizer.padding_side`, mirroring how labels are padded.
    Padded R positions get `fill_value` (default 0.0); they are never used by the loss
    because the corresponding label is -100.
    """

    fill_value: float = 0.0

    def __call__(self, features, return_tensors=None):
        has_R = bool(features) and "renyi_R" in features[0]
        if has_R:
            def _to_list(x):
                if torch.is_tensor(x):
                    return x.detach().cpu().tolist()
                return list(x)
            R_lists = [_to_list(f.pop("renyi_R")) for f in features]

        batch = super().__call__(features, return_tensors=return_tensors)

        if has_R:
            L_max = batch["input_ids"].shape[1]
            padding_side = getattr(self.tokenizer, "padding_side", "right")
            padded = []
            for R_list in R_lists:
                R_list = R_list[:L_max]
                pad_len = L_max - len(R_list)
                if pad_len > 0:
                    if padding_side == "left":
                        R_list = [self.fill_value] * pad_len + R_list
                    else:
                        R_list = R_list + [self.fill_value] * pad_len
                padded.append(R_list)
            batch["renyi_R"] = torch.tensor(padded, dtype=torch.float32)
        return batch


class DataCollatorForSFTWithPlateau(DataCollatorForSFTWithR):
    """Like DataCollatorForSFTWithR, but also pads `plateau_k` and `plateau_topk_ids`.

    Padding values:
      plateau_k        : 0  (long)  → corresponding label is -100, never read
      plateau_topk_ids : 0  (long)  → corresponding label is -100, never read

    Args:
      K_save: must match the K_save used in precompute (the inner size of plateau_topk_ids rows).
    """

    K_save: int = 10
    plateau_k_fill: int = 0
    plateau_topk_id_fill: int = 0

    def __init__(self, *args, K_save: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.K_save = int(K_save)

    def __call__(self, features, return_tensors=None):
        # Pop plateau fields BEFORE delegating to the parent (which itself pops renyi_R and
        # then calls DataCollatorForSeq2Seq.__call__ on the remaining fields).
        has_plat = bool(features) and ("plateau_k" in features[0]) and ("plateau_topk_ids" in features[0])
        if has_plat:
            def _to_list_1d(x):
                if torch.is_tensor(x):
                    return x.detach().cpu().tolist()
                return list(x)

            def _to_list_2d(x):
                if torch.is_tensor(x):
                    return x.detach().cpu().tolist()
                return [list(row) for row in x]

            k_lists = [_to_list_1d(f.pop("plateau_k")) for f in features]
            topk_ids_lists = [_to_list_2d(f.pop("plateau_topk_ids")) for f in features]

        # Parent pads renyi_R and runs the seq2seq collator
        batch = super().__call__(features, return_tensors=return_tensors)

        if has_plat:
            L_max = batch["input_ids"].shape[1]
            padding_side = getattr(self.tokenizer, "padding_side", "right")
            K_save = self.K_save

            padded_k = []
            padded_topk = []
            for k_list, topk_rows in zip(k_lists, topk_ids_lists):
                k_list = list(k_list)[:L_max]
                topk_rows = [list(r) for r in topk_rows][:L_max]
                pad_len = L_max - len(k_list)
                if pad_len > 0:
                    if padding_side == "left":
                        k_list = [self.plateau_k_fill] * pad_len + k_list
                        topk_rows = [[self.plateau_topk_id_fill] * K_save for _ in range(pad_len)] + topk_rows
                    else:
                        k_list = k_list + [self.plateau_k_fill] * pad_len
                        topk_rows = topk_rows + [[self.plateau_topk_id_fill] * K_save for _ in range(pad_len)]
                padded_k.append(k_list)
                padded_topk.append(topk_rows)
            batch["plateau_k"] = torch.tensor(padded_k, dtype=torch.long)
            batch["plateau_topk_ids"] = torch.tensor(padded_topk, dtype=torch.long)
        return batch


class DataCollatorForSFTWithRefAlign(DataCollatorForSFTWithPlateau):
    """Like DataCollatorForSFTWithPlateau, but also pads `ref_topk_logits` and
    `ref_label_logit` for lp_sft.

    Padding values:
      ref_topk_logits  : 0.0  (float)  → corresponding label is -100, never read
      ref_label_logit  : 0.0  (float)  → corresponding label is -100, never read

    Args:
      K_save: must match the K_save used in precompute (the inner size of
              ref_topk_logits / plateau_topk_ids rows).
    """

    ref_topk_logit_fill: float = 0.0
    ref_label_logit_fill: float = 0.0

    def __call__(self, features, return_tensors=None):
        # Pop ref-align fields BEFORE delegating to the parent so they don't reach
        # DataCollatorForSeq2Seq.__call__ (which would choke on non-int tensors).
        has_ref = bool(features) and ("ref_topk_logits" in features[0]) and ("ref_label_logit" in features[0])
        if has_ref:
            def _to_list_1d(x):
                if torch.is_tensor(x):
                    return x.detach().cpu().tolist()
                return list(x)

            def _to_list_2d(x):
                if torch.is_tensor(x):
                    return x.detach().cpu().tolist()
                return [list(row) for row in x]

            ref_topk_lg_lists = [_to_list_2d(f.pop("ref_topk_logits"))  for f in features]
            ref_label_lg_lists = [_to_list_1d(f.pop("ref_label_logit")) for f in features]

        # Parent pads renyi_R, plateau_k, plateau_topk_ids and runs the seq2seq collator
        batch = super().__call__(features, return_tensors=return_tensors)

        if has_ref:
            L_max = batch["input_ids"].shape[1]
            padding_side = getattr(self.tokenizer, "padding_side", "right")
            K_save = self.K_save

            padded_topk_lg = []
            padded_label_lg = []
            for topk_rows, label_lg in zip(ref_topk_lg_lists, ref_label_lg_lists):
                topk_rows = [list(r) for r in topk_rows][:L_max]
                label_lg  = list(label_lg)[:L_max]
                pad_len = L_max - len(topk_rows)
                if pad_len > 0:
                    if padding_side == "left":
                        topk_rows = (
                            [[self.ref_topk_logit_fill] * K_save for _ in range(pad_len)]
                            + topk_rows
                        )
                        label_lg = [self.ref_label_logit_fill] * pad_len + label_lg
                    else:
                        topk_rows = topk_rows + [
                            [self.ref_topk_logit_fill] * K_save for _ in range(pad_len)
                        ]
                        label_lg = label_lg + [self.ref_label_logit_fill] * pad_len
                padded_topk_lg.append(topk_rows)
                padded_label_lg.append(label_lg)
            batch["ref_topk_logits"] = torch.tensor(padded_topk_lg, dtype=torch.float32)
            batch["ref_label_logit"] = torch.tensor(padded_label_lg, dtype=torch.float32)
        return batch
