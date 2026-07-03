#!/usr/bin/env python3
"""
对训练数据预先计算 token 级 Rényi entropy ratio R = N_2 / N_1, 用 base 模型 (frozen).

R 定义 (跟 analysis/src/metrics_cache.py 一致):
    p     = softmax(logits) over top-K (默认 30, 长尾贡献小可忽略)
    H1    = -Σ p log p                  (Shannon entropy)
    H2    = -log Σ p²                   (Collision entropy, alpha=2 Rényi)
    N1    = exp(H1) = effective # of candidates (Shannon)
    N2    = exp(H2) = effective # of candidates (Collision)
    R     = N2 / N1                     ∈ (0, 1], 越大越接近 uniform

输出文件 (每行一个 sample):
    {"sample_idx": int, "R": [float, ...], "n_tokens": int}
其中 R 的长度 == sum(labels != ignore_index) for that sample, 跟训练时 mask 后的
有效 token 序列一一对应. 训练时按 sample_idx 索引, 把 R 数组对齐到 per-token loss 上.

如果传 --save_topk, 额外保存 plateau loss 训练所需的 ref top-K_save 候选信息:
    {"sample_idx": int,
     "R":        [float, ...],
     "n_tokens": int,
     "k":        [int, ...],          # k_t = clamp(ceil(N_method), 1, K_save), 由 --set_method 决定
     "k_n1":     [int, ...],          # k_t from Shannon: clamp(ceil(N1), 1, K_save) (always saved)
     "k_n2":     [int, ...],          # k_t from Collision: clamp(ceil(N2), 1, K_save) (always saved)
     "topk_ids": [[int x K_save],...] # ref top-K_save token ids, sorted desc by ref prob
    }
其中 K_save 是 --topk_save_ids (默认 10), 同时也是 |S_t'| 的硬上限 (含 y_real).
k_n1 和 k_n2 总是同时保存, 一次 precompute 即可服务 N1 和 N2 两种 set_method 实验.

如果同时传 --save_ref_logits, 还会额外保存:
    {"topk_logits":     [[float x K_save], ...],  # ref logits at topk_ids (sorted desc)
     "label_ref_logit": [float, ...],             # ref logit at the true label y_t
    }
这是 lp_sft 所需的 (用来构造 q_ref^S over S_t' = S_t ∪ {y_t}).
若 y_t ∉ S_t (即 label not in topk_ids), 训练时会把 label_ref_logit 拼到 S_t 末尾形成 S_t'.
若 y_t ∈ S_t, label_ref_logit 不被读取 (但仍保存以避免 collator 路径分支).

用法 (单卡测试):
    python losses/precompute_R.py \\
        --model_path $MODELS_DIR/Qwen3-4B-Base \\
        --data_file  $DATA_DIR/metamath/metamath_sft_train_qwen3-4b_tokenized.jsonl \\
        --out_file   $DATA_DIR/metamath/metamath_R_qwen3_4b_base.jsonl \\
        --max_samples 1000

用法 (8 卡并行):
    torchrun --nproc_per_node=8 losses/precompute_R.py \\
        --model_path $MODELS_DIR/Qwen3-4B-Base \\
        --data_file  $DATA_DIR/metamath/metamath_sft_train_qwen3-4b_tokenized.jsonl \\
        --out_file   $DATA_DIR/metamath/metamath_R_qwen3_4b_base.jsonl \\
        --max_samples 100000 --topk 30

Resume: 如果 --out_file 已存在, 会读取已完成的 sample_idx 集合, 跳过它们.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--model_path", required=True, help="base 模型路径 (frozen)")
    ap.add_argument("--data_file",  required=True, help="tokenized 训练数据 (.jsonl, 含 input_ids/labels)")
    ap.add_argument("--out_file",   required=True, help="R cache 输出路径 (.jsonl)")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="只算前 N 个 sample (None=全量). 训练用 100K, 应跟 SMOKE/full 对齐.")
    ap.add_argument("--topk", type=int, default=30,
                    help="top-K 用于近似 H1/H2 (默认 30, 跟 analysis pipeline 一致)")
    ap.add_argument("--max_seq_len", type=int, default=4096,
                    help="截断长度, 跟 tokenize_sft.py 的 MAX_SEQ_LENGTH 一致 (默认 4096)")
    ap.add_argument("--ignore_index", type=int, default=-100)
    ap.add_argument("--flush_every", type=int, default=200,
                    help="每写多少 sample flush 一次 (防 OOM/crash)")
    # ---- plateau loss extras ----
    ap.add_argument("--save_topk", action="store_true",
                    help="额外保存 ref top-K_save ids 与 k_t, 用于 plateau loss 训练")
    ap.add_argument("--topk_save_ids", type=int, default=10,
                    help="保存的 ref top-K_save 数量, 同时 = |S_t'| 上限 = k_t clamp 上限")
    # ---- lp_sft extras ----
    ap.add_argument("--save_ref_logits", action="store_true",
                    help="额外保存 ref top-K_save logits 与 label_ref_logit, 用于 "
                         "lp_sft 训练. 必须同时启用 --save_topk.")
    # ---- set-size 选择 (Rényi N1 vs N2) ----
    ap.add_argument("--set_method", choices=["N1", "N2"], default="N2",
                    help="计算 k_t 用哪个 effective size: N1=Shannon (exp(H1)), "
                         "N2=Collision (exp(H2)). N1 ≥ N2 总成立, 所以 N1 模式 "
                         "set 普遍更大、更宽松. 默认 N2 (跟现有 cache 兼容).")
    return ap.parse_args()


def setup_distributed():
    """初始化分布式 (torchrun 启动)。单卡时 rank=0, world_size=1."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    return rank, world_size, device


def load_done_idx(out_file: Path) -> set:
    """读取已完成 sample_idx 集合, 用于 resume."""
    done = set()
    if not out_file.exists():
        return done
    with open(out_file) as f:
        for line in f:
            try:
                d = json.loads(line)
                done.add(int(d["sample_idx"]))
            except Exception:
                continue
    return done


def load_data(data_file, max_samples, ignore_index, max_seq_len):
    """流式读取 jsonl, 返回 list of (sample_idx, input_ids, labels)."""
    out = []
    with open(data_file) as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            d = json.loads(line)
            input_ids = d["input_ids"][:max_seq_len]
            labels    = d["labels"][:max_seq_len]
            # 必须有至少 1 个 trainable token, 否则没意义
            if not any(l != ignore_index for l in labels):
                continue
            out.append((i, input_ids, labels))
    return out


@torch.no_grad()
def compute_R_for_sample(
    model, input_ids, labels, topk, device,
    ignore_index=-100,
    save_topk=False,
    topk_save_ids=10,
    save_ref_logits=False,
    set_method="N2",
):
    """对一个 sample 跑 forward, 返回每个 valid label 位置的 R 值 (list of float).

    长度 = sum(labels != ignore_index), 跟训练时 mask 后的 token 数对齐.

    如果 save_topk=True, 同时返回 (k_list, topk_ids_list):
      - k_list:        每个 valid token 的 k_t = clamp(ceil(N_method), 1, topk_save_ids)
                       N_method = N1 (Shannon) 或 N2 (collision), 由 set_method 控制.
                       N1 ≥ N2 总成立, 所以 N1 模式 set 普遍更大.
      - topk_ids_list: 每个 valid token 的 ref top-topk_save_ids ids (sorted desc by ref prob)
    长度均与 R 一致.

    如果同时 save_ref_logits=True (要求 save_topk=True), 多返回 (topk_logits_list, label_ref_logit_list):
      - topk_logits_list:     每个 valid token 在 top-topk_save_ids 上的 ref logits (sorted desc)
      - label_ref_logit_list: 每个 valid token 在真实标签 y_t 上的 ref logit (用于构造 S_t' 时拼接)
    """
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    lab = torch.tensor([labels],    dtype=torch.long, device=device)

    logits = model(input_ids=ids).logits[0]  # [seq_len, V]

    # 标准 SFT shift: predict t+1 from t
    shift_logits = logits[:-1]               # [seq_len-1, V]
    shift_labels = lab[0, 1:]                # [seq_len-1]

    mask = shift_labels != ignore_index
    if mask.sum() == 0:
        empty: list = []
        if save_topk and save_ref_logits:
            return empty, empty, empty, empty, empty, empty, empty
        if save_topk:
            return empty, empty, empty, empty, empty
        return empty

    logits_valid = shift_logits[mask]        # [n_valid, V]
    labels_valid = shift_labels[mask]        # [n_valid]

    # 用 top-K 近似 H1/H2 (跟 metrics_cache._compute 一致)
    # 注意: log_softmax 是 monotonic, 所以 top-K 的 indices 等价于 logits 的 top-K
    log_probs_full = F.log_softmax(logits_valid, dim=-1)
    topk_logprobs, topk_indices = log_probs_full.topk(topk, dim=-1)  # [n_valid, K]
    # 重新归一化到 top-K 内 (mass 不到 1, 长尾忽略)
    p_raw  = topk_logprobs.exp()              # [n_valid, K]
    mass   = p_raw.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    p      = p_raw / mass                     # 归一化

    # H1 = -Σ p log p; H2 = -log Σ p²
    H1 = -(p * (p + 1e-12).log()).sum(dim=-1)
    H2 = -((p * p).sum(dim=-1) + 1e-12).log()
    N1 = H1.exp()
    N2 = H2.exp()
    R  = torch.where(N1 > 0, N2 / N1, torch.zeros_like(N1))
    # R ∈ (0, 1]; numerical safety
    R = R.clamp(min=0.0, max=1.0)

    R_list = R.float().cpu().tolist()
    if not save_topk:
        return R_list

    # ---- plateau loss extras ----
    # k_t = clamp(ceil(N2), 1, K_save), per valid token
    K_save = int(topk_save_ids)
    if K_save < 1:
        raise ValueError(f"topk_save_ids must be >= 1, got {K_save}")
    if K_save > topk:
        raise ValueError(
            f"topk_save_ids ({K_save}) cannot exceed --topk ({topk}); "
            f"top-K_save ids must be a subset of the top-K used for R."
        )

    # Always compute BOTH k_n1 (Shannon) and k_n2 (Collision) so one cache run
    # serves experiments with either set_method.  The legacy `k` field is kept
    # for backward compat and equals k_n1 or k_n2 depending on --set_method.
    k_n1_t = torch.round(N1).long().clamp_(min=1, max=K_save)    # [n_valid]
    k_n2_t = torch.round(N2).long().clamp_(min=1, max=K_save)    # [n_valid]
    k_n1_list = k_n1_t.cpu().tolist()
    k_n2_list = k_n2_t.cpu().tolist()

    if set_method == "N1":
        k_list = k_n1_list
    elif set_method == "N2":
        k_list = k_n2_list
    else:
        raise ValueError(f"set_method must be N1 or N2, got {set_method!r}")

    # 取 ref top-K_save ids (前面已经按 prob 降序 topk 过, 直接切片)
    topk_ids_save = topk_indices[:, :K_save].cpu().tolist()  # list of list[int]

    # Save raw N1/N2 float values so training code can apply any k-rounding policy
    # (ceil, round, floor, custom) without re-running precompute.
    n1_list = N1.float().cpu().tolist()   # list[float], per valid token
    n2_list = N2.float().cpu().tolist()   # list[float], per valid token

    if not save_ref_logits:
        return R_list, k_list, topk_ids_save, k_n1_list, k_n2_list, n1_list, n2_list

    # ---- lp_sft extras: ref logits at S_t (top-K_save) and at y_t ----
    # We re-gather raw logits (NOT log_probs) since softmax(z/tau) is shift-invariant
    # under any constant in z, so saving raw logits is sufficient and most natural.
    topk_indices_save = topk_indices[:, :K_save]                         # [n_valid, K_save]
    topk_logits_save  = logits_valid.gather(-1, topk_indices_save)       # [n_valid, K_save]
    label_ref_logit   = logits_valid.gather(
        -1, labels_valid.unsqueeze(-1)
    ).squeeze(-1)                                                         # [n_valid]
    topk_logits_list      = topk_logits_save.float().cpu().tolist()      # list of list[float]
    label_ref_logit_list  = label_ref_logit.float().cpu().tolist()       # list of float

    return R_list, k_list, topk_ids_save, topk_logits_list, label_ref_logit_list, k_n1_list, k_n2_list, n1_list, n2_list


def main():
    args = parse_args()
    rank, world_size, device = setup_distributed()

    if args.save_ref_logits and not args.save_topk:
        raise ValueError("--save_ref_logits requires --save_topk (S_t comes from the same top-K_save).")

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # ---------------- load done set (resume) ----------------
    done = load_done_idx(out_file)
    if rank == 0:
        print(f"[precompute_R] data            = {args.data_file}", flush=True)
        print(f"[precompute_R] model           = {args.model_path}", flush=True)
        print(f"[precompute_R] out             = {args.out_file}", flush=True)
        print(f"[precompute_R] topk (K0)       = {args.topk}", flush=True)
        print(f"[precompute_R] save_topk       = {args.save_topk}", flush=True)
        if args.save_topk:
            print(f"[precompute_R] K_save          = {args.topk_save_ids} (= |S_t'| cap, k_t clamp)", flush=True)
            print(f"[precompute_R] set_method      = {args.set_method} (k_t = clamp(ceil({args.set_method}), 1, K_save))", flush=True)
        print(f"[precompute_R] save_ref_logits = {args.save_ref_logits}", flush=True)
        print(f"[precompute_R] max_seq         = {args.max_seq_len}", flush=True)
        print(f"[precompute_R] world_size      = {world_size}", flush=True)
        print(f"[precompute_R] resume          = {len(done)} samples already done", flush=True)

    # ---------------- load model ----------------
    # 用 bf16 (足够准, 跟 analysis pipeline 一致)
    AutoTokenizer.from_pretrained(args.model_path)  # 预触发下载/缓存
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",     # 跟 analysis pipeline 一致 (避免精度差异)
    ).to(device).eval()

    # ---------------- load and shard data ----------------
    all_samples = load_data(args.data_file, args.max_samples,
                            args.ignore_index, args.max_seq_len)
    if rank == 0:
        print(f"[precompute_R] total samples loaded = {len(all_samples)}", flush=True)

    # 每个 rank 处理 (sample_idx % world_size == rank) 的子集
    shard = [(idx, ids, lab) for (idx, ids, lab) in all_samples
             if (idx % world_size) == rank and idx not in done]

    if rank == 0:
        print(f"[precompute_R] rank 0 will process {len(shard)} samples (skipping {len([1 for (i,_,_) in all_samples if i % world_size == 0 and i in done])} resumed)", flush=True)

    # ---------------- per-rank append to shared file ----------------
    # 每个 rank 写自己的临时文件, 最后 rank 0 合并 (避免 fd 竞争)
    tmp_file = out_file.with_suffix(out_file.suffix + f".rank{rank}")
    fw = open(tmp_file, "w")

    iterator = tqdm(shard, desc=f"rank {rank}", disable=(rank != 0))
    t0 = time.time()
    n_written = 0
    n_skipped_short = 0
    for sample_idx, input_ids, labels in iterator:
        try:
            if args.save_topk and args.save_ref_logits:
                R, k_list, topk_ids_list, topk_logits_list, label_ref_logit_list, k_n1_list, k_n2_list, n1_list, n2_list = compute_R_for_sample(
                    model, input_ids, labels, args.topk, device,
                    args.ignore_index,
                    save_topk=True, topk_save_ids=args.topk_save_ids,
                    save_ref_logits=True,
                    set_method=args.set_method,
                )
            elif args.save_topk:
                R, k_list, topk_ids_list, k_n1_list, k_n2_list, n1_list, n2_list = compute_R_for_sample(
                    model, input_ids, labels, args.topk, device,
                    args.ignore_index,
                    save_topk=True, topk_save_ids=args.topk_save_ids,
                    set_method=args.set_method,
                )
            else:
                R = compute_R_for_sample(model, input_ids, labels, args.topk,
                                         device, args.ignore_index)
        except Exception as e:
            print(f"[precompute_R][rank {rank}][WARN] sample {sample_idx} failed: {e}", flush=True)
            continue
        if len(R) == 0:
            n_skipped_short += 1
            continue
        record = {"sample_idx": sample_idx, "R": R, "n_tokens": len(R)}
        if args.save_topk:
            record["k"] = k_list
            record["k_n1"] = k_n1_list
            record["k_n2"] = k_n2_list
            record["n1_vals"] = n1_list   # raw Shannon effective count, for flexible k policy
            record["n2_vals"] = n2_list   # raw collision effective count
            record["topk_ids"] = topk_ids_list
        if args.save_ref_logits:
            record["topk_logits"] = topk_logits_list
            record["label_ref_logit"] = label_ref_logit_list
        fw.write(json.dumps(record) + "\n")
        n_written += 1
        if n_written % args.flush_every == 0:
            fw.flush()

    fw.close()
    dt = time.time() - t0
    if rank == 0:
        rate = n_written / max(dt, 1e-9)
        print(f"[precompute_R][rank 0] wrote {n_written} samples in {dt:.0f}s ({rate:.1f}/s), skipped {n_skipped_short} short", flush=True)

    # ---------------- merge shards ----------------
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        print(f"[precompute_R] merging {world_size} shard(s) ...", flush=True)
        # append-mode: 保留旧已 done 的行, 加上新写的
        merged_idx = set(done)
        # 先把现有 out_file 备份/读出来, 然后重写
        existing_lines = []
        if out_file.exists():
            with open(out_file) as f:
                existing_lines = f.readlines()

        new_lines = []
        for r in range(world_size):
            tmp = out_file.with_suffix(out_file.suffix + f".rank{r}")
            if not tmp.exists():
                continue
            with open(tmp) as f:
                for line in f:
                    new_lines.append(line)
            tmp.unlink()

        # 合并并按 sample_idx 排序
        all_records = []
        seen = set()
        for line in existing_lines + new_lines:
            try:
                d = json.loads(line)
                if d["sample_idx"] in seen:
                    continue
                seen.add(d["sample_idx"])
                all_records.append(d)
            except Exception:
                continue
        all_records.sort(key=lambda x: x["sample_idx"])

        with open(out_file, "w") as f:
            for d in all_records:
                f.write(json.dumps(d) + "\n")

        print(f"[precompute_R] DONE -> {out_file}", flush=True)
        print(f"[precompute_R] total samples in cache: {len(all_records)}", flush=True)
        if all_records:
            n_total_tokens = sum(d["n_tokens"] for d in all_records)
            print(f"[precompute_R] total tokens: {n_total_tokens:,}", flush=True)
            # quick stats
            import statistics
            all_R = [r for d in all_records for r in d["R"]]
            if all_R:
                print(f"[precompute_R] R distribution: mean={statistics.mean(all_R):.3f}  "
                      f"median={statistics.median(all_R):.3f}  "
                      f"p10={sorted(all_R)[len(all_R)//10]:.3f}  "
                      f"p90={sorted(all_R)[9*len(all_R)//10]:.3f}", flush=True)
            if args.save_topk and all_records and "k" in all_records[0]:
                all_k = [k for d in all_records for k in d["k"]]
                if all_k:
                    K_save = args.topk_save_ids
                    k_eq1 = sum(1 for k in all_k if k == 1)
                    k_eq_max = sum(1 for k in all_k if k == K_save)
                    print(f"[precompute_R] k distribution: mean={statistics.mean(all_k):.2f}  "
                          f"median={statistics.median(all_k):.0f}  "
                          f"k=1: {100.0*k_eq1/len(all_k):.1f}%  "
                          f"k={K_save}: {100.0*k_eq_max/len(all_k):.1f}%", flush=True)
                    # R distribution restricted to plateau-active subset (k>1).
                    # k=1 forces N1=N2=1 -> R≈1 by degeneracy and pollutes the global R stats.
                    # The k>1 slice reflects R on the tokens where plateau loss actually fires.
                    R_active = [
                        r for d in all_records
                        for r, k in zip(d["R"], d["k"]) if k > 1
                    ]
                    n_active = len(R_active)
                    n_total = len(all_k)
                    active_ratio = 100.0 * n_active / n_total if n_total else 0.0
                    if R_active:
                        R_active_sorted = sorted(R_active)
                        n = len(R_active_sorted)
                        print(f"[precompute_R] R | k>1 ({active_ratio:.1f}% of tokens, n={n_active:,}): "
                              f"mean={statistics.mean(R_active):.3f}  "
                              f"median={R_active_sorted[n//2]:.3f}  "
                              f"p10={R_active_sorted[n//10]:.3f}  "
                              f"p25={R_active_sorted[n//4]:.3f}  "
                              f"p75={R_active_sorted[3*n//4]:.3f}  "
                              f"p90={R_active_sorted[9*n//10]:.3f}", flush=True)
                    else:
                        print(f"[precompute_R] R | k>1: no active tokens (all k=1)", flush=True)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
