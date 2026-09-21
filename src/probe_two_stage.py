#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 8 步 · 两阶段检索验证（IVF 召回 → 精确重排）

为什么想到这个
--------------
单看 faiss IVF-SQ8 的成绩：

| nprobe | 延迟 | recall@10 |
|---|---|---|
| 1 | 2.6 ms | 0.48 |
| 32 | 51 ms | 0.87 |
| 128 | 209 ms | 0.95 |
| **4096（全扫）** | **5.4 s** | **0.98** ← 上限就是 0.98 |

**全扫都只有 0.98**，说明 SQ8 量化掉了 2% —— 而暴力精确检索要 263ms（若能常驻内存）
就能拿到 1.000。也就是说：**花 208ms 换 0.95 是亏的**。

正解是工业界的标准姿势 —— **两阶段**：

    第一阶段（召回）：用近似索引快速取 **top-1000 候选**（不在乎顺序，只要别漏）
    第二阶段（重排）：对这 1000 条用**原始 fp16 向量**算精确内积，取真正的 top-10

好处：1000 条候选只有 2MB，从 `emb.npy` 随机读出来 + 精确算，几乎不花时间。
      最终分数是**精确的**（不再是 SQ8 的近似值），只有「候选有没有召回到」这一步是近似的。

本脚本就是要量化：**用多少 nprobe 召回 top-1000，能让真 top-10 的覆盖率 > 0.99**。

用法
----
    "...python313\\python.exe" src\\probe_two_stage.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vectorize import find_model  # noqa: E402

PROJECT = HERE.parent
SRC = PROJECT / "data" / "index" / "bge-large-zh-v1.5"
# 默认索引：2026-09-19 起改为 IVF Flat。
# 原默认是 faiss_ivf_sq8.index（本脚本当初就是为验证「SQ8 量化损失」而写的），
# 但该结论已被 Flat 版推翻（nprobe=512 时 recall 1.000），两个 SQ8 索引已删除。
# 若要复现脚本头部那组 SQ8 数据，先重建：
#   python src\build_faiss_index.py --quantizer sq8   （约 8 分钟，3.4 GB）
IDX = PROJECT / "data" / "index" / "faiss_ivf_flat.index"

EVAL_QUERIES = [
    "台灣東部開發於古時的人行道路",
    "中國的化學工程學家",
    "這個地方氣候怎麼樣",
    "苏花公路的历史",
    "南开大学化工系的创始人",
    "上海的行政区划",
    "量子力学的奠基人",
    "长江有多长",
    "红楼梦的作者是谁",
    "乒乓球世界冠军",
]


def brute_topk(emb, qv, k, block=200_000):
    n = emb.shape[0]
    bs = np.full(k, -np.inf, dtype=np.float32)
    bi = np.zeros(k, dtype=np.int64)
    for s in range(0, n, block):
        e = min(s + block, n)
        sc = np.asarray(emb[s:e], dtype=np.float32) @ qv
        kk = min(k, e - s)
        part = np.argpartition(-sc, kk - 1)[:kk]
        cs = np.concatenate([bs, sc[part]])
        ci = np.concatenate([bi, part.astype(np.int64) + s])
        keep = np.argpartition(-cs, k - 1)[:k]
        bs, bi = cs[keep], ci[keep]
    return bs[np.argsort(-bs)], bi[np.argsort(-bs)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--index", default=str(IDX))
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--candidates", type=int, default=1000)
    ap.add_argument("--model-kind", default="large", choices=["large", "small"])
    args = ap.parse_args()

    import faiss
    print("=" * 76)
    print("两阶段检索验证：IVF 召回 top-K  →  精确重排 top-10")
    print("=" * 76)

    emb = np.load(Path(args.src) / "emb.npy", mmap_mode="r")
    index = faiss.read_index(args.index)
    print(f"原始向量 : {emb.shape[0]:,} × {emb.shape[1]} {emb.dtype}（用于精确重排）")
    print(f"faiss    : nlist={index.nlist}  ntotal={index.ntotal:,}  SQ8 量化")

    from FlagEmbedding import FlagModel
    model = FlagModel(str(find_model(args.model_kind)), use_fp16=True)
    qv_all = np.ascontiguousarray(
        np.asarray(model.encode(EVAL_QUERIES, batch_size=8, max_length=512)).astype(np.float32))

    print(f"\n计算暴力 ground truth（{len(EVAL_QUERIES)} 题）…")
    t = time.time()
    gt = np.array([brute_topk(emb, qv, args.topk)[1] for qv in qv_all])
    gt_sec = time.time() - t
    print(f"  用时 {gt_sec:.1f} 秒（平摊 {gt_sec / len(EVAL_QUERIES):.2f} 秒/题）← 这就是要摆脱的成本")

    print()
    print("=" * 76)
    print(f"结果（候选池 A={args.candidates}，最终 top-{args.topk}）")
    print("=" * 76)
    print(f"{'nprobe':>7} | {'召回(ms)':>9} | {'重排(ms)':>9} | {'合计(ms)':>9} | "
          f"{'候选覆盖率':>10} | {'最终recall':>10}")
    print("-" * 76)

    rows = []
    for npb in (1, 4, 8, 16, 32, 64):
        index.nprobe = npb
        t0 = time.time()
        _, I = index.search(qv_all, args.candidates)
        rec_ms = (time.time() - t0) / len(qv_all) * 1000

        t0 = time.time()
        final = np.empty((len(qv_all), args.topk), dtype=np.int64)
        cover_hits, recall_hits = 0, 0
        for qi in range(len(qv_all)):
            cand = I[qi]                                   # 1000 个候选行号
            vecs = np.asarray(emb[cand], dtype=np.float32)  # 只取 1000 × 1024 = 4MB
            sc = vecs @ qv_all[qi]                          # 精确内积（原始 fp16 向量）
            keep = cand[np.argsort(-sc)[:args.topk]]
            final[qi] = keep
            cover_hits += len(set(cand.tolist()) & set(gt[qi].tolist()))
            recall_hits += len(set(keep.tolist()) & set(gt[qi].tolist()))
        rr_ms = (time.time() - t0) / len(qv_all) * 1000

        cover = cover_hits / (len(qv_all) * args.topk)
        recall = recall_hits / (len(qv_all) * args.topk)
        rows.append({"nprobe": npb, "recall_ms": round(rec_ms, 2), "rerank_ms": round(rr_ms, 2),
                     "total_ms": round(rec_ms + rr_ms, 2),
                     "candidate_coverage": round(cover, 4), "final_recall": round(recall, 4)})
        print(f"{npb:>7} | {rec_ms:>9.2f} | {rr_ms:>9.2f} | {rec_ms + rr_ms:>9.2f} | "
              f"{cover:>10.4f} | {recall:>10.4f}")

    print("-" * 76)
    print(f"{'暴力':>7} | {'—':>9} | {'—':>9} | {gt_sec / len(EVAL_QUERIES) * 1000:>9.1f} | "
          f"{1.0:>10.4f} | {1.0:>10.4f}   ← 精确但慢")

    best = next((r for r in rows if r["final_recall"] >= 0.99), None)
    print()
    if best:
        print(f"✅ 找到达标配置：nprobe={best['nprobe']}  →  "
              f"延迟 {best['total_ms']:.1f} ms，最终 recall {best['final_recall']:.4f}")
        print(f"   相比暴力精确检索（{gt_sec / len(EVAL_QUERIES) * 1000:.0f} ms），"
              f"快 **{(gt_sec / len(EVAL_QUERIES) * 1000) / best['total_ms']:.0f} 倍**，"
              f"且召回几乎无损。")
    else:
        print("⚠️  6 个 nprobe 里没有一个让最终 recall ≥ 0.99 —— 需要加大候选池（--candidates 2000+）。")

    # ---- 抽检 Q1 的最终结果 ----
    index.nprobe = (best or rows[-1])["nprobe"]
    _, I = index.search(qv_all[0][None, :], args.candidates)
    cand = I[0]
    sc = np.asarray(emb[cand], dtype=np.float32) @ qv_all[0]
    top5 = cand[np.argsort(-sc)[:5]]
    print(f"\n抽检 Q1「{EVAL_QUERIES[0]}」两阶段 top-5 行号：{top5.tolist()}")
    print(f"     暴力答案行号：                              {gt[0][:5].tolist()}")

    print("=" * 76)
    print("TWO_STAGE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
