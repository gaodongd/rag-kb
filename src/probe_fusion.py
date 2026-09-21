#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 8 步 · T7.5 融合策略对比 —— RRF「免调参」的代价，用数据量出来

背景：T7 的三模式对照结果（3 题）是 **1 胜 2 负**
---------------------------------------------------
| Q | 向量 top1 | BM25 top1 | RRF(60) top1 | 判定 |
|---|---|---|---|---|
| 台灣東部開發於古時的人行道路 | 東寧路(臺南市) ❌ | 蘇花古道 ✅ | **蘇花古道** ✅ | 混合大胜 |
| 中國的化學工程學家 | 张克忠 ✅ | 八田四郎次(日本) ❌ | 朱汝瑾 ⚠️ | 混合变差 |
| 這個地方氣候怎麼樣 | 温带海洋性气候 ✅ | 韩东君·評價 ❌ | 韩东君 ❌ | 混合变差 |

**根因**：RRF 只用排名、**完全丢弃分数强度**：

- BM25「强匹配」：蘇花古道 **27.05** 分，领先第二名 **33%** → 该信
- BM25「弱匹配」：韩东君 **19.90** 分，仅领先第二名 **2.7%** → 不该信

两者在 RRF 眼里都是「rank 1」，拿一样的权重 `1/(k+1)`。于是弱匹配的垃圾结果
被抬到和向量强命中同等的地位，甚至靠"两路都在中游"的共识分反超单路的第一名。
（Q2 就是如此：朱汝瑾 vec#22+bm25#18 = 0.025 击败了张克忠 vec#1 = 0.016。**多路共识赢了强信号**。）

本脚本对比 4 种策略
------------------
1. `rrf60`  —— RRF k=60（原论文/ES 默认）
2. `rrf10`  —— RRF k=10（放大头部排名差异：1/11 vs 1/61）
3. `cc-mm`  —— Convex Combination，min-max 归一化后加权（利用分数，但对离群值敏感）
4. `cc-z`   —— Convex Combination，z-score 归一化后加权（利用分数分布形状）

用 faiss 后端跑（0.25 秒/次），所以可以放开试。

用法
----
    "...python313\\python.exe" src\\probe_fusion.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from search_bm25 import build_idf, load_bm25  # noqa: E402
from search_vector import DEFAULT_QUERIES, fetch_texts, load_index  # noqa: E402

PROJECT = HERE.parent
VEC_INDEX = PROJECT / "data" / "index" / "bge-large-zh-v1.5"
BM25_INDEX = PROJECT / "data" / "index" / "bm25"
PARQUET = PROJECT / "data" / "processed" / "chunks.parquet"
FAISS_INDEX = PROJECT / "data" / "index" / "faiss_ivf_flat.index"


# ---------------------------------------------------------------- 融合策略

def fuse_rrf(vr, br, k=60.0, w_vec=1.0, w_bm25=1.0):
    """纯排名融合。分数完全不参与。"""
    acc = {}
    for rows, w in ((vr, w_vec), (br, w_bm25)):
        if w == 0:
            continue
        for pos, r in enumerate(rows, start=1):
            r = int(r)
            acc[r] = acc.get(r, 0.0) + w / (k + pos)
    return sorted(acc.items(), key=lambda x: (-x[1], x[0]))


def _norm(x, mode):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x
    if mode == "minmax":
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-12)
    # zscore
    return (x - x.mean()) / (x.std() + 1e-12)


def fuse_cc(vr, vs, br, bs, w_vec=0.5, mode="minmax"):
    """
    Convex Combination：归一化后加权求和。
    **利用了分数强度** —— 强匹配（27 分领先 33%）和弱匹配（19.9 分领先 2.7%）
    归一化后确实会拉开差距（后者差距被压缩）。
    """
    vn = _norm(vs, mode)
    bn = _norm(bs, mode)
    acc = {}
    for r, s in zip(vr, vn):
        r = int(r)
        acc[r] = acc.get(r, 0.0) + w_vec * float(s)
    for r, s in zip(br, bn):
        r = int(r)
        acc[r] = acc.get(r, 0.0) + (1.0 - w_vec) * float(s)
    return sorted(acc.items(), key=lambda x: (-x[1], x[0]))


def fuse_rrf_gate(vr, br, vs, bs, k=10, w_strong=1.0, w_weak=0.3,
                  margin=0.10, w_vec=1.0):
    """
    RRF + 「领先幅度」门控。

    观察（来自本脚本 3 题实测）：

        Q1  BM25 top1 领先次名 **33.4%**  → 强匹配（蘇花古道，正文首句就是原话）
        Q2  BM25 top1 领先次名 **6.9%**   → 弱
        Q3  BM25 top1 领先次名 **2.8%**   → 极弱（韩东君，实际完全无关）

    领先幅度能区分「BM25 真的捞到了强匹配」和「BM25 只是在泛词上排了个序」。
    后者不该和前者拿同样的权重 —— 这是纯 RRF 做不到的（它连分数都不看）。

    ⚠️ **本门控是在 3 条 query 上调出来的，样本量严重不足，极可能过拟合。**
    写在这里是为了记录「有这么一个信号存在」，不是宣称它可用。
    要真正采用，需要先标注几十条 query 做验证。
    返回 (融合结果, 领先幅度, 实际用的 BM25 权重)。
    """
    lead = (float(bs[0]) - float(bs[1])) / max(abs(float(bs[1])), 1e-9) if len(bs) > 1 else 0.0
    w_b = w_strong if lead >= margin else w_weak
    return fuse_rrf(vr, br, k=k, w_vec=w_vec, w_bm25=w_b), lead, w_b


def rank_of(rows, r):
    """r 在 rows 里的排名，没有则 None。"""
    for i, x in enumerate(rows, 1):
        if int(x) == int(r):
            return i
    return None


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topn", type=int, default=100)
    ap.add_argument("--show", type=int, default=5)
    ap.add_argument("--nprobe", type=int, default=512)
    ap.add_argument("--query", action="append", default=None)
    args = ap.parse_args()
    queries = args.query or DEFAULT_QUERIES

    print("=" * 78)
    print("第 8 步 · 融合策略对比（4 种）")
    print("=" * 78)

    import faiss
    from FlagEmbedding import FlagModel
    from vectorize import find_model

    ids = (VEC_INDEX / "ids.txt").read_text(encoding="utf-8").split()
    n = len(ids)
    index = faiss.read_index(str(FAISS_INDEX))
    index.nprobe = args.nprobe
    print(f"faiss     : {index.ntotal:,} 条，nprobe={args.nprobe}")

    meta, vocab, indptr, doc_ids, tf_norm = load_bm25(BM25_INDEX)
    idf = build_idf(indptr, n)
    print(f"BM25      : {meta['vocab_size']:,} 词 / {meta['postings']:,} postings")

    model = FlagModel(str(find_model("large")), use_fp16=True)
    qv_all = np.asarray(model.encode(queries, batch_size=8, max_length=512)).astype(np.float32)
    print(f"模型      : bge-large-zh，{len(queries)} 条 query 已编码")

    from search_bm25 import bm25_search

    STRATEGIES = ["rrf60", "rrf10", "cc-mm", "cc-z", "gate"]
    for q, qv in zip(queries, qv_all):
        t = time.time()
        # 一次 search 同时拿行号和分数（IP 度量下 D 就是内积）
        D, I = index.search(qv[None, :], args.topn)
        vr = np.asarray(I[0], dtype=np.int64)
        vs = D[0].astype(np.float64)
        t_v = time.time() - t

        t = time.time()
        br, bs, _ = bm25_search(q, vocab, indptr, doc_ids, tf_norm, idf, n, args.topn)
        br = np.asarray(br, dtype=np.int64)
        bs = np.asarray(bs, dtype=np.float64)
        t_b = time.time() - t

        print()
        print("=" * 78)
        print(f"Q: {q}")
        print(f"   [向量 {t_v * 1000:.0f} ms / BM25 {t_b * 1000:.0f} ms]")
        print("=" * 78)
        print(f"   向量 top1 : row={vr[0]:<9} 分数 {vs[0]:.4f}"
              f"（领先次名 {(vs[0] - vs[1]) / max(abs(vs[1]), 1e-9) * 100:.1f}%）")
        print(f"   BM25 top1 : row={br[0]:<9} 分数 {bs[0]:.4f}"
              f"（领先次名 {(bs[0] - bs[1]) / max(abs(bs[1]), 1e-9) * 100:.1f}%）")

        gate_out, gate_lead, gate_w = fuse_rrf_gate(vr, br, vs, bs, k=10)
        fused = {
            "rrf60": fuse_rrf(vr, br, k=60),
            "rrf10": fuse_rrf(vr, br, k=10),
            "cc-mm": fuse_cc(vr, vs, br, bs, w_vec=0.5, mode="minmax"),
            "cc-z": fuse_cc(vr, vs, br, bs, w_vec=0.5, mode="zscore"),
            "gate": gate_out,
        }
        print(f"   [门控] BM25 领先幅度 {gate_lead * 100:.1f}% → 权重取 {gate_w}"
              f"（阈值 10%）")

        # 所有策略的候选并起来，一次回查原文
        cand = set()
        for out in fused.values():
            cand.update(r for r, _ in out[:args.show])
        got = fetch_texts(PARQUET, [ids[r] for r in cand])

        for name in STRATEGIES:
            out = fused[name]
            print(f"\n   ── {name} ──")
            for rank, (r, sc) in enumerate(out[:args.show], 1):
                rec = got.get(ids[r], {})
                v_p = rank_of(vr, r)
                b_p = rank_of(br, r)
                tag = f"vec#{v_p if v_p else '—':<3} bm25#{b_p if b_p else '—':<3}"
                print(f"   {rank}. {sc:>8.5f}  ({tag})  {rec.get('title', '?')}"
                      f"  ·  {rec.get('section') or '—'}")

    print()
    print("=" * 78)
    print("PROBE_FUSION_OK")
    print("=" * 78)
    print("怎么读：重点看每个策略的 top1 是不是那个「唯一正确答案」。")
    print("  · rrf 系只吃排名 → 弱匹配的 rank1 与强匹配的 rank1 等价")
    print("  · cc 系吃分数 → 强匹配能体现出来，但归一化方式（minmax/zscore）影响很大")
    return 0


if __name__ == "__main__":
    sys.exit(main())
