#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 8 步 · T6 前置：BM25 倒排索引在这个规模上到底多大？

为什么不直接写建索引脚本
------------------------
331.6 万块的倒排是**亿级 postings**。先量清楚三件事再动手，否则很容易写到一半 OOM：

  1. 平均每块有多少个「有效词」（滤掉单字 / 标点 / 停用词之后）
  2. 全量 postings 有多少条 → 磁盘与内存要多少
  3. 词表（vocab）有多大 → 查询时要加载多少

**为什么要滤单字**：中文里「的/了/是/在/和」这类单字占了 postings 的一大半，
但 IDF 极低、几乎不提供区分度。砍掉它们能省掉一大半体积，且检索质量几乎无损。
这是中文 BM25 和英文最大的工程差异之一（英文靠空格分词，天然没有「单字高频」问题）。

用法
----
    "...python313\\python.exe" src\\probe_bm25_scale.py --n 50000
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

# 极简停用词：只挡最高频的那批。真正的停用词表要另找，但这一版够用 ——
# 因为单字过滤已经挡掉大头，二字停用词（"可以""因为"）对 BM25 影响有限。
STOP = {
    "可以", "因为", "所以", "但是", "并且", "以及", "或者", "如果", "这个", "那个",
    "这些", "那些", "一个", "一种", "我们", "他们", "你们", "自己", "之后", "之前",
    "同时", "also", "the", "and", "for", "with", "from", "that", "this",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(PROJECT / "data" / "processed" / "chunks.parquet"))
    ap.add_argument("--total", type=int, default=3_316_395, help="全量块数（用于外推）")
    ap.add_argument("--n", type=int, default=50_000, help="采样块数")
    args = ap.parse_args()

    print("=" * 74)
    print("T6 前置调研 · BM25 倒排索引规模估算")
    print("=" * 74)
    print(f"采样块数 : {args.n:,}   全量块数 : {args.total:,}")

    import jieba
    import pyarrow.dataset as ds_mod

    t = time.time()
    d = ds_mod.dataset(args.input, format="parquet")
    tbl = d.head(args.n, columns=["chunk_text"])
    texts = tbl.column("chunk_text").to_pylist()
    print(f"读样本   : {len(texts):,} 块（{time.time() - t:.1f} 秒）")
    print("⚠️  局限：取的是 parquet 开头的一段，不是随机抽样 —— 词分布可能有偏，")
    print("         但量级判断足够（误差不会到一个数量级）。")

    # ---- 分词 ----
    print("\n分词中（jieba 首次会加载词典，稍等）…")
    t = time.time()
    df = Counter()          # 词 → 出现在多少个块里
    post_total = 0          # 全量 postings 数（= Σ 每块的 unique 词数）
    raw_tokens = 0          # 未去重的总词数
    kept_tokens = 0
    lengths = []

    for txt in texts:
        txt = txt or ""
        toks = jieba.lcut(txt)
        raw_tokens += len(toks)
        keep = set()
        for w in toks:
            w = w.strip()
            if len(w) < 2:                 # 砍单字（含标点、数字）
                continue
            if w in STOP:
                continue
            if w.isdigit() or not any(c.isalnum() for c in w):
                continue
            keep.add(w)
        kept_tokens += len(set(toks))
        post_total += len(keep)
        df.update(keep)
        lengths.append(len(keep))

    dt = time.time() - t
    n = len(texts)
    print(f"完成     : {dt:.1f} 秒（{raw_tokens / dt / 1e6:.2f} M token/秒，单进程）")

    # ---- 统计 ----
    avg_word = post_total / n
    avg_raw = raw_tokens / n
    vocab = len(df)
    print()
    print("-" * 74)
    print("单块统计")
    print("-" * 74)
    print(f"  原始分词数/块        : {avg_raw:8.1f}")
    print(f"  过滤后 uv 词数/块    : {avg_word:8.1f}   ← 每块贡献的 postings 数")
    print(f"  过滤掉的比例          : {1 - avg_word / max(avg_raw, 1):8.1%}")

    print()
    print("-" * 74)
    print(f"全量外推（× {args.total / n:.1f}）")
    print("-" * 74)
    est_post = int(avg_word * args.total)
    print(f"  postings 总数        : {est_post:,}  （{est_post / 1e8:.2f} 亿）")
    print(f"  磁盘（int32+fp16=6B）: {est_post * 6 / 1e9:8.2f} GB")
    print(f"  磁盘（int32+fp32=8B）: {est_post * 8 / 1e9:8.2f} GB")

    # ---- df 分布 ----
    print()
    print("-" * 74)
    print("词表与 df 分布（采样内）")
    print("-" * 74)
    print(f"  采样词表规模         : {vocab:,}")
    print(f"  外推全量词表         : {int(vocab * (args.total / n) ** 0.5):,} "
          f"~ {int(vocab * args.total / n):,}（Heaps 定律：词表增长慢于语料）")
    buckets = [(1, 1), (2, 4), (5, 19), (20, 99), (100, 999), (1000, 10**9)]
    print(f"  {'df 区间':>14} | {'词数':>10} | {'占词表':>7} | {'贡献 postings':>14}")
    for lo, hi in buckets:
        ws = [w for w, c in df.items() if lo <= c <= hi]
        pp = sum(df[w] for w in ws)
        print(f"  {str(lo) + '-' + (str(hi) if hi < 10**9 else '∞'):>14} | {len(ws):>10,} "
              f"| {len(ws) / max(vocab, 1):>6.1%} | {pp:>14,}")

    lo_df = sum(c for c in df.values() if c == 1)
    print()
    print(f"  ⚠️ 只出现 1 次的词（df=1）：{lo_df:,} 个，白占 {(df and lo_df * 6 / 1e9) or 0:.3f} GB。")
    print("     这些词**对检索几乎无用**（只有 1 个块有它，任何查询都不会问它），")
    print("     但**不能直接砍 df=1** —— 生僻专名正是 BM25 相对向量的优势所在。")
    print("     折中：保留但用更紧凑的编码（见结论）。")

    print()
    print("=" * 74)
    print("结论")
    print("=" * 74)
    gb = est_post * 6 / 1e9
    if gb < 3:
        print(f"  ✅ 预计 {gb:.2f} GB —— 可以直接建，用 CSR 结构落盘，查询时 mmap。")
    elif gb < 8:
        print(f"  ⚠️  预计 {gb:.2f} GB —— 可行，但要分批构建 + 落盘，别全塞内存。")
    else:
        print(f"  🔴 预计 {gb:.2f} GB —— 太大，需要换方案（BM25 只索引 title+前 128 字，")
        print("     或改用 bm25s 库的稀疏矩阵方案）。")
    print(f"  耗时预估：全量单进程分词 ≈ {args.total / n * dt / 60:.0f} 分钟，"
          f"用 {min(16, (__import__('os').cpu_count() or 8))} 进程 ≈ "
          f"{args.total / n * dt / min(16, (__import__('os').cpu_count() or 8)) / 60:.0f} 分钟")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
