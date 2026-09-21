#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 8 步 · T5 向量检索器（全量 331.6 万块）

一句话问进去，返回最相关的 top-k 原文。

核心：为什么必须分块
--------------------
向量已 L2 归一化 → 内积 == 余弦相似度，所以检索就是 `emb @ q`。

但**不能直接写** `qv @ np.asarray(emb, dtype=np.float32).T`：
`emb.npy` 是 3,316,395 × 1024 的 **fp16（6.79GB）**，一旦整体 `astype(np.float32)`
就膨胀成 **13.6GB**，加上 memmap 本身，内存峰值 20GB 起步 —— 实机上会开始换页，慢一个数量级。

正确做法：**分块扫**，每块 20 万行（200,000 × 1024 × 4B = 819MB 临时内存），
块内先 `argpartition` 取局部 top-k，再与全局 top-k 归并。扫完 17 块就是**精确**的全量结果。

这是 exact search（暴力检索），**没有近似损失**。在 331.6 万这个规模上，
numpy 单线程扫完约 1 秒 —— 所以本步**不需要 faiss/Chroma**（决策见 `第8步-任务清单.md`）。

用法
----
    :: 默认三个语义问题
    "...python313\\python.exe" src\\search_vector.py

    :: 自定义
    "...python313\\python.exe" src\\search_vector.py --query "苏花古道" --topk 5
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
DEFAULT_INDEX = PROJECT / "data" / "index" / "bge-large-zh-v1.5"
DEFAULT_PARQUET = PROJECT / "data" / "processed" / "chunks.parquet"

DEFAULT_QUERIES = [
    "台灣東部開發於古時的人行道路",
    "中國的化學工程學家",
    "這個地方氣候怎麼樣",
]


def load_index(index_dir: Path):
    """读索引。ids.txt 的第 i 行 == emb.npy 的第 i 行，这是整个系统的对齐契约。"""
    ids_path, emb_path = index_dir / "ids.txt", index_dir / "emb.npy"
    if not ids_path.exists() or not emb_path.exists():
        raise SystemExit(f"[错误] 索引不完整：{index_dir}\n"
                         f"        需要 emb.npy + ids.txt（第 7 步产物）")
    ids = ids_path.read_text(encoding="utf-8").split()
    emb = np.load(emb_path, mmap_mode="r")
    if emb.shape[0] != len(ids):
        raise SystemExit(f"[错误] emb.npy 有 {emb.shape[0]:,} 行，ids.txt 有 {len(ids):,} 行 —— 索引已损坏。\n"
                         f"        先跑 src\\check_vectors.py 定位问题。")
    return ids, emb


def scan_topk(emb, qv: np.ndarray, topk: int, block: int = 200_000):
    """
    分块内积 + 归并 top-k。**返回 (rows, scores)**，已按分数降序。

    ⚠️ 返回顺序是「行号在前」。曾经写成 (scores, rows)，结果被调用方解包反了 ——
    分数 0.6 被当成行号 `int()` 成 0，100 个候选全塌成 row=0，
    在 RRF 里同一个 key 被累加 100 次，伪造出 0.99 的假高分。
    **全程不报错、不崩溃**，只在结果里表现为"命中了一条无关的 row=0"。
    凡是「同 dtype 的两个数组」成对返回，调用点必须显式命名，别用 `a, _ =`。
    """
    n, _ = emb.shape
    best_s = np.full(topk, -np.inf, dtype=np.float32)
    best_i = np.zeros(topk, dtype=np.int64)

    for s in range(0, n, block):
        e = min(s + block, n)
        blk = np.asarray(emb[s:e], dtype=np.float32)   # 819MB 临时，用完即释放
        sc = blk @ qv                                   # 局部相似度
        del blk

        k = min(topk, e - s)
        part = np.argpartition(-sc, k - 1)[:k]
        cand_s = np.concatenate([best_s, sc[part]])
        cand_i = np.concatenate([best_i, part.astype(np.int64) + s])
        keep = np.argpartition(-cand_s, topk - 1)[:topk]
        best_s, best_i = cand_s[keep], cand_i[keep]

    order = np.argsort(-best_s)
    return best_i[order], best_s[order]


def fetch_texts(parquet: Path, chunk_ids):
    """
    按需回查原文 —— **只取命中的那几条**。

    反例：`isin(list(全部 331.6 万个 id))`（smoke_search.py 里那个写法）在这个规模会卡死。
    顺序纪律：pyarrow 的 isin **不保证返回顺序**，所以必须建字典、再按 ids 顺序取（§7 坑 D）。
    """
    import pyarrow.compute as pc
    import pyarrow.dataset as ds_mod

    d = ds_mod.dataset(str(parquet), format="parquet")
    tbl = d.to_table(
        columns=["chunk_id", "title", "section", "chunk_text"],
        filter=pc.field("chunk_id").isin(list(chunk_ids)),
    )
    return {r["chunk_id"]: r for r in tbl.to_pylist()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(DEFAULT_INDEX))
    ap.add_argument("--input", default=str(DEFAULT_PARQUET))
    ap.add_argument("--query", action="append", default=None)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--block", type=int, default=200_000)
    ap.add_argument("--model-kind", default="large", choices=["large", "small"])
    ap.add_argument("--show", type=int, default=5, help="打印前几条")
    args = ap.parse_args()
    queries = args.query or DEFAULT_QUERIES

    print("=" * 76)
    print("第 8 步 · 向量检索（全量暴力 / exact）")
    print("=" * 76)

    t = time.time()
    ids, emb = load_index(Path(args.index))
    print(f"索引      : {args.index}")
    print(f"规模      : {emb.shape[0]:,} 条 × {emb.shape[1]} 维  dtype={emb.dtype}")
    print(f"加载耗时  : {time.time() - t:.1f} 秒（memmap，不占内存）")

    # ---- 编码 query ----
    t = time.time()
    from FlagEmbedding import FlagModel
    model = FlagModel(str(find_model(args.model_kind)), use_fp16=True)
    qv_all = np.asarray(
        model.encode(queries, batch_size=8, max_length=512)
    ).astype(np.float32)
    print(f"模型加载+编码: {time.time() - t:.1f} 秒（{len(queries)} 条 query）")
    print(f"分块大小  : {args.block:,} 行（峰值临时内存 ≈ {args.block * emb.shape[1] * 4 / 1e6:.0f} MB）")

    # ---- 逐条检索 ----
    hit_ids, results, latencies = [], [], []
    print()
    print("=" * 76)
    print("检索延迟实测")
    print("=" * 76)
    for q, qv in zip(queries, qv_all):
        t = time.time()
        rows, scores = scan_topk(emb, qv, args.topk, args.block)
        dt = time.time() - t
        latencies.append(dt)
        print(f"  {dt:6.3f} 秒   {q}")

        results.append((q, scores, rows))
        hit_ids.extend(ids[r] for r in rows)

    # ---- 回查原文 ----
    t = time.time()
    got = fetch_texts(Path(args.input), hit_ids)
    print(f"\n回查原文  : {len(got):,} / {len(set(hit_ids)):,} 条命中，{time.time() - t:.1f} 秒")

    # ---- 打印 ----
    for q, scores, rows in results:
        print()
        print("=" * 76)
        print(f"Q: {q}")
        print("=" * 76)
        for rank, (sc, j) in enumerate(zip(scores, rows), 1):
            if rank > args.show:
                break
            cid = ids[j]
            r = got.get(cid)
            if r is None:
                print(f"  {rank}. {sc:.4f}  [缺失] chunk_id={cid} row={j}")
                continue
            sec = r["section"] or "—"
            text = (r["chunk_text"] or "").replace("\n", " ")[:76]
            print(f"  {rank}. {sc:.4f}  {r['title']}  ·  {sec}")
            print(f"       {text}")
            print(f"       chunk_id={cid}  row={j}")

    print()
    print("=" * 76)
    print("SEARCH_VECTOR_OK")
    print(f"  索引条数 : {emb.shape[0]:,}")
    print(f"  topk     : {args.topk}")
    print(f"  查询数   : {len(queries)}")
    print(f"  延迟     : 平均 {float(np.mean(latencies)):.3f}s / 最大 {float(np.max(latencies)):.3f}s"
          f"  （阈值 2.0s —— 超了就上 faiss IVF，没超就不引入）")
    print("=" * 76)
    print("自检提示：top1 语义相关才算通过。若相似度很高但内容对不上，")
    print("         八成是 ids.txt 与 emb.npy 行号错位 —— 跑 check_vectors.py，别调参。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
