#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 7 步验收 · 向量语义冒烟：随便问几个问题，看最近邻是不是真的相关。

为什么必须做这一步
------------------
"形状对、范数对、无 NaN" 只能证明**代码没写错**，证明不了**向量可用**。
检索系统里最贵的错误是「形状全对但语义全错」—— 典型成因：
  * 拼串顺序搞反（把 section 和正文写颠倒了）
  * 把 chunk_id 和向量的行号错位（续跑时最容易发生）
  * tokenizer 截断把关键内容切掉了
这三种都不报错，只有真查一次才暴露。

用法
----
    :: 用默认几组问题
    "...Python313\\python.exe" src\\smoke_search.py --index eval\\results\\vec_smoke

    :: 自定义问题
    "...python.exe" src\\smoke_search.py --index data\\index\\bge-large-zh-v1.5 ^
        --query "台灣東部的古道" --query "中國化學工程學家" --topk 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vectorize import find_model  # noqa: E402

PROJECT = HERE.parent

DEFAULT_QUERIES = [
    "台灣東部開發於古時的人行道路",
    "中國的化學工程學家",
    "這個地方氣候怎麼樣",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(PROJECT / "eval" / "results" / "vec_smoke"))
    ap.add_argument("--input", default=str(PROJECT / "data" / "processed" / "chunks.parquet"))
    ap.add_argument("--query", action="append", default=None)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--model-kind", default="large", choices=["large", "small"])
    args = ap.parse_args()
    queries = args.query or DEFAULT_QUERIES

    idx = Path(args.index)
    ids = (idx / "ids.txt").read_text(encoding="utf-8").split()
    emb = np.load(idx / "emb.npy", mmap_mode="r")
    assert emb.shape[0] == len(ids), \
        f"向量行数({emb.shape[0]}) 与 ids 行数({len(ids)}) 不一致 —— 索引已损坏"
    print(f"索引：{idx}   {emb.shape[0]:,} 条 × {emb.shape[1]} 维")

    # ---- 按 ids 顺序取回原文（顺序必须与 emb 行号严格对齐） ----
    import pyarrow.compute as pc
    import pyarrow.dataset as ds_mod
    d = ds_mod.dataset(args.input, format="parquet")
    want = set(ids)
    tbl = d.to_table(columns=["chunk_id", "title", "section", "chunk_text"],
                     filter=pc.field("chunk_id").isin(list(want)))
    got = {r["chunk_id"]: r for r in tbl.to_pylist()}
    print(f"从 parquet 取回原文：{len(got):,} / {len(ids):,} 条命中")
    if len(got) < len(ids):
        print(f"  ⚠️ 有 {len(ids) - len(got):,} 条 chunk_id 在 parquet 里找不到")

    # ---- 编码 query ----
    import torch
    from FlagEmbedding import FlagModel
    model = FlagModel(str(find_model(args.model_kind)), use_fp16=True)
    qv = np.asarray(model.encode(queries, batch_size=8, max_length=512)).astype(np.float32)
    print(f"query 已编码：{qv.shape}\n")

    # 向量已 L2 归一化 → 内积即余弦相似度
    sims = qv @ np.asarray(emb, dtype=np.float32).T

    for qi, q in enumerate(queries):
        order = np.argsort(-sims[qi])[:args.topk]
        print("=" * 72)
        print(f"Q{qi + 1}: {q}")
        print("=" * 72)
        for rank, j in enumerate(order, 1):
            r = got.get(ids[j])
            if r is None:
                print(f"  {rank}. [缺失] row={j}")
                continue
            sec = r["section"] or "—"
            text = (r["chunk_text"] or "").replace("\n", " ")[:70]
            print(f"  {rank}. {sims[qi][j]:.4f}  {r['title']}  ·  {sec}")
            print(f"       {text}")
        print()

    print("自检提示：看 top1 是否语义相关。若 top1 明显不相关但相似度很高，")
    print("         八成是「ids.txt 与 emb.npy 行号错位」—— 重跑向量化，别调参。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
