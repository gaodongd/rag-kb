# -*- coding: utf-8 -*-
"""从评测 JSON 生成消融对比表（markdown）。避免手抄数字出错。"""
import json
from pathlib import Path

R = Path(r"E:\AI-learning\projects\rag-kb\eval\results")
CFG = ["vector", "bm25", "hybrid", "hybrid_rerank"]
CN = {"vector": "纯向量", "bm25": "纯 BM25", "hybrid": "混合 RRF", "hybrid_rerank": "混合+重排"}


def load(tag):
    return json.loads((R / f"eval_{tag}.json").read_text(encoding="utf-8"))


def row(d, cfg, key="strict", metric="recall@5"):
    s = d["summary"][cfg]
    return s.get(key, {}).get(metric)


a = load("v1_final")        # 归一化开
b = load("v1_nonorm")       # 归一化关

print("### 表 1 · 字形归一化消融（关掉归一化后重跑同一套题）\n")
print(f"评测集 {a['meta']['testset']} · {a['meta']['n_queries']} 条库内题"
      f" · 池 {a['meta']['pool']}\n")

print("**整体 Recall@5（strict）**\n")
print("| 链路 | 归一化开 | 归一化关 | 差 |")
print("|---|---|---|---|")
for c in CFG:
    x, y = row(a, c), row(b, c)
    print(f"| {CN[c]} | {x:.3f} | {y:.3f} | **{y-x:+.3f}** |")

print("\n**跨字形子集（n={}）Recall@5（strict）—— 这才是归一化真正作用的地方**\n"
      .format(a["by_script"]["cross"][CFG[0]]["n"]))
print("| 链路 | 归一化开 | 归一化关 | 差 |")
print("|---|---|---|---|")
for c in CFG:
    x = a["by_script"]["cross"][c]["strict"]["recall@5"]
    y = b["by_script"]["cross"][c]["strict"]["recall@5"]
    print(f"| {CN[c]} | {x:.3f} | {y:.3f} | **{y-x:+.3f}** |")

print("\n**同字形子集（n={}）—— 对照组，应当几乎不变**\n"
      .format(a["by_script"]["same"][CFG[0]]["n"]))
print("| 链路 | 归一化开 | 归一化关 | 差 |")
print("|---|---|---|---|")
for c in CFG:
    x = a["by_script"]["same"][c]["strict"]["recall@5"]
    y = b["by_script"]["same"][c]["strict"]["recall@5"]
    print(f"| {CN[c]} | {x:.3f} | {y:.3f} | {y-x:+.3f} |")

print("\n**按题型（归一化关，Recall@5 strict）—— 看繁体题是不是全灭**\n")
kinds = ["easy", "hard_anon", "hard_para", "t2s", "s2t"]
KCN = {"easy": "基础事实", "hard_anon": "实体匿名", "hard_para": "释义改写",
       "t2s": "繁问简答", "s2t": "简问繁答"}
print("| 题型 | n | 归一化开·BM25 | 归一化关·BM25 | 差 |")
print("|---|---|---|---|---|")
for k in kinds:
    if k not in a["by_kind"]:
        continue
    n = a["by_kind"][k][CFG[0]]["n"]
    x = a["by_kind"][k]["bm25"]["strict"]["recall@5"]
    y = b["by_kind"][k]["bm25"]["strict"]["recall@5"]
    mark = "  ← 全灭" if y == 0 else ""
    print(f"| {KCN[k]} | {n} | {x:.3f} | {y:.3f} | **{y-x:+.3f}**{mark} |")

print("\n---\n")
print("### 表 2 · 检索链路总表（归一化开，strict）\n")
print("| 链路 | R@1 | R@5 | R@10 | MRR@10 |")
print("|---|---|---|---|---|")
for c in CFG:
    s = a["summary"][c]["strict"]
    print(f"| {CN[c]} | {s['recall@1']:.3f} | {s['recall@5']:.3f} | "
          f"{s['recall@10']:.3f} | {s['mrr@10']:.3f} |")

print("\n### 表 3 · 分题型 Recall@5（strict，归一化开）\n")
print("| 题型 | n | 纯向量 | 纯 BM25 | 混合 RRF | 混合+重排 |")
print("|---|---|---|---|---|---|")
for k in ["easy", "hard_anon", "hard_para", "t2s", "s2t"]:
    if k not in a["by_kind"]:
        continue
    blk = a["by_kind"][k]
    n = blk[CFG[0]]["n"]
    cells = " | ".join(f"{blk[c]['strict']['recall@5']:.3f}" for c in CFG)
    print(f"| {KCN[k]} | {n} | {cells} |")

print("\n### 表 4 · 延迟（毫秒，归一化开）\n")
print("| 环节 | 均值 | 中位 |")
print("|---|---|---|")
for k, v in a["latency_ms"].items():
    print(f"| {k} | {v['mean']:.1f} | {v['p50']:.1f} |")
rs = a["reranker_stats"]
print(f"\nreranker：加载 {rs['load_seconds']}s · {rs['total_pairs']} 对 · {rs['ms_per_pair']} ms/对")
