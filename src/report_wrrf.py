# -*- coding: utf-8 -*-
"""
加权 RRF 权重扫描的报表（读评测 JSON，避免手抄数字）。

背景（第 11 步 11.5）：等权 RRF 在「实体匿名」子集上 **反而低于纯 BM25**
（0.781 vs 0.812）—— 弱路（向量在该子集只有 0.375）的候选稀释了强路结果。
本脚本量：把向量路权重降下来，能不能救回匿名子集、同时不伤基础事实题。

自带对照组校验：向量路 / BM25 路**不依赖融合权重**，所以四组跑出来的
这两个数字必须**逐位相同**。不同就说明实验被别的变量污染了，直接报错。

用法：python src\\report_wrrf.py
"""
import json
from pathlib import Path

R = Path(r"E:\AI-learning\projects\rag-kb\eval\results")
CFG = ["vector", "bm25", "hybrid"]
CN = {"vector": "纯向量", "bm25": "纯 BM25", "hybrid": "混合 RRF"}
KINDS = ["easy", "hard_anon", "hard_para", "t2s", "s2t"]
KCN = {"easy": "基础事实", "hard_anon": "实体匿名", "hard_para": "释义改写",
       "t2s": "繁问简答", "s2t": "简问繁答"}

# (tag, w_vec, w_bm25) —— 只跑 rrf 融合，关重排
RUNS = [("wrrf_1_1", 1.0, 1.0), ("wrrf_1_2", 1.0, 2.0),
        ("wrrf_1_3", 1.0, 3.0), ("wrrf_2_1", 2.0, 1.0)]


def load(tag):
    return json.loads((R / f"eval_{tag}.json").read_text(encoding="utf-8"))


runs = [(t, a, b, load(t)) for t, a, b in RUNS]
base = runs[0][3]


def r5(d, cfg, kind=None):
    blk = d["by_kind"][kind] if kind else d
    return blk[cfg]["strict"]["recall@5"]


# ---------------------------------------------------------------- 对照组合法性
print("## 0 · 对照组合法性自检（加权不该影响单路）\n")
bad = []
for t, _a, _b, d in runs:
    for c in ("vector", "bm25"):
        if d["summary"][c]["strict"] != base["summary"][c]["strict"]:
            bad.append(f"{t} 的 {c}")
if bad:
    raise SystemExit("[失败] 单路指标被改了 → 变量没隔离干净：" + "、".join(bad))
print("✅ 四组的纯向量 / 纯 BM25 指标**逐位相同**（本来就与融合权重无关）")
n = base["summary"]["vector"]["n"]
print(f"   库内题 {n} 条 · 池 {base['meta']['pool']} · RRF k={base['meta']['rrf_k']} · "
      f"nprobe={base['meta']['nprobe']}\n")

# ---------------------------------------------------------------- 主表
print("## 1 · 整体指标（strict）\n")
print("| w_vec : w_bm25 | R@1 | R@5 | R@10 | MRR@10 |")
print("|---|---|---|---|---|")
for (t, wv, wb, d) in runs:
    s = d["summary"]["hybrid"]["strict"]
    tag = f"{wv:g} : {wb:g}" + ("（等权基线）" if (wv, wb) == (1.0, 1.0) else "")
    print(f"| {tag} | {s['recall@1']:.3f} | {s['recall@5']:.3f} | "
          f"{s['recall@10']:.3f} | {s['mrr@10']:.3f} |")
print(f"\n（对照）纯向量 R@5 = {base['summary']['vector']['strict']['recall@5']:.3f} · "
      f"纯 BM25 R@5 = {base['summary']['bm25']['strict']['recall@5']:.3f}")

# ---------------------------------------------------------------- 分题型
print("\n## 2 · 分题型 R@5（strict）—— 关键看「实体匿名」能不能救回来\n")
head = " | ".join(f"{wv:g}:{wb:g}" for _t, wv, wb, _d in runs)
print(f"| 题型 | n | {head} |")
print("|---" * (3 + len(runs)) + "|")
for k in KINDS:
    if k not in base["by_kind"]:
        continue
    nn = base["by_kind"][k]["vector"]["n"]
    cells = " | ".join(f"{r5(d, 'hybrid', k):.3f}" for _t, _wv, _wb, d in runs)
    print(f"| {KCN[k]} | {nn} | {cells} |")
print("\n（参考）同题型下单路表现 —— 单路不随权重变化：\n")
print(f"| 题型 | n | 纯向量 | 纯 BM25 |")
print("|---|---|---|---|")
for k in KINDS:
    if k not in base["by_kind"]:
        continue
    nn = base["by_kind"][k]["vector"]["n"]
    print(f"| {KCN[k]} | {nn} | {r5(base, 'vector', k):.3f} | {r5(base, 'bm25', k):.3f} |")

# ---------------------------------------------------------------- 字形分组
print("\n## 3 · 字形分组 R@5（strict）\n")
print(f"| 字形 | n | {head} |")
print("|---" * (3 + len(runs)) + "|")
for tg, cn in (("same", "同字形"), ("cross", "跨字形")):
    blk = base["by_script"][tg]
    nn = blk["vector"]["n"]
    cells = " | ".join(f"{d['by_script'][tg]['hybrid']['strict']['recall@5']:.3f}"
                       for _t, _wv, _wb, d in runs)
    print(f"| {cn} | {nn} | {cells} |")

# ---------------------------------------------------------------- 逐题胜负
print("\n## 4 · 逐题分析：加权到底改了什么\n")
print("以等权基线为参照，统计**每一题上融合结果相对单路的变化**：\n")
print("| 权重 | 混合比纯BM25差 | 混合比纯向量差 | 混合两路都赢 | 混合两路都输 |")
print("|---|---|---|---|---|")
for (t, wv, wb, d) in runs:
    worse_b = worse_v = both_win = both_lose = 0
    for q in d["per_query"]:
        h = q["ranks"]["hybrid"]["strict"]
        b = q["ranks"]["bm25"]["strict"]
        v = q["ranks"]["vector"]["strict"]
        inf = 10**9
        hh, bb, vv = (h or inf), (b or inf), (v or inf)
        if hh > bb:
            worse_b += 1
        if hh > vv:
            worse_v += 1
        if hh < bb and hh < vv:
            both_win += 1
        if hh > bb and hh > vv:
            both_lose += 1
    print(f"| {wv:g}:{wb:g} | {worse_b} | {worse_v} | {both_win} | {both_lose} |")

# ---------------------------------------------------------------- 结论提示
print("\n## 5 · 自动判读\n")
best = min(runs, key=lambda r: -r[3]["summary"]["hybrid"]["strict"]["recall@5"])
anon = {f"{wv:g}:{wb:g}": r5(d, "hybrid", "hard_anon") for _t, wv, wb, d in runs}
easy = {f"{wv:g}:{wb:g}": r5(d, "hybrid", "easy") for _t, wv, wb, d in runs}
ba = base["by_kind"]["hard_anon"]["bm25"]["strict"]["recall@5"]
print(f"- 整体 R@5 最优权重：**{best[1]:g} : {best[2]:g}**"
      f"（{best[3]['summary']['hybrid']['strict']['recall@5']:.3f}）")
print(f"- 匿名子集：纯 BM25 = {ba:.3f}；等权混合 = {anon['1:1']:.3f}"
      f"（{'低于' if anon['1:1'] < ba else '不低于'}单路）")
for k, v in anon.items():
    flag = "← 已追平/超过纯 BM25" if v >= ba else ""
    print(f"    - {k} → {v:.3f} {flag}")
print(f"- 基础事实题随权重变化：{easy}")
