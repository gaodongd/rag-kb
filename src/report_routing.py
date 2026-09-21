# -*- coding: utf-8 -*-
"""
查询路由实验的报表。

要回答的问题（承接 11.11 的结论「全局权重救不了，两个子集需要相反的权重」）：
    按查询级信号逐题选权重，能不能**两边都不掉**？
    验收标准（11.8 定的）：匿名子集 ≥ 0.812 且 基础事实 ≥ 0.956。

关键设计：**oracle 上界**。
    既然每题的 hybrid 排名在不同权重下都记在 JSON 里，就可以算
    「如果每题都能选对权重，指标最高能到多少」= min(rank_1_1, rank_1_2)。
    没有这个上界，看到路由只涨一点点，无法判断是"信号不好"还是"根本没空间"。

用法：python src\\report_routing.py
"""
import json
from pathlib import Path

R = Path(r"E:\AI-learning\projects\rag-kb\eval\results")
KINDS = ["easy", "hard_anon", "hard_para", "t2s", "s2t"]
KCN = {"easy": "基础事实", "hard_anon": "实体匿名", "hard_para": "释义改写",
       "t2s": "繁问简答", "s2t": "简问繁答"}
INF = 10 ** 9


def load(tag):
    return json.loads((R / f"eval_{tag}.json").read_text(encoding="utf-8"))


EQ = load("wrrf_1_1")        # 等权基线（融合，无 gate）
ST = load("wrrf_1_2")        # 全局降到 1:2（已知：救 anon，伤 easy）
RUNS = [("route_s2_m10", "路由·门限10%"), ("route_s2_m30", "路由·门限30%"),
        ("route_orig_gate", "原版gate·弱×0.3")]


def r5(d, cfg="hybrid", kind=None):
    blk = d["by_kind"][kind] if kind else d
    return blk[cfg]["strict"]["recall@5"]


def overall(d):
    return d["summary"]["hybrid"]["strict"]


print("## 1 · 总览（strict，n=104）\n")
print("| 方案 | R@1 | R@5 | MRR@10 |")
print("|---|---|---|---|")
for d, name in [(EQ, "等权基线（1:1）"), (ST, "全局 1:2（已知：救 anon 伤 easy）")]:
    s = overall(d)
    print(f"| {name} | {s['recall@1']:.3f} | {s['recall@5']:.3f} | {s['mrr@10']:.3f} |")
for tag, name in RUNS:
    try:
        s = overall(load(tag))
    except FileNotFoundError:
        continue
    print(f"| {name} | {s['recall@1']:.3f} | {s['recall@5']:.3f} | {s['mrr@10']:.3f} |")

print("\n## 2 · 两个关键子集（验收标准：anon ≥ 0.812 且 easy ≥ 0.956）\n")
print("| 方案 | 基础事实(45) | 实体匿名(32) | 释义改写(14) |")
print("|---|---|---|---|")
print(f"| 等权基线 | {r5(EQ,'hybrid','easy'):.3f} | {r5(EQ,'hybrid','hard_anon'):.3f} | "
      f"{r5(EQ,'hybrid','hard_para'):.3f} |")
print(f"| 纯 BM25（参考） | {r5(EQ,'bm25','easy'):.3f} | {r5(EQ,'bm25','hard_anon'):.3f} | "
      f"{r5(EQ,'bm25','hard_para'):.3f} |")
print(f"| 全局 1:2 | {r5(ST,'hybrid','easy'):.3f} | {r5(ST,'hybrid','hard_anon'):.3f} | "
      f"{r5(ST,'hybrid','hard_para'):.3f} |")
for tag, name in RUNS:
    try:
        d = load(tag)
    except FileNotFoundError:
        continue
    print(f"| {name} | {r5(d,'hybrid','easy'):.3f} | {r5(d,'hybrid','hard_anon'):.3f} | "
          f"{r5(d,'hybrid','hard_para'):.3f} |")

# ---------------------------------------------------------------- oracle 上界
print("\n## 3 · oracle 上界：完美路由能到多少（判断还有没有空间）\n")
eq = {q["qid"]: q for q in EQ["per_query"]}
st = {q["qid"]: q for q in ST["per_query"]}
common = [k for k in eq if k in st]
print("| 口径 | 等权 1:1 | 全局 1:2 | **oracle（逐题取更优）** |")
print("|---|---|---|---|")
for label, sel in (("全部 104 题", lambda r: True),
                   ("仅基础事实", lambda r: r["kind"] == "easy"),
                   ("仅实体匿名", lambda r: r["kind"] == "hard_anon")):
    a = [eq[k]["ranks"]["hybrid"]["strict"] or INF for k in common if sel(eq[k])]
    b = [st[k]["ranks"]["hybrid"]["strict"] or INF for k in common if sel(st[k])]
    o = [min(x, y) for x, y in zip(a, b)]
    n = len(o)

    def f(v):
        return sum(1 for x in v if x <= 5) / n

    print(f"| {label}（n={n}） | {f(a):.3f} | {f(b):.3f} | **{f(o):.3f}** |")
print("\n（oracle 是**上界**，不是可达方案：它假设每题都知道该用哪个权重。）")

# ---------------------------------------------------------------- 路由分布
print("\n## 4 · 路由实际生效了多少题\n")
for tag, name in RUNS:
    try:
        d = load(tag)
    except FileNotFoundError:
        continue
    m = d["meta"]
    leads = [q["bm25_lead"] for q in d["per_query"] if q.get("bm25_lead") is not None]
    strong = sum(1 for x in leads if x >= m["gate_margin"])
    print(f"- **{name}**（门限 {m['gate_margin']:.0%} · 强×{m['gate_strong_w']} "
          f"弱×{m['gate_weak_w']}）：{len(leads)} 题里 **{strong}** 题走了强匹配分支"
          f"（{strong/len(leads):.1%}），其余 {len(leads)-strong} 题走弱分支")

# ---------------------------------------------------------------- 信号质量
print("\n## 5 · 信号本身有没有区分度（路由的**前提**，比结果更重要）\n")
# ⚠️ 用带 bm25_lead 的那次跑（基线 wrrf_1_1 早于该字段的引入，没有这个数据）。
#    单路排名不随 gate 变化，所以拿它分析信号是等价的。
SIG = None
for tag, _n in RUNS:
    try:
        cand = load(tag)
    except FileNotFoundError:
        continue
    if any(q.get("bm25_lead") is not None for q in cand["per_query"]):
        SIG = cand
        break
rows = [q for q in SIG["per_query"] if q.get("bm25_lead") is not None] if SIG else []
g = {"BM25 更好": [], "向量更好": [], "打平": []}
for q in rows:
    b = q["ranks"]["bm25"]["strict"] or INF
    v = q["ranks"]["vector"]["strict"] or INF
    g["打平" if b == v else ("BM25 更好" if b < v else "向量更好")].append(q["bm25_lead"])
print("| 组 | n | 领先幅度中位 | 领先幅度均值 |")
print("|---|---|---|---|")
for k, v in g.items():
    if v:
        sv = sorted(v)
        print(f"| {k} | {len(v)} | {sv[len(sv)//2]:.1%} | {sum(v)/len(v):.1%} |")
bb, vv = g["BM25 更好"], g["向量更好"]
if bb and vv:
    sb, sv = sorted(bb), sorted(vv)
    print(f"\n中位差：{sb[len(sb)//2] - sv[len(sv)//2]:+.1%} —— "
          f"{'有区分度' if sb[len(sb)//2] > sv[len(sv)//2] else '⚠️ 没有区分度（信号不能用）'}")
print("\n**注意**：这里只用了「哪路把 gold 排得更前」，不等于「哪路对融合更有利」 ——")
print("后者才是路由真正要预测的东西，两者可能不一致。")

# ---------------------------------------------------------------- 显著性
print("\n## 6 · 配对检验：提升是不是噪声\n")
from math import comb  # noqa: E402


def binom_p(b: int, c: int) -> float:
    """McNemar 精确检验（双尾）：只看不一致的那 b+c 对。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


print("以「strict 命中该名次」为一票，做 McNemar 配对检验（只看两边不一致的题）：\n")
print("| 对比 | 名次 | 基线命中 | 新方案命中 | 基线独中 | 新独中 | p 值 | 结论 |")
print("|---|---|---|---|---|---|---|---|")
for tag, name in RUNS:
    try:
        d = load(tag)
    except FileNotFoundError:
        continue
    for K in (1, 5):
        b_i = c_i = both = 0
        for q in EQ["per_query"]:
            new = next((x for x in d["per_query"] if x["qid"] == q["qid"]), None)
            if new is None:
                continue
            a = (q["ranks"]["hybrid"]["strict"] or INF) <= K
            bb = (new["ranks"]["hybrid"]["strict"] or INF) <= K
            if a and bb:
                both += 1
            elif a:
                b_i += 1
            elif bb:
                c_i += 1
        p = binom_p(b_i, c_i)
        verdict = "**显著**" if p < 0.05 else ("接近显著" if p < 0.15 else "噪声范围")
        print(f"| {name} vs 等权 | R@{K} | {both + b_i} | {both + c_i} | {b_i} | {c_i} | "
              f"{p:.3f} | {verdict} |")
print("\n⚠️ 结论必须这样写：**提升量要同时看幅度和显著性**。")
print("   n=104 时 R@1 的 1 个标准差 ≈ 0.048 —— 小于这个数的「提升」不能宣称有效。")
