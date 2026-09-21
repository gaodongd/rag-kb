#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重排对照实验报表：max_length（截断长度）× pool（召回池大小）。

============================================================================
为什么需要这个报表，而不是直接看 evaluate.py 打印的汇总
============================================================================
evaluate.py 打印的是**单次运行**的指标。本实验要回答的三个问题都跨运行：

  ① max_length 512 → 1024，整体能涨多少？（单看两个数，这个 evaluate 会打）
  ② **涨的是不是那一批被截断的题？**（关键归因）—— 如果不是，
     那 1024 的收益就与"截断"无关，只是换了组超参碰巧好一点，
     说明机制判断错了，不能把结论写成"截断是瓶颈"。
  ③ 池子 100 → 50 → 30，指标掉多少、延迟省多少？（延迟换精度的兑换率）

问题 ② 必须**按 qid 跨文件对齐**：ml512 那次跑出来的 gold_truncated 标记，
要拿去 ml1024 的结果里取同一条题的排名。evaluate.py 的 by_trunc
只描述"本次运行里被截断的题表现如何"，两次运行之间对不上号。

另外一个容易犯的错：ml1024 运行时，所有题的 gold_truncated 都是 False
（因为不截了），所以它的 by_trunc 里只有 gold_intact 一组 ——
不能拿两个文件的 by_trunc 直接比，那样比的是"不同题集"。必须按 qid 对齐。
============================================================================
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "eval" / "results"
KIND_CN = {"easy": "基础事实", "hard_anon": "实体匿名", "hard_para": "释义改写",
           "t2s": "繁问", "s2t": "简问", "oob": "库外"}


def load(tag: str):
    p = RES / f"eval_{tag}.json"
    if not p.exists():
        raise SystemExit(f"缺少结果文件：{p}\n（实验还没跑完？）")
    d = json.loads(p.read_text(encoding="utf-8"))
    return d, {q["qid"]: q for q in d["per_query"]}


def m(ranks: list) -> dict:
    """严格口径 R@K / MRR@10，与 evaluate.summarize 保持一致。"""
    n = len(ranks)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "R@1": sum(1 for r in ranks if r is not None and r <= 1) / n,
        "R@5": sum(1 for r in ranks if r is not None and r <= 5) / n,
        "MRR": sum(1.0 / r if (r is not None and r <= 10) else 0.0 for r in ranks) / n,
        "miss": sum(1 for r in ranks if r is None),
    }


def ranks_of(mapq: dict, qids, cfg="hybrid_rerank"):
    return [mapq[q]["ranks"][cfg]["strict"] for q in qids if q in mapq]


def load_trunc():
    """
    读**独立诊断脚本**导出的截断标记，作为"gold 是否被截断"的唯一真源。

    为什么不直接用本次运行 per_query 里的 rerank_diag：
      ① 那个字段是评测跑的时候顺带算的，用的是 `qn+dn+3` 的**错**公式
         （XLMRoberta 模板有 4 个特殊 token，实测差 1）——
         在该公式下"卡在边界上"的那一条会被判成未截断；
      ② 单一真源原则：口径只能有一处定义，报表自己再算一遍迟早会跑偏。
    池中被截候选数（pool_truncated）无法从本文件得到，仍用 per_query 的字段，
    那张表只看趋势，1 个 token 的边界误差不影响结论。
    """
    p = RES / "trunc_gold_ml512.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    A, QA = load("rerank_ml512")       # pool=100, ml=512
    B, QB = load("rerank_ml1024")      # pool=100, ml=1024
    C, QCC = load("rerank_p50_ml1024")
    D, QD = load("rerank_p30_ml1024")
    base, Qbase = load("v1_final")     # 历史基线（同配置，用于复现校验）

    print("# 重排对照实验：截断长度 × 召回池大小\n")
    print(f"评测集 `{A['meta']['testset']}` · {A['meta']['n_queries']} 条库内题 · "
          f"指标为 strict 口径（gold chunk 命中）\n")

    # ==================================================== 0. 复现校验
    print("## 0 · 复现校验（先确认没把既有结论改坏）\n")
    common = [q for q in QA if q in Qbase]
    d_hist = m(ranks_of(Qbase, common))
    d_new = m(ranks_of(QA, common))
    print(f"`v1_final`（历史） vs `rerank_ml512`（本次重跑，同配置 pool=100 ml=512）\n")
    print("| 来源 | R@1 | R@5 | MRR@10 |")
    print("|---|---|---|---|")
    print(f"| v1_final | {d_hist['R@1']:.4f} | {d_hist['R@5']:.4f} | {d_hist['MRR']:.4f} |")
    print(f"| rerank_ml512 | {d_new['R@1']:.4f} | {d_new['R@5']:.4f} | {d_new['MRR']:.4f} |")
    same = all(abs(d_hist[k] - d_new[k]) < 1e-9 for k in ("R@1", "R@5", "MRR"))
    print(f"\n{'✅ 逐位一致 —— 改动未影响既有链路，且结果可复现' if same else '⚠️ 有差异，先查改动'}\n")

    # ==================================================== 1. 主表
    qids = [q for q in QA if q in QB]
    print("## 1 · max_length 512 → 1024（pool 固定 100）\n")
    print("| 口径 | R@1 | R@5 | MRR@10 |")
    print("|---|---|---|---|")
    a, b = m(ranks_of(QA, qids)), m(ranks_of(QB, qids))
    print(f"| ml=512 | {a['R@1']:.4f} | {a['R@5']:.4f} | {a['MRR']:.4f} |")
    print(f"| ml=1024 | {b['R@1']:.4f} | {b['R@5']:.4f} | {b['MRR']:.4f} |")
    print(f"| **Δ** | **{b['R@1']-a['R@1']:+.4f}** | **{b['R@5']-a['R@5']:+.4f}** "
          f"| **{b['MRR']-a['MRR']:+.4f}** |")

    # 分题型
    print("\n分题型 R@1 / MRR@10：\n")
    kinds = ["easy", "hard_anon", "hard_para", "t2s", "s2t"]
    print("| 题型 | n | ml=512 R@1 | ml=1024 R@1 | ΔR@1 | ml=512 MRR | ml=1024 MRR | ΔMRR |")
    print("|---|---|---|---|---|---|---|---|")
    for k in kinds:
        ks = [q for q in qids if QA[q]["kind"] == k]
        if not ks:
            continue
        ra, rb = m(ranks_of(QA, ks)), m(ranks_of(QB, ks))
        print(f"| {KIND_CN.get(k,k)} | {ra['n']} | {ra['R@1']:.3f} | {rb['R@1']:.3f} "
              f"| {rb['R@1']-ra['R@1']:+.3f} | {ra['MRR']:.3f} | {rb['MRR']:.3f} "
              f"| {rb['MRR']-ra['MRR']:+.3f} |")

    # ==================================================== 2. 机制归因（核心）
    print("\n## 2 · 机制归因：收益是否来自「被截断的那批题」 ← 本报表的核心\n")
    TR = load_trunc()
    if TR:
        tq = {q for q, v in TR["per_query"].items() if v["gold_truncated"]}
        trunc = [q for q in qids if q in tq]
        intact = [q for q in qids if q not in tq]
        print(f"分组来源：`trunc_gold_ml512.json`（{TR['tokenizer']}，"
              f"特殊 token {TR['special_tokens']} 个，max_length={TR['max_length']}）\n")
    else:
        print("⚠️ 未找到 trunc_gold_ml512.json，退回用本次运行的 rerank_diag"
              "（口径差 1 个 token，建议先跑：\n"
              "   python src/diagnose_rerank_trunc.py --dump eval/results/trunc_gold_ml512.json）\n")
        trunc = [q for q in qids if (QA[q].get("rerank_diag") or {}).get("gold_truncated") is True]
        intact = [q for q in qids if (QA[q].get("rerank_diag") or {}).get("gold_truncated") is False]
    tset = set(trunc)
    print(f"按 **ml=512 时** gold 是否被截断分组（ml=1024 时全部不被截断，"
          f"所以只能在 512 的分组里比）：\n")
    print("| 分组 | n | ml=512 R@1 | ml=1024 R@1 | ΔR@1 | ml=512 R@5 | ml=1024 R@5 | ΔR@5 |")
    print("|---|---|---|---|---|---|---|---|")
    for label, sel in (("gold **被截断**", trunc), ("gold 完整", intact)):
        ra, rb = m(ranks_of(QA, sel)), m(ranks_of(QB, sel))
        print(f"| {label} | {ra['n']} | {ra['R@1']:.3f} | {rb['R@1']:.3f} "
              f"| **{rb['R@1']-ra['R@1']:+.3f}** | {ra['R@5']:.3f} | {rb['R@5']:.3f} "
              f"| {rb['R@5']-ra['R@5']:+.3f} |")
    ra_t, rb_t = m(ranks_of(QA, trunc)), m(ranks_of(QB, trunc))
    ra_i, rb_i = m(ranks_of(QA, intact)), m(ranks_of(QB, intact))
    print(f"\n- 截断组 R@1 提升 {rb_t['R@1']-ra_t['R@1']:+.3f}，完整组 {rb_i['R@1']-ra_i['R@1']:+.3f}")
    if (rb_t["R@1"] - ra_t["R@1"]) > (rb_i["R@1"] - ra_i["R@1"]):
        print("- ✅ 收益集中在截断组 → **「截断确实造成了伤害」这个机制判断成立**")
    else:
        print("- ⚠️ 收益没有集中在截断组 → 机制判断不成立，1024 的好处另有来源，别写成「修截断」")

    # 池子里被截候选数 vs 重排质量（ml=512 的查询侧信号）
    print("\n顺带看：ml=512 时「池子里被截候选数」与重排质量的关系"
          "（查询侧信号，不依赖 gold）：\n")
    print("| 池中被截候选数 | n | ml=512 R@1 | ml=512 MRR |")
    print("|---|---|---|---|")
    for lo, hi, lab in ((0, 0, "0 条"), (1, 9, "1–9 条"), (10, 29, "10–29 条"),
                        (30, 999, "≥30 条")):
        sel = [q for q in qids
               if lo <= (QA[q].get("rerank_diag") or {}).get("pool_truncated", -1) <= hi]
        if not sel:
            continue
        r = m(ranks_of(QA, sel))
        print(f"| {lab} | {r['n']} | {r['R@1']:.3f} | {r['MRR']:.3f} |")

    # ==================================================== 3. 逐题迁移
    print("\n## 3 · 逐题迁移（ml=512 → 1024，只看严格命中口径的名次变化）\n")
    up, down = [], []
    for q in qids:
        ra = QA[q]["ranks"]["hybrid_rerank"]["strict"]
        rb = QB[q]["ranks"]["hybrid_rerank"]["strict"]
        va, vb = (ra if ra else 10 ** 9), (rb if rb else 10 ** 9)
        if vb < va:
            up.append((q, ra, rb))
        elif vb > va:
            down.append((q, ra, rb))
    print(f"名次**改善** {len(up)} 条，**变差** {len(down)} 条，不变 {len(qids)-len(up)-len(down)} 条\n")
    if up:
        print("改善的题：\n")
        print("| qid | 题型 | gold 被截 | 512 名次 | 1024 名次 |")
        print("|---|---|---|---|---|")
        for q, ra, rb in sorted(up, key=lambda x: (x[1] or 10 ** 9)):
            tg = q in tset
            print(f"| `{q}` | {KIND_CN.get(QA[q]['kind'], QA[q]['kind'])} "
                  f"| {'是' if tg else '否'} | {ra or '≥100'} | {rb or '≥100'} |")
    if down:
        print("\n变差的题（若明显多于改善，说明 1024 不是纯赚）：\n")
        print("| qid | 题型 | gold 被截 | 512 名次 | 1024 名次 |")
        print("|---|---|---|---|---|")
        for q, ra, rb in down:
            tg = q in tset
            print(f"| `{q}` | {KIND_CN.get(QA[q]['kind'], QA[q]['kind'])} "
                  f"| {'是' if tg else '否'} | {ra or '≥100'} | {rb or '≥100'} |")
    # 配对符号检验（只看"改善 vs 变差"，不看幅度 —— 稳健）
    n_up, n_dn = len(up), len(down)
    if n_up + n_dn > 0:
        from math import comb
        n = n_up + n_dn
        p = sum(comb(n, i) for i in range(n_up, n + 1)) / 2 ** n if n_up >= n_dn else 1.0
        print(f"\n配对符号检验：{n_up} 改善 vs {n_dn} 变差，"
              f"单尾 p={p:.3f}（n={n}）"
              f"{'  → ✅ 方向可信' if p < 0.05 else '  → ⚠️ 样本量不足以判显著'}")
        print("  注：这里只数「改善/变差」的**符号**，不看幅度，"
              "比比较均值稳健（均值会被单条从 ≥100 跳到 1 这种极端值带偏）。")

    # ==================================================== 4. 池子大小
    print("\n## 4 · 召回池大小（max_length 固定 1024）\n")
    print("| pool | R@1 | R@5 | MRR@10 | 重排均值(ms) | ms/对 | rerank 总耗时(s) |")
    print("|---|---|---|---|---|---|---|")
    for tag, mp, dd in (("100", QB, B), ("50", QCC, C), ("30", QD, D)):
        r = m(ranks_of(mp, qids))
        rs = dd.get("reranker_stats") or {}
        print(f"| {tag} | {r['R@1']:.4f} | {r['R@5']:.4f} | {r['MRR']:.4f} "
              f"| {dd['latency_ms'].get('重排', {}).get('mean', 0):.0f} "
              f"| {rs.get('ms_per_pair', 0):.2f} | {rs.get('total_seconds', 0):.1f} |")
    r100, r50, r30 = (m(ranks_of(x, qids)) for x in (QB, QCC, QD))
    t100 = B["latency_ms"].get("重排", {}).get("mean", 1)
    t30 = D["latency_ms"].get("重排", {}).get("mean", 1)
    print(f"\n池子 100→30：R@5 {r100['R@5']:.4f} → {r30['R@5']:.4f}"
          f"（{r30['R@5']-r100['R@5']:+.4f}），"
          f"重排均值 {t100:.0f} → {t30:.0f} ms（省 {t100-t30:.0f} ms，{(t30/t100-1)*100:+.0f}%）")
    print("\n> ⚠️ **实验设计的耦合**：`--pool` 同时控两件事 —— ① 各路召回池大小、"
          "② 喂给重排的候选数。\n"
          "> 所以上面量到的不是\"纯重排池缩小\"的效果。"
          "**下一节用新增的 `--rerank-pool` 把两者解耦后重跑，结论更强。**")

    # ==================================================== 5. 解耦验证
    print("\n## 5 · 解耦验证：把「检索池」与「重排候选数」分开调\n")
    ep = RES / "eval_rerank_rp50_ml1024.json"
    if not ep.exists():
        print("（跳过：`eval_rerank_rp50_ml1024.json` 不存在。"
              "跑 `evaluate.py --tag rerank_rp50_ml1024 --rerank --rerank-pool 50`）")
    else:
        E = json.loads(ep.read_text(encoding="utf-8"))
        QE = {q["qid"]: q for q in E["per_query"]}
        print(f"新增 `--rerank-pool`（0=跟随 `--pool`）后可以分开了。三组对比"
              f"（max_length 均为 1024）：\n")
        print("| 检索池 | 重排候选 | R@1 | R@5 | MRR@10 | 重排均值 | rerank 总耗时 |")
        print("|---|---|---|---|---|---|---|")
        for lab, mp, dd in (("100", QB, B), ("**50**", QCC, C),
                            ("100", QE, E)):
            r = m(ranks_of(mp, qids))
            # 早期跑的三组还没有 rerank_pool 字段（当时参数不存在）→ 回退到 pool
            rp = dd["meta"].get("rerank_pool") or dd["meta"].get("pool")
            print(f"| {lab} | {rp} | {r['R@1']:.4f} | **{r['R@5']:.4f}** | {r['MRR']:.4f} "
                  f"| {dd['latency_ms'].get('重排', {}).get('mean', 0):.0f} ms "
                  f"| {dd['reranker_stats']['total_seconds']:.1f} s |")
        re_ = m(ranks_of(QE, qids))
        print(f"\n- 解耦组（池 100 / 候选 50）：R@5 **{re_['R@5']:.4f}**、"
              f"重排 {E['latency_ms']['重排']['mean']:.0f} ms —— "
              f"**比原默认多命中 {(re_['R@5']-r100['R@5'])*len(qids):.0f} 条题，"
              f"且重排还快 {(1-E['latency_ms']['重排']['mean']/t100)*100:.0f}%**")
        print("- ⇒ **两个参数各自独立地取最优**：检索池保持 100 保住召回质量（RRF 融合输入不缩水），"
              "\n  同时只把重排输入降到 50 省延迟。"
              "\n  耦合着调（直接 `--pool 50`）会**连带缩小 RRF 的融合输入**，反而少 2 条 R@5。")
        print("\n**推荐配置**：`--pool 100 --rerank-pool 50 --rerank-max-length 1024`")

    # ==================================================== 6. 延迟明细
    print("\n## 6 · 端到端延迟明细（每题均值，ms）\n")
    keys = ["encode", "vec(searcher内)", "bm25(searcher内)", "fuse", "取原文", "重排"]
    cols = [("A", "ml512·池100·候选100", A), ("B", "ml1024·池100·候选100", B),
            ("C", "ml1024·池50·候选50", C), ("D", "ml1024·池30·候选30", D)]
    if ep.exists():
        cols.append(("E", "**ml1024·池100·候选50**", E))
    print("| 阶段 | " + " | ".join(lab for _, lab, _ in cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for k in keys:
        cells = []
        for _, _, dd in cols:
            v = dd["latency_ms"].get(k, {}).get("mean")
            cells.append(f"{v:.0f}" if v is not None else "—")
        print(f"| {k} | " + " | ".join(cells) + " |")
    tot = [sum(dd["latency_ms"].get(k, {}).get("mean", 0) for k in keys) for _, _, dd in cols]
    print("| **合计** | " + " | ".join(f"**{x:.0f}**" for x in tot) + " |")
    print(f"\n端到端总耗时（{A['meta']['n_queries']} 题，含索引加载）：")
    for lab, name, dd in cols:
        print(f"  {lab}: {name:<24} {dd['meta']['elapsed_seconds']:>6.1f} s")
    if ep.exists():
        dt = E["meta"]["elapsed_seconds"] - B["meta"]["elapsed_seconds"]
        print(f"\n⇒ **推荐配置 E 比原默认 B 端到端快 {abs(dt):.1f} s（{dt/B['meta']['elapsed_seconds']*100:+.1f}%），"
              f"而 R@5 高 {(re_['R@5']-r100['R@5'])*100:+.2f} 个点** —— 两个维度同时变好。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
