#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评测集人工抽检辅助（**只读**，不改任何数据）。

用途：第 11 步定稿了 129 题评测集，废题率自报 2.9%。但"自报"来自生成侧，
      不能自己证明自己。这个脚本做两件事：

  ① **分层随机抽样**（seed 固定 → 可复现），按题型比例抽 30 条打印全字段，
     供人工逐条判断"问题自然 / 答案在原文 / gold 正确"三件事。
  ② **自动可疑项检测** —— 分成两级：
     - **报警**：疑似废题（指代不明 / 标签矛盾 / 匿名失败 / 答案缺失 / 题目重复 …）
     - **提示**：不是缺陷，但**会影响结果解读**（例：easy 题词面零重合 → 对 BM25 不友好）
     人工只看"报警"+ 抽样即可，不必逐条看 129 条。

⚠️ 设计原则：**只报警，不自动删**。判据本身也会骗人（第 10 步踩过：
   "答案必须是原文连续子串"这个判据让 32 条好题被误杀）。
   所以这里每条报警都印出证据，由人裁决。

📌 判据能力边界（2026-09-21 补）：**一个判据抓不住它没被设计来抓的东西。**
   生成侧有个 `answer_coverage`（字符覆盖率，查"答案里的字有没有出处"），
   它对 `v1-d27c79` 给出 **1.0 满分** —— 而那题的答案是错的（问两个量只答一个：
   答「2.6毫米」，原文是「喙宽度约2.6毫米，喙厚度约2.6毫米」）。
   覆盖率类判据查的是「**每个字有没有出处**」，天生查不出「**该答的量答全没有**」。
   ⇒ 补法是加一条**正交**的新判据（E2：题干问 N 个量 vs 答案给了几个数值），
   而不是去调老判据的阈值 —— **换个阈值永远抓不到它。**

用法：
    python src\\audit_testset.py                     # 抽样 30 条 + 全部报警
    python src\\audit_testset.py --sample 40
    python src\\audit_testset.py --testset qa_testset_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
EVAL = PROJECT / "eval"

KIND_CN = {
    "easy": "基础事实", "hard_anon": "实体匿名", "hard_para": "释义改写",
    "t2s": "繁问简答", "s2t": "简问繁答", "oob": "库外问题",
}

# 指代不明的典型信号：问题里没有任何具名锚点，读者无法知道"这/该"指什么
DEIXIS = ["这本书", "这部书", "该条目", "此条目", "该片", "这部电影", "这部剧",
          "该公司", "该组织", "该人物", "这个人物", "该建筑", "该事件", "该作品",
          "这所", "这间", "该馆", "该站", "该条目", "该地区", "该物种"]
# 具名锚点：书名号 / 引号 / 括号里的原名
ANCHOR_RE = re.compile(r"[《〈「『\"“]|（[^）]{2,}）|\([^)]{2,}\)")
# 「问多个量」的强标记（供 E2 判据用）。
# **刻意不用「和／与／、」** —— 实测误报太多：像
# 「一种体细长侧扁、银色体色带深色纵带…的鱼，它最长能长到多少厘米」
# 里既有「、」也有「多少」，但它只问一个量。判据宁可窄，也不要制造噪音。
MULTI_ASK = re.compile(r"各是|各为|各自|各多少|各有多|分别是|分别多少|分别有多")


def flag(kind: str, qid: str, msg: str, evidence: str, level: str = "报警") -> dict:
    """level: 报警 = 疑似废题，需裁决；提示 = 不是缺陷，但影响结果解读，值得知道。"""
    return {"kind": kind, "qid": qid, "msg": msg, "evidence": evidence, "level": level}


def audit(items: list[dict]) -> list[dict]:
    flags: list[dict] = []
    for it in items:
        q, a = it["question"], (it["answer"] or "").strip()
        k, qid = it["kind"], it["qid"]
        title = it["gold_title"] or ""
        leak = it.get("leak_ratio")
        cross = it.get("script_cross")
        qf, cf = it.get("q_fanti"), it.get("c_fanti")

        # A. 指代不明
        for d in DEIXIS:
            if d in q and not ANCHOR_RE.search(q) and title not in q:
                flags.append(flag(k, qid, "指代不明：有指代词但无具名锚点", q))
                break

        # B. 泄漏过高（问题几乎照抄原文 → 不能算"检索"，只能算"字符串匹配"）
        if leak is not None and leak >= 0.85:
            flags.append(flag(k, qid, f"泄漏过高 leak={leak}", q))

        # C.【提示，非缺陷】easy 题词面完全不重合
        #    ⚠️ 一开始我把它当"废题信号"，是错的：`leak_ratio` 的定义是
        #    「问题 4-gram 命中率，只比 chunk_text」，实测 104 题里有 18 题就是 0.000，
        #    中位也只有 0.25。**零重合不代表题目有问题**（例：「康托尔用什么字母表示序数？」
        #    是好题），它只说明**这题对 BM25 不友好** —— 是解读检索指标时要的信息。
        #    （这条修正的由来见 第11步-任务清单.md §11.10）
        if k == "easy" and leak is not None and leak == 0.0:
            flags.append(flag(k, qid, "easy 题词面零重合（对 BM25 不友好，非废题）", q,
                              level="提示"))

        # D. 匿名题里出现 gold 条目名 → 匿名失败（本轮实测的重灾区）
        if k == "hard_anon" and title and (title in q or (len(title) >= 3 and title[:3] in q)):
            flags.append(flag(k, qid, f"匿名失败：问题里出现 gold 条目名「{title}」", q))

        # E. 答案缺失/过短（⚠️ 库外题按设计就没有答案 —— 它考的是"能不能拒答"，
        #    所以必须排除，否则 25 条 oob 全部误报。判据也是会骗人的。）
        if k != "oob" and len(a) < 2:
            flags.append(flag(k, qid, f"答案过短或为空（{a!r}）", q))

        # E2.【提示】题干问「多个量」、答案却只给一个数值
        #     ⚠️ 定稿评测集里**没有 chunk_text**，没法像 build_testset 的
        #     multi_ask_one_answer() 那样用"原文含几个数值"交叉验证 ——
        #     所以这里只报**疑点**，让人回原文核对，级别定"提示"不搞成硬报警。
        #
        #     由来（2026-09-21）：`v1-d27c79` 问「喙宽度和厚度各是多少毫米」，
        #     答案只有「2.6毫米」——**而它的 answer_coverage 是 1.0**，
        #     因为"2.6毫米"确实一字不差在原文里。
        #     ⇒ 覆盖率类判据查的是"每个字有没有出处"，**天生查不出"答漏"**。
        #     这类缺陷只能靠"数量对比"型判据抓（见 第11步-任务清单.md §11.10）。
        if k != "oob" and a:
            if MULTI_ASK.search(q) and len(re.findall(r"\d+(?:\.\d+)?", a)) == 1:
                flags.append(flag(
                    k, qid, "题干问多个量，答案却是单个数值（回原文核对是否答漏）",
                    f"Q: {q} ／ A: {a}", level="提示"))

        # F. 字形标签自相矛盾：script_cross 应等于 (q_fanti != c_fanti)
        if cross is not None and qf is not None and cf is not None:
            if bool(cross) != (bool(qf) != bool(cf)):
                flags.append(flag(k, qid, "字形标签矛盾",
                                  f"script_cross={cross} 但 q_fanti={qf} c_fanti={cf}"))

        # G. 库外题其实库内有
        if k == "oob" and it.get("answerable_in_kb") is True:
            flags.append(flag(k, qid, "标记为库外，但判据认为库内可答", q))

    # H. 题目文本重复（同题出现两次 → 等权重被放大）
    by_q = defaultdict(list)
    for it in items:
        by_q[re.sub(r"\s+", "", it["question"])].append(it["qid"])
    for qn, qids in by_q.items():
        if len(qids) > 1:
            flags.append(flag("—", ",".join(qids), f"题目重复（{len(qids)} 次）", qn[:60]))

    # I. gold 过度集中（同一 chunk 被多题当答案 → 该 chunk 主导指标）
    by_c = Counter(it["gold_chunk_id"] for it in items if it["kind"] != "oob")
    for cid, c in by_c.items():
        if c >= 3:
            t = next(it["gold_title"] for it in items if it["gold_chunk_id"] == cid)
            flags.append(flag("—", cid[:12], f"同一 gold chunk 被 {c} 题共用", t))
    return flags


def strat_sample(items: list[dict], n: int, seed: int) -> list[dict]:
    """按题型比例分层抽样，保证每个题型都被覆盖。"""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for it in items:
        groups[it["kind"]].append(it)
    total = len(items)
    out = []
    for k, g in sorted(groups.items()):
        take = max(1, round(n * len(g) / total))
        out += rng.sample(g, min(take, len(g)))
    rng.shuffle(out)
    return out[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description="评测集人工抽检（只读）")
    ap.add_argument("--testset", default="qa_testset_v1.jsonl")
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="testset_audit_v1.md")
    args = ap.parse_args()

    p = EVAL / args.testset
    if not p.exists():
        raise SystemExit(f"[错误] 找不到 {p}")
    items = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    flags = audit(items)
    sample = strat_sample(items, args.sample, args.seed)

    L = []
    L.append(f"# 评测集抽检报告 · {args.testset}\n")
    L.append(f"共 **{len(items)}** 条："
             + " · ".join(f"{KIND_CN.get(k, k)} {v}" for k, v in sorted(Counter(i['kind'] for i in items).items())))
    L.append(f"\n抽样 **{len(sample)}** 条（分层随机，seed={args.seed}）· "
             f"报警 {sum(1 for f in flags if f['level'] == '报警')} 条 · "
             f"提示 {sum(1 for f in flags if f['level'] == '提示')} 条\n")

    for lv, title, note in (
        ("报警", "一、报警（疑似废题，需人工裁决）", ""),
        ("提示", "二、提示（不是缺陷，但会影响结果解读）", ""),
    ):
        sub = [f for f in flags if f["level"] == lv]
        L.append(f"\n---\n\n## {title}\n")
        if not sub:
            L.append("无。\n")
            continue
        L.append("| 类型 | 条数 |")
        L.append("|---|---|")
        for m, c in Counter(f["msg"].split("：")[0] for f in sub).most_common():
            L.append(f"| {m} | {c} |")
        cur = None
        for f in sorted(sub, key=lambda x: x["msg"]):
            t = f["msg"].split("：")[0]
            if t != cur:
                cur = t
                L.append(f"\n### {t}\n")
            L.append(f"- `{f['qid']}` [{KIND_CN.get(f['kind'], f['kind'])}] **{f['msg']}**")
            L.append(f"  - 证据：{f['evidence']}")

    L.append("\n---\n\n## 三、抽样逐条（人工看：问题自然 / 答案在原文 / gold 正确）\n")
    for i, it in enumerate(sample, 1):
        cid = it.get("gold_chunk_id") or "—（库外题无 gold）"
        if len(cid) > 12 and "—" not in cid:
            cid = cid[:12] + "…"
        L.append(f"### {i}. `{it['qid']}` [{KIND_CN.get(it['kind'], it['kind'])}]")
        L.append(f"- **问题**：{it['question']}")
        L.append(f"- **答案**：{it['answer']}")
        L.append(f"- **gold**：`{it.get('gold_title')}` ｜ 章节 `{it.get('gold_section')}` ｜ "
                 f"chunk `{cid}`")
        L.append(f"- 泄漏率 {it.get('leak_ratio')} · 字形 cross={it.get('script_cross')} "
                 f"(q_fanti={it.get('q_fanti')} c_fanti={it.get('c_fanti')}) · "
                 f"库内可答={it.get('answerable_in_kb')}")
        L.append("")

    outp = EVAL / "results" / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(L), encoding="utf-8")

    print(f"评测集 {len(items)} 条 · 抽样 {len(sample)} · "
          f"报警 {sum(1 for f in flags if f['level'] == '报警')} · "
          f"提示 {sum(1 for f in flags if f['level'] == '提示')}")
    if flags:
        print("分布：")
        for lv in ("报警", "提示"):
            for m, c in Counter(f["msg"].split("：")[0] for f in flags
                                if f["level"] == lv).most_common():
                print(f"  [{lv}] {c:>3}  {m}")
    print(f"\n报告已写：{outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
