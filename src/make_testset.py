#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 `build_testset.py` 的候选（candidates）过滤成**正式评测集**。

============================================================================
为什么要有这一步（而不是让 evaluate.py 直接读 candidates）
============================================================================
① **定稿是一个独立动作**。candidates 是"原料"，含已知废题；
   评测集是"标尺"，一旦用上就不该再变 —— 否则不同实验的分数不可比。
   分成两个文件，等于把"标尺的修订"留下痕迹。

② 评测集要能**脱离生成脚本被读懂**。字段改成统一命名
   （gold_chunk_id / gold_title 而不是 chunk_id / title），
   并去掉中间产物字段（answer_coverage 之类只对生成阶段有意义）。

③ 人工校对的黑名单要**可累积**。`--exclude` 把人工否决的题号写进
   `eval/qa_testset_excluded.txt`，重跑时自动生效，不需要改代码。

============================================================================
过滤规则（三道，全部是"证据不足就剔"）
============================================================================
| 规则 | 剔什么 | 依据 |
|---|---|---|
| 库内题 pass_local=false | 4 条：泄漏率过高 / 匿名化失败 / 答案疑似概括 | build_testset 的本地预筛 |
| 库外题 answerable_in_kb != False | 5 条：4 条"其实在库内" + 1 条"未判定" | 检索验证：库外题的合格标准是**确认它不在库内** |
| 人工黑名单 | 抽检否决的 | --exclude |

⚠️ 注意库内题和库外题的**合格判据完全不同**：
   库内题看"答案是否真在 gold chunk 里"，库外题看"检索是否确认它不在库内"。
   拿库内题的判据去卡库外题，会把 25 条有效题全判死 —— 这个坑见
   `第10步-任务清单.md`（build_testset 的 `--recheck` 里已按 kind 分支处理）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
EVAL = PROJECT / "eval"

# 题型的中文名（写报告用）
KIND_CN = {
    "easy": "基础事实",
    "hard_anon": "实体匿名",
    "hard_para": "释义改写",
    "t2s": "繁问简答",
    "s2t": "简问繁答",
    "oob": "库外问题",
}


def parse_args():
    ap = argparse.ArgumentParser(description="候选 → 正式评测集")
    ap.add_argument("--candidates", default=str(EVAL / "qa_testset_candidates_v1.jsonl"))
    ap.add_argument("--out", default=str(EVAL / "qa_testset_v1.jsonl"))
    ap.add_argument("--exclude-file", default=str(EVAL / "qa_testset_excluded.txt"),
                    help="人工否决清单，每行一个 qid，支持 # 注释")
    ap.add_argument("--tag", default="v1")
    return ap.parse_args()


def load_excluded(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(line.split()[0])       # 只取第一个 token，行尾可写理由
    return out


def main() -> int:
    args = parse_args()
    cand = Path(args.candidates)
    if not cand.exists():
        print(f"[错误] 候选文件不存在：{cand}")
        return 1

    rows = [json.loads(l) for l in cand.read_text(encoding="utf-8").splitlines() if l.strip()]
    excluded = load_excluded(Path(args.exclude_file))

    print("=" * 72)
    print(f"评测集定稿 · 候选 {len(rows)} 条")
    print("=" * 72)

    kept, dropped = [], []
    for i, r in enumerate(sorted(rows, key=lambda x: (x.get("kind", ""), x.get("question", ""))), 1):
        # qid 用**问题文本的哈希**，不用行号 —— 行号会随排序方式变化
        # （比如修正了 kind 标签，排序就变了，所有 qid 全漂移，
        #   而人工黑名单和实验结果的对照都是按 qid 索引的）。
        qid = f"{args.tag}-" + hashlib.md5(
            r.get("question", "").encode("utf-8")).hexdigest()[:6]
        kind = r.get("kind")
        reason = None

        # ---- 跨字形题：按**实测字形**重新定标签，不信任原来的 kind ----
        # 原因：build_testset 里 t2s/s2t 的键名与实现相反（已修，但本批数据已生成）。
        # q_fanti / c_fanti 是生成后实测的，不会说谎 —— 用事实推导，
        # 这样即使上游标签再错，评测报告的方向也不会错。
        # t2s = Traditional→Simplified = 繁体提问 → 简体 gold
        # ⚠️ 必须放在下面的"剔除判定"之前 —— 同字形的情形要靠它设 reason 剔掉。
        if kind in ("t2s", "s2t"):
            qf, cf = bool(r.get("q_fanti")), bool(r.get("c_fanti"))
            if qf and not cf:
                kind = "t2s"
            elif cf and not qf:
                kind = "s2t"
            else:
                # 两侧同字形 = 字形转换没生效，它测不出跨字形能力
                reason = (f"标为 {r.get('kind')} 但问题与 gold 同字形"
                          f"（转换未生效），测不出跨字形能力")

        # ---- 规则 1/2：按题型走**不同**的合格判据 ----
        if reason is None:
            if kind == "oob":
                if r.get("answerable_in_kb") is not False:
                    reason = ("其实在库内" if r.get("answerable_in_kb") is True
                              else "库外性未判定")
            else:
                if not r.get("pass_local"):
                    reason = r.get("warn") or "本地预筛未通过"

        # ---- 规则 3：人工黑名单 ----
        if reason is None and qid in excluded:
            reason = "人工校对否决"

        if reason:
            dropped.append({"qid": qid, "kind": kind, "question": r.get("question", ""),
                            "reason": reason})
            continue

        kept.append({
            "qid": qid,
            "question": r["question"],
            "answer": r.get("answer"),
            "gold_chunk_id": r.get("chunk_id"),
            "gold_title": r.get("title"),
            "gold_section": r.get("section") or "",
            "kind": kind,
            "script_cross": bool(r.get("script_cross")),
            "q_fanti": bool(r.get("q_fanti")),
            "c_fanti": bool(r.get("c_fanti")),
            "leak_ratio": r.get("leak_ratio"),
            "answerable_in_kb": r.get("answerable_in_kb"),
        })

    # ---- 排序：库内题按难度（easy → anon → para → 跨字形），库外题最后 ----
    order = {"easy": 0, "hard_anon": 1, "hard_para": 2, "t2s": 3, "s2t": 4, "oob": 9}
    kept.sort(key=lambda x: (order.get(x["kind"], 8), x["qid"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- 报告 ----
    in_kb = [r for r in kept if r["kind"] != "oob"]
    oob = [r for r in kept if r["kind"] == "oob"]

    print(f"\n保留 {len(kept)} 条 → {out}")
    print(f"  库内 {len(in_kb)} 条 · 库外 {len(oob)} 条")
    print("\n  按题型：")
    for k, n in sorted(Counter(r["kind"] for r in kept).items(),
                       key=lambda x: order.get(x[0], 8)):
        print(f"    {k:10s} {KIND_CN.get(k, k):6s} {n:>3d} 条")

    n_cross = sum(1 for r in in_kb if r["script_cross"])
    print(f"\n  字形交叉题 {n_cross} 条（{n_cross/len(in_kb)*100:.1f}% 的库内题）")
    print(f"    ├ 繁体问→简体答 {sum(1 for r in in_kb if r['q_fanti'] and not r['c_fanti'])} 条")
    print(f"    ├ 简体问→繁体答 {sum(1 for r in in_kb if r['c_fanti'] and not r['q_fanti'])} 条")
    print(f"    └ 两侧同字形但跨条目 {n_cross - sum(1 for r in in_kb if r['q_fanti'] != r['c_fanti'])} 条")

    print(f"\n剔除 {len(dropped)} 条：")
    for d in dropped:
        print(f"  [{d['kind']:9s}] {d['question'][:42]}")
        print(f"              原因：{d['reason']}")

    if n_cross == 0:
        print("\n⚠️ 警告：没有字形交叉题，繁简归一化的效果将无法评测")

    print("\n" + "=" * 72)
    print("下一步：")
    print(f'  python src\\evaluate.py --testset {out.name} --tag baseline')
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
