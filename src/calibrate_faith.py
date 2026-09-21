#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Faithfulness 判据校准（§11.14）
================================

把人工逐条判定过的子句当"金标准"，量化比较两套判据：

  v1/v2  2-gram 覆盖率（句子级 / 子句级）
  v3     **实词覆盖率**（子句级 + 按字数加权，准备作为主判据）

用法
----
    "...python313\\python.exe" src/calibrate_faith.py

标签文件 `eval/faith_labels.jsonl` 每行：
  {"qid": "...", "clause": "子句原文", "label": "yes|no|skip", "note": "为什么"}

label 取值：
  yes  = 有依据（原文确实支持这句话）
  no   = 无依据（模型自己加的/幻觉）
  skip = 不该参与判定（结构句、纯评论、切分残渣）—— 用来检查判据有没有把它们灌进分母

为什么要单独写一个脚本而不是"看两眼就行"：
本项目已经四次栽在"判据看起来很合理"上（§11.10 的字符覆盖率、§11.14 的数字抽取……）。
既然已经花了人力逐条判过，就必须让结论**可以重跑**，否则下次改判据又要重新人肉一遍。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))

from evaluate_gen import (CITE_RE, PUNCT_RE, context_texts,  # noqa: E402
                          content_coverage, coverage, digits_supported,
                          cited_chunk_texts, is_struct_clause)


def main() -> int:
    labels_path = PROJECT / "eval" / "faith_labels.jsonl"
    gen_path = PROJECT / "eval" / "results" / "gen_dashscope_testset_rr.jsonl"
    if not labels_path.exists():
        raise SystemExit(f"[错误] 标签文件不存在：{labels_path}")

    labels = [json.loads(l) for l in labels_path.read_text(encoding="utf-8").splitlines()
              if l.strip()]
    rows = {}
    for line in gen_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r.get("qid")] = r

    print("=" * 78)
    print("Faithfulness 判据校准")
    print("=" * 78)
    print(f"标签 {len(labels)} 条（yes={sum(1 for x in labels if x['label']=='yes')}"
          f" no={sum(1 for x in labels if x['label']=='no')}"
          f" skip={sum(1 for x in labels if x['label']=='skip')}）")

    # ---- 逐条计算两套判据 ----（阈值扫描）
    calc = []
    for it in labels:
        r = rows.get(it["qid"])
        if r is None:
            print(f"  [跳过] 找不到 {it['qid']}")
            continue
        cl = it["clause"]
        nums = [int(x) for x in CITE_RE.findall(cl)]
        ctx = context_texts(r)
        cited = cited_chunk_texts(r)
        own = [c for n, c in enumerate(ctx, 1) if n in nums]
        src = own if nums else (cited or ctx)

        cov = max((coverage(cl, c) for c in src), default=0.0)
        ccov = max((content_coverage(cl, c)[0] for c in src), default=0.0)
        ok_num = True
        if own:
            ok_num = any(digits_supported(cl, c)[0] for c in own)
        calc.append({"qid": it["qid"], "clause": cl, "label": it["label"],
                     "note": it.get("note", ""), "cov": cov, "ccov": ccov,
                     "ok_num": ok_num, "struct": is_struct_clause(cl)})

    yes = [c for c in calc if c["label"] == "yes"]
    no = [c for c in calc if c["label"] == "no"]
    skip = [c for c in calc if c["label"] == "skip"]

    print()
    print("── 逐条明细 ──")
    for c in calc:
        print(f"  [{c['label']:>4}] 2g={c['cov']:.2f} 实词={c['ccov']:.2f}"
              f" {'结构句' if c['struct'] else '      '} {c['clause'][:56]}")
        if c["note"]:
            print(f"          └ {c['note']}")

    print()
    print("── 阈值扫描：在 yes/no 上的准确率 ──")
    print(f"{'阈值':>6} {'判据':<10}{'yes 判对':>10}{'no 判对':>10}{'合计准确率':>12}")
    best = {}
    for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        for name, key in (("2-gram", "cov"), ("实词", "ccov")):
            tp = sum(1 for c in yes if c[key] >= th and c["ok_num"])
            tn = sum(1 for c in no if not (c[key] >= th and c["ok_num"]))
            acc = (tp + tn) / max(1, len(yes) + len(no))
            print(f"{th:>6.2f} {name:<10}{tp:>7}/{len(yes)}{tn:>8}/{len(no)}{acc:>11.1%}")
            best.setdefault(name, []).append((acc, th))

    print()
    print("── 结构句识别（应被 is_struct_clause 剔除）──")
    print(f"  标为 skip 的 {len(skip)} 条里，判据剔除 {sum(1 for c in skip if c['struct'])} 条")
    fp = [c for c in yes + no if c["struct"]]
    print(f"  标为 yes/no 的 {len(yes)+len(no)} 条里，被误剔除 {len(fp)} 条"
          + ("（应为 0）" if not fp else ""))
    for c in fp[:5]:
        print(f"    ⚠️ [{c['label']}] {c['clause'][:50]}")

    print()
    for name, lst in best.items():
        lst.sort(reverse=True)
        print(f"  最优（{name}）：阈值 {lst[0][1]:.2f} → 准确率 {lst[0][0]:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
