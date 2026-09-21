#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 6 步辅助工具：给 audit_sample.md 加一张「总览表」，把 50 条压缩成一屏能扫完的表格。

为什么需要它
------------
人工审计的真实成本不在"读"，而在"翻"。50 条正文铺满 60KB，
逐条往下滚会失去整体感 —— 你不知道后面还有多少条、也不知道哪几条字数异常。
加一张表插在开头，先扫全貌、再逐条判，效率差好几倍。

用法
----
    python src/make_audit_sheet.py --json eval/audit_sample.seed42.json \
                                  --md   eval/audit_sample.md

它做的事：
  1. 读 audit_sample.md，把旧的「总览表」区块（如果存在）整段替换掉，可重复运行
  2. 表里每行 = 一条样本，附「机器预检」结果，方便你只重点看可疑的
  3. 原文件的逐条详情一个字不动

注意：HTML 注释标记用来划定插入边界，不要手删。
"""

import argparse
import json
import re
from pathlib import Path

BEGIN = "<!-- AUDIT-SHEET-BEGIN -->"
END = "<!-- AUDIT-SHEET-END -->"


def build_table(meta: dict) -> str:
    rows = meta["rows"]
    dirty = meta.get("n_dirty_by_machine", 0)
    lines = [
        BEGIN,
        "",
        f"## 总览表（{len(rows)} 条，先扫这张表再逐条判）",
        "",
        f"- 机器预检：**{len(rows) - dirty} 条干净 / {dirty} 条带残留标记**",
        "- 「机器预检」只查得出「无标记」这一条判据，**其余三条只能你读**",
        "- 逐条详情在下面，编号与下表一一对应",
        "",
        "| # | 标题 | 章节 | 字数 | 中文占比 | 机器预检 | 通顺 | 自洽 | 有信息量 | 无标记 | 结论 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        title = str(r.get("title", "")).replace("|", "\\|")
        if len(title) > 22:
            title = title[:21] + "…"
        # 章节列不能省：判「自洽」时要看它（检索串 = title + section + 正文）。
        # v1 漏了这一列，`蓬镇 / 媒体` 那条被误判成「标题讲 A、正文讲 B」。
        sec = str(r.get("section") or "").strip().replace("|", "\\|") or "—"
        if len(sec) > 12:
            sec = sec[:11] + "…"
        nmark = " ".join(r.get("residual") or [])
        mark = f"⚠️ {nmark}" if nmark else "✅ 干净"
        lines.append(
            f"| {r['no']} | {title} | {sec} | {r['char_len']} | {r['cn_ratio']:.2f} | {mark} "
            f"| `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |"
        )
    lines += [
        "",
        "> 填完这把表，**通过率 = 结论为「通过」的行数 ÷ 50**。",
        "> 报数给助手时把有问题的编号一并报上，比只说一个比率有用得多。",
        "> **报编号时请连「卡在哪条判据」一起说** —— 同一类问题出现 3 次以上才是真 bug。",
        "",
        END,
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="eval/audit_sample.seed42.json")
    ap.add_argument("--md", default="eval/audit_sample.md")
    args = ap.parse_args()

    meta = json.loads(Path(args.json).read_text(encoding="utf-8"))
    md_path = Path(args.md)
    text = md_path.read_text(encoding="utf-8")

    table = build_table(meta)
    block = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)

    if block.search(text):
        text = block.sub(table, text)
        action = "已替换旧总览表"
    else:
        # 插在第一个 "## 1. " 之前（各脚本生成的样本文件都是这个结构）
        anchor = re.search(r"^## 1\. ", text, re.M)
        if not anchor:
            raise SystemExit("[错误] 找不到 '## 1. ' 锚点，样本文件格式变了？")
        text = text[: anchor.start()] + table + "\n\n---\n\n" + text[anchor.start():]
        action = "已插入总览表"

    md_path.write_text(text, encoding="utf-8")
    print(f"[OK] {action} -> {md_path}")
    print(f"     条数={len(meta['rows'])}  机器预检脏数据={meta.get('n_dirty_by_machine', 0)}")


if __name__ == "__main__":
    main()
