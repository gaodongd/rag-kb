# -*- coding: utf-8 -*-
"""
逐步追踪：clean_v2 的哪一步让 {| 残留？（不猜，直接打每步结果）
"""
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(here / "src"))
import build_pipeline as bp  # noqa: E402
import diag_clean2 as d2  # noqa: E402

TARGET = "法国"
raw = None
with open(here / "data/raw/wiki.jsonl", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        if d.get("title") == TARGET:
            raw = d["text"]
            break
if raw is None:
    print("没找到", TARGET)
    sys.exit(1)

print(f"原文长度 {len(raw):,}，表格数 {raw.count('{|')}")

steps = [
    ("原始", raw),
    ("RE_COMMENT", bp.RE_COMMENT.sub("", raw)),
]
t = steps[-1][1]
t = bp.RE_REF.sub("", t);                                 steps.append(("+RE_REF", t))
t_old = bp.RE_TABLE.sub("", t);                           steps.append(("+RE_TABLE(旧)", t_old))
t_new = d2.RE_TABLE2.sub("", t);                          steps.append(("+RE_TABLE2(新)", t_new))
t = d2.RE_LANGVAR2.sub(bp._pick_langvar, t);              steps.append(("+RE_LANGVAR", t))
t = bp.RE_LANG_TPL.sub(r"\1", t);                         steps.append(("+RE_LANG_TPL", t))
for i in range(3):
    t, n = d2.RE_TEMPLATE2.subn("", t)
    steps.append((f"+RE_TEMPLATE2 第{i+1}轮(n={n})", t))
    if n == 0:
        break

for name, s in steps:
    print(f"  {name:<26} 表格残留数={s.count('{|'):<4} 长度={len(s):,}")

print()
print("=== 关键对比：旧 RE_TABLE vs 新 RE_TABLE2 ===")
print("旧 RE_TABLE.pattern    =", repr(bp.RE_TABLE.pattern), "re.S =", bool(bp.RE_TABLE.flags & 16))
print("新 RE_TABLE2.pattern   =", repr(d2.RE_TABLE2.pattern), "re.S =", bool(d2.RE_TABLE2.flags & 16))
print()
# 找一个真实表格片段做单点测试
i = raw.find("{|")
frag = raw[i:i + 400]
print("片段测试：")
print("  片段开头:", repr(frag[:150]))
print("  旧匹配 :", bool(bp.RE_TABLE.search(frag)), "->", repr(bp.RE_TABLE.search(frag).group(0)[:80]) if bp.RE_TABLE.search(frag) else "")
m2 = d2.RE_TABLE2.search(frag)
print("  新匹配 :", bool(m2), "->", repr(m2.group(0)[:80]) if m2 else "")
