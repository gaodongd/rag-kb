# -*- coding: utf-8 -*-
"""
第 6 步数据审计衍生的清洗改进 —— 离线 A/B 实测。

背景（2026-09-17 全量审计实测，3,926,261 块）。第 6 步人工抽样一共挖出 **两类** 新问题：

【一】删过头 —— 模板被整段删掉，留下空壳
    空括号 （）    310,941 块   7.902%
    连续标点 ，。    76,587 块   1.946%
    空书名号 《》    49,986 块   1.270%
    空引号 「」『』   14,532 块   0.369%
    对照：残留标记最多的一项（{{）只有 702 块 = 0.018%
  → 修法见 RE_EMPTY_SHELL / RE_DUP_PUNCT

【二】命名空间被当正文 —— [[Category:X]] 被剥成纯文本
    含 `Category:` 的块           492,872 块  12.553%
    整块只有分类标签、零正文的块   143,683 块   3.660%   ← 纯噪声
  → 修法见 RE_CATEGORY（必须赶在 RE_LINK_* 之前整块删除）

本脚本对同一批样本跑**现在的** clean_wikitext，三类指标放一起看：
    * 空壳（SIDE_EFFECTS）        期望大幅下降
    * 命名空间（NAMESPACE_RESIDUE）期望归零
    * 残留标记（RESIDUAL）        期望**不退化** —— 零回归才准上全量

用法：
    python src/diag_clean3.py --n 20000
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pipeline import clean_wikitext  # noqa: E402
from verify_chunks import NAMESPACE_RESIDUE, RESIDUAL, SIDE_EFFECTS  # noqa: E402

HERE = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(HERE / "data/raw/wiki.jsonl"))
    ap.add_argument("--n", type=int, default=20000)
    args = ap.parse_args()

    texts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            try:
                texts.append(json.loads(line)["text"])
            except Exception:
                continue

    print(f"样本 {len(texts):,} 篇（{args.input} 前 {args.n} 行）")
    print("=" * 68)

    # ---- 原始文本（未清洗）里本来就有多少？这是基线，不能全算到清洗器头上 ----
    raw_shell = {k: 0 for k in SIDE_EFFECTS}
    raw_ns = {k: 0 for k in NAMESPACE_RESIDUE}
    for t in texts:
        for name, pat in SIDE_EFFECTS.items():
            raw_shell[name] += len(re.findall(pat, t))
        for name, pat in NAMESPACE_RESIDUE.items():
            raw_ns[name] += len(re.findall(pat, t))

    cleaned = [clean_wikitext(t) for t in texts]

    new_shell = {k: 0 for k in SIDE_EFFECTS}
    new_ns = {k: 0 for k in NAMESPACE_RESIDUE}
    new_res = {k: 0 for k in RESIDUAL}
    n_bad_doc = 0            # 清洗后仍含任一类问题的文档数
    fixed_doc = 0            # 清洗前有、清洗后没有的文档数（真正被修的）
    samples = []             # 待展示的修复样例

    for raw, t in zip(texts, cleaned):
        raw_has = any(re.search(p, raw) for p in SIDE_EFFECTS.values()) or \
                  any(re.search(p, raw) for p in NAMESPACE_RESIDUE.values())
        new_has = False
        for name, pat in SIDE_EFFECTS.items():
            c = len(re.findall(pat, t))
            new_shell[name] += c
            new_has = new_has or bool(c)
        for name, pat in NAMESPACE_RESIDUE.items():
            c = len(re.findall(pat, t))
            new_ns[name] += c
            new_has = new_has or bool(c)
        for name, marker in RESIDUAL.items():
            if marker in t:
                new_res[name] += 1
        if new_has:
            n_bad_doc += 1
        elif raw_has and len(samples) < 5:
            # 清洗前有问题、清洗后没了 → 找一段上下文展示
            for p in list(SIDE_EFFECTS.values()) + list(NAMESPACE_RESIDUE.values()):
                m = re.search(p, raw)
                if m:
                    s = max(0, m.start() - 45)
                    samples.append(raw[s:m.end() + 45].replace("\n", " "))
                    break
        if raw_has and not new_has:
            fixed_doc += 1

    print("\n[一 · 空壳] 删过头留下的")
    print(f"  {'项':<12} {'清洗前':>10} {'清洗后':>10}   判定")
    for name in SIDE_EFFECTS:
        a, b = raw_shell[name], new_shell[name]
        note = "OK 归零" if b == 0 else ("↓ 下降" if b < a else "!! 未改善")
        print(f"  {name:<12} {a:>10,} {b:>10,}   {note}")

    print("\n[二 · 命名空间] [[Category:X]] 被当正文留下的")
    print(f"  {'项':<12} {'清洗前':>10} {'清洗后':>10}   判定")
    for name in NAMESPACE_RESIDUE:
        a, b = raw_ns[name], new_ns[name]
        note = "OK 归零" if b == 0 else ("↓ 下降" if b < a else "!! 未改善")
        print(f"  {name:<12} {a:>10,} {b:>10,}   {note}")

    print("\n[三 · 回归检查] 残留标记 —— 只许变好，不许变差")
    for name, marker in RESIDUAL.items():
        print(f"  含 {marker:<6} 的文档 : {new_res[name]:,}")

    print(f"\n[文档级]")
    print(f"  清洗后仍含问题的文档   : {n_bad_doc:,} / {len(texts):,} "
          f"({n_bad_doc / len(texts) * 100:.2f}%)")
    print(f"  问题被修掉（前有后无） : {fixed_doc:,}")

    print("\n[修复样例]  —— 清洗前长这样，现在没了")
    for s in samples:
        print("  ..." + s)

    print("=" * 68)
    ok = all(v == 0 for v in new_shell.values()) and all(v == 0 for v in new_ns.values())
    print("VERDICT:", "PASS 空壳 + 命名空间全部清零" if ok
          else "还有未归零项，看上面哪一行标了 !!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
