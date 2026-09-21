# -*- coding: utf-8 -*-
"""
定位：为什么缓释点号版 RE_TABLE2 反而比旧版多残留 8 倍的 {| 。

只做一件事：找出"旧版清干净了、新版没清干净"的篇章，把**原文**对应位置打出来。
"""
import argparse
import json
import re
import sys
from pathlib import Path

here = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(here / "src"))
import build_pipeline as bp  # noqa: E402
from diag_clean2 import clean_v2  # noqa: E402

RE_TABLE_OLD = bp.RE_TABLE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(here / "data/raw/wiki.jsonl"))
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--show", type=int, default=4)
    args = ap.parse_args()

    shown = 0
    with open(args.input, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.n or shown >= args.show:
                break
            doc = json.loads(line)
            raw = doc.get("text") or ""
            c1 = bp.clean_wikitext(raw)
            c2 = clean_v2(raw)
            if "{|" in c2 and "{|" not in c1:
                idx = c2.find("{|")
                print("=" * 70)
                print("TITLE:", doc.get("title"))
                print("--- 新版输出里的上下文 ---")
                print(repr(c2[max(0, idx - 120): idx + 160]))
                # 找原文里对应的片段
                probe = c2[max(0, idx - 20): idx + 40].strip()
                j = raw.find(probe[:20]) if probe else -1
                print("--- 原文对应位置 ---")
                if j >= 0:
                    print(repr(raw[max(0, j - 200): j + 400]))
                else:
                    print("(没在原文里直接找到，说明是清洗过程中产生的)")
                    print(repr(raw[:600]))
                shown += 1

    # 顺便：-{ 残留样例（新版旧版都有）
    print("\n" + "=" * 70)
    print("-{ 残留样例（新旧都残留，说明是另一类问题）")
    shown = 0
    with open(args.input, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.n or shown >= 2:
                break
            doc = json.loads(line)
            raw = doc.get("text") or ""
            c2 = clean_v2(raw)
            if "-{" in c2:
                idx = c2.find("-{")
                print("-" * 70)
                print("TITLE:", doc.get("title"))
                print("新版输出:", repr(c2[max(0, idx - 80): idx + 120]))
                j = raw.find("-{")
                print("原文    :", repr(raw[max(0, j - 60): j + 200]))
                shown += 1


if __name__ == "__main__":
    main()
