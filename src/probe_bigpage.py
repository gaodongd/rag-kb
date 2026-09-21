"""诊断：找出维基 XML 中体量异常巨大的 <text> 节点。

背景（2026-09-16）：
  wiki_extract.py（iterparse）每次都在**完全相同的位置**崩：
      n_page=328,850 / 采纳 153,123 / 已解压 1.35 G字符
      ExpatError: out of memory: line 28421794, column 44
  确定性的崩溃 + 机器有 18GB 空闲内存 → 怀疑是某个超大 text 节点让 expat 的
  内部缓冲膨胀。本脚本纯行扫描（不建 DOM），量出每个 <text> 节点的大小。

用法：
    python src/probe_bigpage.py --stop-gchars 2.0 --top 10
"""
import argparse
import bz2
import re
import sys
import time
from pathlib import Path

RE_TITLE = re.compile(r"<title>(.*?)</title>")
RE_TEXT_OPEN = re.compile(r"<text[^>]*>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",
                    default="E:/AI-learning/data/zhwiki-latest-pages-articles.xml.bz2")
    ap.add_argument("--stop-gchars", type=float, default=2.0, help="扫到多少 G字符就停")
    ap.add_argument("--top", type=int, default=10, help="报告最大的 N 个 text 节点")
    ap.add_argument("--crash-offset", type=float, default=1.35,
                    help="关注哪个 G字符位置（复现点）")
    args = ap.parse_args()

    stop_chars = args.stop_gchars * 1024 ** 3
    crash_chars = args.crash_offset * 1024 ** 3

    print(f"扫描：{args.input}")
    print(f"目标：{args.stop_gchars} G字符   关注点：{args.crash_offset} G字符")
    print("-" * 72)

    t0 = time.time()
    total = 0
    lineno = 0
    n_page = 0
    cur_title = "?"
    in_text = False
    text_len = 0
    text_start = 0
    biggest = []          # (len, title, start_chars, end_chars)
    page_open_chars = 0
    crash_page = None

    with bz2.open(args.input, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            lineno += 1
            n = len(line)
            prev_total = total
            total += n

            if in_text:
                text_len += n
                if "</text>" in line:
                    biggest.append((text_len, cur_title, text_start, total))
                    in_text = False
            else:
                stripped = line.lstrip()
                if stripped.startswith("<page>"):
                    n_page += 1
                    page_open_chars = total
                elif "<title>" in line:
                    m = RE_TITLE.search(line)
                    if m:
                        cur_title = m.group(1)
                elif stripped.startswith("<text"):
                    m = RE_TEXT_OPEN.search(line)
                    if m:
                        in_text = True
                        text_len = n - m.end()
                        text_start = prev_total + m.end()
                        if "</text>" in line[m.end():]:
                            biggest.append((text_len, cur_title, text_start, total))
                            in_text = False

            if crash_page is None and total >= crash_chars:
                crash_page = (n_page, cur_title, total, "（当前所在条目）")

            if n_page % 50000 == 0 and prev_total < total:
                pass
            if total >= stop_chars:
                break
            if lineno % 4000000 == 0:
                el = time.time() - t0
                print(f"  ... {lineno:,} 行 / {total / 1024**3:.2f} G字符 / "
                      f"{n_page:,} page / {total / 1024**2 / max(el, 1e-6):.1f} M字符/s",
                      flush=True)

    el = time.time() - t0
    print(f"\n扫完：{lineno:,} 行 / {total / 1024**3:.3f} G字符 / {n_page:,} page / {el:.0f}s")
    print(f"平均每行 {total / max(lineno, 1):.0f} 字符")
    print("=" * 72)

    if crash_page:
        print(f"\n【崩点定位】第 {crash_chars / 1024**3:.2f} G字符 落在：")
        print(f"  page 序号 ~{crash_page[0]:,}  标题 = {crash_page[1]}")
        print(f"  位置 = {crash_page[2] / 1024**3:.2f} G字符  {crash_page[3]}")

    biggest.sort(reverse=True)
    print(f"\n【最大的 {args.top} 个 <text> 节点】")
    print(f"{'字符数':>14}  {'占比':>7}  标题")
    for ln, title, s, e in biggest[:args.top]:
        print(f"{ln:>14,}  {ln / total * 100:>6.2f}%  {title[:50]}")

    if biggest:
        tot = sum(b[0] for b in biggest)
        print(f"\n  {len(biggest):,} 个 text 节点，合计 {tot / 1024**3:.3f} G字符 "
              f"（占已扫 {tot / total * 100:.1f}%）")
        print(f"  平均 {tot / len(biggest):,.0f} 字符，最大 {biggest[0][0]:,} 字符")
    return 0


if __name__ == "__main__":
    sys.exit(main())
