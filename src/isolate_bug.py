"""隔离实验：按「整页边界」截出崩溃点附近的 XML，看能否独立复现 expat 的 out of memory。

背景（2026-09-16）：
  全量解析每次都在**完全相同**位置崩：
      n_page=328,850 / 采纳 153,123 / 已解压 1.35 G字符 / 第 28,421,794 行
      ExpatError: out of memory: line 28421794, column 44
  已排除的原因：
      * 进程 RSS 全程 25→60 MB，无泄漏、无内存压力（机器 18GB 空闲）
      * 前 2.0 G字符里最大的 <text> 节点只有 46.6 万字符，不存在超大节点
  → 是**特定内容**触发。本脚本按 <page> 边界切一段出来独立验证。

用法：
    python src/isolate_bug.py --start-line 27000000 --take-lines 3000000
"""
import argparse
import bz2
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

PROLOGUE = ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">\n')
EPILOGUE = '</mediawiki>\n'


def build_slice(src: Path, dst: Path, start_line: int, take_lines: int):
    """按行切：从 start_line 之后的第一个 <page> 开始，写到够 take_lines 行且落在 </page> 结束。"""
    t0 = time.time()
    lineno = 0
    written = 0
    started = False
    buf = ""
    n_chars = 0

    with bz2.open(src, "rt", encoding="utf-8", errors="replace") as f, \
            dst.open("w", encoding="utf-8") as out:
        while True:
            chunk = f.read(1 << 22)          # 每次 4M 字符
            if not chunk:
                break
            n_chars += len(chunk)
            buf += chunk
            parts = buf.split("\n")
            buf = parts.pop()                # 最后一段可能不完整，留回

            for line in parts:
                lineno += 1
                if not started:
                    if lineno < start_line:
                        continue
                    if line.lstrip().startswith("<page>"):
                        out.write(PROLOGUE)
                        started = True
                    else:
                        continue
                out.write(line + "\n")
                written += 1
                if written >= take_lines and line.lstrip().startswith("</page>"):
                    out.write(EPILOGUE)
                    started = False
                    print(f"  切到 {lineno:,} 行处收尾（共写 {written:,} 行）")
                    print(f"  用时 {time.time() - t0:.0f}s，读入 {n_chars / 1024**3:.2f} G字符")
                    print(f"  切片：{dst}  {dst.stat().st_size / 1024**2:.1f} MB")
                    return written
            if lineno % 2000000 == 0:
                print(f"  ... 已扫 {lineno:,} 行 / 写入 {written:,} 行", flush=True)

        out.write("\n" + EPILOGUE)
    print(f"  到达文件末尾。共 {lineno:,} 行，写入 {written:,} 行，"
          f"{dst.stat().st_size / 1024**2:.1f} MB，用时 {time.time() - t0:.0f}s")
    return written


def _local(tag):
    if isinstance(tag, str) and tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def try_parse(path: Path):
    """用与 wiki_extract.py 相同的 iterparse 路径跑切片。"""
    n_page = 0
    chars = 0
    t0 = time.time()
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            ctx = ET.iterparse(f, events=("end",))
            for _ev, elem in ctx:
                if elem.tag != "page" and _local(elem.tag) != "page":
                    continue
                n_page += 1
                title = ""
                for child in elem:
                    if _local(child.tag) == "title":
                        title = child.text or ""
                elem.clear()
                if n_page % 20000 == 0:
                    print(f"  page {n_page:>8,} | {time.time() - t0:>5.0f}s | 最后标题 {title[:30]}")
    except EOFError:
        print("\n  [EOFError]")
    except ET.ParseError as e:
        print(f"\n  **复现** ParseError：page={n_page:,} 用时 {time.time() - t0:.0f}s")
        print(f"  报错：{e}")
        return True
    print(f"\n  完成：page={n_page:,} 用时 {time.time() - t0:.0f}s —— **未复现**")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",
                    default="E:/AI-learning/data/zhwiki-latest-pages-articles.xml.bz2")
    ap.add_argument("--out", default="E:/AI-learning/data/_repro.xml")
    ap.add_argument("--start-line", type=int, default=27000000)
    ap.add_argument("--take-lines", type=int, default=3000000)
    ap.add_argument("--skip-parse", action="store_true")
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.out)

    print(f"截取：第 {args.start_line:,} 行起，取 {args.take_lines:,} 行（按 <page> 边界对齐）")
    build_slice(src, dst, args.start_line, args.take_lines)
    print("=" * 70)

    if args.skip_parse:
        return 0
    print("用同一套 iterparse 跑切片：")
    reproduced = try_parse(dst)
    print("=" * 70)
    if reproduced:
        print("结论：**独立复现** —— 触发点在切片内，可继续二分定位")
    else:
        print("结论：**未复现** —— 与解析器累积状态有关（比如总处理量/页数），不是纯内容触发")
    return 2 if reproduced else 0


if __name__ == "__main__":
    sys.exit(main())
