"""诊断工具：定位维基 XML 中触发 expat 'out of memory' 的位置。

背景（2026-09-16）：
  bzip2 -t 全流校验通过（文件字节级完整），但 wiki_extract.py 在
  `ExpatError: out of memory: line 28421794, column 44` 处中止，只处理了约 20% 的文件。
  机器有 31.7GB 内存、18.3GB 可用，不存在真实内存不足。

本脚本直接把那一行附近的内容打出来，看是什么东西让 expat 爆掉。
只做流式扫描，不构建 DOM，内存占用恒定。

用法：
    python src/probe_xml.py --target 28421794
"""
import argparse
import bz2
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",
                    default="E:/AI-learning/data/zhwiki-latest-pages-articles.xml.bz2")
    ap.add_argument("--target", type=int, default=28421794, help="要查看的行号")
    ap.add_argument("--before", type=int, default=6, help="向前看几行")
    ap.add_argument("--after", type=int, default=6, help="向后看几行")
    ap.add_argument("--max-preview", type=int, default=300, help="每行最多预览多少字符")
    args = ap.parse_args()

    lo = args.target - args.before
    hi = args.target + args.after

    print(f"扫描：{args.input}")
    print(f"目标行：{args.target}（区间 {lo} ~ {hi}）")
    print("-" * 70)

    t0 = time.time()
    lineno = 0
    nchars = 0          # 尚未含换行的字符数
    total_chars = 0
    hits = []

    with bz2.open(args.input, "rt", encoding="utf-8", errors="replace") as f:
        while True:
            line = f.readline()
            if not line:
                break
            lineno += 1
            nchars += len(line)
            total_chars += len(line)

            if lo <= lineno <= hi:
                hits.append((lineno, line))

            if lineno % 2000000 == 0:
                mb = total_chars / 1024 ** 2
                el = time.time() - t0
                print(f"  ... {lineno:,} 行 / 解压出 {mb:,.0f} MB / "
                      f"{mb / max(el, 1e-6):.1f} MB/s", flush=True)

            if lineno > hi:
                break

    el = time.time() - t0
    print(f"扫到第 {lineno:,} 行，解压出 {total_chars / 1024 ** 3:.2f} GB，用时 {el:.0f} 秒")
    print("=" * 70)

    if not hits:
        print("没扫到目标行（文件可能比目标行短）")
        return 1

    for ln, text in hits:
        body = text.rstrip("\n")
        preview = body[:args.max_preview]
        mark = "  <<<<< 目标行" if ln == args.target else ""
        print(f"\n[{ln}] 长度 {len(body):,} 字符{mark}")
        print(f"  开头: {preview!r}")
        if len(body) > args.max_preview:
            print(f"  结尾: ...{body[-200:]!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
