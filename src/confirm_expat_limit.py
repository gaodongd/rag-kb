"""确认实验：Expat 的「动态内存异常放大保护」是不是真凶。

线索链（2026-09-16）：
  1. 全量解析每次在**完全相同**位置崩：ExpatError: out of memory: line 28421794
  2. 崩溃时进程 RSS 只有 60 MB，机器有 18GB 空闲 —— 不是真的内存不足
  3. 崩溃点所在内容单独切出来跑完全正常 —— 不是内容触发
  4. 本机 expat 是 2.8.2，暴露了 SetAllocTrackerActivationThreshold /
     SetAllocTrackerMaximumAmplification —— Expat 2.6+ 新增的
     「动态内存异常放大保护」，默认激活阈值 64 MiB、最大放大倍数 100.0
  5. 该保护的实现方式就是**让分配返回 NULL** —— 于是对外报 XML_ERROR_NO_MEMORY
     （即 "out of memory"），但进程其实没吃多少内存

本脚本把这两个限制放开，重跑同一份数据：
  * 若跑过原崩点（1.35 G字符）→ 真凶确认，且修复方案就是放开限制 / 换解析器
  * 若仍在同一位置崩 → 推翻该假设，换方向

用法：
    python src/confirm_expat_limit.py
"""
import argparse
import bz2
import sys
import time
import xml.etree.ElementTree as ET

import xml.parsers.expat as expat

TARGET_CHARS = 1.6 * 1024 ** 3      # 原崩点是 1.35 G字符，跑到这里就算成功
CRASH_CHARS = 1.35 * 1024 ** 3


def build_parser(relax: bool):
    """建一个 expat 解析器；relax=True 时放开两项保护。"""
    p = expat.ParserCreate()
    if relax:
        for name, val in [
            ("SetAllocTrackerActivationThreshold", 1 << 42),      # 4 TiB
            ("SetAllocTrackerMaximumAmplification", 1e6),
            ("SetBillionLaughsAttackProtectionActivationThreshold", 1 << 42),
            ("SetBillionLaughsAttackProtectionMaximumAmplification", 1e6),
        ]:
            fn = getattr(p, name, None)
            if fn is None:
                print(f"    [警告] 本解释器没有 {name}，跳过")
                continue
            try:
                fn(val)
                print(f"    已设置 {name} = {val}")
            except Exception as e:
                print(f"    [警告] 设置 {name} 失败: {type(e).__name__}: {e}")
    return p


def run(path, relax, target_chars=TARGET_CHARS, quiet=False):
    p = build_parser(relax)
    n_page = 0
    chars = 0
    t0 = time.time()

    def on_start(name, attrs):
        nonlocal n_page
        if name == "page":
            n_page += 1

    p.StartElementHandler = on_start

    try:
        with bz2.open(path, "rt", encoding="utf-8", errors="replace") as f:
            while chars < target_chars:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                chars += len(chunk)
                p.Parse(chunk, False)
                if not quiet and chars % (200 << 20) < (1 << 20):
                    el = time.time() - t0
                    print(f"    ... {chars / 1024**3:.2f} G字符 | page {n_page:,} | "
                          f"{chars / 1024**2 / max(el, 1e-6):.1f} M字符/s", flush=True)
            p.Parse(b"", True)          # 收尾
    except expat.ExpatError as e:
        print(f"\n    **崩了** {type(e).__name__}: {e}")
        print(f"    位置：{chars / 1024**3:.2f} G字符 / page {n_page:,} / "
              f"用时 {time.time() - t0:.0f}s")
        return False, chars

    el = time.time() - t0
    print(f"\n    跑完目标：{chars / 1024**3:.2f} G字符 / page {n_page:,} / {el:.0f}s")
    return True, chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",
                    default="E:/AI-learning/data/zhwiki-latest-pages-articles.xml.bz2")
    ap.add_argument("--target-gchars", type=float, default=1.6)
    ap.add_argument("--only", default="", choices=["", "relax", "strict"])
    args = ap.parse_args()

    target = int(args.target_gchars * 1024 ** 3)
    print(f"目标：跑到 {args.target_gchars} G字符（原崩点 {CRASH_CHARS / 1024**3:.2f} G字符）")
    print("=" * 74)

    if args.only in ("", "strict"):
        print("\n【A】保持默认保护（预期在 1.35 G字符 崩）")
        ok_a, _ = run(args.input, relax=False, target_chars=target)
        print(f"  A 结论：{'跑过崩点' if ok_a else '仍在原位置附近崩'}")
        print("=" * 74)

    if args.only in ("", "relax"):
        print("\n【B】放开两项保护（若真凶是它，应能跑过崩点）")
        ok_b, _ = run(args.input, relax=True, target_chars=target)
        print(f"  B 结论：{'**跑过崩点 —— 真凶确认**' if ok_b else '仍崩 —— 两个假设都被推翻'}")
        print("=" * 74)

    return 0


if __name__ == "__main__":
    sys.exit(main())
