"""诊断：量出 expat 崩溃时的**精确解压字节偏移**。

已知：
  * 崩点随喂入块大小漂移（iterparse 16KB 块 → 第 28,421,794 行；
    expat 直连 1MB 块 → 第 28,405,697 行）
  * 崩点换算成字符都是 ~1.35 G字符
  * 放开 SetAllocTracker* / SetBillionLaughs* 两项保护后**仍然崩**

→ 触发条件像是「输入累计到某个量」，而非某段内容。
   本脚本量出精确的 UTF-8 字节偏移，看它是否撞上 32 位边界：
   2^31 = 2,147,483,648   2^32 = 4,294,967,296

用法：
    python src/probe_offset.py
"""
import bz2
import sys
import time

import xml.parsers.expat as expat


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "E:/AI-learning/data/zhwiki-latest-pages-articles.xml.bz2"
    chunk_chars = int(sys.argv[2]) if len(sys.argv) > 2 else (1 << 20)

    print(f"输入：{path}")
    print(f"块大小：{chunk_chars:,} 字符")
    print(f"参照：2^31 = {2**31:,}   2^32 = {2**32:,}   2^33 = {2**33:,}")
    print("-" * 70)

    p = expat.ParserCreate()
    p.StartElementHandler = lambda n, a: None

    chars = 0
    nbytes = 0          # 喂给 expat 的 UTF-8 字节累计
    t0 = time.time()
    mark = 0

    try:
        with bz2.open(path, "rt", encoding="utf-8", errors="replace") as f:
            while True:
                chunk = f.read(chunk_chars)
                if not chunk:
                    p.Parse(b"", True)
                    break
                chars += len(chunk)
                nbytes += len(chunk.encode("utf-8"))
                p.Parse(chunk, False)
                if nbytes // (1 << 30) > mark:
                    mark = nbytes // (1 << 30)
                    print(f"  {mark} GiB 字节 | {chars / 1024**3:.3f} G字符 | "
                          f"{time.time() - t0:.0f}s", flush=True)
    except expat.ExpatError as e:
        print("\n" + "=" * 70)
        print(f"**崩了**：{e}")
        print(f"  精确偏移 chars = {chars:,}")
        print(f"  精确偏移 bytes = {nbytes:,}   ({nbytes / 1024**3:.4f} GiB)")
        print(f"  用时 {time.time() - t0:.0f}s")
        print()
        for name, v in [("2^31", 2**31), ("2^32", 2**32), ("2^33", 2**33)]:
            print(f"  与 {name} 相差 {nbytes - v:+,} 字节")
        print()
        print("  判读：若 bytes 恰好≈某个 2 的幂，就是 32 位计数器溢出；")
        print("        若远离所有整数边界，则是别的原因（需继续查）。")
        return 2

    print(f"\n跑完全程：{chars / 1024**3:.3f} G字符 / {nbytes / 1024**3:.3f} GiB 字节 / "
          f"{time.time() - t0:.0f}s —— 未崩")
    return 0


if __name__ == "__main__":
    sys.exit(main())
