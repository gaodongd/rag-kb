"""隔离实验第 3 步：测「总处理量」假设。

已确认的事实（2026-09-16）：
  * 全量解析每次都在**完全相同**位置崩（328,850 页 / 1.35 G字符 / RSS 仅 60MB）
  * 崩溃点所在的内容（第 27,000,000–30,000,535 行）单独切出来跑**完全正常**
  * Python 是 64 位，不存在 32 位地址空间耗尽

→ 剩下的假设：**失败与"累计处理了多少"有关，而与具体内容无关。**

本脚本把切片内容复制 N 份拼成一个 XML（内容完全相同，只是量变大），
如果它在某个累计量附近崩掉，就证实了"总量触发"。

用法：
    python src/test_volume.py --input E:/AI-learning/data/_repro8.xml
"""
import argparse
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import psutil
    _PROC = psutil.Process()
    def mem():
        m = _PROC.memory_info()
        return m.rss / 1024 ** 2, m.vms / 1024 ** 2
except Exception:
    def mem():
        return -1.0, -1.0


def _local(tag):
    if isinstance(tag, str) and tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="E:/AI-learning/data/_repro8.xml")
    ap.add_argument("--every", type=int, default=50000)
    args = ap.parse_args()

    path = Path(args.input)
    print(f"输入：{path}  ({path.stat().st_size / 1024**2:.1f} MB)")
    r0, v0 = mem()
    print(f"起点：RSS {r0:.0f} MB / VMS {v0:.0f} MB")
    print("-" * 72)

    n_page = 0
    n_ok = 0
    root = None
    chars = 0
    t0 = time.time()
    last_rss, last_vms = r0, v0

    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            ctx = ET.iterparse(f, events=("end",))
            for _ev, elem in ctx:
                if _local(elem.tag) != "page":
                    continue
                n_page += 1
                if root is None:
                    root = getattr(ctx, "root", None)

                title = ns = rev_text = ""
                for child in elem:
                    name = _local(child.tag)
                    if name == "title":
                        title = child.text or ""
                    elif name == "ns":
                        ns = (child.text or "").strip()
                    elif name == "revision":
                        for sub in child:
                            if _local(sub.tag) == "text":
                                rev_text = sub.text or ""
                                break
                elem.clear()

                if ns == "0" and title and rev_text and len(rev_text) >= 200:
                    n_ok += 1

                # 与原脚本一致的清理策略
                if n_ok and n_ok % 2000 == 0 and root is not None:
                    root.clear()

                if n_page % args.every == 0:
                    r, v = mem()
                    el = time.time() - t0
                    print(f"  page {n_page:>9,} | 采纳 {n_ok:>8,} | "
                          f"RSS {r:>7.0f} MB | VMS {v:>8.0f} MB | {n_page / max(el, 1e-6):>7.0f} p/s",
                          flush=True)
                    last_rss, last_vms = r, v
    except EOFError:
        print("\n  [EOFError] 正常结束")
    except ET.ParseError as e:
        r, v = mem()
        print(f"\n  **复现** ParseError：page={n_page:,} 采纳={n_ok:,}，"
              f"RSS {r:.0f} MB / VMS {v:.0f} MB，用时 {time.time() - t0:.0f}s")
        print(f"  报错：{e}")
        print(f"  最后采样点：RSS {last_rss:.0f} MB / VMS {last_vms:.0f} MB")
        return 2
    except MemoryError as e:
        print(f"\n  [MemoryError] page={n_page:,} -> {e}")
        return 2

    r, v = mem()
    print(f"\n  完成：page={n_page:,} / 采纳 {n_ok:,} / 用时 {time.time() - t0:.0f}s")
    print(f"  结束：RSS {r:.0f} MB / VMS {v:.0f} MB —— **未复现**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
