"""诊断：iterparse 在维基大 XML 上的内存行为对照实验。

背景（2026-09-16）：
  wiki_extract.py 在 20% 处报 `ExpatError: out of memory`（机器有 18GB 可用内存）。
  怀疑是 root 节点上的元素累积没被有效释放。本脚本用两种清理策略跑同一条数据流，
  每 20k 条打印一次进程 RSS，直接看出内存是不是线性增长。

用法：
    # 跑 20 万条，对比两种清理策略
    python src/diag_iterparse.py --accepted 200000

    # 一直跑到崩（复现原问题）
    python src/diag_iterparse.py --accepted 0
"""
import argparse
import bz2
import ctypes
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

NS = "{http://www.mediawiki.org/xml/export-"


def _local(tag):
    if isinstance(tag, str) and tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


# ---------- RSS 读取（Windows，不依赖 psutil） ----------
class _PMC(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def rss_mb():
    """当前进程 RSS（MB）。psutil 优先，失败回退到 ctypes。"""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 ** 2
    except Exception:
        pass
    try:
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            return pmc.WorkingSetSize / 1024 ** 2
    except Exception:
        pass
    return -1.0


def peak_rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().peak_wset / 1024 ** 2
    except Exception:
        return -1.0


SKIP = ("Wikipedia:", "Template:", "Category:", "File:", "Help:", "Portal:",
        "MediaWiki:", "Module:", "Draft:", "Talk:", "User:",
        "模块:", "模板:", "分类:", "文件:", "帮助:", "维基百科:", "用户:")


class _Counter:
    """包一层，统计从 bz2 读到多少字符 / 多少字节。"""
    def __init__(self, f):
        self._f = f
        self.chars = 0
        self.bytes = 0
        self.max_read = 0

    def read(self, n=-1):
        data = self._f.read(n)
        if data:
            self.chars += len(data)
            self.bytes += len(data.encode("utf-8", "replace"))
            self.max_read = max(self.max_read, len(data))
        return data


def run(path: Path, mode: str, limit_accepted: int, every: int = 20000):
    """mode: 'batch' = 原策略（每 2000 条被采纳的清了 root）
             'each'  = 每个 page 都清 root
             'none'  = 完全不清 root（对照，预期必炸）"""
    n_page = 0        # 处理过的 page 元素总数（含被过滤掉的）
    n_ok = 0          # 被采纳的
    t0 = time.time()
    samples = []
    cnt = None

    print(f"\n{'=' * 70}")
    print(f"策略 = {mode}   起点 RSS = {rss_mb():.0f} MB")
    print(f"{'=' * 70}")

    with bz2.open(path, "rt", encoding="utf-8", errors="replace") as raw:
        f = _Counter(raw)
        cnt = f
        ctx = ET.iterparse(f, events=("end",))
        root = None
        try:
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

                if ns == "0" and title and rev_text and len(rev_text) >= 200 \
                        and not title.startswith(SKIP) \
                        and not rev_text.lstrip()[:9].upper().startswith("#REDIRECT"):
                    n_ok += 1
                    if limit_accepted and n_ok >= limit_accepted:
                        print(f"\n  达到 {limit_accepted:,} 条，停止")
                        break

                # ---- 三种清理策略 ----
                if mode == "each":
                    if root is not None:
                        root.clear()
                elif mode == "batch":
                    if n_ok and n_ok % 2000 == 0 and root is not None:
                        root.clear()

                if n_page % every == 0:
                    el = time.time() - t0
                    mem = rss_mb()
                    samples.append((n_page, n_ok, mem))
                    print(f"  page {n_page:>9,} | 采纳 {n_ok:>8,} | RSS {mem:>7.0f} MB | "
                          f"解压 {f.chars / 1024**3:>6.2f} GB | {n_page / max(el, 1e-6):>7.0f} page/s")
        except EOFError:
            print("\n  [EOFError] 正常结束")
        except ET.ParseError as e:
            el = time.time() - t0
            print(f"\n  [ParseError] n_page={n_page:,} n_ok={n_ok:,} "
                  f"RSS={rss_mb():.0f} MB 用时 {el:.0f}s")
            print(f"  报错内容：{e}")
            if cnt is not None:
                print(f"  已读 {cnt.chars / 1024**3:.2f} GB 解压字符 / "
                      f"单次最大 read = {cnt.max_read:,} 字符")
        except MemoryError as e:
            print(f"\n  [MemoryError] n_page={n_page:,} RSS={rss_mb():.0f} MB -> {e}")

    el = time.time() - t0
    print(f"\n  小结：page {n_page:,} / 采纳 {n_ok:,} / {el:.0f}s / RSS {rss_mb():.0f} MB")
    if cnt is not None:
        print(f"  解压总量 {cnt.chars / 1024**3:.2f} GB / 单次最大 read {cnt.max_read:,} 字符")
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",
                    default="E:/AI-learning/data/zhwiki-latest-pages-articles.xml.bz2")
    ap.add_argument("--accepted", type=int, default=200000, help="采够多少条就停，0=不限")
    ap.add_argument("--modes", default="each,batch,none")
    args = ap.parse_args()

    path = Path(args.input)
    for mode in args.modes.split(","):
        run(path, mode.strip(), args.accepted)
    return 0


if __name__ == "__main__":
    sys.exit(main())
