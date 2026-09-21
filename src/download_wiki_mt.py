"""中文维基百科 dump 下载器（多线程分片版）

为什么用多线程：
  本机实测（2026-09-16）单连接被限速在 ~107 KB/s，多开连接总速度能线性叠加。
  单线程下 3.2GB 要 8 小时；4 并发约 2 小时。

⚠️ 为什么并发不能调太高（实测踩过）：
  开 12 并发全速跑 15 分钟，服务端开始返回 HTTP 429 Too Many Requests，
  203 个分片挂掉 107 个。Wikimedia 的 CDN 有反滥用限速，"并发越快越好"
  只在没有反滥用机制时成立。
  所以：**默认 4 并发**，并且脚本内置全局冷却 —— 收到 429 时读 Retry-After
  让所有 worker 一起退避（只让单个 worker 退避是没用的，其他连接还在打服务端）。
  ⚠️ 短时间测速（几十秒）看不出这个问题，必须跑够久才知道真实上限。

为什么默认只下 700MB：
  完整维基 3.2GB（压缩）解压后约 25GB XML、400 万条目。按每条目平均 6 个
  chunk 算 → 2400 万个 chunk。在你的 4060 上跑 bge-large-zh 向量化
  （约 400 chunk/s）要 16 小时以上，而且信息高度冗余。
  700MB 约等于 60-80 万条目 → 约 300-400 万 chunk，向量化 2-3 小时，
  是个人项目的合理规模。想要全量加 --full。

用法（CMD）：
  :: 默认：下 700MB
  "C:\\Users\\搞懂\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" src\\download_wiki_mt.py

  :: 自定义大小
  "C:\\Users\\搞懂\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" src\\download_wiki_mt.py --mb 400

  :: 全量 3.2GB
  "C:\\Users\\搞懂\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" src\\download_wiki_mt.py --full

  :: 断了重跑同一条命令，已下完的分片自动跳过
"""
import argparse
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

URL = "https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2"
DATA_DIR = Path(r"E:\AI-learning\data")
DEST = DATA_DIR / "zhwiki-latest-pages-articles.xml.bz2"
PART = DEST.with_suffix(DEST.suffix + ".part")
CHUNK_DIR = DATA_DIR / "_chunks"

UA = "rag-kb-study/1.0 (educational RAG project; local use only)"
CHUNK_SIZE = 16 * 1024 * 1024   # 每片 16MB
WORKERS = 4                      # ⚠️ 别调太高：12 并发全速跑 15 分钟就被 429 限流
MAX_RETRY = 6                    # 普通错误的重试次数
MAX_THROTTLE_RETRY = 40          # 429 单独给大预算，不算进上面那个

_lock = threading.Lock()
_state = {"done": 0, "total": 0, "bytes": 0, "t0": time.time(), "throttled": 0}

# ---- 全局冷却：收到 429 时让所有 worker 一起停，否则退避没意义 ----
_cooldown_lock = threading.Lock()
_cooldown_until = 0.0


def _trigger_cooldown(seconds):
    global _cooldown_until
    with _cooldown_lock:
        _cooldown_until = max(_cooldown_until, time.time() + seconds)


def _respect_cooldown():
    while True:
        with _cooldown_lock:
            wait = _cooldown_until - time.time()
        if wait <= 0:
            return
        time.sleep(min(wait, 3))


def human(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def remote_size():
    req = urllib.request.Request(URL, method="HEAD", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers.get("Content-Length") or 0)


def fetch_range(start, end, path):
    """下载 [start, end] 区间到 path。已存在且大小正确则跳过。

    429 的处理是本脚本的核心：
      Wikimedia 的 CDN 有反滥用限速。实测 12 并发全速冲 15 分钟后开始返回
      HTTP 429，203 个分片挂掉 107 个。所以收到 429 时要：
        ① 读 Retry-After 头，按它指定的秒数退避；
        ② 所有 worker 一起退避（全局冷却），而不是各自傻试 —— 否则退避期间
           其他连接还在打服务端，等于没退避；
        ③ 429 有独立的、更大的重试预算（不占用普通错误的 6 次）。
    """
    want = end - start + 1
    if path.exists() and path.stat().st_size == want:
        with _lock:
            _state["done"] += 1
            _state["bytes"] += want
        return "skip"

    last_err = None
    normal_tries = 0
    throttled_tries = 0

    while True:
        try:
            _respect_cooldown()
            have = path.stat().st_size if path.exists() else 0
            if have >= want:
                break
            headers = {"User-Agent": UA, "Range": f"bytes={start + have}-{end}"}
            req = urllib.request.Request(URL, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                with open(path, "ab") as f:
                    while True:
                        buf = resp.read(1024 * 256)
                        if not buf:
                            break
                        f.write(buf)
                        with _lock:
                            _state["bytes"] += len(buf)
            if path.stat().st_size != want:
                raise IOError(f"分片大小不对 {path.stat().st_size} != {want}")
            with _lock:
                _state["done"] += 1
            return "ok"

        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429:
                throttled_tries += 1
                if throttled_tries > MAX_THROTTLE_RETRY:
                    raise IOError(f"分片 {path.name} 被限流 {throttled_tries} 次仍未通过")
                try:
                    ra = int(e.headers.get("Retry-After") or 0)
                except (TypeError, ValueError):
                    ra = 0
                # 服务端没给就指数退避，最多 5 分钟
                wait = ra if ra > 0 else min(15 * (2 ** (throttled_tries - 1)), 300)
                wait += random.uniform(0, 5)          # 加抖动，避免所有 worker 同时醒来
                with _lock:
                    _state["throttled"] += 1
                _trigger_cooldown(wait)
                print(f"\n  [限流] 收到 429，全局退避 {wait:.0f}s "
                      f"（第 {throttled_tries} 次）", flush=True)
            else:
                normal_tries += 1
                if normal_tries >= MAX_RETRY:
                    raise IOError(f"分片 {path.name} 失败 {normal_tries} 次：{e}")
                time.sleep(min(2 * normal_tries, 15))

        except Exception as e:
            last_err = e
            normal_tries += 1
            if normal_tries >= MAX_RETRY:
                raise IOError(f"分片 {path.name} 失败 {normal_tries} 次：{last_err}")
            time.sleep(min(2 * normal_tries, 15))


def progress_loop(stop_event):
    while not stop_event.wait(5.0):
        with _lock:
            done, total, nbytes, thr = (_state["done"], _state["total"],
                                        _state["bytes"], _state["throttled"])
        el = time.time() - _state["t0"]
        spd = nbytes / max(el, 0.001)
        line = f"  分片 {done}/{total}  已下 {human(nbytes)}  速度 {human(spd)}/s"
        if nbytes:
            line += f"  剩余约 {(total * CHUNK_SIZE - nbytes) / max(spd, 1) / 60:.0f} 分钟"
        if thr:
            line += f"  [被限流 {thr} 次]"
        print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=700,
                    help="下载多少 MB，默认 700（约 60-80 万条目）")
    ap.add_argument("--full", action="store_true", help="下载完整 3.2GB")
    ap.add_argument("--workers", type=int, default=WORKERS, help=f"并发数，默认 {WORKERS}")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("中文维基百科 dump 下载器（多线程分片）")
    print("=" * 64)

    total_size = remote_size()
    print(f"服务器总大小：{human(total_size)}")

    target = total_size if args.full else min(args.mb * 1024 * 1024, total_size)
    n_chunks = (target + CHUNK_SIZE - 1) // CHUNK_SIZE

    print(f"本次目标：{human(target)}  （{n_chunks} 个分片 × {human(CHUNK_SIZE)}）")
    print(f"并发连接：{args.workers}")
    print(f"分片目录：{CHUNK_DIR}")
    print("-" * 64)
    print("断点续传：已下完的分片会自动跳过，重跑同一条命令即可。")
    print()

    _state["total"] = n_chunks
    stop_event = threading.Event()
    t = threading.Thread(target=progress_loop, args=(stop_event,), daemon=True)
    t.start()

    jobs = []
    for i in range(n_chunks):
        s = i * CHUNK_SIZE
        e = min(s + CHUNK_SIZE, target) - 1
        jobs.append((s, e, CHUNK_DIR / f"{i:05d}.bin"))

    failures = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch_range, s, e, p): p.name for s, e, p in jobs}
            for fu in as_completed(futs):
                try:
                    fu.result()
                except Exception as ex_:
                    failures.append((futs[fu], str(ex_)))
    except KeyboardInterrupt:
        stop_event.set()
        print("\n\n已手动停止。分片都保留着，重跑同一条命令继续。")
        return 130

    stop_event.set()
    time.sleep(0.2)
    print()

    if failures:
        print(f"[!] {len(failures)} 个分片失败：")
        for name, err in failures[:5]:
            print(f"    {name}: {err}")
        print("\n重跑同一条命令会自动重试失败的分片。")
        return 1

    # 拼接
    print("全部分片就绪，开始拼接 ...")
    t0 = time.time()
    with open(PART, "wb") as out:
        for i, (s, e, p) in enumerate(jobs):
            with open(p, "rb") as f:
                while True:
                    buf = f.read(4 * 1024 * 1024)
                    if not buf:
                        break
                    out.write(buf)
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                print(f"  拼接到第 {i + 1}/{len(jobs)} 片", flush=True)
    size = PART.stat().st_size
    print(f"拼接完成：{human(size)}  用时 {time.time() - t0:.0f}s")

    # 校验
    print("校验 bz2（读前 500 行）...")
    import bz2
    n = 0
    try:
        with bz2.open(PART, "rb") as f:
            while n < 500 and f.readline():
                n += 1
        print(f"  OK，能正常解压（读到 {n} 行）")
    except Exception as e:
        print(f"  [!] 解压校验异常：{type(e).__name__}: {e}")
        print("  这是流式截断的正常表现，解析脚本 wiki_extract.py 已做容错处理。")

    DEST.unlink(missing_ok=True)
    PART.rename(DEST)
    print(f"\n完成 → {DEST}")

    # 清理分片
    freed = 0
    for _, _, p in jobs:
        freed += p.stat().st_size
        p.unlink(missing_ok=True)
    try:
        CHUNK_DIR.rmdir()
    except OSError:
        pass
    print(f"已清理分片，释放 {human(freed)}")
    print()
    print("下一步：跑解析")
    print(r'  "C:\Users\搞懂\AppData\Local\Programs\Python\Python313\python.exe" src\wiki_extract.py')
    return 0


if __name__ == "__main__":
    sys.exit(main())
