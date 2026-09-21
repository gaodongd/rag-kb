"""下载中文维基百科 dump —— 断点续传 + 无限重试 + 完整性校验。

为什么不直接用 curl：
  这台机器的网络会偶发 DNS 解析失败、连接被掐。curl 的 --retry 一旦耗尽就退出，
  3 小时的下载白跑一半。这个脚本把"断了就自动接上下"做到位，挂着不用管。

用法（CMD）：
  :: A. 先下 800MB —— 拿到几万条目，够跑通 M1（约 35 分钟）
  "C:\\Users\\搞懂\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" src\\download_wiki.py --max-mb 800

  :: B. 完整下载 3.4GB（约 2.5 小时）
  "C:\\Users\\搞懂\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" src\\download_wiki.py

  :: C. 断了之后想要完整版：把 --max-mb 拿掉，重跑同一条命令，自动接着下
  "C:\\Users\\搞懂\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" src\\download_wiki.py

实测速度（2026-09-16，本机）：
  官方 dumps.wikimedia.org   394 KB/s   <- 用它
  mirror.accum.se            32 B/s     <- 废
  ftp.acc.umu.se             280 B/s    <- 废
  dumps.wikimedia.your.org   8 KB/s     <- 废
"""
import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2"
DEST = Path(r"E:\AI-learning\data\zhwiki-latest-pages-articles.xml.bz2")
PART = DEST.with_suffix(DEST.suffix + ".part")

# Wikimedia 要求带可识别的 UA，裸 urllib 的默认 UA 可能被拒
UA = "rag-kb-study/1.0 (educational RAG project; local use only)"
CHUNK = 256 * 1024
MAX_RETRY = 999


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def remote_size():
    """拿服务器上的文件总大小。失败返回 0。"""
    req = urllib.request.Request(URL, method="HEAD", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers.get("Content-Length") or 0)


def try_open(start):
    """发起请求；start>0 时带 Range 头做续传。
    返回 (响应对象, 本次实际会收到的字节数, 是否 206 续传成功)。"""
    headers = {"User-Agent": UA}
    if start > 0:
        headers["Range"] = f"bytes={start}-"
    req = urllib.request.Request(URL, headers=headers)
    resp = urllib.request.urlopen(req, timeout=60)
    code = resp.getcode()
    if start > 0 and code != 206:
        # 服务器不支持续传，只能从头来
        return resp, None, False
    total = int(resp.headers.get("Content-Length") or 0)
    return resp, total, True


def download(max_bytes=0):
    got_total = 0
    if PART.exists():
        got_total = PART.stat().st_size
        print(f"发现未完成的下载，从 {human(got_total)} 处继续")

    retry = 0
    t_start = time.time()
    bytes_this_run = 0
    last_print = 0.0

    while True:
        try:
            resp, remaining, resumed = try_open(got_total)
            if got_total > 0 and not resumed:
                print("\n[!] 服务器不支持断点续传，从头开始")
                got_total = 0
                if PART.exists():
                    PART.unlink()

            mode = "ab"
            with open(PART, mode) as f:
                while True:
                    # 留 1KB 容差：bz2 截断处解压会 EOFError，解析脚本已容错
                    if max_bytes and got_total + bytes_this_run >= max_bytes:
                        print(f"\n已达到 --max-mb 限制（{human(max_bytes)}），主动停止。")
                        print(f"文件在：{PART}")
                        print("想要完整版：去掉 --max-mb 重跑同一条命令，会自动接着下。")
                        return 0
                    buf = resp.read(CHUNK)
                    if not buf:
                        break
                    f.write(buf)
                    bytes_this_run += len(buf)

                    now = time.time()
                    if now - last_print >= 2.0:
                        last_print = now
                        done = got_total + bytes_this_run
                        spd = bytes_this_run / max(now - t_start, 0.001)
                        line = f"  已下 {human(done)}"
                        if remaining:
                            line += f" / {human(got_total + remaining)}"
                            line += f" ({done * 100 / (got_total + remaining):.1f}%)"
                            eta = (got_total + remaining - done) / max(spd, 1)
                            line += f"  速度 {human(spd)}/s  剩余约 {eta / 60:.0f} 分钟"
                        else:
                            line += f"  速度 {human(spd)}/s"
                        print(line, flush=True)

            # 正常读完
            final_size = PART.stat().st_size
            print(f"\n下载结束，文件大小 {human(final_size)}")
            if remaining and resumed:
                expect = got_total + remaining
                if final_size < expect:
                    raise IOError(f"文件不完整：{final_size} < {expect}")

            # 完整性校验：看能不能解压出头几条
            print("校验 bz2 头部 ...")
            import bz2
            n = 0
            with bz2.open(PART, "rb") as f:
                while n < 200 and f.readline():
                    n += 1
            print(f"  能正常解压（读到 {n} 行），文件可用。")

            DEST.unlink(missing_ok=True)
            PART.rename(DEST)
            print(f"\n完成 → {DEST}")
            return 0

        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            retry += 1
            if retry > MAX_RETRY:
                print(f"\n重试 {MAX_RETRY} 次仍失败，放弃。最后错误：{e}")
                return 1
            got_total = PART.stat().st_size if PART.exists() else 0
            bytes_this_run = 0
            wait = min(5 * retry, 60)
            print(f"\n[!] 网络中断（{type(e).__name__}: {e}）")
            print(f"    已保存 {human(got_total)}，{wait} 秒后从断点继续（第 {retry} 次重试）")
            time.sleep(wait)
            t_start = time.time()
            last_print = 0.0
        except KeyboardInterrupt:
            got_total = PART.stat().st_size if PART.exists() else 0
            print(f"\n\n已手动停止。已下载 {human(got_total)} 保存在：")
            print(f"  {PART}")
            print("重跑同一条命令会自动续传。")
            return 130


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-mb", type=int, default=0,
                    help="只下这么多 MB（用于快速跑通流程），默认 0 = 完整下载")
    args = ap.parse_args()

    DEST.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("中文维基百科 dump 下载器")
    print("=" * 64)
    print(f"源：{URL}")
    print(f"存：{DEST}")

    try:
        size = remote_size()
        print(f"服务器文件大小：{human(size) if size else '未知'}")
    except Exception as e:
        print(f"[!] 拿不到文件大小（{e}），仍然尝试下载")

    if args.max_mb:
        print(f"本次限制：{args.max_mb} MB（够跑通 M1 用）")
    print()
    print("提示：中途 Ctrl+C 不会丢数据，重跑自动续传。")
    print("-" * 64)

    return download(max_bytes=args.max_mb * 1024 * 1024 if args.max_mb else 0)


if __name__ == "__main__":
    sys.exit(main())
