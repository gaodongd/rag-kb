"""在纯 Python 里复现 udf_clean / udf_chunk，找出让 Spark worker 崩掉的那条记录。

Spark 把 Python worker 的 stderr 吞掉了，只留一句
  java.net.SocketException: Connection reset by peer: socket write error
所以必须绕开 Spark，直接在解释器里跑同样的函数。

用法：
    python src/diag_udf.py --input data/_smoke/smoke.jsonl
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

# 直接复用管道里的实现，不复制粘贴 —— 复制会导致两边逻辑漂移
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pipeline import clean_wikitext, split_chunks, cn_ratio, max_char_share  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/_smoke/smoke.jsonl")
    ap.add_argument("--show", type=int, default=3, help="每条异常最多打印几条样例")
    args = ap.parse_args()

    path = Path(args.input)
    n = 0
    bad_clean, bad_chunk = [], []
    weird_utf = []

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            n += 1
            try:
                rec = json.loads(line)
            except Exception as e:
                print(f"[JSON 坏行] line {lineno}: {type(e).__name__}: {e}")
                continue

            text = rec.get("text") or ""

            # --- 检测非法 Unicode（孤立代理项 / 需替换的字符）---
            if "\ud800" <= max(text, default="\0") <= "\udfff" or _has_surrogate(text):
                weird_utf.append((lineno, rec.get("title")))

            # --- 1. clean ---
            try:
                clean = clean_wikitext(text)
            except Exception as e:
                bad_clean.append((lineno, rec.get("title"), type(e).__name__, str(e)[:200]))
                print(f"[clean 崩] line {lineno} title={rec.get('title')!r} "
                      f"{type(e).__name__}: {e}")
                traceback.print_exc()
                continue

            # --- 2. 逐条模拟 UDF 返回值：真正会崩的是"编码回 UTF-8"这一步 ---
            try:
                clean.encode("utf-8")
            except Exception as e:
                print(f"[clean 结果无法编码] line {lineno} title={rec.get('title')!r} "
                      f"{type(e).__name__}: {e}")
                bad_clean.append((lineno, rec.get("title"), "EncodeFail", str(e)[:200]))

            # --- 3. chunk ---
            try:
                chunks = split_chunks(clean)
            except Exception as e:
                bad_chunk.append((lineno, rec.get("title"), type(e).__name__, str(e)[:200]))
                print(f"[chunk 崩] line {lineno} title={rec.get('title')!r} "
                      f"{type(e).__name__}: {e}")
                traceback.print_exc()
                continue

            for sec, body in chunks:
                try:
                    body.encode("utf-8")
                except Exception as e:
                    print(f"[chunk 结果无法编码] line {lineno} title={rec.get('title')!r} "
                          f"{type(e).__name__}: {e}")

    print()
    print("=" * 64)
    print(f"扫描行数            : {n:,}")
    print(f"clean 异常          : {len(bad_clean)}")
    print(f"chunk 异常          : {len(bad_chunk)}")
    print(f"含孤立代理项        : {len(weird_utf)}")
    for lineno, title in weird_utf[:args.show]:
        print(f"    line {lineno}  title={title!r}")
    print("=" * 64)
    return 0 if not (bad_clean or bad_chunk) else 1


def _has_surrogate(s: str) -> bool:
    for ch in s:
        o = ord(ch)
        if 0xD800 <= o <= 0xDFFF:
            return True
    return False


if __name__ == "__main__":
    sys.exit(main())
