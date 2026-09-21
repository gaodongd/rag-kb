"""隔离实验第 2 步：在切片上做一组「每次只改一个变量」的对照测试，定位崩溃层级。

用法：
    python src/test_variants.py --input E:/AI-learning/data/_repro.xml
"""
import argparse
import bz2
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path as _P

import xml.parsers.expat as expat


def _local(tag):
    if isinstance(tag, str) and tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def t_iterparse_text(path, bufsize=None):
    """A：当前做法 —— iterparse + 文本模式 + errors=replace"""
    n = 0
    kw = {} if bufsize is None else {"bufsize": bufsize}
    with open(path, encoding="utf-8", errors="replace") as f:
        for _ev, elem in ET.iterparse(f, events=("end",), **kw):
            if _local(elem.tag) != "page":
                continue
            n += 1
            elem.clear()
    return n


def t_iterparse_binary(path):
    """B：iterparse + 二进制模式（不经过 Python 的 UTF-8 解码）"""
    n = 0
    with open(path, "rb") as f:
        for _ev, elem in ET.iterparse(f, events=("end",)):
            if _local(elem.tag) != "page":
                continue
            n += 1
            elem.clear()
    return n


def t_expat_raw(path):
    """C：直接用 expat，不建树（排除 TreeBuilder 的影响）"""
    p = expat.ParserCreate()
    n = [0]

    def start(name, attrs):
        if name == "page":
            n[0] += 1

    p.StartElementHandler = start
    p.buffer_text = True
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 16)
            if not chunk:
                break
            p.Parse(chunk, False)
        p.Parse(b"", True)
    return n[0]


def t_expat_push(path, bufsize):
    """D：expat 用指定块大小喂（测块边界假设）——但每次都是同一个 parser，要重建"""
    p = expat.ParserCreate()
    n = [0]
    p.StartElementHandler = lambda name, attrs: (n.__setitem__(0, n[0] + 1)
                                                 if name == "page" else None)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(bufsize)
            if not chunk:
                break
            p.Parse(chunk, False)
        p.Parse(b"", True)
    return n[0]


def t_fromstring(path):
    """E：整块 ET.fromstring（完全不同的调用路径，同一 expat）"""
    data = _P(path).read_bytes()
    root = ET.fromstring(data)
    return sum(1 for e in root if _local(e.tag) == "page")


def t_lxml(path):
    """F：lxml（libxml2，不是 expat）—— 若它能过，说明是 expat 的问题"""
    try:
        from lxml import etree
    except ImportError:
        return "lxml 未安装"
    n = 0
    ctx = etree.iterparse(path, events=("end",), tag="{http://www.mediawiki.org/xml/export-0.11/}page")
    for _ev, elem in ctx:
        n += 1
        elem.clear()
    return n


TESTS = [
    ("A iterparse 文本模式(现状)", lambda p: t_iterparse_text(p)),
    ("B iterparse 二进制模式", lambda p: t_iterparse_binary(p)),
    ("C expat 裸用(64KB块)", lambda p: t_expat_raw(p)),
    ("D expat 16KB块", lambda p: t_expat_push(p, 1 << 14)),
    ("E ET.fromstring 整块", lambda p: t_fromstring(p)),
    ("F lxml/libxml2", lambda p: t_lxml(p)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="E:/AI-learning/data/_repro.xml")
    ap.add_argument("--only", default="", help="只跑指定字母，如 BC")
    args = ap.parse_args()

    p = args.input
    print(f"切片：{p}  ({_P(p).stat().st_size / 1024**2:.1f} MB)")
    print("=" * 72)
    print(f"{'测试':<28}{'结果':<34}{'耗时':>8}")
    print("-" * 72)

    results = {}
    for name, fn in TESTS:
        if args.only and name[0] not in args.only:
            continue
        t0 = time.time()
        try:
            n = fn(p)
            out = f"OK  page={n:,}" if isinstance(n, int) else str(n)
            results[name[0]] = "OK"
        except Exception as e:
            out = f"{type(e).__name__}: {str(e)[:60]}"
            results[name[0]] = f"FAIL {type(e).__name__}"
        print(f"{name:<28}{out:<34}{time.time() - t0:>7.0f}s", flush=True)

    print("-" * 72)
    print("判读：")
    print("  A 挂 / B 过  -> 问题出在「文本模式 + errors=replace」这条路径")
    print("  A 挂 / C 过  -> 问题出在 ElementTree 的建树/清理（不是 expat 本身）")
    print("  A 挂 / C 也挂 -> expat 本身在这份数据上就会挂")
    print("  F 过         -> 换个解析器就能绕开（lxml 走 libxml2）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
