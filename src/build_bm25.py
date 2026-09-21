#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 8 步 · T6 建 BM25 倒排索引（预计 1.97 亿 postings / 1.18 GB）

为什么不用 rank_bm25 之类的库
-----------------------------
它们的隐含前提是「语料能整份进内存、能算词频矩阵」。331.6 万块不满足。
而且**倒排索引是检索岗的核心考点**，自己建一遍比调库有价值得多。

结构：CSR（Compressed Sparse Row）
---------------------------------
不要把「词 × 文档」当成二维打分表（那是 386 万 × 331.6 万，天文数字）。
只有极少数位置有值 —— 那就**只存有值的位置**，这正是稀疏矩阵 CSR 的思路：

    vocab.txt     第 i 行 = 第 i 号词（按字典序，查询时二分查找）
    indptr[V+1]   第 t 号词的 postings 落在 [indptr[t], indptr[t+1]) 区间
    doc_ids[P]    这些 postings 各自属于哪个块（int32）
    tf_norm[P]    **预先算好的** BM25 词频分量（fp16）
    doc_len[N]    每块的词数（BM25 的 b 参数要用）

`tf_norm` 预计算是关键优化：BM25 的词频分量
    tf×(k1+1) / (tf + k1×(1-b+b×dl/avgdl))
只依赖 (tf, dl, avgdl)，三者都与查询无关 → 可以离线算好。
查询时就只剩「查表 + 乘 IDF + 累加」，单核百万 postings/秒。

两趟分词
--------
第一趟只为统计 df（每个词出现在多少块里），第二趟才写 postings。
看起来浪费一次分词，但换来的是：**postings 可以按 indptr 直接写入，完全不需要排序**。
否则要把 1.97 亿条记录排序 —— 那才是真的慢。

用法
----
    :: 全量（约 20 分钟）
    "...python313\\python.exe" src\\build_bm25.py

    :: 小规模验通
    "...python313\\python.exe" src\\build_bm25.py --limit 20000 --out eval\\results\\bm25_smoke
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

# 子进程（Windows 用 spawn）会重新 import 本模块，所以 sys.path 要在模块级插好，
# 否则 worker 里的 `from zh import norm_text` 直接 ImportError。
sys.path.insert(0, str(HERE))

from zh import HAVE_ZHCONV, TARGET, norm_text  # noqa: E402

# 停用词：只挡最高频的一批。单字过滤已经挡掉大头，二字停用词影响有限。
STOP = {
    "可以", "因为", "所以", "但是", "并且", "以及", "或者", "如果", "这个", "那个",
    "这些", "那些", "一个", "一种", "我们", "他们", "你们", "自己", "之后", "之前",
    "同时", "由于", "因此", "the", "and", "for", "with", "from", "that", "this",
}

_KEEP_CACHE = {}


def _keep(w: str) -> bool:
    """过滤：砍单字、纯数字、纯符号、停用词。"""
    if len(w) < 2:
        return False
    r = _KEEP_CACHE.get(w)
    if r is not None:
        return r
    ok = (w not in STOP) and (not w.isdigit()) and any(c.isalnum() for c in w)
    if len(_KEEP_CACHE) < 2_000_000:
        _KEEP_CACHE[w] = ok
    return ok


def _init_worker():
    """每个子进程启动时执行一次（Windows 是 spawn，jieba 要在子进程里重新加载）。"""
    import jieba
    jieba.initialize()


def _tokenize_batch(texts):
    """
    每个任务处理**一批**文本，返回 [词频 dict, ...]。

    为什么返回「词→词频」而不是 token 列表：
      ① 去重后体积小一大截（实测平均 167 token → 59.5 个词），多进程 IPC 省 2/3；
      ② pass1 要 df（每个词 +1）、pass2 要 tf —— 两者都只需要去重后的词频，一次就够。
    """
    import jieba
    out = []
    for t in texts:
        c = {}
        # ⚠️ 归一化必须在 jieba.lcut **之前**。
        #    放到分词之后是无效的：繁体串那时已经被切碎成单字，而单字会被 _keep 丢掉。
        #    归一化的开销分摊到 16 个 worker 上，实测可忽略（见 README 的建索引耗时）。
        for w in jieba.lcut(norm_text(t)):
            if _keep(w):
                c[w] = c.get(w, 0) + 1
        out.append(c)
    return out


def iter_texts(parquet: Path, batch: int, limit=None):
    """流式读 parquet，分批 yield list[str]。不把 331.6 万块一次性载入。"""
    import pyarrow.dataset as ds_mod
    d = ds_mod.dataset(str(parquet), format="parquet")
    scanner = d.scanner(columns=["chunk_text"], batch_size=batch)
    seen = 0
    for rb in scanner.to_batches():
        texts = rb.column("chunk_text").to_pylist()
        if limit is not None and seen + len(texts) > limit:
            texts = texts[: limit - seen]
        if texts:
            yield texts
            seen += len(texts)
        if limit is not None and seen >= limit:
            return


def pass1(args, path_df: Path, path_dl: Path):
    """第一趟：统计 df 与每块的词数。结果落盘，便于断点重跑时跳过这一趟。"""
    print()
    print("-" * 76)
    print(f"第一趟 · 统计 df（token 出现在多少块里）  进程数 = {args.procs}")
    print("-" * 76)
    t0 = time.time()
    df = Counter()
    doc_len = []
    n_doc = 0

    with Pool(args.procs, initializer=_init_worker) as pool:
        for batch in pool.imap(_tokenize_batch,
                               iter_texts(args.input, args.read_batch, args.limit),
                               chunksize=args.chunk):
            for c in batch:
                df.update(c.keys())       # df = 词出现在多少**块**里 —— 是 +1，不是加 tf
                doc_len.append(sum(c.values()))
            n_doc += len(batch)
            if n_doc % 200_000 < len(batch):
                el = time.time() - t0
                eta = el / max(n_doc, 1) * (args.total - n_doc)
                print(f"      分词 {n_doc:>9,}/{args.total:,}  {n_doc / args.total:5.1%}"
                      f"   已用 {el / 60:5.1f} 分  剩余约 {eta / 60:5.1f} 分")

    dt = time.time() - t0
    print(f"第一趟完成：{n_doc:,} 块，{len(df):,} 个不同词，{dt / 60:.1f} 分钟")

    dl = np.asarray(doc_len, dtype=np.int32)
    assert len(dl) == n_doc, "doc_len 长度与块数不一致"
    np.save(path_dl, dl)
    print(f"doc_len 落盘：{path_dl}（{dl.nbytes / 1e6:.0f} MB，均值 {dl.mean():.1f} 词/块）")

    # df 落盘：JSON 装 386 万个词会很大，改用「词\t计数」纯文本，流式读回
    with open(path_df, "w", encoding="utf-8") as f:
        for w, c in df.items():
            f.write(f"{w}\t{c}\n")
    print(f"df 落盘：{path_df}（{path_df.stat().st_size / 1e6:.0f} MB）")
    return df, dl


def build_vocab(df: Counter, out_dir: Path, min_df: int):
    """按字典序排好词表 —— 排序后查询时可以用二分查找，省掉 386 万项的 dict 常驻。"""
    print()
    print("-" * 76)
    print(f"构建词表（min_df={min_df}）")
    print("-" * 76)
    t = time.time()
    terms = sorted(w for w, c in df.items() if c >= min_df)
    vocab_path = out_dir / "vocab.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write("\n".join(terms))
    print(f"词表规模：{len(terms):,}（过滤掉 {len(df) - len(terms):,} 个 df<{min_df} 的词，"
          f"{time.time() - t:.1f} 秒）")
    print(f"vocab.txt：{vocab_path.stat().st_size / 1e6:.0f} MB")
    return terms


def pass2(args, out_dir: Path, terms, dl: np.ndarray, path_df: Path):
    """第二趟：分词 → 按 indptr 直接写入 postings（**无需排序**）。"""
    print()
    print("-" * 76)
    print(f"第二趟 · 写 postings  进程数 = {args.procs}")
    print("-" * 76)
    t0 = time.time()

    # df 从文件流式读回（比从内存 Counter 取省事，也支持断点）
    df_arr = np.zeros(len(terms), dtype=np.int64)
    t2i = {w: i for i, w in enumerate(terms)}
    with open(path_df, encoding="utf-8") as f:
        for line in f:
            w, _, c = line.rstrip("\n").partition("\t")
            i = t2i.get(w)
            if i is not None:
                df_arr[i] = int(c)
    print(f"df 表载入：{np.count_nonzero(df_arr):,} 个词有值")

    # indptr = df 的前缀和 —— 这一步决定了每条 posting 写到哪
    indptr = np.zeros(len(terms) + 1, dtype=np.int64)
    np.cumsum(df_arr, out=indptr[1:])
    P = int(indptr[-1])
    print(f"postings 总数：{P:,}（{P / 1e8:.2f} 亿）")

    doc_ids = np.empty(P, dtype=np.int32)
    tf_norm = np.empty(P, dtype=np.float16)
    cursor = indptr[:-1].copy()          # 每个词下一个待写位置

    n_doc, avgdl = len(dl), float(dl.mean())
    k1, b = args.k1, args.b
    n_total = len(dl)
    doc_idx = 0

    with Pool(args.procs, initializer=_init_worker) as pool:
        for batch in pool.imap(_tokenize_batch,
                               iter_texts(args.input, args.read_batch, args.limit),
                               chunksize=args.chunk):
            for c in batch:
                dl_d = sum(c.values())
                norm = k1 * (1.0 - b + b * dl_d / avgdl)
                for w, tf in c.items():
                    ti = t2i.get(w)
                    if ti is None:                       # 被 min_df 过滤掉的词
                        continue
                    pos = cursor[ti]
                    # 越界说明 df 统计和这一趟不一致 —— 宁可炸也不能静默写歪
                    assert pos < indptr[ti + 1], f"postings 越界：term={w} df 统计不一致"
                    doc_ids[pos] = doc_idx
                    tf_norm[pos] = (tf * (k1 + 1.0)) / (tf + norm)
                    cursor[ti] = pos + 1
                doc_idx += 1
            if doc_idx % 200_000 < len(batch):
                el = time.time() - t0
                eta = el / max(doc_idx, 1) * (n_total - doc_idx)
                print(f"      写入 {doc_idx:>9,}/{n_total:,}  {doc_idx / n_total:5.1%}"
                      f"   已用 {el / 60:5.1f} 分  剩余约 {eta / 60:5.1f} 分")

    dt = time.time() - t0
    print(f"第二趟完成：{doc_idx:,} 块，{dt / 60:.1f} 分钟")

    # ---- 一致性自检（第 5 / 7 步的教训：检测口径必须和写入口径对齐）----
    written = cursor - indptr[:-1]
    bad = np.nonzero(written != df_arr)[0]
    if len(bad):
        print(f"🔴 有 {len(bad):,} 个词的写入数与 df 不符，前 5 个：{bad[:5].tolist()}")
        raise SystemExit("postings 写入不完整 —— 不要使用这份索引")
    print(f"✅ 自检通过：每个词的 postings 条数 == df（{P:,} 条）")

    np.save(out_dir / "indptr.npy", indptr)
    np.save(out_dir / "doc_ids.npy", doc_ids)
    np.save(out_dir / "tf_norm.npy", tf_norm)
    print(f"落盘：indptr {indptr.nbytes / 1e6:.0f} MB / doc_ids {doc_ids.nbytes / 1e6:.0f} MB "
          f"/ tf_norm {tf_norm.nbytes / 1e6:.0f} MB")
    return indptr, df_arr, P


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(PROJECT / "data" / "processed" / "chunks.parquet"))
    ap.add_argument("--out", default=str(PROJECT / "data" / "index" / "bm25"))
    ap.add_argument("--total", type=int, default=3_316_395)
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 块（验通用）")
    ap.add_argument("--procs", type=int, default=min(16, (os.cpu_count() or 8)))
    ap.add_argument("--read-batch", type=int, default=2_000,
                    help="每次从 parquet 取多少块交给一个进程（太小则 IPC 次数多，太大则负载不均）")
    ap.add_argument("--chunk", type=int, default=1,
                    help="imap 的 chunksize。每个任务已经是一整批文本，所以设为 1")
    ap.add_argument("--min-df", type=int, default=2,
                    help="词表下限。df=1 的词贡献 14% 的 postings 却几乎不会被查到；"
                         "但裁掉会丢失生僻专名的精确匹配，所以默认只砍 df=1")
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--reuse-df", action="store_true", help="复用已落盘的 df，跳过第一趟")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.limit:
        args.total = args.limit

    print("=" * 76)
    print("第 8 步 · 建 BM25 倒排索引")
    print("=" * 76)
    print(f"输入     : {args.input}")
    print(f"输出     : {out_dir}")
    print(f"块数     : {args.total:,}{'（--limit 限制）' if args.limit else ''}")
    print(f"进程数   : {args.procs}（CPU {os.cpu_count()} 核）")
    print(f"BM25 参数: k1={args.k1}  b={args.b}   min_df={args.min_df}")
    print(f"繁简归一 : {TARGET if HAVE_ZHCONV else '关闭（未装 zhconv —— 繁体查询词将无法命中）'}")

    path_df = out_dir / "_df.tsv"
    path_dl = out_dir / "doc_len.npy"
    t_all = time.time()

    if args.reuse_df and path_df.exists() and path_dl.exists():
        print("\n[--reuse-df] 跳过第一趟，复用已落盘的 df")
        df = Counter()
        with open(path_df, encoding="utf-8") as f:
            for line in f:
                w, _, c = line.rstrip("\n").partition("\t")
                df[w] = int(c)
        dl = np.load(path_dl)
    else:
        df, dl = pass1(args, path_df, path_dl)

    terms = build_vocab(df, out_dir, args.min_df)
    indptr, df_arr, P = pass2(args, out_dir, terms, dl, path_df)

    meta = {
        "type": "bm25_csr",
        "vocab_size": len(terms),
        "postings": P,
        "docs": int(len(dl)),
        "avg_doc_len": round(float(dl.mean()), 2),
        "k1": args.k1,
        "b": args.b,
        "min_df": args.min_df,
        # 归一化状态必须落盘（见 zh.py 的铁律 2）：
        # 不写的话，「新代码 + 旧索引」也能正常跑起来，
        # 表现只是"某些词永远搜不到"，全程没有一行报错。
        "zh_norm": TARGET if HAVE_ZHCONV else None,
        "total_size_mb": round(sum((out_dir / f).stat().st_size
                                   for f in ["vocab.txt", "indptr.npy", "doc_ids.npy",
                                             "tf_norm.npy", "doc_len.npy"]) / 1e6, 1),
        "elapsed_min": round((time.time() - t_all) / 60, 1),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    path_df.unlink(missing_ok=True)          # 中间产物，不留

    print()
    print("=" * 76)
    print("BUILD_BM25_OK")
    print(f"  词表      : {len(terms):,}")
    print(f"  postings  : {P:,}（{P / 1e8:.2f} 亿）")
    print(f"  索引体积  : {meta['total_size_mb']:.0f} MB")
    print(f"  平均块长  : {meta['avg_doc_len']} 词")
    print(f"  总耗时    : {meta['elapsed_min']} 分钟")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
