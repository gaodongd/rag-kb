#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 8 步 · T6 BM25 关键词检索（全量 331.6 万块）

BM25 和向量检索是**互补**的，不是替代关系
------------------------------------------
向量检索擅长「语义」：问"这个地方气候怎么样"（没有任何地名）也能命中气候小节。
BM25 擅长「字面」：专有名词、代码符号、生僻地名 —— 这些词表意太窄，
向量模型没见过就映射不准（第 8 步 T5 就抓到一例：问"臺灣東部開發於古時的人行道路"
命中了「東寧路 (臺南市)」这个语义无关的条目）。

两者融合（见 `search_hybrid.py`）才是正解，这一步先把 BM25 单独跑对。

打分公式
--------
    score(q, d) = Σ_{t ∈ q} IDF(t) × tf_norm[t][d] × qtf(t)

其中
    IDF(t)      = ln(1 + (N - df + 0.5) / (df + 0.5))        ← 查询时才算（只 3 百万次）
    tf_norm[t][d] = tf×(k1+1) / (tf + k1×(1-b+b×dl/avgdl))   ← **建索引时已算好**（1.92 亿次）

关键优化：BM25 里唯一与查询相关的部分是 IDF，而分母（文档长度归一化）与查询无关。
把它离线预计算 → 查询时只剩「查表 + 乘 IDF + 累加」，这就是 tf_norm.npy 存在的理由。

为什么词表按字典序 + 二分查找
-----------------------------
`build_bm25.py` 把词表 sorted 后落盘，所以查询时用 `np.searchsorted` 就够，
**不需要常驻 309 万项的 Python dict**（那个 dict 光对象头就吃 400MB+）。

用法
----
    :: 默认三个问题（与 T5 向量检索用同一组，便于对照）
    "...python313\\python.exe" src\\search_bm25.py

    :: 自定义
    "...python313\\python.exe" src\\search_bm25.py --query "苏花古道" --topk 5

    :: 只看命中详情（不打印逐词 df）
    "...python313\\python.exe" src\\search_bm25.py --quiet
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import jieba
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# ⚠️ 分词口径必须与建索引时**完全同一份代码**。
#    抄一份 _keep 过来是隐患：哪天建索引那边改了规则，查询这边不跟，
#    表现是「某些词永远搜不到」—— 不报错、不崩溃，纯粹静默失灵。
from build_bm25 import STOP, _keep  # noqa: E402
from zh import HAVE_ZHCONV, TARGET, norm_text  # noqa: E402
from search_vector import DEFAULT_QUERIES, fetch_texts  # noqa: E402

PROJECT = HERE.parent
DEFAULT_INDEX = PROJECT / "data" / "index" / "bm25"
DEFAULT_PARQUET = PROJECT / "data" / "processed" / "chunks.parquet"
# 行号 → chunk_id 的映射是**全局契约**（parquet 行序 = 第 7 步 emb.npy 行序 = BM25 postings 行号），
# 所以它属于第 7 步的产物，不在 bm25/ 目录里。这里只是借用。
DEFAULT_IDS = PROJECT / "data" / "index" / "bge-large-zh-v1.5" / "ids.txt"


# ---------------------------------------------------------------- 加载

def load_bm25(index_dir: Path, allow_norm_mismatch: bool = False):
    """
    读倒排索引。全部用 mmap —— 1.2GB 的索引不必真的进内存。

    allow_norm_mismatch：只给**对照实验**用（量繁简归一化值多少）。
      置 True 时会放行"旧索引（未归一化）+ 新代码（查询会归一化）"这种组合 ——
      但那正是要避免的静默失配，所以**必须同时**给 bm25_search(normalize=False)，
      由调用方负责成对使用。默认 False，正常路径不受影响。
    """
    need = ["meta.json", "vocab.txt", "indptr.npy", "doc_ids.npy", "tf_norm.npy"]
    miss = [f for f in need if not (index_dir / f).exists()]
    if miss:
        raise SystemExit(f"[错误] 索引不完整：{index_dir}\n"
                         f"        缺少 {miss}\n"
                         f"        先跑 src\\build_bm25.py 全量建索引")

    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))

    # ---- 归一化状态校验（必须挡在检索之前）----
    # 索引建在"已归一化"的文本上，而查询端如果没装 zhconv（或反过来装了但索引没转），
    # 繁体词就永远查不到 —— 全程没有一行报错，只是"某些词搜不出来"。
    # 这类静默失配只能靠落盘的版本标记挡住（见 zh.py 铁律 2）。
    idx_norm = meta.get("zh_norm")
    cur_norm = TARGET if HAVE_ZHCONV else None
    if idx_norm != cur_norm:
        if not allow_norm_mismatch:
            raise SystemExit(
                f"[错误] 索引的繁简归一化状态与当前环境不一致：\n"
                f"        索引 meta.zh_norm = {idx_norm!r}\n"
                f"        当前环境         = {cur_norm!r}\n"
                f"        两者不一致时检索会静默漏词，所以这里直接拒绝启动。\n"
                f"        处理：确认装好 zhconv 后重建索引 → python src\\build_bm25.py")
        print(f"[警告] 归一化状态不一致（索引 {idx_norm!r} vs 环境 {cur_norm!r}）"
              f"—— 已放行，仅限对照实验。查询侧务必同时关掉 norm_text。")

    terms = (index_dir / "vocab.txt").read_text(encoding="utf-8").split("\n")
    if terms and terms[-1] == "":
        terms.pop()                                  # 末尾换行切出来的空串
    vocab = np.array(terms)

    if len(vocab) != meta["vocab_size"]:
        raise SystemExit(f"[错误] vocab.txt 有 {len(vocab):,} 行，meta 说是 {meta['vocab_size']:,} —— 索引已损坏")

    indptr = np.load(index_dir / "indptr.npy", mmap_mode="r")
    doc_ids = np.load(index_dir / "doc_ids.npy", mmap_mode="r")
    tf_norm = np.load(index_dir / "tf_norm.npy", mmap_mode="r")

    # 一致性契约：postings 数组长度必须等于 indptr 的末位
    P = int(indptr[-1])
    if len(doc_ids) < P or len(tf_norm) < P:
        raise SystemExit(f"[错误] indptr 声称 {P:,} 条 postings，但 "
                         f"doc_ids={len(doc_ids):,} / tf_norm={len(tf_norm):,} —— 索引已损坏")
    return meta, vocab, indptr, doc_ids, tf_norm


def build_idf(indptr, n_docs: int) -> np.ndarray:
    """df = indptr 差分；IDF = ln(1 + (N-df+0.5)/(df+0.5))。只算一次，全查询复用。"""
    df = np.diff(np.asarray(indptr))                # int64，3.09M
    return np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)


# ---------------------------------------------------------------- 查询

def tokenize_query(q: str, normalize: bool = True):
    """
    与建索引**同一口径**分词 → {词: 查询内词频}。

    归一化在这里做，而且必须在 jieba.lcut 之前 —— 因为索引端
    （`build_bm25._tokenize_batch`）就是这个顺序。两边顺序一致才叫"同一口径"。

    例：「張克忠」→（归一化）「张克忠」→ jieba 切成 ['张克忠'] → 命中索引。
        不归一化的话，jieba 会把繁体串切碎成单字，被 `_keep` 全部过滤掉，
        整个 query 在进入检索之前就什么都不剩了（这正是 2026-09-19 抓到的那个 bug）。

    normalize=False：只配 `load_bm25(allow_norm_mismatch=True)` + 旧索引用，
      模拟"修复前"的行为，用来量化归一化的收益。**正常检索永远不要传 False。**
    """
    return dict(Counter(w for w in jieba.lcut(norm_text(q) if normalize else (q or ""))
                        if _keep(w)))


def lookup_term(vocab: np.ndarray, w: str) -> int:
    """词表二分查找（字典序已排好）。未收录返回 -1。"""
    i = int(np.searchsorted(vocab, w))
    if i < len(vocab) and vocab[i] == w:
        return i
    return -1


def bm25_search(q: str, vocab, indptr, doc_ids, tf_norm, idf, n_docs: int,
                topk: int = 5, verbose: bool = False, normalize: bool = True):
    """
    返回 (rows, scores, detail)。rows 是块行号（与 ids.txt 第 i 行对齐），已按分数降序。

    实现要点：**一次 bincount 累加完所有词**。
    逐个词 `scores[docs] += vals` 看起来直观，但 numpy 的散射赋值遇到重复索引只保留最后一个
    （要正确必须用慢得多的 `np.add.at`）。bincount 是「按索引计数累加」的 C 实现，
    天然正确且快 —— 代价是一次 N 长的 float64 数组（26MB）。
    """
    jieba.initialize()                              # 单进程检索，冷启动约 1-2 秒，只做一次

    qtf = tokenize_query(q, normalize=normalize)
    docs_list, vals_list, detail = [], [], []

    for w, tf in qtf.items():
        ti = lookup_term(vocab, w)
        if ti < 0:
            detail.append((w, tf, 0, 0.0, "未收录"))
            continue
        s, e = int(indptr[ti]), int(indptr[ti + 1])
        n = e - s
        if n == 0:
            detail.append((w, tf, 0, 0.0, "df=0"))
            continue
        d = np.asarray(doc_ids[s:e]).astype(np.intp)
        v = np.asarray(tf_norm[s:e]).astype(np.float32) * idf[ti] * tf
        docs_list.append(d)
        vals_list.append(v)
        detail.append((w, tf, n, float(idf[ti]), ""))

    if not docs_list:
        return np.empty(0, np.int64), np.empty(0, np.float32), detail

    all_docs = np.concatenate(docs_list)
    all_vals = np.concatenate(vals_list)
    scores = np.bincount(all_docs, weights=all_vals, minlength=n_docs).astype(np.float32)
    del all_docs, all_vals

    nz = int(np.count_nonzero(scores))
    if nz == 0:
        return np.empty(0, np.int64), np.empty(0, np.float32), detail

    k = min(topk, nz)
    part = np.argpartition(-scores, k - 1)[:k]
    order = np.argsort(-scores[part])
    rows = part[order].astype(np.int64)
    if verbose:
        return rows, scores[rows], detail, nz
    return rows, scores[rows], detail


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(DEFAULT_INDEX))
    ap.add_argument("--input", default=str(DEFAULT_PARQUET))
    ap.add_argument("--ids", default=str(DEFAULT_IDS),
                    help="行号 → chunk_id 映射（第 7 步产物，全局契约）")
    ap.add_argument("--query", action="append", default=None)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--show", type=int, default=5)
    ap.add_argument("--quiet", action="store_true", help="不打印逐词 df 明细")
    args = ap.parse_args()
    queries = args.query or DEFAULT_QUERIES

    print("=" * 76)
    print("第 8 步 · BM25 关键词检索（全量倒排）")
    print("=" * 76)

    t = time.time()
    meta, vocab, indptr, doc_ids, tf_norm = load_bm25(Path(args.index))
    n_docs = int(meta["docs"])
    print(f"索引      : {args.index}")
    print(f"词表      : {meta['vocab_size']:,} 词（字典序 + 二分查找，无常驻 dict）")
    print(f"postings  : {meta['postings']:,}（{meta['postings'] / 1e8:.2f} 亿）")
    print(f"块数      : {n_docs:,}  平均块长 {meta['avg_doc_len']} 词")
    print(f"参数      : k1={meta['k1']}  b={meta['b']}  min_df={meta['min_df']}")
    print(f"加载耗时  : {time.time() - t:.1f} 秒（全部 memmap）")

    t = time.time()
    idf = build_idf(indptr, n_docs)
    print(f"IDF 预算  : {time.time() - t:.2f} 秒（{len(idf):,} 项，df 来自 indptr 差分）")

    # ---- 逐条检索 ----
    hit_ids, results, latencies = [], [], []
    print()
    print("=" * 76)
    print("检索延迟实测")
    print("=" * 76)
    for q in queries:
        t = time.time()
        rows, sc, detail, nz = bm25_search(q, vocab, indptr, doc_ids, tf_norm, idf,
                                           n_docs, args.topk, verbose=True)
        dt = time.time() - t
        latencies.append(dt)
        n_hit = len(rows)
        print(f"  {dt * 1000:7.1f} ms   {q}   → 命中 {nz:,} 块，取 top{n_hit}")

        if not args.quiet:
            terms = [(w, tf, n, i) for w, tf, n, i, err in detail if not err]
            miss = [w for w, tf, n, i, err in detail if err]
            terms.sort(key=lambda x: -x[2])
            for w, tf, n, i in terms[:8]:
                print(f"            {w:<12} qtf={tf}  df={n:>9,}  idf={i:.3f}")
            if miss:
                print(f"            未收录/无 postings：{', '.join(miss)}")

        results.append((q, sc, rows))
        hit_ids.extend(int(x) for x in rows)

    # ---- 回查原文 ----
    t = time.time()
    ids_path = Path(args.ids)
    id_list = None
    if ids_path.exists():
        id_list = ids_path.read_text(encoding="utf-8").split()   # 只读一次，下面复用
        print(f"\n行号表    : {len(id_list):,} 行（{ids_path.name}），{time.time() - t:.1f} 秒")
    else:
        print(f"\n[警告] 缺少 {ids_path} —— 行号无法映射到 chunk_id，只能显示行号")

    if id_list is None:
        got = {}
    else:
        t = time.time()
        cids = [id_list[int(r)] for r in hit_ids]
        got = fetch_texts(Path(args.input), cids)
        print(f"回查原文  : {len(got):,} / {len(set(cids)):,} 条命中，{time.time() - t:.1f} 秒")

    # ---- 打印 ----
    for q, sc, rows in results:
        print()
        print("=" * 76)
        print(f"Q: {q}")
        print("=" * 76)
        for rank, (score, j) in enumerate(zip(sc, rows), 1):
            if rank > args.show:
                break
            cid = id_list[int(j)] if id_list else str(int(j))
            r = got.get(cid) if got else None
            if r is None:
                print(f"  {rank}. {score:.4f}  row={j}  chunk_id={cid}  [原文缺失]")
                continue
            sec = r["section"] or "—"
            text = (r["chunk_text"] or "").replace("\n", " ")[:76]
            print(f"  {rank}. {score:.4f}  {r['title']}  ·  {sec}")
            print(f"       {text}")
            print(f"       chunk_id={cid}  row={j}")

    print()
    print("=" * 76)
    print("SEARCH_BM25_OK")
    print(f"  索引条数 : {n_docs:,}")
    print(f"  查询数   : {len(queries)}")
    print(f"  延迟     : 平均 {float(np.mean(latencies)) * 1000:.1f} ms / "
          f"最大 {float(np.max(latencies)) * 1000:.1f} ms")
    print("=" * 76)
    print("自检提示：BM25 分数量纲与余弦不同（这里可 >1），跨检索器比较必须用排名，")
    print("         这就是混合检索用 RRF（排名融合）而不是分数加权的直接原因。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
