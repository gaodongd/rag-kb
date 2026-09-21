#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 8 步 · T5.5 用 faiss 建 IVF 索引（把 8 秒压到毫秒）

为什么需要这一步
----------------
T5 暴力检索**语义全对**，但延迟 **8.2 秒**（阈值 2 秒）。诊断脚本 `bench_search_io.py` 给出真因：

| 环节 | 耗时 | 说明 |
|---|---|---|
| 纯内存 fp32 内积（全量 331.6 万） | **263 ms** | 数据在内存、dtype 已对 |
| 实测 mmap + fp16→fp32 分块扫 | **8.2 秒** | **31 倍差距** |
| 冷扫 vs 热扫 | 差 5% | 不是 page cache 的事 |
| 磁盘顺序读 | 587 MB/s | 也不是盘太慢 |

→ 瓶颈是 **「每次查询都要把 6.79GB 读出来、再逐元素 fp16→fp32」**：
  3.4e9 次 dtype 转换，numpy 只有约 4e8 元素/秒，这一项就吃掉 8 秒。

两条解法
--------
1. 全量常驻内存 fp32（13.6GB）→ 263 ms，但服务启动要读 23 秒盘、常驻 13.6GB 内存
2. **faiss IVF** ← 本脚本：
   - 先用 k-means 把 331.6 万个向量聚成 `nlist` 个桶（倒排）
   - 查询时只在 `nprobe` 个最近的桶里扫 → 扫描量降到 `nprobe/nlist`
   - faiss 内部用 SIMD，且索引可落盘复用（不用每次读原始 emb.npy）

代价是**近似的**（可能漏掉落在没扫的桶里的真答案）—— 所以脚本最后会用
**暴力检索的结果当 ground truth**，实测 IVF 的 recall@k，把损失量化出来。
"用什么参数换多少召回"是检索岗的日常，不能拍脑袋。

用法
----
    :: 建索引（20~40 分钟，内存峰值约 4GB）
    "...python313\\python.exe" src\\build_faiss_index.py

    :: 只测已有索引的延迟 / 召回
    "...python313\\python.exe" src\\build_faiss_index.py --skip-build --nprobe 32
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ⚠️ 重定向到文件时 Python 默认用**块缓冲**（8KB），日志会长时间空白，看不出跑到哪。
# 实测踩过：跑了 2 分钟日志仍是 0 字节，无法判断进度也不能诊断。
# `line_buffering=True` 强制逐行刷 —— 长任务脚本必须加这一段。
try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vectorize import find_model  # noqa: E402

PROJECT = HERE.parent
DEFAULT_SRC = PROJECT / "data" / "index" / "bge-large-zh-v1.5"
DEFAULT_OUT = PROJECT / "data" / "index" / "faiss_ivf_sq8.index"

PROBES = [1, 4, 16, 32, 64, 128, 512, 1024, 4096]

EVAL_QUERIES = [
    "台灣東部開發於古時的人行道路",
    "中國的化學工程學家",
    "這個地方氣候怎麼樣",
    "苏花公路的历史",
    "南开大学化工系的创始人",
    "上海的行政区划",
    "量子力学的奠基人",
    "长江有多长",
    "红楼梦的作者是谁",
    "乒乓球世界冠军",
]


def load_vectors(src: Path):
    ids = (src / "ids.txt").read_text(encoding="utf-8").split()
    emb = np.load(src / "emb.npy", mmap_mode="r")
    if emb.shape[0] != len(ids):
        raise SystemExit("[错误] emb/ids 行数不一致，先跑 check_vectors.py")
    return ids, emb


def build(args, emb):
    """训练 + 灌入 + 落盘。分批转 fp32 以免一次性吃 13.6GB 内存。"""
    import faiss

    n, d = emb.shape
    print()
    print("-" * 74)
    print(f"建索引：nlist={args.nlist}  quantizer={args.quantizer}  d={d}  N={n:,}")
    print("-" * 74)

    # ---- 1. 训练样本：随机采，别用前 N 个（维基 dump 是按标题排序的，前 N 个全是 A 开头的条目）----
    t = time.time()
    rng = np.random.RandomState(42)
    idx_train = np.sort(rng.choice(n, size=min(args.train_size, n), replace=False))
    Xt = np.empty((len(idx_train), d), dtype=np.float32)
    for i, j in enumerate(idx_train):
        Xt[i] = emb[j]
    print(f"[1/4] 训练样本 {len(idx_train):,} × {d} fp32 = {Xt.nbytes / 1e9:.2f} GB"
          f"（随机采样，{time.time() - t:.1f} 秒）")

    # ---- 2. 建结构 + 训练 ----
    t = time.time()
    quantizer = faiss.IndexFlatIP(d) if args.metric == "ip" else faiss.IndexFlatL2(d)
    metric = faiss.METRIC_INNER_PRODUCT if args.metric == "ip" else faiss.METRIC_L2
    if args.quantizer == "sq8":
        index = faiss.IndexIVFScalarQuantizer(
            quantizer, d, args.nlist,
            faiss.ScalarQuantizer.QT_8bit, metric)
    else:
        index = faiss.IndexIVFFlat(quantizer, d, args.nlist, metric)
    index.train(Xt)
    print(f"[2/4] k-means 训练完成（{time.time() - t:.1f} 秒）")
    del Xt

    # ---- 3. 分批灌入 ----
    t = time.time()
    B = args.add_batch
    for s in range(0, n, B):
        e = min(s + B, n)
        index.add(np.ascontiguousarray(emb[s:e], dtype=np.float32))
        if (s // B) % 4 == 0 or e == n:
            pct = e / n * 100
            el = time.time() - t
            eta = el / max(e, 1) * (n - e)
            print(f"      add {e:>9,}/{n:,}  {pct:5.1f}%   已用 {el / 60:5.1f} 分  剩余约 {eta / 60:5.1f} 分")
    print(f"[3/4] 灌入完成（{time.time() - t:.1f} 秒）")
    print(f"      ntotal = {index.ntotal:,}   内存中的索引体积 ≈ {index.ntotal * d / 1e9:.2f} GB")

    # ---- 4. 落盘 ----
    t = time.time()
    out = Path(args.out)                      # ⚠️ args.out 是 str，必须先转 Path（踩过：AttributeError: 'str' object has no attribute 'parent'）
    out.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out))
    size = out.stat().st_size / 1e9
    print(f"[4/4] 落盘 {out}  = {size:.2f} GB（{time.time() - t:.1f} 秒）")
    return index


def ground_truth(emb, qv_all, k, block=200_000):
    """暴力精确 top-k，作为 recall 的基准。"""
    n = emb.shape[0]
    gt = np.empty((len(qv_all), k), dtype=np.int64)
    for qi, qv in enumerate(qv_all):
        best_s = np.full(k, -np.inf, dtype=np.float32)
        best_i = np.zeros(k, dtype=np.int64)
        for s in range(0, n, block):
            e = min(s + block, n)
            sc = np.asarray(emb[s:e], dtype=np.float32) @ qv
            kk = min(k, e - s)
            part = np.argpartition(-sc, kk - 1)[:kk]
            cs = np.concatenate([best_s, sc[part]])
            ci = np.concatenate([best_i, part.astype(np.int64) + s])
            keep = np.argpartition(-cs, k - 1)[:k]
            best_s, best_i = cs[keep], ci[keep]
        gt[qi] = best_i[np.argsort(-best_s)]
    return gt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--nlist", type=int, default=4096)
    ap.add_argument("--nprobe", type=int, default=32)
    ap.add_argument("--quantizer", default="sq8", choices=["sq8", "flat"],
                    help="sq8 省内存(4x)，flat 无损但占 13.6GB")
    ap.add_argument("--metric", default="ip", choices=["ip", "l2"],
                    help="ip=内积；l2=欧氏。**向量已归一化时两者排序完全等价**"
                         "（‖a-b‖²=2-2a·b），但 L2 的 k-means 质心更新更稳，"
                         "实测 IVF 候选覆盖率明显更高 —— 这是本步一个反直觉的坑")
    ap.add_argument("--train-size", type=int, default=300_000)
    ap.add_argument("--add-batch", type=int, default=200_000)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--eval-k", type=int, default=10)
    ap.add_argument("--model-kind", default="large", choices=["large", "small"])
    args = ap.parse_args()

    import faiss
    print("=" * 74)
    print("第 8 步 · faiss IVF 索引")
    print("=" * 74)
    print(f"faiss 版本 : {faiss.__version__ if hasattr(faiss, '__version__') else '?'}")
    ids, emb = load_vectors(Path(args.src))
    print(f"源向量     : {emb.shape[0]:,} × {emb.shape[1]}  {emb.dtype}")

    out = Path(args.out)
    if args.skip_build:
        if not out.exists():
            raise SystemExit(f"[错误] {out} 不存在，不能 --skip-build")
        t = time.time()
        index = faiss.read_index(str(out))
        print(f"读索引     : {out}  ({out.stat().st_size / 1e9:.2f} GB, {time.time() - t:.1f} 秒)")
    else:
        index = build(args, emb)
    index.nprobe = args.nprobe

    # ---- 编码评测 query ----
    print()
    print("-" * 74)
    print(f"评测：{len(EVAL_QUERIES)} 条 query，k={args.eval_k}")
    print("-" * 74)
    t = time.time()
    from FlagEmbedding import FlagModel
    model = FlagModel(str(find_model(args.model_kind)), use_fp16=True)
    qv_all = np.ascontiguousarray(
        np.asarray(model.encode(EVAL_QUERIES, batch_size=8, max_length=512)).astype(np.float32))
    print(f"编码完成（{time.time() - t:.1f} 秒）")

    # ---- ground truth（暴力）----
    print("\n计算暴力 ground truth（这就是 T5 的慢查询，一次约 8 秒 × 题数）…")
    t = time.time()
    gt = ground_truth(emb, qv_all, args.eval_k)
    print(f"  暴力检索 {len(EVAL_QUERIES)} 题共 {time.time() - t:.1f} 秒"
          f"（平摊 {(time.time() - t) / len(EVAL_QUERIES):.2f} 秒/题）")

    # ---- 单题暴力延迟（对照）----
    tv = []
    for qv in qv_all[:3]:
        t = time.time()
        _ = ground_truth(emb, qv[None, :], args.eval_k)
        tv.append(time.time() - t)
    brute_ms = float(np.mean(tv)) * 1000

    # ---- 扫 nprobe，看 延迟 / 召回 的权衡 ----
    print()
    print("=" * 74)
    print("nprobe 权衡表（核心结论，写进 README）")
    print("=" * 74)
    print(f"{'nprobe':>7} | {'延迟(ms)':>9} | {'recall@' + str(args.eval_k):>10} | {'扫描比例':>9}")
    print("-" * 74)
    rows = []
    for np_ in PROBES:
        if np_ > args.nlist:
            continue
        index.nprobe = np_
        lat = []
        hit = 0
        for qi, qv in enumerate(qv_all):
            t = time.time()
            _, I = index.search(qv[None, :], args.eval_k)
            lat.append(time.time() - t)
            hit += len(set(I[0].tolist()) & set(gt[qi].tolist()))
        rec = hit / (len(qv_all) * args.eval_k)
        ms = float(np.mean(lat)) * 1000
        ratio = np_ / args.nlist
        rows.append({"nprobe": np_, "latency_ms": round(ms, 2), "recall": round(rec, 4)})
        print(f"{np_:>7} | {ms:>9.2f} | {rec:>10.3f} | {ratio:>8.1%}")
    print("-" * 74)
    print(f"{'暴力':>7} | {brute_ms:>9.2f} | {1.0:>10.3f} | {'100%':>9}   ← ground truth 本身")

    best = next((r for r in rows if r["recall"] >= 0.95), rows[-1] if rows else None)
    if best:
        print()
        print(f"→ recall ≥ 95% 的最小 nprobe = **{best['nprobe']}**，"
              f"延迟 {best['latency_ms']} ms，比暴力快 **{brute_ms / max(best['latency_ms'], 1e-9):.0f} 倍**")

        # 用该 nprobe 打印 top-5 看语义
        index.nprobe = best["nprobe"]
        _, I = index.search(qv_all[0][None, :], 5)
        print(f"\n抽检 Q1「{EVAL_QUERIES[0]}」top-5 行号：{I[0].tolist()}")

    meta = {
        "index": str(out),
        "nlist": args.nlist,
        "metric": args.metric,
        "quantizer": args.quantizer,
        "ntotal": int(index.ntotal),
        "dim": int(emb.shape[1]),
        "index_size_gb": round(out.stat().st_size / 1e9, 3),
        "brute_force_ms": round(brute_ms, 2),
        "probes": rows,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    mp = out.with_suffix(".meta.json")
    mp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n元数据 → {mp}")
    print("=" * 74)
    print("FAISS_BUILD_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
