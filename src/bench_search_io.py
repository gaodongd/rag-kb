#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 8 步诊断 · 向量检索的延迟瓶颈在哪？

背景
----
T5 实测全量检索 **8~16 秒**，超了 2 秒阈值 8 倍。可能的瓶颈有三个，
它们对应的解法完全不同，所以**必须先测出来是哪个，再动手优化**：

  A. 磁盘 IO   —— mmap 读 6.79GB。解法：换成 faiss IVF（只读一部分数据）
  B. dtype 转换 —— fp16 → fp32，3.4e9 次转换。解法：预转 fp32 落盘 / 用 int8 量化
  C. 矩阵乘法  —— 6.8 GFLOP。解法：BLAS 已经很快，基本不可能是这里

测法
----
1. **冷扫 vs 热扫**：连扫 3 遍同一条数据。第 1 遍要读盘，第 2/3 遍数据已在
   page cache → 后面明显变快 = IO 是瓶颈。若三遍一样慢 = 瓶颈在计算。
2. **纯计算基准**：拿已经躺在内存里的 fp32 数组做同样的内积，得到"没有 IO 干扰"的下限。
3. **顺序读速度**：单独读 1GB 测盘速，给出 MB/s。

用法
----
    "...python313\\python.exe" src\\bench_search_io.py
"""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
EMB = PROJECT / "data" / "index" / "bge-large-zh-v1.5" / "emb.npy"
BLOCK = 200_000


def main() -> int:
    if not EMB.exists():
        raise SystemExit(f"[错误] 找不到 {EMB}")
    emb = np.load(EMB, mmap_mode="r")
    n, d = emb.shape
    gb = emb.nbytes / 1e9
    print("=" * 72)
    print("向量检索延迟诊断")
    print("=" * 72)
    print(f"emb.npy   : {n:,} × {d}  {emb.dtype}  = {gb:.2f} GB（磁盘上的原始体积）")

    rng = np.random.RandomState(0)
    q = rng.randn(d).astype(np.float32)
    q /= np.linalg.norm(q)

    # ---------- 1. 冷 / 热扫对照 ----------
    print()
    print("-" * 72)
    print("1. mmap 分块扫（每块先 fp16→fp32，再内积）—— 连扫 3 遍")
    print("-" * 72)
    times = []
    for i in range(1, 4):
        t = time.time()
        for s in range(0, n, BLOCK):
            blk = np.asarray(emb[s:s + BLOCK], dtype=np.float32)
            _ = blk @ q
            del blk
        dt = time.time() - t
        times.append(dt)
        note = "（含磁盘 IO）" if i == 1 else "（page cache 已部分命中）"
        print(f"   第 {i} 遍: {dt:6.2f} 秒   吞吐 {gb / dt * 1000:6.0f} MB/s  {note}")
    print(f"   → 第1遍 vs 第3遍 差 {times[0] - times[2]:.2f} 秒 "
          f"= {gb / max(times[0] - times[2], 1e-9) * 1000:.0f} MB/s 的 IO 缺口")

    # ---------- 2. 纯计算下限（数据已在内存） ----------
    print()
    print("-" * 72)
    print("2. 纯计算下限：数据已在内存的 fp32 数组（无 IO、无 dtype 转换）")
    print("-" * 72)
    m = 1_000_000                       # 100 万行 = 4.1 GB fp32，能在内存里放下
    X = np.ascontiguousarray(emb[:m], dtype=np.float32)
    print(f"   样本: {m:,} × {d} fp32 = {X.nbytes / 1e9:.2f} GB（已常驻内存）")
    for i in range(2):
        t = time.time()
        _ = X @ q
        dt = time.time() - t
        print(f"   第 {i + 1} 次: {dt * 1000:7.1f} ms  → {m * d * 2 / dt / 1e9:.1f} GFLOP/s")
        if i == 0:
            first = dt
    dt = first
    scaled = dt * n / m
    print(f"   → 若全量 {n:,} 行都能常驻内存，预计 **{scaled * 1000:.0f} ms**")
    del X

    # ---------- 3. 磁盘顺序读速度 ----------
    print()
    print("-" * 72)
    print("3. 磁盘顺序读（1 GB，绕过 page cache 影响：读文件开头 1GB 的两段）")
    print("-" * 72)
    need = 1_000_000_000
    rows = need // (d * 2)               # fp16 = 2 字节
    t = time.time()
    _ = np.asarray(emb[:rows]).sum()
    dt = time.time() - t
    gb_read = rows * d * 2 / 1e9
    print(f"   读 {gb_read:.2f} GB 用时 {dt:.2f} 秒 → **{gb_read / dt * 1000:.0f} MB/s**")

    # ---------- 结论 ----------
    print()
    print("=" * 72)
    print("结论判据")
    print("=" * 72)
    io_gap = times[0] - times[2]
    print(f"  冷热差 = {io_gap:.2f} 秒（占冷扫 {io_gap / times[0] * 100:.0f}%）")
    if io_gap > times[0] * 0.3:
        print("  → 瓶颈以 **磁盘 IO** 为主。上 faiss IVF（只扫一部分倒排桶）收益最大。")
    else:
        print("  → 瓶颈不在 IO，重点看 dtype 转换 / 矩阵乘。考虑落盘 fp32 或 int8 量化。")
    print(f"  内存内全量理论值 ≈ {scaled * 1000:.0f} ms。常驻内存若可行，也是解。")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
