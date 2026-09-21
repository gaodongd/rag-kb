"""向量索引体检 —— 续跑后 / 进检索前都该跑一次。

为什么单独写一个
----------------
`emb.npy` 的形状对、范数对，**不能证明 ids.txt 与它对齐**。
续跑逻辑一旦写错（见 操作手册 §7.5），表现是「静默错位」：
两件产物各自都"正常"，只有放在一起才错。

本脚本专门查**跨文件的三个不变量**：
  1. ids.txt 行数 == emb.npy 行数（错位的最直接证据）
  2. ids.txt 里的 chunk_id 无重复（重复 = 续跑把同一段又 append 了一遍）
  3. emb.npy 抽样范数 ≈ 1.0、无 NaN（真的写过，不是预分配的空壳）

用法（**必须 Python 3.13**，3.11 没装 numpy 之外的这些）
------
    "...Python313\\python.exe" src\\check_vectors.py
    "...Python313\\python.exe" src\\check_vectors.py --index data\\index\\bge-large-zh-v1.5

退出码 0 = 三项全过。
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(PROJECT / "data" / "index" / "bge-large-zh-v1.5"))
    ap.add_argument("--sample", type=int, default=2000, help="每段抽多少行查范数")
    args = ap.parse_args()

    idx = Path(args.index)
    ids_path = idx / "ids.txt"
    emb_path = idx / "emb.npy"
    print("=" * 66)
    print("向量索引体检")
    print("=" * 66)
    print(f"目录    : {idx}")

    bad = 0

    # ---- 1. emb.npy ----
    # open_memmap('r') 而不是 np.load(mmap_mode='r')：前者不会把整个文件读进内存
    emb = np.lib.format.open_memmap(emb_path, mode="r")
    n_emb, dim = emb.shape
    print(f"emb.npy : {n_emb:,} × {dim}  dtype={emb.dtype}")
    if emb.dtype != np.float16:
        print("  ⚠️ dtype 不是 float16")

    # ---- 2. ids.txt 行数 ----
    raw = ids_path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        print("  🔴 ids.txt 末行不完整（断电截断）")
        bad += 1
    lines = [x for x in raw.decode("utf-8").split("\n") if x]
    n_ids = len(lines)
    print(f"ids.txt : {n_ids:,} 行")
    # ⚠️ 口径要和事实对称：emb.npy 是**预分配**到 N 行的（open_memmap w+），
    #    未跑完时尾部就是一片零。所以「ids < emb 行数」是**正常的未完成状态**，
    #    不是错位。真正该报警的是 **ids > emb 行数** ——
    #    那意味着 ids 写到了 emb 还没写的地方，才叫错位。
    if n_ids > n_emb:
        print(f"  🔴 ids 比 emb 多 {n_ids - n_emb:,} 行 → 真错位（见手册 §7.5）")
        bad += 1
    elif n_ids < n_emb:
        print(f"  ✅ ids ≤ emb（未跑完，尾部 {n_emb - n_ids:,} 行仍是预分配空洞）")
    else:
        print("  ✅ 行数一致（已跑完）")

    # ---- 3. id 唯一性（查"续跑重复 append"） ----
    # 2.5M 条字符串，Counter 内存约 300MB，可以接受
    cnt = Counter(lines)
    dups = [(k, v) for k, v in cnt.items() if v > 1]
    if dups:
        print(f"  🔴 有 {len(dups)} 个重复 chunk_id（前 3 个示例）：")
        for k, v in dups[:3]:
            print(f"       {k}  ×{v}")
        print("     典型成因：续跑起点读错 → 同一段被 append 两遍（见手册 §7.5）")
        bad += 1
    else:
        print(f"  ✅ 无重复 id（{len(cnt):,} 个唯一）")

    # ---- 4. 抽样范数：只在**已写入区间** [0, n_ids) 内抽 ----
    # 抽到预分配空洞（尾部零向量）是必然的假警报，不是问题。
    valid = min(n_ids, n_emb)
    k = min(args.sample, valid)
    if k == 0:
        print("已写入区间为空，跳过范数抽样")
    else:
        spots = sorted({0, max(0, valid // 4), max(0, valid // 2),
                        max(0, 3 * valid // 4), max(0, valid - k)})
        print(f"范数抽样: 已写入 {valid:,} 行内，每段 {k:,} 行，共 {len(spots)} 段")
        nan_total = 0
        for s in spots:
            blk = np.asarray(emb[s:s + k]).astype(np.float32)
            if blk.size == 0:
                continue
            nn = int(np.isnan(blk).sum())
            nan_total += nn
            nr = np.linalg.norm(blk, axis=1)
            zero = int((nr < 1e-6).sum())
            flag = "✅" if (nn == 0 and zero == 0 and 0.99 < nr.min() and nr.max() < 1.01) else "🔴"
            print(f"  {flag} 行 {s:>9,}~{s + k:>9,}  范数 [{nr.min():.4f}, {nr.max():.4f}]"
                  f"  NaN={nn}  零向量={zero}")
            if flag == "🔴":
                bad += 1
                if zero:
                    print("      ⚠️ 已写入区间内出现零向量 = 真的没被写过，不是空洞")

    print("-" * 66)
    if bad == 0:
        print("VECTORS_OK  三项不变量全过，可以用于检索或继续续跑")
    else:
        print(f"VECTORS_FAIL  有 {bad} 项不合格")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
