#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
取原文延迟诊断 —— **不加载 faiss、不加载 BM25、不加载模型**（跑完约 1 分钟）。

============================================================================
为什么能这么便宜
============================================================================
取原文的耗时只取决于两件事：**要取哪些行号**，以及**存储的物理布局**。
这两件事都不需要索引：行号可以从历史评测结果里回收（`hit_chunk_ids`），
布局可以从 parquet 元数据里读（只读元数据，不解压数据）。

这也是本仓库反复用到的一条方法：**先找一个"不花代价就能量的量"**。
（同类的还有 `diagnose_rerank_trunc.py` —— 只加载 tokenizer 就量出重排截断率。）
不先诊断就直接改，很容易花两小时优化一个只占 3% 的环节。

============================================================================
三个问题
============================================================================
① 现状到底多慢？—— 分「冷」（首轮，含磁盘读）与「热」（页缓存命中）两个数。
   两者差别说明瓶颈是**磁盘**还是**CPU 解压**，这决定了优化方向完全不同。
② 慢在哪一层？—— 按"碰到几个分片"分组统计：4 个分片全碰 vs 只碰 1 个。
   如果每碰一个分片固定花约 470 ms，那就是"每分片全扫"的锅（而不是"取的行多"）。
③ 换成随机访问能省多少？—— 侧车建好后用 `--impl blob` 复跑，同一批行号直接对比。

用法
----
    python src/diagnose_fetch.py                      # 量现状（parquet 全扫）
    python src/diagnose_fetch.py --impl blob --repeat 50   # 量侧车（mmap 随机访问）
============================================================================
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))

from fetch_store import (BLOB, OFFS, BlobFetcher, RowTextFetcher,  # noqa: E402
                         list_parts, read_ids)


def layout(parquet: Path) -> None:
    """① 物理布局：分片 / row group 行数与字节（只读元数据，零 IO 成本）。"""
    import pyarrow.parquet as pq

    print("=" * 78)
    print("【布局】parquet 是为「扫全表」设计的")
    print("=" * 78)
    parts = list_parts(parquet)
    tot_rows = tot_enc = tot_unc = 0
    for f in parts:
        md = pq.ParquetFile(f).metadata
        enc = f.stat().st_size
        unc = sum(md.row_group(i).total_byte_size for i in range(md.num_row_groups))
        tot_rows += md.num_rows
        tot_enc += enc
        tot_unc += unc
        print(f"  {f.name[:14]}  行 {md.num_rows:>9,}  row_group {md.num_row_groups}  "
              f"磁盘 {enc / 1e6:>6.0f} MB  解压后 {unc / 1e6:>6.0f} MB  "
              f"（压缩比 {unc / enc:.1f}×）")
    rg = pq.ParquetFile(parts[0]).metadata.row_group(0)
    print(f"  单个 row group：{rg.num_rows:,} 行 / 解压 {rg.total_byte_size / 1e6:.0f} MB")
    print(f"  合计 {tot_rows:,} 行 / 磁盘 {tot_enc / 1e9:.2f} GB / 解压后 {tot_unc / 1e9:.2f} GB")
    print("  ⇒ 要取 100 行，至少要打开 1 个 179 MB 的 row group；")
    print("    而 chunk_id 是随机哈希，min/max 统计剪不掉任何行组 —— 4 个分片全开。")


def payload_size(parquet: Path, sample_rows: int = 20_000) -> None:
    """② 侧车体积预测：抽样算每行 JSON 字节数（真读一个 row group，约 2 s）。"""
    import pyarrow.parquet as pq

    print()
    print("=" * 78)
    print("【体积】侧车要多大（决定这笔交易划不划算）")
    print("=" * 78)
    f = list_parts(parquet)[0]
    t = pq.ParquetFile(f).read_row_group(0, columns=["chunk_id", "title", "section", "chunk_text"])
    n = min(sample_rows, t.num_rows)
    d = t.slice(0, n).to_pydict()
    tot = 0
    for i in range(n):
        tot += len(json.dumps({"id": d["chunk_id"][i], "title": d["title"][i],
                               "section": d["section"][i], "text": d["chunk_text"][i]},
                              ensure_ascii=False, separators=(",", ":")).encode()) + 1
    avg = tot / n
    print(f"  抽样 {n:,} 行：平均 {avg:.0f} 字节/行（JSON，id+title+section+text）")
    print(f"  ⇒ 全量 331.6 万行约 {avg * 3_316_395 / 1e9:.2f} GB 磁盘")
    print(f"  ⇒ 偏移数组 331.6 万 × 8 字节 ≈ 26.5 MB（常驻内存）")
    print("  ⚠️ 注意：体积**没有变小**（和 parquet 差不多）—— 换来的是随机访问。")
    del t


def load_workload(ids_path: Path, results: Path) -> tuple[list[int], list[int]]:
    """
    ③ 真实行号分布 —— 从历史评测结果里回收 hit_chunk_ids，再反查行号。

    为什么要用真实行号而不是随机行：真实命中**集中在少数热门条目**上，
    分布比均匀随机更偏 —— 用均匀随机会低估"同一分片被反复打开"的代价。
    """
    need = set()
    for line in results.read_text(encoding="utf-8").splitlines():
        if line.strip():
            need.update(json.loads(line).get("hit_chunk_ids") or [])
    idx = {}
    with open(ids_path, encoding="utf-8") as f:
        for i, ln in enumerate(f):
            cid = ln.strip()
            if cid in need:
                idx[cid] = i
    hit_rows = sorted(idx.values())
    # 把 645 条真实命中扩成 pool 规模的批次（pool=100）：循环取用，保留偏斜
    return hit_rows, sorted(need)


def part_bounds(parquet: Path) -> list[tuple[int, int]]:
    """各分片的行号区间（只读元数据）。用来统计"一批 100 行摊到了几个分片"。"""
    import pyarrow.parquet as pq

    out, cum = [], 0
    for f in list_parts(parquet):
        n = pq.ParquetFile(f).metadata.num_rows
        out.append((cum, cum + n))
        cum += n
    return out


def _n_shards(rows, bounds) -> int:
    return len({i for r in rows for i, (lo, hi) in enumerate(bounds) if lo <= int(r) < hi})


def _snap():
    """
    采样本进程的 (CPU秒, 磁盘读字节)。

    ⚠️ 为什么要这两个数，而不是只看墙钟：**墙钟受页缓存状态影响，不可比**。
       本机实测过同一份工作"冷盘比热盘还快"（后台进程干扰），
       所以「冷/热」这种分法得不出可靠结论。
       而 CPU 时间与缓存无关：解压就是解压，缓存命中也要烧 CPU；
       read_bytes 则告诉我们"这次到底从盘上拿了多少字节"。
       两个数一起看，才能回答"瓶颈是磁盘还是 CPU"。
    """
    try:
        import psutil
    except ImportError:
        return None, None
    p = psutil.Process()
    c = p.cpu_times()
    return c.user + c.system, p.io_counters().read_bytes


def bench(fetcher, batches: list[list[int]], label: str, bounds) -> dict:
    """按批计时（调用前应已热身一次，避免把首次导入/冷盘算进来）。"""
    ts, cpus, mbs, n_shards = [], [], [], []
    for rows in batches:
        c0, b0 = _snap()
        t = time.time()
        got = fetcher.fetch(rows)
        ts.append((time.time() - t) * 1000)
        c1, b1 = _snap()
        if c0 is not None:
            cpus.append((c1 - c0) * 1000)
            mbs.append((b1 - b0) / 1e6)
        assert len(got) == len(set(rows)), f"{label}: 取回条数 {len(got)} != 请求 {len(set(rows))}"
        n_shards.append(_n_shards(rows, bounds))
    return {"label": label, "med": st.median(ts), "p90": sorted(ts)[int(len(ts) * 0.9) - 1],
            "min": min(ts), "max": max(ts), "shards": st.mean(n_shards),
            "cpu": st.median(cpus) if cpus else None, "mb": st.median(mbs) if mbs else None}


def main() -> int:
    ap = argparse.ArgumentParser(description="取原文延迟诊断（不加载索引）")
    ap.add_argument("--impl", default="parquet", choices=["parquet", "blob"])
    ap.add_argument("--repeat", type=int, default=8, help="批次数（每批 pool 条行号）")
    ap.add_argument("--pool", type=int, default=100)
    ap.add_argument("--skip-layout", action="store_true")
    ap.add_argument("--parquet", default=str(PROJECT / "data" / "processed" / "chunks.parquet"))
    ap.add_argument("--ids", default=str(PROJECT / "data" / "index" / "bge-large-zh-v1.5" / "ids.txt"))
    ap.add_argument("--blob", default=str(BLOB), help="侧车路径（量冷读时可指向一份新拷贝）")
    ap.add_argument("--offs", default=str(OFFS))
    ap.add_argument("--results", default=str(PROJECT / "eval" / "results" / "gen_dashscope_testset_rr.jsonl"))
    args = ap.parse_args()

    parquet, ids_path = Path(args.parquet), Path(args.ids)

    if not args.skip_layout:
        layout(parquet)
        payload_size(parquet)

    print()
    print("=" * 78)
    print(f"【计时】impl={args.impl}  pool={args.pool}  repeat={args.repeat}")
    print("=" * 78)
    t_ids = time.time()
    id_list = read_ids(ids_path)
    hit_rows, _ = load_workload(ids_path, Path(args.results))
    print(f"  读 ids.txt {len(id_list):,} 行 / {time.time() - t_ids:.1f}s；"
          f"回收真实命中行号 {len(hit_rows):,} 个")
    if len(hit_rows) < args.pool:
        raise SystemExit("[错误] 真实命中行号不足一批，检查 --results")

    rng = np.random.default_rng(0)
    # ⚠️ 真实命中批次必须**随机抽样**，不能按行号切连续的段：
    #    hit_rows 是排序过的，连续 100 个往往落在同一个分片里 ——
    #    那样量出来会比实际快一倍（第一版就是这么骗了我一次，冷热差甚至是负的）。
    hit_batches = [rng.choice(hit_rows, size=args.pool, replace=False).tolist()
                   for _ in range(args.repeat)]
    rand_batches = [rng.integers(0, len(id_list), size=args.pool).tolist()
                    for _ in range(args.repeat)]

    if args.impl == "parquet":
        fetcher = RowTextFetcher(parquet, id_list)
        print(f"  分片：{fetcher.summary()}")
    else:
        fetcher = BlobFetcher(Path(args.blob), Path(args.offs), id_list)
        print(f"  侧车：{fetcher.summary()}")

    # 热身：① 让延迟导入（parquet 路径会 import search_vector）发生在计时之外；
    #       ② 用一条**代表性**批次把磁盘读成页缓存 —— 冷热之差才是"磁盘 vs CPU"。
    t = time.time()
    fetcher.fetch(rand_batches[0])
    t_cold = (time.time() - t) * 1000
    print(f"  冷启动首轮（含导入 + 落盘读）：{t_cold:.0f} ms")

    bounds = part_bounds(parquet)
    res = [bench(fetcher, rand_batches[1:], "均匀随机行号", bounds),
           bench(fetcher, hit_batches, "真实命中行号", bounds)]
    print()
    print(f"  {'批次来源':<14}{'中位':>10}{'p90':>10}{'CPU':>10}{'磁盘读':>10}{'均摊分片':>10}")
    for r in res:
        cpu = f"{r['cpu']:.0f}ms" if r["cpu"] is not None else "—"
        mb = f"{r['mb']:.0f}MB" if r["mb"] is not None else "—"
        print(f"  {r['label']:<14}{r['med']:>9.1f}ms{r['p90']:>9.1f}ms"
              f"{cpu:>10}{mb:>10}{r['shards']:>9.1f}")

    med = res[0]["med"]
    cpu = res[0]["cpu"]
    print()
    print("=" * 78)
    print("【结论】")
    print("=" * 78)
    print(f"  取 {args.pool} 条原文的中位耗时：{med:.0f} ms（impl={args.impl}，均匀随机行号）")
    if cpu is not None:
        share = cpu / max(med, 1e-9) * 100
        print(f"  同批 CPU 时间 {cpu:.0f} ms（占墙钟 {share:.0f}%）+ 磁盘读 {res[0]['mb']:.0f} MB")
        if med < 5:
            print("  ⇒ 墙钟已到**噪声量级**（<5 ms），CPU 与磁盘都测不出来 —— 不用再归因了")
        else:
            print("  ⇒ " + ("**CPU 密集**：墙钟基本被 CPU 吃满，而且是解压吃掉的 —— "
                            "换 SSD / 加大缓存都没用"
                            if share > 60 else
                            "CPU 占比不高 ⇒ 大头在 IO 或调度，先查磁盘"))
    print("  ⚠️ 不要再细分「冷盘/热盘」：本机实测出现过冷盘更快（后台进程干扰），"
          "这个分法得不出可靠结论；用 CPU + read_bytes 归因才是稳的。")
    per_shard = med / max(res[0]["shards"], 1)
    print(f"  两批都摊到 {res[0]['shards']:.1f} 个分片 ⇒ 单分片约 {per_shard:.0f} ms")
    print("  它还解释了一件反直觉的事：**代价与「取几条」无关** ——"
          "取 5 条和取 100 条一样慢（都得把分片解压一遍）。")
    print("  ⇒ 这是「全表扫描」的代价，不是「点查」的代价。")
    if args.impl == "parquet":
        print("  预期：换成侧车（--impl blob）后应降到 1 ms 量级；")
        print(f"        端到端按 4.78 s 算，相当于省掉约 {med / 4780 * 100:.0f}% 的端到端延迟。")
    else:
        print("  对照基线见 --impl parquet 的输出（同一批行号、同一台机器）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
