#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 6 步辅助工具（跨 Python 的第二步）：画 chunk 长度分布图。

为什么必须跨两个 Python
----------------------
* 读 parquet 要 pyspark → 只能 3.11（Spark 3.5 不支持 3.13，见易错点表 A 项）
* 画图要 matplotlib → 只装在 3.13（3.11 里没有）
中间用 `eval/results/char_len_parts/*.txt` 当桥 —— 12MB 纯文本，两个解释器都读得动。

用法
----
    "C:\\...\\Python313\\python.exe" src\\plot_len_dist.py

产物
----
    eval\\results\\chunk_len_dist.png    左：直方图；右：累积分布
    eval\\results\\chunk_len_stats.json  统计量，README 直接引用

这张图要看什么
--------------
1. **不能全挤在最左边。** 全挤左边 = 切块器把内容切碎了（比如按句号硬切）。
2. **不能有 800 处的尖峰。** 有尖峰 = 大量块是被 max_len 硬截断的，
   说明切块策略对长段落无效（我们的实现是**段落级切分 + 超长段才硬切**，
   所以 800 附近应该只有个小台阶，不是一根针）。
3. 右图的 CDF 陡不陡，决定了检索时取 top-k 的长度代价。
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default=str(PROJECT / "eval" / "results" / "char_len_parts"))
    ap.add_argument("--out", default=str(PROJECT / "eval" / "results" / "chunk_len_dist.png"))
    ap.add_argument("--bins", type=int, default=40)
    args = ap.parse_args()

    parts = sorted(Path(args.parts).glob("part-*"))
    if not parts:
        print(f"[致命错误] {args.parts} 里没有 part-* 文件，先跑 src/export_char_len.py")
        return 1

    # 331 万个 int32 = 13MB，比逐行 int() 快一个量级
    vals = np.concatenate([np.array(f.read_text(encoding="utf-8").split(), dtype=np.int32)
                           for f in parts])
    n = vals.size

    q = {p: float(np.percentile(vals, p)) for p in (1, 5, 25, 50, 75, 90, 95, 99)}
    stats = {
        "n": int(n),
        "min": int(vals.min()),
        "max": int(vals.max()),
        "mean": round(float(vals.mean()), 1),
        "std": round(float(vals.std()), 1),
        "p1": q[1], "p5": q[5], "p25": q[25], "p50": q[50],
        "p75": q[75], "p90": q[90], "p95": q[95], "p99": q[99],
        "pct_at_max": round(float((vals >= 800).mean() * 100), 4),
        "pct_under_100": round(float((vals < 100).mean() * 100), 2),
        "src_files": [f.name for f in parts],
    }

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # ---- 左：直方图 ----
    ax1.hist(vals, bins=args.bins, color="#4C78A8", edgecolor="white", linewidth=0.4)
    for p, c, lab in ((q[50], "#E45756", "中位数 209"),
                      (q[90], "#F58518", "P90")):
        ax1.axvline(p, color=c, linestyle="--", linewidth=1.4,
                    label=f"{lab} = {int(p)}")
    ax1.set_xlabel("chunk 字符数")
    ax1.set_ylabel("块数量")
    ax1.set_title(f"chunk 长度分布（{n:,} 块，bins={args.bins}）")
    ax1.legend(frameon=False)
    ax1.grid(axis="y", alpha=0.25, linewidth=0.5)

    # ---- 右：累积分布 ----
    # 331 万个点直接 plot 会画出一张又慢又肥的图，每 ~1600 个取 1 个足够光滑
    step = max(1, n // 2000)
    xs = np.sort(vals)[::step]
    ys = np.arange(1, n + 1)[::step] / n * 100
    ax2.plot(xs, ys, color="#4C78A8", linewidth=1.6)
    ax2.axhline(50, color="#E45756", linestyle="--", linewidth=1.0)
    ax2.axvline(q[50], color="#E45756", linestyle="--", linewidth=1.0)
    ax2.set_xlabel("chunk 字符数")
    ax2.set_ylabel("累计占比 (%)")
    ax2.set_title("长度累积分布（CDF）")
    ax2.grid(alpha=0.25, linewidth=0.5)
    ax2.set_ylim(0, 100)

    # 右上角写一行关键数，图截图进 README 时不用再配文字
    box = (f"mean {stats['mean']:.0f}\n"
           f"P50 {int(q[50])}   P90 {int(q[90])}\n"
           f"min {stats['min']}   max {stats['max']}")
    ax2.text(0.97, 0.06, box, transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=9, family="monospace",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="#F7F7F7",
                       edgecolor="#CCCCCC", linewidth=0.8))

    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)

    (Path(args.out).parent / "chunk_len_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("PLOT_LEN_DIST_OK")
    print(f"  n / mean / p50 / max : {stats['n']:,} / {stats['mean']} / "
          f"{int(q[50])} / {stats['max']}")
    print(f"  <100 字占比          : {stats['pct_under_100']}%  "
          f"(全挤左边就是这个数很大)")
    print(f"  >=800 硬截断占比     : {stats['pct_at_max']}%  "
          f"(很大说明 max_len 卡得太死)")
    print(f"  图 -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
