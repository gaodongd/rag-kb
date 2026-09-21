# -*- coding: utf-8 -*-
"""
第 5 步产物验收脚本 —— 校验 chunks.parquet 的数据质量。

为什么要单独写一个：管道跑完只是"没报错"，不等于"数据对"。
合格的验收必须能回答三个问题：
  1. 规模对不对？（总块数、唯一 doc_id 数）
  2. 内容干不干净？（残留 wiki 标记、乱码符）
  3. 字段有没有越界？（char_len / cn_ratio 是否落在设计区间）

实测踩过的坑（别重犯）：
  * 用 `df.cache()` 缓存 2.3GB 的 chunk 表 → Java heap OOM。验收是只读扫描，
    本来就不需要缓存；而且用**一次 agg** 把所有计数算完，比跑 8 次 count() 快得多。
  * `--driver-memory` 必须给够（本项目给 6g），默认 1g 在 400 万行上必炸。

用法：
    python src/verify_chunks.py --input data/processed/chunks.parquet
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pipeline import ensure_hadoop_home, ensure_java_home  # noqa: E402

JAVA_HOME = ensure_java_home()
ensure_hadoop_home()
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession, functions as F  # noqa: E402

# 残留标记 —— 每一项对应清洗器里的一条正则，漏哪个一眼能看出来
RESIDUAL = {
    "LBRACKET": "[[",       # 内部链接没洗掉
    "TEMPLATE": "{{",       # 模板没洗掉
    "REF": "<ref",          # 引用标签没洗掉
    "LANGCONV": "-{",       # 语言变体标记（RE_LANG 管这个）
    "HTMLDIV": "<div",      # HTML 标签没洗掉
    "TABLE": "{|",          # 表格没洗掉
}

# 清洗副作用 —— 和上面的「残留标记」是**两类不同的问题**，别混在一起看：
#   残留标记   = 该删的没删干净（正则漏了）
#   清洗副作用 = 删得太干净，把成对符号的内容掏空，留下空壳（{{lang|xx|…}} 被删掉后
#               剩下的 `（）`）。早期版本没检测这一类，靠人工抽样才发现的。
# 用正则（rlike）而不是 contains：要匹配「一对符号中间什么都没有」。
SIDE_EFFECTS = {
    # 括号里只剩标点 （，） （；） （、） （-）   ← 第四轮才加的
    #   教训：EMPTYPAREN 只匹配「括号里什么都没有」，抓不到「括号里还剩一个逗号」。
    #   原文 `山田宗昌（{{lang-ja|やまだ むねなさ}}，1562年—1636年）` ——
    #   模板和生卒年被删掉、逗号留下，于是验收表报了 EMPTYPAREN = 0 的**假绿**。
    #   全量实测 89,143 块（2.685%），比 EMPTYPAREN 声称的 0 严重得多。
    #   ⚠️ 一条检测规则写得太窄，会让验收表报出「0」这种虚假安全感。
    #   ⚠️ 检测口径必须和修复口径**逐字符对齐**（见 build_pipeline.SHELL_INNER）：
    #      刻意不收 `.` 和 `…` —— `（...）` 是合法省略用法，收进来就会永远报红。
    "PAREN_PUNCT": r"[（(][\s，。；：、,;:\-–—]*[）)]",
    "EMPTYPAREN": r"[（(]\s*[）)]",       # 空括号（）（已被上面那条完全覆盖，留作对照）
    "EMPTYBOOK": r"[《][\s，。；：、,;:\-–—]*[》]",   # 空书名号《》及《，》
    "EMPTYQUOTE": r"[「『][\s，。；：、,;:\-–—]*[」』]",
    "EMPTYBRACKET": r"[【][\s，。；：、,;:\-–—]*[】]",  # 空方头括号【】
    "EMPTYDBLQUOTE": r"[“][\s，。；：、,;:\-–—]*[”]",  # 空双引号“”
    "DUP_PUNCT": r"[，。；、]{2,}",        # 连续重复标点
}

# 命名空间残留 —— 第三类问题，和上面两类都不同：
#   残留标记   = 该删的没删干净（正则没匹配上）
#   清洗副作用 = 删过头，留下空壳
#   命名空间   = **该整块删的东西被当正文留下了**（[[Category:X]] 被剥成 `Category:X`）
# 全量实测（2026-09-17，3,926,261 块）：
#   含 `Category:` 的块          492,872 条  12.553%
#   整块只有分类标签、零正文的块  143,683 条   3.660%  ← 纯噪声，检索时只污染召回
NAMESPACE_RESIDUE = {
    "CATEGORY": r"(?i)category\s*:",
    "PURE_CAT": r"(?i)^\s*(category\s*:[^\n]*\n?)+\s*$",
    # 行首的 File:/Image: 行 —— 两个来源：<gallery> 里的无方括号文件行，
    # 以及"嵌套 File 链接被 RE_LINK_BARE 抢剥"留下的内容。只查行首，避免误伤
    # 英文单词（profile: 之类）。
    "FILEXT": r"(?im)^\s*(?:file|image|文件|图像|圖片)\s*:[^\n]*",
}


def main():
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(here / "data/processed/chunks.parquet"))
    ap.add_argument("--driver-memory", default="6g")
    ap.add_argument("--master", default="local[4]")
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()

    spark = (
        SparkSession.builder
        .appName("verify-chunks")
        .master(args.master)
        .config("spark.driver.memory", args.driver_memory)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.parquet(args.input)

    # ---- 一次 agg 拿到所有计数（不要分开 count()，会重复扫 8 遍）----
    exprs = [
        F.count(F.lit(1)).alias("total"),
        F.countDistinct("doc_id").alias("uniq_doc"),
        F.countDistinct("title").alias("uniq_title"),
        F.min("char_len").alias("len_min"),
        F.max("char_len").alias("len_max"),
        F.expr("percentile_approx(char_len, 0.5)").alias("len_p50"),
        F.round(F.avg("char_len"), 1).alias("len_avg"),
        F.round(F.avg("cn_ratio"), 4).alias("cn_avg"),
        F.sum((F.col("char_len") > 800).cast("int")).alias("len_gt_800"),
        F.sum((F.col("char_len") < 50).cast("int")).alias("len_lt_50"),
        F.sum((F.col("cn_ratio") < 0.30).cast("int")).alias("cn_lt_30"),
        F.sum((F.col("section") == "").cast("int")).alias("empty_section"),
        F.sum((F.col("title") == "").cast("int")).alias("empty_title"),
        F.sum(F.col("chunk_text").contains("\ufffd").cast("int")).alias("has_fffd"),
    ]
    for name, pat in RESIDUAL.items():
        exprs.append(F.sum(F.col("chunk_text").contains(pat).cast("int")).alias("res_" + name))
    for name, pat in SIDE_EFFECTS.items():
        exprs.append(F.sum(F.col("chunk_text").rlike(pat).cast("int")).alias("side_" + name))
    for name, pat in NAMESPACE_RESIDUE.items():
        exprs.append(F.sum(F.col("chunk_text").rlike(pat).cast("int")).alias("ns_" + name))

    row = df.agg(*exprs).first()
    total = row["total"]
    print("=" * 64)
    print("chunks.parquet 验收")
    print("=" * 64)

    print("\n[规模]")
    print(f"  总块数            : {total:,}")
    print(f"  唯一 doc_id       : {row['uniq_doc']:,}")
    print(f"  唯一 title        : {row['uniq_title']:,}")
    print(f"  平均块/文档       : {total / max(row['uniq_doc'], 1):.2f}")

    print("\n[char_len 分布]")
    print(f"  min / p50 / avg / max : {row['len_min']} / {row['len_p50']} / "
          f"{row['len_avg']} / {row['len_max']}")
    print(f"  >800 的块         : {row['len_gt_800']:,}  "
          f"({row['len_gt_800'] / max(total, 1) * 100:.3f}%)")
    print(f"  <50 的块          : {row['len_lt_50']:,}  "
          f"({row['len_lt_50'] / max(total, 1) * 100:.3f}%)")

    print("\n[清洗残留]  —— 这一栏是本次修复的重点，越低越好")
    for name in RESIDUAL:
        c = row["res_" + name]
        print(f"  含 {RESIDUAL[name]:<6} 的块  : {c:,}  ({c / max(total, 1) * 100:.3f}%)")
    print(f"  含 U+FFFD 的块    : {row['has_fffd']:,}  "
          f"({row['has_fffd'] / max(total, 1) * 100:.3f}%)")
    print(f"  cn_ratio < 0.30   : {row['cn_lt_30']:,}  "
          f"（设计上已过滤，应为 0）")

    print("\n[清洗副作用]  —— 删得太干净留下的空壳（第 6 步人工审计发现的新问题）")
    for name, pat in SIDE_EFFECTS.items():
        c = row["side_" + name]
        print(f"  {name:<12} : {c:,}  ({c / max(total, 1) * 100:.3f}%)")

    print("\n[命名空间残留]  —— [[Category:X]] 被剥成纯文本 `Category:X` 留下的")
    for name, pat in NAMESPACE_RESIDUE.items():
        c = row["ns_" + name]
        print(f"  {name:<12} : {c:,}  ({c / max(total, 1) * 100:.3f}%)")

    print("\n[空字段]")
    print(f"  section 为空      : {row['empty_section']:,}")
    print(f"  title 为空        : {row['empty_title']:,}")

    print("\n[抽样]")
    for r in df.select("chunk_id", "title", "section", "char_len", "cn_ratio",
                       "chunk_text").limit(args.samples).collect():
        print(f"  --- {r['title']} / {r['section']} / len={r['char_len']} "
              f"/ cn={r['cn_ratio']:.3f} / id={r['chunk_id'][:12]}")
        print("      " + r["chunk_text"][:180].replace("\n", " "))

    print("\n[残留样例]  —— 如果上面计数不为 0，这里看它长什么样")
    cond = None
    for pat in RESIDUAL.values():
        c = F.col("chunk_text").contains(pat)
        cond = c if cond is None else (cond | c)
    for r in df.filter(cond).select("title", "chunk_text").limit(3).collect():
        t = r["chunk_text"]
        idx = min([i for i in (t.find(p) for p in RESIDUAL.values()) if i >= 0] or [0])
        print(f"  {r['title']}: ...{t[max(0, idx - 50): idx + 120]}".replace("\n", " "))

    print("=" * 64)
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
