#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 6 步辅助工具（跨 Python 的第一步）：把 chunks.parquet 的 char_len 列导成纯文本。

为什么单独写成文件
------------------
手册初版把这段塞在 `python -c "..."` 里，有两个问题：
  1. 331 万条 `collect()` 到 driver，在 Python 里就是 331 万个 int 对象（约 100MB+），
     一旦后面还要 join 别的操作就容易把 driver 顶爆。导出成文件是零 driver 压力的做法。
  2. cmd.exe 的多行内联脚本引号地狱（手册 §6.1 已经踩过这个坑）。
所以：**这个脚本用 3.11 跑（因为有 pyspark），产出的文本给 3.13 画图用。**

用法
----
    "C:\\...\\Python311\\python.exe" src\\export_char_len.py

产物
----
    eval\\results\\char_len_parts\\part-*.txt   （每行一个字符数）

下一步：用 3.13 跑 `python src/plot_len_dist.py` 画直方图。
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

sys.path.insert(0, str(HERE))
from build_pipeline import ensure_hadoop_home, ensure_java_home  # noqa: E402

# ⚠️ 必须在 import pyspark 之前调 —— JVM 一旦起来，环境变量再设也晚了。
# 不设的后果实测过一次：Spark 能启动，但一读 parquet 就抛
#   java.lang.UnsatisfiedLinkError: NativeIO$Windows.access0
# 因为它要加载 winutils.exe 里的原生方法，而 hadoop.home.dir 没指到
# E:\AI-learning\hadoop。手册 §5.0 / 易错点表 K 项说的就是这个。
ensure_java_home()
ensure_hadoop_home()

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",
                    default=str(PROJECT / "data" / "processed" / "chunks.parquet"))
    ap.add_argument("--out",
                    default=str(PROJECT / "eval" / "results" / "char_len_parts"))
    ap.add_argument("--master", default="local[4]",
                    help="local[N]；N 不是越大越快，parquet 读一列是 IO 活")
    args = ap.parse_args()

    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    # 溢写目录一定放 E 盘：默认落在 C:\\Users\\...\\AppData\\Local\\Temp，
    # C 盘本来就紧，331 万行 shuffle 一次就能把它顶满（第 5 步踩过）。
    tmp = str(PROJECT / "data" / "_spark_tmp")

    spark = (SparkSession.builder
             .appName("export-char-len")
             .master(args.master)
             .config("spark.local.dir", tmp)
             .config("spark.sql.shuffle.partitions", "8")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)

    t0 = time.time()
    df = spark.read.parquet(args.input).select(F.col("char_len").cast("string"))

    # 只读一列 → parquet 列存只需扫这一个 column chunk，比整表快一个量级。
    # coalesce(1)：14MB 的文本没必要散成 4 个文件，合一个便于下一步读。
    (df.coalesce(1).write.mode("overwrite").text(str(out)))
    n = df.count()  # 导出后复点一次，防止写出过程静默丢行

    files = sorted(out.glob("part-*"))
    size_mb = sum(f.stat().st_size for f in files) / 1024 / 1024

    print("EXPORT_CHAR_LEN_OK")
    print(f"  rows        : {n:,}")
    print(f"  files       : {len(files)} 个 -> {out}")
    print(f"  size        : {size_mb:.1f} MB")
    print(f"  elapsed     : {time.time() - t0:.1f} s")

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
