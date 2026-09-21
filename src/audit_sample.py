# -*- coding: utf-8 -*-
"""
第 6 步 · 数据质量审计 —— 抽样导出脚本。

为什么要单独做这一步：管道报"完成"只证明它没崩，证明不了数据好。
面试官问"你的数据真实吗、质量怎么样"，唯一有说服力的答案是**一张审计表**，
而审计表里最硬的一行是「人工抽样通过率」—— 这一行机器算不出来。

本脚本只干机械活：捞样本、排版成人能读的样子、附上机器能判的那部分预检。
判「通顺 / 自洽 / 有信息量」必须你自己看 —— 这是设计意图，不是偷懒。

为什么不用手册里那条 `python -c "..."` 内联命令：
  在 cmd.exe 里，`-c "` 的外层双引号会被正文里的 `"` 提前闭合，
  报 `SyntaxError` 或把内容截断。多行 Python 一律写进文件再跑。

实测踩过的坑：
  * DataFrame 没有 `takeSample()` —— 那是 RDD 的方法。手册 6.1 初版就是这么写的，
    直接报 `AttributeError: 'DataFrame' object has no attribute 'takeSample'`。
    改用 `orderBy(F.rand(seed)).limit(n)`，见下面的注释。
  * 别 `df.cache()` —— 400 万行 × 800 字 ≈ 2.3GB，必 OOM。

用法：
    python src/audit_sample.py --n 50 --seed 42

输出：
    eval/audit_sample.md    给人看的（逐条打勾）
    eval/audit_sample.json  给机器读的（含 chunk_id，便于复查和对比）
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pipeline import ensure_hadoop_home, ensure_java_home  # noqa: E402

# 注意：这两个 ensure 必须在 import pyspark **之前**调用，
# 否则 JVM 已经按错误的环境变量启动了，再设也没用。
ensure_java_home()
ensure_hadoop_home()
import os  # noqa: E402

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession, functions as F  # noqa: E402

# 复用验收脚本里的残留标记定义，保证「机器预检」和「产物验收」口径一致。
# 口径不一致是数据审计里最容易被抓的漏洞：同一批数据两份报告给两个结论。
from verify_chunks import RESIDUAL  # noqa: E402

# 人工判据 —— 和手册 6.1 的表一一对应
CRITERIA = ["通顺", "自洽", "有信息量", "无标记"]


def find_residual(text: str) -> list:
    """返回命中的残留标记名列表（空列表 = 机器判定这一条干净）。"""
    return [name for name, marker in RESIDUAL.items() if marker in text]


def main():
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(here / "data/processed/chunks.parquet"))
    ap.add_argument("--outdir", default=str(here / "eval"))
    ap.add_argument("--n", type=int, default=50, help="抽样条数")
    ap.add_argument("--seed", type=int, default=42, help="固定随机种子，保证可复现")
    ap.add_argument("--driver-memory", default="6g")
    ap.add_argument("--master", default="local[4]")
    args = ap.parse_args()

    spark = (
        SparkSession.builder
        .appName("audit-sample")
        .master(args.master)
        .config("spark.driver.memory", args.driver_memory)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # 不 cache：400 万行 × 800 字 ≈ 2.3GB，cache 必 OOM（第 5 步验收时踩过）。
    #
    # ⚠️ DataFrame 没有 takeSample() —— 那是 RDD 的方法（手册 6.1 初版就写错了，
    #    报 AttributeError: 'DataFrame' object has no attribute 'takeSample'）。
    # 用 orderBy(F.rand(seed)).limit(n)：
    #   * rand 带 seed → 结果可复现
    #   * Catalyst 把 Sort+Limit 优化成 TakeOrderedAndProject：每个分区本地先取
    #     n 条候选再归并，**不做全表 shuffle**，所以 400 万行也很快
    # 别改用 df.sample(fraction).limit(n)：fraction 太小则实际只命中前几个分区，
    # 抽出来不是全表均匀样本，审计结论会偏。
    df = spark.read.parquet(args.input)
    rows = (
        df.select("chunk_id", "doc_id", "title", "section",
                  "char_len", "cn_ratio", "chunk_text")
        .orderBy(F.rand(args.seed))
        .limit(args.n)
        .collect()
    )

    if not rows:
        print("[致命错误] 抽样 0 条 —— 输入为空或 schema 不匹配，停止。")
        return 1

    # 按字符数升序：看 50 条的过程中你能自然感受到「长度 vs 质量」的关系
    rows = sorted(rows, key=lambda r: r["char_len"])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    md_path = outdir / "audit_sample.md"
    json_path = outdir / f"audit_sample.seed{args.seed}.json"

    n_dirty = 0
    json_rows = []

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 第 6 步 · 抽样人工核查\n\n")
        f.write(f"- 来源：`data/processed/chunks.parquet`\n")
        f.write(f"- 抽样：随机 {args.n} 条，seed={args.seed}（**固定种子，结果可复现**）\n")
        f.write(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 排序：按字符数升序\n")
        f.write(f"- 材料版本：**v2**"
                f"（v1 漏印 `section` 字段，把一条正常的按节切分误判成「不自洽」）\n\n")
        f.write("## 先看懂这三行，再动手判\n\n")
        f.write("```\n")
        f.write("## 8. 蓬镇  ·  88 字  ·  中文占比 0.80  ·  ✅ 无标记   ← 标题 + 机器预检\n")
        f.write("**章节**：`媒体`  ·  **检索串**：`蓬镇 / 媒体 / 正文`    ← 判「自洽」的依据\n")
        f.write("法国主要的媒体均可在蓬镇接收……                        ← 正文（只含「媒体」这一节）\n")
        f.write("```\n\n")
        f.write("**向量化时喂给模型的是「检索串」那一整条，不是单独的正文。**"
                "所以判「自洽」要按检索串看：标题 `蓬镇` + 章节 `媒体` + 正文讲媒体 = 自洽。\n\n")
        f.write("## 判据定义（照这个判，别凭感觉）\n\n")
        f.write("**四条全中 = 通过。任一条不中 = 不通过。** 不打半个勾。\n\n")
        f.write("### 1. 通顺 —— 这句话能不能一口气读完\n\n")
        f.write("| | 标准 |\n|---|---|\n")
        f.write("| ✅ | 完整句子，标点正常，主谓齐全，读起来不卡 |\n")
        f.write("| ✅ | **列表型**：每条自己能读通即可（`*马家砭镇强家沟村` 读得通） |\n")
        f.write("| ❌ | 有空壳符号 `（，）` `《》` `「」`；句子半截就断；"
                "以 `、。` 开头；连续 `，，` |\n\n")
        f.write("判法：**默读一遍**。卡住的地方，是原文本来如此，还是数据被删坏了？"
                "被删坏的判不通过。\n\n")
        f.write("`[别搞混]` **列表型不等于不通过。** 全语料约 17% 的块是列表/年表"
                "（历史沿革、作品年表、行政区划名单），它们本身就是这个形态。"
                "判断点在于**条目里有没有实体和事实**，而不在于有没有主谓宾。\n\n")
        f.write("### 2. 自洽 —— 正文在不在讲标题那件事\n\n")
        f.write("| | 标准 |\n|---|---|\n")
        f.write("| ✅ | 正文第一句里能找到标题的实体（或同义说法） |\n")
        f.write("| ✅ | **标题带消歧义后缀**：`明珠广场站 (绍兴市)` 正文只写 `明珠广场站` "
                "→ 算找到。后缀是 Wikipedia 用来区分同名条目的，本来就不会出现在正文里 |\n")
        f.write("| ✅ | **正文讲的正是它那一节**：标题 `蓬镇` + 章节 `媒体` + 正文讲媒体 "
                "→ 自洽。**先看「章节」再看正文** |\n")
        f.write("| ❌ | 标题讲 A、正文讲 B；正文是导航/目录，跟标题无关 |\n\n")
        f.write("判法：**先在正文里搜标题词，再对照「章节」那一行**。搜不到就看是不是同义"
                "（「田邊站」→「該站」）。\n\n")
        f.write("`[必读]` **只看正文，会把「正常的按节切分」误判成「不自洽」。**\n\n")
        f.write("> 长条目的正文被切成多块，每块带一个 `section`（小节名）。"
                "块内文本只包含**那一小节**的内容，"
                "所以「标题=地名、正文=媒体」这种组合是**设计如此**，不是拼接错误。\n"
                "> 检索时喂给向量模型的是 `title + section + 正文` 拼起来的串，"
                "上下文由 `section` 补上（这一点手册 §5.5 末尾讲透了）。\n"
                "> **实测教训**：审计材料第一版没印 `section`，"
                "导致一条 `蓬镇 / 媒体` 的正常块被判成「不自洽」——"
                "问题出在材料，不在数据。\n\n")
        f.write("### 3. 有信息量 —— 这条能不能回答一个真问题\n\n")
        f.write("| | 标准 |\n|---|---|\n")
        f.write("| ✅ | 含实体 + 事实/关系/属性。"
                "例：「田邊站是位于…的铁路车站，车站编号是T30」 |\n")
        f.write("| ❌ | 纯外链列表（`*XX官方网站`）；纯名单无上下文；全是空行和孤立符号 |\n")
        f.write("| ⚠️ | 列表/年表型**看实质**：`*三碘化铬，CrI3` 有对应关系→通过；"
                "`*烏山頭水庫管理計畫` 只有链接文字→不通过 |\n\n")
        f.write("判法：**问自己「有人搜什么问题时这条能派上用场？」** 答不上来 → 不通过。\n\n")
        f.write("### 4. 无标记 —— 有没有 wiki 语法残留\n\n")
        f.write("看总览表「机器预检」列：`✅ 干净` = 这条过了；`⚠️` = 直接判不通过。\n\n")
        f.write("> ⚠️ **但机器干净 ≠ 真干净。** 上面三条只能你读出来。\n")
        f.write("> 实测教训：机器预检报「50 条全干净」，人工一读就发现 `（，）` 空壳 ——\n")
        f.write("> 因为检测规则写窄了，只认「括号里什么都没有」，"
                "认不出「括号里还剩一个逗号」。\n\n")
        f.write("## 判定纪律（别跳过）\n\n")
        f.write("1. **先校准 5 条再连判。** 前 5 条边判边对齐标准，"
                "之后**不要中途改标准** —— 改了通过率就没意义。\n")
        f.write("2. **判据是筛子，不是秤。** 目标不是每条都判对，"
                "是**发现重复出现的模式**。判错 2~3 条对通过率只有几个百分点影响。\n")
        f.write("3. **同一类问题出现 3 次以上，停下来记编号。** "
                "那才是真 bug，比通过率本身值钱得多。\n\n")
        f.write("---\n\n")

        for i, r in enumerate(rows, 1):
            res = find_residual(r["chunk_text"])
            if res:
                n_dirty += 1
            flag = f"⚠️ 含 {'/'.join(res)}" if res else "✅ 无标记"

            # section 必须印出来 —— 第一版漏了它，代价是人工审计把正常的
            # 按节切分判成「不自洽」（`蓬镇 / 媒体` 那条）。
            # 检索串 = title + section + 正文，判「自洽」时要按这个整体看。
            sec = (r["section"] or "").strip()
            sec_show = f"`{sec}`" if sec else "（空 = 该条目的导语段）"

            f.write(
                f"## {i}. {r['title']}  ·  {r['char_len']} 字  ·  "
                f"中文占比 {r['cn_ratio']:.2f}  ·  {flag}\n\n"
            )
            f.write(f"`chunk_id={r['chunk_id']}`\n\n")
            f.write(f"**章节**：{sec_show}  ·  "
                    f"**检索串**：`{r['title']} / {sec} / 正文`\n\n")
            f.write(r["chunk_text"].strip() + "\n\n")
            f.write("**判据**：" + "  ".join(f"`[ ] {c}`" for c in CRITERIA))
            f.write("  →  **`[ ] 通过`**  **`[ ] 不通过`**\n\n")
            f.write("---\n\n")

            json_rows.append({
                "no": i,
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "title": r["title"],
                "section": r["section"],
                "char_len": r["char_len"],
                "cn_ratio": round(float(r["cn_ratio"]), 4),
                "residual": res,
                "chunk_text": r["chunk_text"],
            })

        # 结尾留一张统计表：你打完勾回来填，这个数直接进 README
        f.write("## 统计（打完勾回填，这一行进 README）\n\n")
        f.write("| 项 | 值 |\n|---|---|\n")
        f.write(f"| 样本总数 | {args.n} |\n")
        f.write("| 通过 | ____ |\n")
        f.write("| 不通过 | ____ |\n")
        f.write("| **通过率** | **____%** |\n")
        f.write(f"| 机器预检：带残留标记 | {n_dirty} 条（这些可直接判不通过） |\n")
        f.write("| 机器预检：无残留标记 | "
                f"{args.n - n_dirty} 条（仍需人工判前 3 条判据） |\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "input": args.input,
            "n": args.n,
            "seed": args.seed,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_dirty_by_machine": n_dirty,
            "rows": json_rows,
        }, f, ensure_ascii=False, indent=1)

    lens = [r["char_len"] for r in rows]
    print("AUDIT_SAMPLE_OK")
    print(f"  samples          : {len(rows)}")
    print(f"  len_min          : {min(lens)}")
    print(f"  len_max          : {max(lens)}")
    print(f"  len_avg          : {sum(lens) / len(lens):.0f}")
    print(f"  dirty_by_machine : {n_dirty}  ({n_dirty / len(rows) * 100:.1f}%)")
    print(f"  markdown         : {md_path}")
    print(f"  json             : {json_path}")
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
