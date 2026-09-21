#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 7 步 · 向量化：chunks.parquet → bge 向量 + id 映射。

用哪个 Python
-------------
**Python 3.13**（要 torch / FlagEmbedding）。注意：3.11 里没装 torch，
所以这一步**不能**像第 5 步那样交给 3.11 —— 但反过来，读 parquet 也不需要 Spark：
3.13 里有 pyarrow，`pyarrow.dataset` 直接读 Spark 输出的目录即可。
**结论：第 7 步全程只用 3.13，不需要跨 Python，也不需要 winutils。**

待编码文本怎么拼（这一步最容易被做错的地方）
--------------------------------------------
    title ｜ section ｜ chunk_text

* `section` 是小节名（如 `媒体`），**必须拼进去**。实测例子：
  `陕西省历史文化名村 / 子长市` 那块正文里没有一个字提到「子长市」，
  全靠 `section` 把父标题上下文补回来。丢了它，title-aware 切块白做。
* 分隔符用**全角竖线 `｜`**：中文维基正文里几乎不出现，不会被误认成内容。
* `section` 为空（短条目导语段，占 21.5%）时退化成 `title ｜ chunk_text`，
  不留空段造成 `title ｜  ｜ text` 这种双分隔。

产物（--out 目录下）
--------------------
    emb.npy        float16 矩阵 (N, 1024)，**已 L2 归一化**（内积=余弦）
    ids.txt        每行一个 chunk_id，**行号 == emb.npy 的行号**（回查用）
    meta.json      模型/维度/耗时/吞吐/拼串格式/截断统计
    progress.json  进度快照（**仅供参考，续跑不读它** —— 每 10 万行才落盘，天然滞后）

为什么必须能续跑
----------------
全量 331.6 万块用 bge-large 在 4060 上要跑好几个小时。
**没有续跑能力的脚本，一次断电/一次误触全废** —— 这种长任务必须能断点续做。
（真发生过：跑到 20.8% 时用户关机，进程随关机结束，日志里一个报错都没有。）

`--resume` **数 ids.txt 的行数**当起点，跳过已编码的行（仍会扫过数据，但不重复过 GPU）。
不能读 progress.json：它滞后，拿它续跑会重复 append → ids.txt 与 emb.npy 错位，
而且全程不报错，直到检索时才发现"相似度很高的块内容对不上"。
选 ids.txt 当基准的理由见正文第 3 节的注释。

用法
----
    :: 先小样本验通（约 1 分钟）
    "...Python313\\python.exe" src\\vectorize.py --limit 5000 --out eval\\results\\vec_smoke

    :: 确认无误再全量（可随时 Ctrl+C，之后 --resume 接着跑）
    "...Python313\\python.exe" src\\vectorize.py --resume
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

SEP = "｜"                       # 全角竖线
MODEL_ROOT = Path(r"E:\AI-learning\ms-cache\models")


def find_model(kind: str) -> Path:
    """在 ModelScope 缓存里找本地模型目录（结构：models/<组织>--<名>/snapshots/master）。"""
    pat = f"bge-{kind}-zh"
    for p in MODEL_ROOT.glob(f"*{pat}*"):
        for cand in (p / "snapshots" / "master", p):
            if (cand / "config.json").exists():
                return cand
    raise FileNotFoundError(
        f"找不到 {pat} 的本地目录（在 {MODEL_ROOT} 下）。"
        f"先跑 src/download_models.py 下载。")


def build_text(title: str, section: str, text: str) -> str:
    """标题 + 小节 + 正文。section 为空时不产生空段。"""
    head = f"{title}{SEP}{section}" if section else f"{title}"
    return f"{head}{SEP}{text}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(PROJECT / "data" / "processed" / "chunks.parquet"))
    ap.add_argument("--out", default=str(PROJECT / "data" / "index" / "bge-large-zh-v1.5"))
    ap.add_argument("--model", default=None, help="本地模型目录；默认自动找 bge-large-zh")
    ap.add_argument("--model-kind", default="large", choices=["large", "small"])
    ap.add_argument("--limit", type=int, default=0, help="只编码前 N 行（0=全量）；小样本验通用")
    ap.add_argument("--read-batch", type=int, default=2048, help="每次从 parquet 取多少行")
    ap.add_argument("--batch", type=int, default=64, help="送 GPU 的批大小")
    ap.add_argument("--max-length", type=int, default=512,
                    help="BGE 上限就是 512，改大无效（超长会被截断）")
    ap.add_argument("--token-stats", type=int, default=20000,
                    help="对前 N 条统计 token 长度与截断率（0=关）")
    ap.add_argument("--log-every", type=int, default=20000)
    ap.add_argument("--resume", action="store_true",
                    help="从 ids.txt 的行数接着跑（不读 progress.json，原因见文件头）")
    args = ap.parse_args()

    import pyarrow.dataset as ds_mod

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    emb_path = out / "emb.npy"
    ids_path = out / "ids.txt"
    meta_path = out / "meta.json"
    prog_path = out / "progress.json"

    # ---- 1. 数行数（读 footer，不扫数据） ----
    dataset = ds_mod.dataset(args.input, format="parquet")
    total_rows = dataset.count_rows()
    N = total_rows if args.limit <= 0 else min(args.limit, total_rows)
    print("=" * 68)
    print("第 7 步 · 向量化")
    print("=" * 68)
    print(f"输入      : {args.input}")
    print(f"parquet   : {total_rows:,} 行  →  本次编码 {N:,} 行"
          + (f"（--limit {args.limit}）" if args.limit > 0 else "（全量）"))
    print(f"输出      : {out}")

    # ---- 2. 模型与维度 ----
    model_path = Path(args.model) if args.model else find_model(args.model_kind)
    cfg = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    dim = int(cfg.get("hidden_size", 1024))
    print(f"模型      : {model_path.name}  dim={dim}  "
          f"max_pos={cfg.get('max_position_embeddings', '?')}")

    # ---- 3. 续跑：确定起点 ----
    #
    # ⚠️ 起点必须数 ids.txt 的行数，**不能**读 progress.json。
    #    原因：progress.json 每 log_every*5 行才落一次盘（默认 10 万行），
    #    关机那一刻它必然滞后。实测：ids.txt 有 689,687 行，progress.json 只记 601,827。
    #    拿 601,827 续跑会把这 8.8 万行再 append 一遍 → ids.txt 与 emb.npy 行号错位，
    #    而且**不报任何错**，直到第 8 步按 id 回查全文才发现内容全对不上。
    #
    #    为什么 ids.txt 是可靠的地面真值？因为主循环是「先写 emb、后写 ids」，
    #    所以 ids.txt 的行数永远不会超过 emb 里已写好的行数（最多落后一批）。
    done = 0
    if args.resume:
        emb_ok = False
        if emb_path.exists():
            try:
                probe = np.load(emb_path, mmap_mode="r")
                emb_ok = (probe.shape == (N, dim))
                del probe
            except Exception as e:
                print(f"[resume]  emb.npy 不可用（{type(e).__name__}），从头跑")
        if emb_ok and ids_path.exists():
            raw = ids_path.read_bytes()
            if raw and not raw.endswith(b"\n"):
                # 关机/断电可能把最后一行写了一半。宁可丢掉这半行，
                # 也不能留一行残缺 id —— 那会污染第 8 步的 id→原文映射。
                raw = raw[: raw.rfind(b"\n") + 1]
                ids_path.write_bytes(raw)
                print("[resume]  ids.txt 末行不完整（疑似断电截断），已丢弃")
            done = raw.count(b"\n")
            print(f"[resume]  起点 = ids.txt 行数 {done:,} / {N:,}"
                  f"（{done / N * 100:.1f}%）")
        else:
            print("[resume]  无可用续跑产物（emb.npy / ids.txt），从头跑")

    # ---- 4. 预分配 memmap：331 万 × 1024 × 2B ≈ 6.8GB，绝不能放内存 ----
    if done == 0 or not emb_path.exists():
        emb = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float16,
                                        shape=(N, dim))
        ids_mode = "w"
    else:
        emb = np.lib.format.open_memmap(emb_path, mode="r+")
        ids_mode = "a" if ids_path.exists() else "w"

    if done == 0:
        print(f"预分配    : emb.npy  {N:,} × {dim} × fp16 ≈ "
              f"{N * dim * 2 / 1024 ** 3:.2f} GB")

    # ---- 5. 加载模型 ----
    import torch
    from FlagEmbedding import FlagModel
    print(f"加载模型  : torch {torch.__version__} | CUDA {torch.cuda.is_available()}"
          + (f" | {torch.cuda.get_device_properties(0).name}" if torch.cuda.is_available() else ""))
    model = FlagModel(str(model_path), use_fp16=True)

    # tokenizer 用来量"有多少块会被 512 截断"——这个数不测出来，
    # 后面检索效果差你都不知道是模型问题还是截断问题。
    tok = None
    tok_lens = []
    if args.token_stats > 0:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(model_path))

    # ---- 6. 流式编码 ----
    t0 = time.time()
    pos = 0                 # 已遍历行数
    n_done = done           # 已写入行数
    seen_sec = 0
    last_log = time.time()
    trunc_over = 0
    with open(ids_path, ids_mode, encoding="utf-8") as f_ids:
        scanner = dataset.scanner(batch_size=args.read_batch,
                                  columns=["chunk_id", "title", "section", "chunk_text"])
        for rb in scanner.to_batches():
            titles = rb.column("title").to_pylist()
            sections = rb.column("section").to_pylist()
            texts = rb.column("chunk_text").to_pylist()
            cids = rb.column("chunk_id").to_pylist()

            texts = [build_text(t or "", s or "", x or "")
                     for t, s, x in zip(titles, sections, texts)]
            for s in sections:
                if s:
                    seen_sec += 1

            # token 长度统计（只测前 N 条，CPU 上很快）
            if tok is not None and len(tok_lens) < args.token_stats:
                room = args.token_stats - len(tok_lens)
                enc = tok(texts[:room], add_special_tokens=True, truncation=False)
                lens = [len(x) for x in enc["input_ids"]]
                tok_lens.extend(lens)
                trunc_over += sum(1 for x in lens if x > args.max_length)

            # 每行都要推进 pos，但只有 pos >= done 的行才写 memmap（续跑跳过）
            if pos + len(texts) <= done:
                pos += len(texts)
                continue

            # ⚠️ 最后一批必须切齐到 N：否则 emb[pos:pos+len] 会越界，
            #    numpy 直接抛 "could not broadcast"（小样本 --limit 时必踩）
            if pos + len(texts) > N:
                k = N - pos
                texts, cids = texts[:k], cids[:k]

            # ⚠️ 别传 show_progress_bar=False：新版 FlagEmbedding 会把它透传给
            #    tokenizer.pad()，直接报 TypeError: unexpected keyword argument。
            #    （tqdm 进度条走 stderr 原地刷新，重定向到日志后不会淹掉正文）
            vecs = model.encode(texts, batch_size=args.batch,
                                max_length=args.max_length)
            vecs = np.asarray(vecs, dtype=np.float16)

            emb[pos:pos + len(vecs)] = vecs
            # ⚠️ 续跑时这一批很可能跨越起点（done 不是 read_batch 的整数倍）：
            #    只 append 起点之后的部分。整批都写的话 ids.txt 会比 emb.npy 多出
            #    几十行，且 emb 那几行是重复覆盖（内容一样，看不出问题），
            #    于是错位被完美隐藏 —— 只在检索时表现为"相似度高的块内容对不上"。
            skip = max(0, done - pos)
            f_ids.write("\n".join(cids[skip:]) + "\n")
            pos += len(vecs)
            n_done = pos

            # 日志 + 检查点
            if n_done % args.log_every < len(vecs):
                el = time.time() - t0
                tps = (n_done - done) / el if el > 0 else 0
                eta = (N - n_done) / tps / 60 if tps > 0 else 0
                print(f"  {n_done:>10,}/{N:,}  {n_done / N * 100:5.1f}%  "
                      f"{tps:7.1f} chunk/s  已用 {el / 60:5.1f} 分  "
                      f"剩余约 {eta:5.1f} 分")
            if n_done % (args.log_every * 5) < len(vecs):
                emb.flush()
                prog_path.write_text(json.dumps(
                    {"done": int(n_done), "total": int(N), "dim": dim,
                     "updated": time.strftime("%Y-%m-%d %H:%M:%S")},
                    ensure_ascii=False, indent=2), encoding="utf-8")

            if pos >= N:
                break

    emb.flush()
    dt = time.time() - t0

    # ---- 6.5 接缝自检（续跑专属） ----
    # 只查开头 1000 行是查不出续跑问题的 —— 那部分是上一轮写的，本来就没问题。
    # 行号错位唯一会发生的点是 done 那道缝，所以直接查缝两侧。
    n_ids_lines = ids_path.read_bytes().count(b"\n") if ids_path.exists() else 0
    seam = {}
    if done > 0:
        lo, hi = max(0, done - 3), min(n_done, done + 3)
        s = np.asarray(emb[lo:hi]).astype(np.float32)
        sn = np.linalg.norm(s, axis=1)
        seam = {"rows": f"{lo}~{hi}", "norm_min": round(float(sn.min()), 4),
                "norm_max": round(float(sn.max()), 4)}
        print(f"  接缝自检      : 第 {lo:,}~{hi:,} 行  范数 "
              f"[{sn.min():.4f}, {sn.max():.4f}]")
    print(f"  ids.txt 行数  : {n_ids_lines:,}   (必须 == 已编码 {n_done:,})")
    if n_ids_lines != n_done:
        print("  🔴 行数不一致 → ids.txt 与 emb.npy 已错位，第 8 步回查全文会全错！")
        return 3

    # ---- 7. 自检（跑完立刻验一次，别等下游报错） ----
    checked = min(1000, n_done)
    probe = np.asarray(emb[:checked])
    norms = np.linalg.norm(probe.astype(np.float32), axis=1)
    n_nan = int(np.isnan(probe.astype(np.float32)).sum())

    tok_summary = {}
    if tok_lens:
        a = np.array(tok_lens)
        tok_summary = {
            "checked": int(a.size),
            "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)),
            "max": int(a.max()),
            "over_max_length": int(trunc_over),
            "over_pct": round(trunc_over / a.size * 100, 3),
        }

    meta = {
        "input": args.input,
        "total_rows_in_parquet": int(total_rows),
        "encoded": int(n_done),
        "model_dir": str(model_path),
        "dim": dim,
        "dtype": "float16",
        "normalized": True,
        "text_format": "title｜section｜chunk_text（section 空则省略该段；分隔符=全角竖线）",
        "max_length": args.max_length,
        "gpu_batch": args.batch,
        "section_filled_pct": round(seen_sec / max(pos, 1) * 100, 2),
        "elapsed_sec": round(dt, 1),
        "chunks_per_sec": round((n_done - done) / dt, 1) if dt > 0 else 0,
        "self_check": {
            "checked": checked,
            "norm_min": round(float(norms.min()), 4),
            "norm_max": round(float(norms.max()), 4),
            "nan_count": n_nan,
        },
        "token_stats": tok_summary,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 68)
    print("VECTORIZE_OK")
    print(f"  已编码        : {n_done:,}")
    print(f"  耗时 / 吞吐   : {dt:.1f}s  /  {meta['chunks_per_sec']:.1f} chunk/s")
    print(f"  自检范数      : [{norms.min():.4f}, {norms.max():.4f}]  "
          f"（应≈1.0）  NaN={n_nan}")
    if tok_summary:
        print(f"  token 长度    : P50 {tok_summary['p50']:.0f} / P90 {tok_summary['p90']:.0f} "
              f"/ max {tok_summary['max']}")
        print(f"  ⚠️ 超 {args.max_length} 被截断: "
              f"{tok_summary['over_max_length']:,}/{tok_summary['checked']:,} "
              f"= {tok_summary['over_pct']}%")
    print(f"  产物          : {emb_path.name} / {ids_path.name} / {meta_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
