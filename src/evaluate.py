#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M4 检索评测：Recall@K / MRR@10 / Hit Rate@K，含消融对比。

============================================================================
三个容易搞错的地方（先说清楚，否则数字会误导人）
============================================================================

**① Recall@K 和 Hit Rate@K 在单 gold 场景下是同一个数。**
   评测集每题的 gold 只有 1 个 chunk，所以
       Recall@K = 命中题数 / 总题数 = HitRate@K
   两个都报是**为了对齐术语**（面试官可能问其中任一个），不是两个独立指标。
   只有当 gold 有多个（多跳问答）时二者才会分开。这里明确标注，不假装有两个数。

**② 主指标是 strict（chunk 级），relaxed（条目级）只作解释。**
   - strict：检索结果里必须出现**gold 那一条 chunk**（chunk_id 精确匹配）
   - relaxed：出现**同一个条目**（title 匹配）就算命中
   为什么主指标用 strict：标准答案是**从 gold chunk 里提取**的，
   同条目的**另一条** chunk 未必含答案 —— 用 relaxed 当主指标会虚高。
   relaxed 的用途是**诊断**：strict 低而 relaxed 高 = "方向对了但没精确定位"，
   这是 reranker 该解决的问题；两个都低 = 检索本身就没找到，得先改召回。

**③ 三种检索模式**一次检索就能全部拿到。
   `HybridSearcher.search(mode="hybrid")` 返回的 per_path 里同时含
   向量路和 BM25 路各自的完整排名 —— 所以 vector / bm25 / hybrid
   不用跑三遍（省 3 倍时间），也**保证三路用的是同一批查询向量**。

    ⚠️ 前提：per_path 的深度要够（topn ≥ 池大小）。池子设 100，topn 也得 100。
============================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))

EVAL = PROJECT / "eval"
RESULTS = EVAL / "results"

# 参与对比的四条链路。hybrid_rerank 依赖 --rerank 才会出现。
CONFIGS = ["vector", "bm25", "hybrid", "hybrid_rerank"]
CONFIG_CN = {
    "vector": "纯向量",
    "bm25": "纯 BM25",
    "hybrid": "混合 RRF",
    "hybrid_rerank": "混合 + 重排",
}

KIND_CN = {
    "easy": "基础事实", "hard_anon": "实体匿名", "hard_para": "释义改写",
    "t2s": "繁问简答", "s2t": "简问繁答", "oob": "库外问题",
}


# ==================================================================== 指标
def first_rank(haystack: list, needle) -> int | None:
    """needle 在 haystack 里的 1-based 排名；找不到返回 None。"""
    for i, x in enumerate(haystack):
        if x == needle:
            return i + 1
    return None


def summarize(ranks: list[dict], ks=(1, 5, 10), mrr_k: int = 10) -> dict:
    """
    把一堆 {"strict": rank|None, "relaxed": rank|None} 汇总成指标。

    MRR@K 的口径：排名超出 K 的算 0 分（而不是忽略），
    否则"排到 500 名"会比"完全没找到"得分还高——那是错的。
    """
    n = len(ranks)
    if n == 0:
        return {"n": 0}
    out = {"n": n}
    for tag in ("strict", "relaxed"):
        row = {}
        for k in ks:
            hit = sum(1 for r in ranks if r[tag] is not None and r[tag] <= k)
            row[f"recall@{k}"] = round(hit / n, 4)
        rr = [1.0 / r[tag] if (r[tag] is not None and r[tag] <= mrr_k) else 0.0
              for r in ranks]
        row[f"mrr@{mrr_k}"] = round(sum(rr) / n, 4)
        out[tag] = row
    # 同一份定义，换个常见叫法，方便和外部口径对齐
    out["hit_rate@5"] = out["strict"]["recall@5"]
    return out


# ==================================================================== 主流程
def parse_args():
    ap = argparse.ArgumentParser(description="RAG 检索评测")
    ap.add_argument("--testset", default="qa_testset_v1.jsonl",
                    help="eval/ 下的评测集文件名")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--pool", type=int, default=100,
                    help="召回池大小（同时也是各路的 topn，必须一致，见文件头 ③）")
    ap.add_argument("--rerank", action="store_true", help="开 cross-encoder 重排")
    ap.add_argument("--rerank-batch", type=int, default=32)
    # max_length 是 **(query, passage) 对**的总长上限，不是 passage 单独的长度。
    # 实测 1 token ≈ 1.38 字符（中文 + title/section 里的数字英文），
    # 512 只够约 370 字的 passage，而 gold passage 均值 413 字 → 20.2% 被截。
    # 截断伤的是 R@1（0.762 vs 0.916），不伤 R@5（0.952 vs 0.964）——
    # 见 src/diagnose_rerank_trunc.py 的诊断，以及 §11.13 的实测。
    # 1024 已完全消除截断（最大 pair 718 token），所以默认就开到 1024。
    ap.add_argument("--rerank-max-length", type=int, default=1024,
                    help="重排 (query, passage) 对的 token 上限；1024 可完全消除截断"
                         "（历史默认 512 会截掉 20.2%% 的 gold）")
    # 把"重排候选数"从 --pool 里解耦出来。为什么要解耦：
    #   --pool 同时控 ① 各路召回池 ② 重排输入，两者绑在一起就没法单独量"重排池"。
    #   实测（pool=100, ml=1024）：重排输入 100 → 50，重排延迟 -21% 而指标不降。
    # 默认 0 = 跟随 --pool（保持旧行为，不改变任何既有结论的可复现性）。
    ap.add_argument("--rerank-pool", type=int, default=0,
                    help="喂给重排的候选数；0 = 跟随 --pool。实测 50 相比 100："
                         "重排延迟 -21%%、指标不降（max_length=1024 时）")
    ap.add_argument("--nprobe", type=int, default=512)
    ap.add_argument("--rrf-k", type=int, default=10)
    # ---- 加权融合（量"等权 RRF 在部分子集上是不是负收益"）----
    # 第 11 步发现：匿名子集上纯 BM25(0.812) > 等权混合(0.781)，
    # 即弱路（向量）的候选在稀释强路结果。这里把权重暴露出来做扫描。
    ap.add_argument("--rrf-weights", nargs=2, type=float, default=[1.0, 1.0],
                    metavar=("W_VEC", "W_BM25"),
                    help="加权 RRF 的两路权重（默认 1.0 1.0 = 等权基线）")
    ap.add_argument("--fuse", default="rrf", choices=["rrf", "gate"],
                    help="rrf=加权排名融合；gate=按「BM25 top1 领先幅度」动态调权")
    ap.add_argument("--gate-margin", type=float, default=0.10,
                    help="gate 阈值：BM25 top1 领先次名低于此比例视为弱匹配（仅 --fuse gate）")
    ap.add_argument("--gate-weak-w", type=float, default=0.3,
                    help="gate 判定为弱匹配时，BM25 权重要乘的系数（仅 --fuse gate）")
    ap.add_argument("--gate-strong-w", type=float, default=1.0,
                    help="gate 判定为强匹配（BM25 领先 ≥ margin）时 BM25 权重的系数。"
                         "1.0 = 只有降权分支（原行为）；>1 = 加权的**查询路由**")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（试跑用）")
    ap.add_argument("--include-oob", action="store_true",
                    help="把库外题也算进检索指标（默认排除，它们没有 gold）")
    # ---- 字形归一化消融（量"第 9 步修的那个 bug 值多少"）----
    ap.add_argument("--bm25-index", default=None,
                    help="BM25 索引目录，默认 data/index/bm25")
    ap.add_argument("--no-norm", action="store_true",
                    help="关掉繁简归一化（对照实验，须配 --bm25-index 指向未归一化的旧索引）")
    return ap.parse_args()


def check_norm_pairing(no_norm: bool, bm25_dir: Path):
    """
    拦住"关了一边"的静默失配。

    归一化必须索引侧和查询侧**成对**：只关查询侧（用归一化索引）= 繁体 query 漏词；
    只关索引侧（用未归一化索引 + 归一化查询）= 简体 query 里被归一化过的词也漏词。
    两种都不报错，只是分数悄悄掉 —— 那会得出"归一化没用"的**错误结论**。
    所以这里宁可拒绝启动。
    """
    from zh import TARGET, HAVE_ZHCONV
    meta_p = bm25_dir / "meta.json"
    if not meta_p.exists():
        raise SystemExit(f"[错误] {bm25_dir} 里没有 meta.json")
    idx_norm = json.loads(meta_p.read_text(encoding="utf-8")).get("zh_norm")
    env_norm = TARGET if HAVE_ZHCONV else None
    want = None if no_norm else env_norm
    if idx_norm != want:
        raise SystemExit(
            f"[错误] 归一化开关与索引不匹配：\n"
            f"        --no-norm = {no_norm}  →  期望索引 zh_norm = {want!r}\n"
            f"        实际索引        zh_norm = {idx_norm!r}  ({bm25_dir})\n"
            f"        用错组合会静默漏词，得出的结论是反的。\n"
            f"        未归一化的旧索引若还在 data\\index\\bm25_old_normoff，"
            f"请 --bm25-index 指向它。")
    return idx_norm


def load_testset(name: str) -> list[dict]:
    p = Path(name)
    if not p.is_absolute():
        p = EVAL / name
    if not p.exists():
        raise SystemExit(f"[错误] 评测集不存在：{p}\n"
                         f"先跑：python src\\make_testset.py")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    args = parse_args()
    from generator import get_fetcher          # 按行号回查原文（第 9 步的优化）
    from search_hybrid import HybridSearcher   # 索引只加载一次
    from vectorize import build_text           # 与索引侧同一个拼接函数

    items = load_testset(args.testset)
    n_total = len(items)
    if not args.include_oob:
        items = [it for it in items if it["kind"] != "oob"]
    n_oob = n_total - len(items)
    if args.limit:
        items = items[:args.limit]

    rr_pool = args.rerank_pool or args.pool      # 0 = 跟随 --pool（旧行为）
    print("=" * 78)
    print(f"M4 检索评测 · {args.testset} · tag={args.tag}")
    print("=" * 78)
    print(f"库内题 {len(items)} 条" + (f"（另有 {n_oob} 条库外题，检索指标不适用，已排除）"
                                     if n_oob else ""))
    print(f"召回池 {args.pool} · faiss nprobe={args.nprobe} · RRF k={args.rrf_k}"
          f" · 重排 {'开' if args.rerank else '关'}"
          + (f"（候选 {rr_pool} · max_length={args.rerank_max_length}）"
             if args.rerank else ""))
    print(f"融合 {args.fuse} · w_vec={args.rrf_weights[0]:g} · w_bm25={args.rrf_weights[1]:g}"
          + (f" · gate[margin={args.gate_margin} 弱×{args.gate_weak_w} "
             f"强×{args.gate_strong_w}]" if args.fuse == "gate" else ""))

    # ---- 索引只加载一次 ----
    from pathlib import Path as _P
    bm25_dir = _P(args.bm25_index) if args.bm25_index else (PROJECT / "data" / "index" / "bm25")
    if args.no_norm or args.bm25_index:
        check_norm_pairing(args.no_norm, bm25_dir)

    ns = SimpleNamespace(mode="hybrid", vec_backend="faiss", nprobe=args.nprobe,
                         fuse=args.fuse, gate_margin=args.gate_margin,
                         gate_weak_w=args.gate_weak_w, gate_strong_w=args.gate_strong_w,
                         bm25_index=str(bm25_dir), no_norm=args.no_norm, quiet=True)
    searcher = HybridSearcher(ns)
    fetcher = get_fetcher(searcher.ids)

    reranker = None
    if args.rerank:
        from rerank import Reranker
        reranker = Reranker(verbose=True, max_length=args.rerank_max_length)

    configs = [c for c in CONFIGS if c != "hybrid_rerank" or args.rerank]

    per_query = []
    lat = defaultdict(list)
    t_start = time.time()

    for i, it in enumerate(items, 1):
        q = it["question"]
        gold_cid, gold_title = it["gold_chunk_id"], it["gold_title"]

        t0 = time.time()
        qv = searcher._encode([q])[0]
        t_enc = time.time() - t0

        t0 = time.time()
        rows, _scores, per_path, _detail, (t_vec, t_bm, t_fuse) = searcher.search(
            q, qv, "hybrid", args.pool, args.pool,
            args.rrf_weights[0], args.rrf_weights[1], args.rrf_k)
        t_search = time.time() - t0

        paths = {
            "vector": list(per_path.get("vector", [])),
            "bm25": list(per_path.get("bm25", [])),
            "hybrid": list(rows),
        }

        # 取原文：既给重排用，也给 relaxed 口径拿 title 用
        t0 = time.time()
        got = fetcher.fetch(rows)
        t_fetch = time.time() - t0
        titles = {r: (got.get(searcher.ids[r]) or {}).get("title") for r in rows}

        t_rr = 0.0
        rr_diag = {}
        if reranker is not None:
            # 只在 hybrid 的前 rr_pool 个候选上重排（与 --pool 解耦，见参数说明）
            rr_rows = rows[:rr_pool]
            passages = []
            for r in rr_rows:
                rec = got.get(searcher.ids[r])
                passages.append(build_text(rec["title"], rec["section"], rec["chunk_text"])
                                if rec else "")
            t0 = time.time()
            order = reranker.rerank(q, passages, topk=None, batch_size=args.rerank_batch)
            t_rr = time.time() - t0
            paths["hybrid_rerank"] = [rr_rows[j] for j, _ in order]

            # 截断诊断（query 侧信号，不依赖 gold）：池子里多少候选被截；
            # 以及 gold 本身在不在池里、有没有被截 —— 用于事后定位
            # 「1024 修好了哪几条题」，避免只看整体均值说不清机制。
            flags = reranker.truncation_flags(q, passages)
            gidx = next((k for k, r in enumerate(rr_rows) if searcher.ids[r] == gold_cid), None)
            rr_diag = {"pool_truncated": sum(flags),
                       "gold_in_pool": gidx is not None,
                       "gold_truncated": bool(flags[gidx]) if gidx is not None else None}

        lat["encode"].append(t_enc)
        lat["vec(searcher内)"].append(t_vec)
        lat["bm25(searcher内)"].append(t_bm)
        lat["fuse"].append(t_fuse)
        lat["取原文"].append(t_fetch)
        if reranker is not None:
            lat["重排"].append(t_rr)

        rec = {"qid": it["qid"], "kind": it["kind"], "question": q,
               "gold_title": gold_title, "gold_chunk_id": gold_cid,
               "script_cross": it["script_cross"], "leak_ratio": it.get("leak_ratio"),
               # 查询级置信度信号：BM25 top1 领先次名的幅度（查询路由要用）
               "bm25_lead": (round(searcher.last_lead, 4)
                             if getattr(searcher, "last_lead", None) is not None else None),
               "rerank_diag": rr_diag or None,
               "ranks": {}}
        for name in configs:
            rws = paths[name][:args.pool]
            cids = [searcher.ids[r] for r in rws]
            tls = [titles.get(r) for r in rws]
            rec["ranks"][name] = {
                "strict": first_rank(cids, gold_cid),
                "relaxed": first_rank(tls, gold_title),
                "top1_title": tls[0] if tls else None,
            }
        per_query.append(rec)

        if i % 10 == 0 or i == len(items):
            el = time.time() - t_start
            print(f"  {i:>3d}/{len(items)}  已用 {el:5.1f}s  "
                  f"（预计总 {el/i*len(items):5.1f}s）", flush=True)

    # ---- 汇总 ----
    summary = {c: summarize([r["ranks"][c] for r in per_query]) for c in configs}

    by_kind = {}
    for kind in sorted({r["kind"] for r in per_query}):
        sub = [r for r in per_query if r["kind"] == kind]
        by_kind[kind] = {c: summarize([r["ranks"][c] for r in sub]) for c in configs}

    by_script = {}
    for tag, sel in (("same", [r for r in per_query if not r["script_cross"]]),
                     ("cross", [r for r in per_query if r["script_cross"]])):
        if sel:
            by_script[tag] = {c: summarize([r["ranks"][c] for r in sel]) for c in configs}

    # 按「gold 是否被截断」分组 —— 这是判断 max_length 作用的**机制性**证据：
    # 若某一组在 max_length 调大后明显改善，而另一组纹丝不动，才算真的归因。
    # （只看整体均值无法区分「截断造成伤害」与「长 passage 的题本来就更难」。）
    by_trunc = {}
    if args.rerank:
        for tag, want in (("gold_truncated", True), ("gold_intact", False)):
            sub = [r for r in per_query
                   if (r.get("rerank_diag") or {}).get("gold_truncated") is want]
            if sub:
                by_trunc[tag] = {"n": len(sub),
                                 **{c: summarize([r["ranks"][c] for r in sub]) for c in configs}}

    out = {
        "meta": {"testset": args.testset, "tag": args.tag, "n_queries": len(items),
                 "pool": args.pool, "nprobe": args.nprobe, "rrf_k": args.rrf_k,
                 "fuse": args.fuse, "rrf_weights": list(args.rrf_weights),
                 "gate_margin": args.gate_margin, "gate_weak_w": args.gate_weak_w,
                 "gate_strong_w": args.gate_strong_w,
                 "rerank": args.rerank, "configs": configs,
                 "rerank_max_length": args.rerank_max_length if args.rerank else None,
                 "rerank_pool": rr_pool if args.rerank else None,
                 "no_norm": args.no_norm, "bm25_index": bm25_dir.name,
                 "elapsed_seconds": round(time.time() - t_start, 1)},
        "summary": summary,
        "by_kind": by_kind,
        "by_script": by_script,
        "by_trunc": by_trunc,
        "latency_ms": {k: {"mean": round(sum(v) / len(v) * 1000, 1),
                           "p50": round(sorted(v)[len(v) // 2] * 1000, 1)}
                       for k, v in lat.items()},
        "reranker_stats": reranker.stats() if reranker else None,
        "per_query": per_query,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    outp = RESULTS / f"eval_{args.tag}.json"
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 打印 ----
    print_report(out, configs, by_kind, by_script)
    print(f"\n明细已存：{outp}")
    return 0


def print_report(out, configs, by_kind, by_script):
    sm = out["summary"]

    print("\n" + "=" * 78)
    print("总览（strict 口径 = 命中 gold chunk 本身）")
    print("=" * 78)
    hdr = f"{'链路':<14}{'Recall@1':>10}{'Recall@5':>10}{'Recall@10':>10}{'MRR@10':>9}"
    print(hdr)
    print("-" * 78)
    for c in configs:
        s = sm[c]["strict"]
        print(f"{CONFIG_CN[c]:<14}{s['recall@1']:>10.3f}{s['recall@5']:>10.3f}"
              f"{s['recall@10']:>10.3f}{s['mrr@10']:>9.3f}")

    print("\n宽松口径（relaxed = 命中同一条目即可，用于诊断）")
    print("-" * 78)
    print(hdr)
    for c in configs:
        s = sm[c]["relaxed"]
        print(f"{CONFIG_CN[c]:<14}{s['recall@1']:>10.3f}{s['recall@5']:>10.3f}"
              f"{s['recall@10']:>10.3f}{s['mrr@10']:>9.3f}")

    print("\n" + "=" * 78)
    print(f"分题型 · Recall@5（strict）  n={sm[configs[0]]['n']}")
    print("=" * 78)
    print(f"{'题型':<12}{'n':>5}" + "".join(f"{CONFIG_CN[c]:>14}" for c in configs))
    print("-" * 78)
    for kind in sorted(by_kind, key=lambda k: -by_kind[k][configs[0]]["n"]):
        row = by_kind[kind]
        n = row[configs[0]]["n"]
        print(f"{KIND_CN.get(kind, kind):<12}{n:>5}"
              + "".join(f"{row[c]['strict']['recall@5']:>14.3f}" for c in configs))

    if by_script:
        print("\n" + "=" * 78)
        print("字形分组 · Recall@5（strict）")
        print("=" * 78)
        print(f"{'字形':<12}{'n':>5}" + "".join(f"{CONFIG_CN[c]:>14}" for c in configs))
        print("-" * 78)
        for tag, cn in (("same", "同字形"), ("cross", "跨字形")):
            if tag not in by_script:
                continue
            row = by_script[tag]
            n = row[configs[0]]["n"]
            print(f"{cn:<12}{n:>5}"
                  + "".join(f"{row[c]['strict']['recall@5']:>14.3f}" for c in configs))
        print("\n⚠️ 跨字形组是**归一化生效后**的结果。要量出归一化的价值，")
        print("   需另跑 --tag no_norm 并对比 —— 见 第10步-任务清单.md")

    print("\n" + "=" * 78)
    print("延迟（每题，毫秒）")
    print("=" * 78)
    for k, v in out["latency_ms"].items():
        print(f"  {k:<18} 均值 {v['mean']:>8.1f}  中位 {v['p50']:>8.1f}")
    if out.get("reranker_stats"):
        rs = out["reranker_stats"]
        print(f"  reranker: 加载 {rs['load_seconds']}s · "
              f"{rs['total_pairs']} 对 · {rs['ms_per_pair']} ms/对")

    print_signal(out)


def print_signal(out):
    """
    查询级信号诊断：**「BM25 top1 领先幅度」能不能预测"哪条路更强"**。

    这是查询路由的前提。如果这个信号和"谁赢"没关系，路由就无从做起；
    如果关系强，就可以按它逐题选权重（11.11 已证明全局权重救不了）。

    判据要诚实：这里量的是"信号 vs 单路谁更好"，不是"信号 vs 最终指标"。
    """
    rows = [r for r in out["per_query"] if r.get("bm25_lead") is not None]
    if not rows:
        return
    INF = 10 ** 9

    def med(v):
        v = sorted(v)
        return v[len(v) // 2] if v else None

    print("\n" + "=" * 78)
    print("查询级信号 · BM25 top1 领先次名的幅度（中位）")
    print("=" * 78)
    print(f"{'题型':<12}{'n':>5}{'中位领先':>12}")
    print("-" * 78)
    for kind in sorted({r["kind"] for r in rows}):
        sub = [r["bm25_lead"] for r in rows if r["kind"] == kind]
        print(f"{KIND_CN.get(kind, kind):<12}{len(sub):>5}{med(sub):>11.1%}")

    # BM25 领先幅度 vs "哪条路把 gold 排得更前"
    groups = {"BM25 更好": [], "向量更好": [], "打平": []}
    for r in rows:
        b = r["ranks"]["bm25"]["strict"] or INF
        v = r["ranks"]["vector"]["strict"] or INF
        key = "打平" if b == v else ("BM25 更好" if b < v else "向量更好")
        groups[key].append(r["bm25_lead"])
    print("\n按「单路谁把 gold 排得更前」分组，看领先幅度是否有区分度：")
    print(f"{'组':<12}{'n':>5}{'中位领先':>12}")
    print("-" * 78)
    for k, v in groups.items():
        if v:
            print(f"{k:<12}{len(v):>5}{med(v):>11.1%}")

    # 规则准确率：lead >= 门限 就赌 BM25 更好
    print("\n若拿它当路由规则（领先 ≥ 门限 → 赌 BM25 更好）：")
    print(f"{'门限':>8}{'赌BM25':>9}{'命中':>8}{'准确率':>10}")
    print("-" * 78)
    for thr in (0.0, 0.05, 0.10, 0.20, 0.30, 0.50):
        picked = [r for r in rows if r["bm25_lead"] >= thr]
        if not picked:
            continue
        hit = sum(1 for r in picked
                  if (r["ranks"]["bm25"]["strict"] or INF)
                  <= (r["ranks"]["vector"]["strict"] or INF))
        print(f"{thr:>8.2f}{len(picked):>9}{hit:>8}{hit / len(picked):>9.1%}")


if __name__ == "__main__":
    sys.exit(main())
