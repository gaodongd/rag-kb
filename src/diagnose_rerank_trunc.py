#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重排截断诊断 —— 只加载 tokenizer，不加载模型、不加载索引（跑完约 20 秒）。

============================================================================
为什么需要这个诊断（它是"要不要改 max_length"的判据，不是结果本身）
============================================================================
bge-reranker-v2-m3 的 `max_length=512` 是 **(query, passage) 对**的总长上限，
不是 passage 单独的长度。可用给 passage 的部分是：

    max_length - len(query_tokens) - 3        # [CLS] q [SEP] d [SEP]

chunk 拼接后（title｜section｜正文）p90=674 字符、18.6% 超 512 ——
也就是说这批 passage **相当一部分会被截掉尾巴**。
gold 信息如果恰好在尾部，重排模型根本看不见它，会把它排到后面去。

⚠️ 为什么不能用「字符数」估：中文虽近似 1 字 1 token，
   但 BGE 的 tokenizer 会把连续数字（如 1787、141178）和英文词组并成更少的 token，
   实测偏差可达 20%。而"要不要处理"的判据正好卡在这个偏差量级上 ——
   估算出来的结论可能是反的。必须用真实 tokenizer。

============================================================================
三个问题，按重要性排序
============================================================================
① 有多少条 gold 会被截断？（决定实验规模：<5% 就不值得跑）
② 被截断的题，重排后的排名是否更差？（决定"截断是否真的造成伤害"
   —— 被截的题可能本来就简单，排名照样好，那就白担心了）
③ 各题型（easy / hard_anon / hard_para / 字形交叉）的截断率差异

输出写到 stdout，可直接 tee 成报表。
============================================================================
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))

KIND_CN = {"easy": "基础事实", "hard_anon": "实体匿名",
           "hard_para": "释义改写", "t2s": "繁问", "s2t": "简问", "oob": "库外"}


def load_gold_rows(ids_path: Path, gold_ids: set[str]) -> dict[str, int]:
    """
    从 ids.txt 里找出 gold_id 对应的**行号**。

    为什么这样写：ids.txt 有 330 万行（112 MB）。建完整 dict 要 ~500 MB 内存，
    但这里只需要 104 条 —— 逐行比对 set，内存 O(目标数)。
    （想建全表映射时再用 dict，别为了省事把 330 万条都塞进内存。）
    """
    found: dict[str, int] = {}
    with open(ids_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            cid = line.rstrip("\n")
            if cid in gold_ids:
                found[cid] = i
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description="重排截断诊断（零成本）")
    ap.add_argument("--testset", default=str(PROJECT / "eval" / "qa_testset_v1.jsonl"))
    ap.add_argument("--eval-json", default=str(PROJECT / "eval" / "results" / "eval_v1_final.json"),
                    help="交叉分析用的既有评测结果（用它看截断与排名的关系）")
    ap.add_argument("--index", default=str(PROJECT / "data" / "index" / "bge-large-zh-v1.5"))
    ap.add_argument("--max-lengths", type=int, nargs="+", default=[512, 1024],
                    help="要评估的 max_length 取值")
    ap.add_argument("--dump", default=None,
                    help="把「每条题在指定 max_length 下是否被截断」导出成 JSON，"
                         "供 report_rerank.py 分组用（**唯一真源**，避免报表自己再算一遍）")
    ap.add_argument("--dump-max-length", type=int, default=512)
    args = ap.parse_args()

    # ---- 1. 评测集（只用库内题：库外题没有 gold） ----
    items = [json.loads(l) for l in open(args.testset, encoding="utf-8") if l.strip()]
    in_kb = [it for it in items if it["kind"] != "oob"]
    print(f"评测集 {Path(args.testset).name}：共 {len(items)} 条，其中库内（有 gold）{len(in_kb)} 条")

    gold_ids = {it["gold_chunk_id"] for it in in_kb}

    # ---- 2. 行号定位 ----
    ids_path = Path(args.index) / "ids.txt"
    print(f"读 {ids_path.name} 定位 gold 行号…", flush=True)
    row_of = load_gold_rows(ids_path, gold_ids)
    miss = gold_ids - set(row_of)
    if miss:
        print(f"⚠️ {len(miss)} 条 gold_chunk_id 在 ids.txt 里找不到：{sorted(miss)[:3]}")
    print(f"定位到 {len(row_of)}/{len(gold_ids)} 条\n", flush=True)

    # ---- 3. 取 gold 原文 ----
    # 不用 generator.RowTextFetcher：它会断言「parquet 行数 == id_list 长度」，
    # 而我们这里只构造了稀疏列表（只填 gold 所在的行号），过不了那个断言。
    # 改成复用同一套「按行号定分片 + 按 chunk_id 查」逻辑，但只查目标行。
    from search_hybrid import PARQUET          # 它才是 PARQUET 的定义处
    from search_vector import fetch_texts
    from vectorize import build_text

    id_list = [None] * (max(row_of.values()) + 1)   # fetcher 只在下标处索引
    for cid, r in row_of.items():
        id_list[r] = cid

    import pyarrow.dataset as ds_mod
    import pyarrow.parquet as pq

    parts, cum = [], 0
    for f in sorted(ds_mod.dataset(str(PARQUET), format="parquet").files):
        n = pq.ParquetFile(f).metadata.num_rows
        parts.append((Path(f), cum, cum + n))
        cum += n

    def locate(row: int) -> Path:
        for f, lo, hi in parts:
            if lo <= row < hi:
                return f
        raise IndexError(row)

    got: dict[str, dict] = {}
    by_part: dict[Path, list[int]] = {}
    for cid, r in row_of.items():
        by_part.setdefault(locate(r), []).append(r)
    for f, rows in by_part.items():
        got.update(fetch_texts(f, [id_list[r] for r in rows]))
    print(f"取回 {len(got)} 条 gold 原文（分片 {len(parts)} 个）\n", flush=True)

    # ---- 4. tokenizer ----
    from transformers import AutoTokenizer
    from rerank import find_ms_model
    tok_dir = find_ms_model("bge-reranker-v2-m3")
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    print(f"tokenizer: {tok_dir.name}\n")

    # ---- 5. 逐题算长度 ----
    rows = []
    for it in in_kb:
        rec = got.get(it["gold_chunk_id"])
        if rec is None:
            continue
        passage = build_text(rec["title"], rec["section"], rec["chunk_text"])
        qn = len(tok.encode(it["question"], add_special_tokens=False))
        dn = len(tok.encode(passage, add_special_tokens=False))
        rows.append({
            "qid": it["qid"], "kind": it["kind"], "script_cross": it.get("script_cross"),
            "q_tokens": qn, "p_tokens": dn, "p_chars": len(passage),
            # XLMRoberta 模板 `<s> q </s></s> d </s>` → 4 个特殊 token（不是 BERT 的 3 个）。
            # 实测核对：qn=15, dn=488 → 真实 507 = 15+488+4。记成 +3 会差 1 个 token。
            "pair_tokens": qn + dn + 4,
        })

    print("=" * 78)
    print("① passage（title｜section｜正文）token 长度分布")
    print("=" * 78)
    P = [r["p_tokens"] for r in rows]
    Q = [r["q_tokens"] for r in rows]
    print(f"  passage : 均值 {st.mean(P):6.1f}  中位 {st.median(P):6.1f}  "
          f"p90 {sorted(P)[int(.9*len(P))]:5d}  p99 {sorted(P)[int(.99*len(P))]:5d}  最大 {max(P)}")
    print(f"  query   : 均值 {st.mean(Q):6.1f}  中位 {st.median(Q):6.1f}  最大 {max(Q)}")
    print(f"  字符/token 比：{st.mean([r['p_chars'] for r in rows]) / st.mean(P):.3f}"
          f"  （看中文是不是真的 1 字 1 token）")

    print("\n" + "=" * 78)
    print("② 各 max_length 下的截断率")
    print("=" * 78)
    print(f"{'max_length':>11} {'total':>7} {'截断条数':>9} {'截断率':>8} {'可用passage长度(中位)':>22}")
    trunc_by_ml: dict[int, set[str]] = {}
    for ml in args.max_lengths:
        tr = [r for r in rows if r["pair_tokens"] > ml]
        trunc_by_ml[ml] = {r["qid"] for r in tr}
        avail = [ml - r["q_tokens"] - 3 for r in rows]
        print(f"{ml:>11} {len(rows):>7} {len(tr):>9} {len(tr)/len(rows):>7.1%} "
              f"{st.median(avail):>22.0f}")

    # ---- 6. 交叉既有评测：截断的题排名是否更差 ----
    ev_path = Path(args.eval_json)
    if not ev_path.exists():
        print(f"\n（跳过交叉分析：{ev_path.name} 不存在）")
        return 0
    ev = json.load(open(ev_path, encoding="utf-8"))
    rank = {}
    for q in ev["per_query"]:
        hr = q.get("ranks", {}).get("hybrid_rerank")
        if hr:
            rank[q["qid"]] = hr["strict"]
    print(f"\n交叉 {ev_path.name}（tag={ev['meta']['tag']}，"
          f"pool={ev['meta']['pool']}）的 hybrid_rerank strict 排名")

    print("\n" + "=" * 78)
    print("③ 截断 vs 未截断：重排后排名是否更差（关键判据）")
    print("=" * 78)
    INF = 10 ** 9

    def stats(qids):
        # strict 排名：0 表示"池子里没找到"，换算成 INF（没命中）
        raw = [rank[q] or INF for q in qids if q in rank]
        if not raw:
            return None
        n = len(raw)
        return {
            "n": n,
            "R@1": sum(1 for x in raw if x == 1) / n,
            "R@5": sum(1 for x in raw if x <= 5) / n,
            "MRR": sum(1 / x if x < INF else 0 for x in raw) / n,
            "miss": sum(1 for x in raw if x >= INF),
        }

    for ml in args.max_lengths:
        tr = trunc_by_ml[ml]
        rows_tr = [r for r in rows if r["qid"] in tr]
        s_tr, s_ok = stats(tr), stats([r["qid"] for r in rows if r["qid"] not in tr])
        print(f"\n  【max_length={ml}】")
        for label, s in (("gold 被截断", s_tr), ("gold 未截断", s_ok)):
            if s is None:
                continue
            print(f"    {label:<12} n={s['n']:>3}  R@1={s['R@1']:.3f}  R@5={s['R@5']:.3f}  "
                  f"MRR={s['MRR']:.3f}  完全没进池 {s['miss']}")
        if s_tr and s_ok:
            d = s_ok["R@5"] - s_tr["R@5"]
            print(f"    → R@5 差值（未截断 − 截断）= {d:+.3f}"
                  f"  {'✅ 截断确有伤害' if d > 0.05 else '⚠️ 差异小，截断可能不是瓶颈'}")

    # ---- 7. 分题型 ----
    print("\n" + "=" * 78)
    print("④ 各题型的截断率（看是否集中在某类题上）")
    print("=" * 78)
    print(f"{'题型':<12}{'n':>5}" + "".join(f"{'截断@'+str(ml):>14}" for ml in args.max_lengths))
    for k in ["easy", "hard_anon", "hard_para", "t2s", "s2t"]:
        sub = [r for r in rows if r["kind"] == k]
        if not sub:
            continue
        cells = "".join(
            f"{sum(1 for r in sub if r['pair_tokens'] > ml)/len(sub):>13.1%} " for ml in args.max_lengths)
        print(f"{KIND_CN.get(k,k):<12}{len(sub):>5}{cells}")

    print("\n" + "=" * 78)
    print("⑤ 最长的 10 条 gold（截断风险最高，可逐条看 gold 信息在不在尾部）")
    print("=" * 78)
    for r in sorted(rows, key=lambda x: -x["pair_tokens"])[:10]:
        print(f"  {r['pair_tokens']:>5} tok (q{r['q_tokens']:>3}+d{r['p_tokens']:>5})  "
              f"{KIND_CN.get(r['kind'],r['kind']):<6} rank@rerank="
              f"{rank.get(r['qid']) or '≥100'}")

    # ---- 8. 导出截断标记（给 report_rerank.py 当唯一真源用） ----
    if args.dump:
        dump = {
            "model": tok_dir.name,
            "tokenizer": type(tok).__name__,
            "special_tokens": 4,          # XLMRoberta `<s>q</s></s>d</s>`
            "max_length": args.dump_max_length,
            "per_query": {r["qid"]: {
                "gold_truncated": r["pair_tokens"] > args.dump_max_length,
                "q_tokens": r["q_tokens"],
                "p_tokens": r["p_tokens"],
                "pair_tokens": r["pair_tokens"],
                "kind": r["kind"],
            } for r in rows},
        }
        p = Path(args.dump)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
        n_tr = sum(1 for v in dump["per_query"].values() if v["gold_truncated"])
        print(f"\n✅ 截断标记已导出：{p}")
        print(f"   max_length={args.dump_max_length} · 截断 {n_tr}/{len(rows)} 条 "
              f"· 供 report_rerank.py 分组（避免报表自己再算一遍、口径跑偏）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
