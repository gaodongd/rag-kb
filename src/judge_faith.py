#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Faithfulness 的 LLM 判官（§11.14）
==================================

为什么词面判据不够（先看结论，再决定要不要上 LLM）
--------------------------------------------------
`src/calibrate_faith.py` 用 18 条人工标注（含 4 条**合成对照**）量过：
  · 2-gram 覆盖率：最好也只有 72%（阈值 0.3），阈值 0.6 时只有 33%
  · 实词覆盖率：最好 72%
  · 更致命的是**漏检方向**：合成对照「永恆帝國由瓦爾族人所建立[1]」
    与原文用词几乎全同（实词覆盖 1.00），只是把"建立者"说反了 ——
    词面判据判它「有依据」。**用词全对、关系说反**正是最该抓的一类幻觉。

结论：词面判据只能当**下界**用，真值要靠 LLM 判官。
这不算"换个模型碰运气"—— 它同样有验收标准：
`--calibrate` 会拿**同一套人工标签**量它的一致率，达不到就不能用。

设计上的几个决定
----------------
1. **按答案整批判，不按句子单发。**
   一个答案的引用块和上下文是共享的，分批发既省钱又给判官完整语境。
   要求它按给定的编号逐条回判，编号由我们指定 —— 这样结果能精确对回子句。
2. **把"允许的宽容"写进 prompt。**
   同义改写、繁简差异、单位换算（公分/厘米）、概括归纳都算"有依据"；
   无依据有三种：原文没有、与原文矛盾、模型自己的评价推断。
   不写清楚，判官会把一切改写都判成幻觉（这正是词面判据的病）。
3. **结果落盘 + 缓存。** 340 个子句的判定要能复用，重跑不花钱。
4. **合成对照进校准集。** 判官必须先过对照样本，否则它的"通过"是假的
   （第 9 步的教训：验证工具自己会骗人）。

用法
----
    :: 先校准（只判人工标注过的那些答案，约 20 次调用）
    "...python313\\python.exe" src\\judge_faith.py --calibrate

    :: 通过后跑全量（约 106 次调用）
    "...python313\\python.exe" src\\judge_faith.py ^
        --input eval\\results\\gen_dashscope_testset_rr.jsonl --tag rr
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))

from evaluate_gen import (CITE_RE, context_texts, faithfulness,  # noqa: E402
                          is_struct_clause, norm_verdict)
from cloud_models import DEFAULT_CLOUD_MODEL                      # noqa: E402
from zh import norm_text                                          # noqa: E402

RESULTS = PROJECT / "eval" / "results"

# 判官 verdict 归一化统一走 evaluate_gen.norm_verdict —— 只认 yes/no 是不够的：
# 判官偶尔用中文回"有依据/无依据"（实测 20 个子句），语义等价却会被静默丢弃。
_norm = norm_verdict

SYSTEM = """你是严格的中文事实核查员。给定【参考资料】和一份【模型答案】，你要逐句判断答案里的每一句话能不能由参考资料支持。

判断标准：
- 判定「有依据」的情况：参考资料里有对应的事实表述。**允许**同义改写、繁简字形差异、单位换算（如 公分/厘米）、把若干句原文概括成一句。
- 判定「无依据」的情况，只要沾上一条就是无依据：
  ① 参考资料里根本没有这个信息（模型用自己的知识补的）；
  ② 与参考资料矛盾（包括把关系说反：谁建立谁、谁是谁的上级、谁在谁之前）；
  ③ 模型自己的评价、推测、总结性判断（如"体现了其政治野心""整体风格偏向保守"这类原文没写的定性）。
- 不要因为答案用了参考资料里的词就判「有依据」——**要看这句话断言的事实是不是真的在参考资料里**。用词全对但关系说反的，是无依据。
- **只判断这句话的内容能不能由参考资料支持，不要判断它是否回答了用户的问题**（那是"答得对不对"，不是"有没有依据"）。
  ⚠️ 实测反例：问"跨境学童什么时候恢复面授"，答案写"最早在2020年5月27日开始分阶段恢复面授课堂[3]"——
  参考资料[3]里确有"2020年5月27日复课"这句话，所以**是有依据的**；它没答对问题（问的是跨境学童）
  属于 correctness 问题，**不要因此判 no**。
- 答案末尾的引用编号 [1][2] 与判断无关，不要被它影响；也不要因为整段没有编号就判无依据。

只输出 JSON，格式：
{"verdicts": [{"i": 1, "verdict": "yes", "why": "不超过20字的理由"}, ...]}
其中 i 是下面给你的句子的序号，必须把每一条都判完。"""

USER_TMPL = """【参考资料】
{context}

【模型答案的待判句】（共 {n} 条，逐条判）
{clauses}"""


def build_messages(query: str, chunks: list[str], clauses: list[str]) -> list[dict]:
    ctx = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(chunks, 1))
    cl = "\n".join(f"{i}. {c}" for i, c in enumerate(clauses, 1))
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"【问题】{query}\n\n" + USER_TMPL.format(
            context=ctx, n=len(clauses), clauses=cl)},
    ]


class Judge:
    def __init__(self, model: str, timeout: int = 120):
        from openai import OpenAI
        key = os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise SystemExit("[错误] 环境变量 DASHSCOPE_API_KEY 未设置")
        self.model = model
        self.client = OpenAI(api_key=key,
                             base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                             timeout=timeout)
        self.calls = 0
        self.tok_in = 0
        self.tok_out = 0

    def judge(self, query: str, chunks: list[str], clauses: list[str]) -> list[dict]:
        msgs = build_messages(query, chunks, clauses)
        resp = self.client.chat.completions.create(
            model=self.model, messages=msgs, temperature=0.0, max_tokens=2048,
            response_format={"type": "json_object"},
        )
        self.calls += 1
        if resp.usage:
            self.tok_in += resp.usage.prompt_tokens or 0
            self.tok_out += resp.usage.completion_tokens or 0
        txt = resp.choices[0].message.content or ""
        try:
            obj = json.loads(txt)
            out = obj.get("verdicts") or []
        except Exception:
            m = re.search(r"\{[\s\S]*\}", txt)
            out = (json.loads(m.group(0)).get("verdicts") if m else []) or []
        # 按序号对齐；缺失的标 None（**不猜**，让上层如实报"未判"）
        got = {int(v["i"]): v for v in out if str(v.get("i", "")).isdigit()}
        return [got.get(i) for i in range(1, len(clauses) + 1)]


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def clauses_of(r: dict) -> list[str]:
    return [c["sentence"] for c in faithfulness(r, 0.6)["clauses"]]


def run(judge: Judge, rows: list[dict], cache: dict, key_fn) -> None:
    todo = [r for r in rows if key_fn(r) not in cache]
    print(f"需判 {len(rows)} 条答案，缓存命中 {len(rows) - len(todo)}，本次调用 {len(todo)}")
    for i, r in enumerate(todo, 1):
        cl = clauses_of(r)
        if not cl:
            cache[key_fn(r)] = []
            continue
        try:
            v = judge.judge(r["query"], context_texts(r), cl)
        except Exception as e:
            print(f"  [{i}/{len(todo)}] {r.get('qid')} 失败：{type(e).__name__}: {e}")
            continue
        cache[key_fn(r)] = [{"clause": c, "verdict": (x or {}).get("verdict"),
                             "why": (x or {}).get("why")} for c, x in zip(cl, v)]
        yes = sum(1 for z in cache[key_fn(r)] if norm_verdict(z["verdict"]) == "yes")
        print(f"  [{i}/{len(todo)}] {r.get('qid')} {yes}/{len(cl)} 有依据")


def calibrate(judge: Judge, cache: dict, force: bool = False) -> int:
    """
    用人工标签量判官的一致率。达不到标准就不能拿去跑全量。

    ⚠️ 必须**逐条直接判**，不能"判整条答案再按文本查回标签里的那句"：
    校准集里含**合成对照**（"永恆帝國由瓦爾族人所建立"这种故意写错的句子），
    它们根本不在任何答案里 —— 按文本查回会一律查不到，
    于是判据看起来"漏判"，实际是校准程序自己错了。
    （2026-09-21 实测：第一版就是这样，18 条里 7 条显示 `?`。）
    """
    labels_path = PROJECT / "eval" / "faith_labels.jsonl"
    gen_path = RESULTS / "gen_dashscope_testset_rr.jsonl"
    labels = [json.loads(l) for l in labels_path.read_text(encoding="utf-8").splitlines()
              if l.strip()]
    rows = {r.get("qid"): r for r in load_rows(gen_path)}

    print(f"逐条判定 {len(labels)} 条标签（每条一次调用，上下文用该题的实际引用块）")
    for i, it in enumerate(labels, 1):
        r = rows.get(it["qid"])
        if r is None:
            print(f"  [{i}] {it['qid']} 找不到记录，跳过")
            continue
        ck = f"judge1::{judge.model}::{it['qid']}::{it['clause']}"
        if ck in cache and not force:
            continue
        try:
            v = judge.judge(r["query"], context_texts(r), [it["clause"]])[0] or {}
        except Exception as e:
            print(f"  [{i}] 失败：{type(e).__name__}: {e}")
            continue
        cache[ck] = {"verdict": v.get("verdict"), "why": v.get("why")}
        print(f"  [{i}/{len(labels)}] [{it['label']:>4}] → {v.get('verdict')}"
              f"  {it['clause'][:38]}")

    print()
    print("=" * 78)
    print("判官校准（对人工标签）")
    print("=" * 78)
    agree = tot = 0
    errs = []
    for it in labels:
        if it["label"] == "skip":
            continue
        ck = f"judge1::{judge.model}::{it['qid']}::{it['clause']}"
        v = _norm((cache.get(ck) or {}).get("verdict"))
        ok = (v == it["label"])
        agree += ok
        tot += 1
        if not ok:
            errs.append((it, v, cache.get(ck) or {}))
        print(f"{it['label']:<6}{v:<6}{'✅' if ok else '❌'} {it['clause'][:44]}")
    print("-" * 78)
    print(f"一致率 {agree}/{tot} = {agree / max(1, tot):.1%}"
          f"（合成对照也算在内 —— 对照样本是专门用来抓'漏检'的）")
    for it, v, j in errs:
        print(f"  ❌ 标签={it['label']} 判官={v}：{it['clause'][:40]}")
        print(f"     理由：{j.get('why')}   备注：{it.get('note', '')[:60]}")
    return 0 if tot and agree / tot >= 0.85 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Faithfulness LLM 判官")
    ap.add_argument("--calibrate", action="store_true", help="先跑校准（约 20 次调用）")
    ap.add_argument("--input", default=None, help="结果 jsonl（全量判定时用）")
    ap.add_argument("--tag", default="", help="输出文件名后缀")
    # ⚠️ 换模型必须**重过校准**（`--calibrate`）：判官是这套指标的量具，
    #    量具换了不重新标定，前后的数就不可比。2026-09-21 实测 qwen-plus 与
    #    qwen-turbo 在同一批 104 条上差 0.4 pp（结论不敏感），但这是**那次**测出来的，
    #    不代表下次也一样。
    ap.add_argument("--model", default=DEFAULT_CLOUD_MODEL,
                    help=f"判官模型，默认 {DEFAULT_CLOUD_MODEL}（与生成侧同一个常量）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="忽略缓存重判")
    args = ap.parse_args()

    cache_path = RESULTS / "faith_judge_cache.json"
    cache = {}
    if cache_path.exists() and not args.force:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"[缓存] 已有 {len(cache)} 条答案的判定")

    judge = Judge(args.model)
    print(f"[判官] {args.model} @ dashscope（temperature=0，json 输出）")

    if args.calibrate:
        rc = calibrate(judge, cache, args.force)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                              encoding="utf-8", newline="\n")
        print(f"\n[判官统计] 调用 {judge.calls} 次 · 输入 {judge.tok_in:,} tok"
              f" · 输出 {judge.tok_out:,} tok")
        print("CALIBRATE_PASS" if rc == 0 else "CALIBRATE_FAIL")
        return rc

    if not args.input:
        ap.error("--input 或 --calibrate 必须给一个")
    p = Path(args.input)
    if not p.is_absolute():
        p = PROJECT / args.input
    rows = load_rows(p)
    if args.limit:
        rows = rows[:args.limit]

    # ⚠️ 缓存键必须带**文件标识**：两个配置（重排开/关）跑的是同一套 qid，
    # 只用 qid 做键会让后一份直接读到前一份的判定 —— 看起来"跑过了"，其实结果全串了。
    series = f"{p.stem}"
    # 缓存键还要带**判官模型名**：换模型后判定会变，不带就把两个模型的结论混在一起了
    key = lambda r: f"judge::{judge.model}::{series}::{r.get('qid')}"   # noqa: E731
    t0 = time.time()
    run(judge, rows, cache, key)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                          encoding="utf-8", newline="\n")

    out = RESULTS / f"gen_faith_judge_{args.tag or p.stem}.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            rec = cache.get(key(r))
            if rec is None:
                continue
            f.write(json.dumps({"qid": r.get("qid"), "clauses": rec},
                               ensure_ascii=False) + "\n")
    # verdict 归一化后再统计：判官偶尔用中文回"有依据/无依据"，
    # 语义等价，不归一化就会在统计里被当成"没判"而静默消失（实测 20 个子句）。
    yes = sum(1 for r in rows for z in (cache.get(key(r)) or []) if _norm(z.get("verdict")) == "yes")
    allv = sum(1 for r in rows for z in (cache.get(key(r)) or []) if _norm(z.get("verdict")))
    print()
    print("=" * 78)
    print("JUDGE_OK")
    print(f"  答案数      : {len(rows)}")
    print(f"  子句数      : {allv}（判官判为有依据 {yes} = {yes / max(1, allv):.1%}）")
    print(f"  调用/耗时   : {judge.calls} 次 · {time.time() - t0:.0f} s")
    print(f"  输入/输出   : {judge.tok_in:,} / {judge.tok_out:,} tok")
    print(f"  结果        : {out}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
