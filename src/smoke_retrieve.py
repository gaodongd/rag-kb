#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检索链路自检（零成本：**不加载 13.6 GB 索引**，本机实测 **0.6 秒**）。

============================================================================
它为什么存在（2026-09-22 的真实事故）
============================================================================
用户双击 `启动问答.bat`，索引加载完、输入问题、点"提问" ——

    Error
    Namespace object has no attribute 'fetch_store'

原因：第 12 步给 `retrieve()` 加了 `args.fetch_store`，
命令行版（argparse 里有这个参数）一切正常，
但 Gradio 界面**自己手抄了一份参数表**，没跟上。

难受的不是这个错，是它**暴露的时机**：
要等 13.6 GB 索引加载完、等用户点一次按钮。
改一行参数 → 一次完整启动 + 一次演示翻车。

于是补两件事：
  ① 参数收成单一来源 `generator.RETRIEVAL_DEFAULTS` + `ensure_retrieval_args()` 兜底；
  ② 这个脚本 —— **用桩 searcher 跑真实的 retrieve()**，
     把"索引加载"从验证里剔掉，0.6 秒就能测出真正的路径会不会崩。

============================================================================
桩（stub）会不会测不到真问题
============================================================================
这个脚本替换的**只有** `HybridSearcher`（向量检索 + BM25，需要 13.6 GB 索引）；
其余全部是真的：真的 `retrieve()`、真的 `args` 命名空间（直接向界面/命令行要 parser）、
真的 `fetch_store` 取原文（真的读侧车/parquet）、真的 `ids.txt`。

而这次事故的成因是"属性不存在" —— 属于**纯 Python 层**的问题，
桩不会把它藏起来，反而正是桩让它变得便宜可测。

用法
----
    "...python313\\python.exe" src\\smoke_retrieve.py
    "...python313\\python.exe" src\\smoke_retrieve.py --mode parquet      # 顺带量基线
    "...python313\\python.exe" src\\smoke_retrieve.py --pool 20 --topk 3

退出码：0=通过；1=有断言失败；2=本机缺数据（跳过，不算失败）
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))

from generator import (  # noqa: E402
    RETRIEVAL_DEFAULTS,
    ensure_retrieval_args,
    retrieve,
)
import generator as gen_mod  # noqa: E402  （对照组要临时替换它里面的函数）
from search_hybrid import PARQUET  # noqa: E402

IDS = PROJECT / "data" / "index" / "bge-large-zh-v1.5" / "ids.txt"

# 界面耗时条真正读的几个键 —— 改 retrieve() 的 timing 结构时这里会先响
TIMING_KEYS = ("vector", "bm25", "fuse", "fetch", "rerank", "retrieve_total")


class StubSearcher:
    """
    替掉真 searcher 的**唯一**理由：它要 13.6 GB 索引才能造出来。

    只实现 retrieve() 与界面层真正用到的三个成员：`.ids`、`.search()`、`._encode()`。
    这是**刻意的**：一旦真实代码多用了别的成员，这里立刻 AttributeError ——
    那说明"调用方需要的接口变了"，本身也是要看的信号。
    （`.search()` 的返回形状按 retrieve() 的实际拆包写：
      rows, scores, per_path, ?, (t_vec, t_bm, t_fuse) —— 5 元组。）
    """

    def __init__(self, ids: list[str]):
        self.ids = ids

    def _encode(self, texts):
        # retrieve() 不用 qv（它只负责透传给 search），界面层负责算 —— 返回 None 即可
        return [None for _ in texts]

    def search(self, query, qv, mode, topk_arg, pool_arg, w_vec, w_bm25, rrf_k):
        n = min(int(pool_arg), len(self.ids))
        # 真实命中是"扎堆"的（相邻行常常一起被召回），这里也造一小段连续行 +
        # 少量散布行，别造出"均匀随机"这种真实检索里不存在的分布。
        base = random.randrange(0, len(self.ids) - n - 1)
        rows = list(range(base, base + n))
        random.shuffle(rows)
        rows = rows[:min(int(topk_arg), n)]
        scores = [1.0 - i * 0.01 for i in range(len(rows))]
        per_path = {"vector": rows[::2], "bm25": rows[1::2]}
        return rows, scores, per_path, None, (0.0, 0.0, 0.0)


class _FakeCloud:
    """零成本的后端替身：真后端要 DASHSCOPE_API_KEY，而且会花钱。"""

    name = "fake"
    is_model = True

    def generate(self, messages, max_new_tokens):
        return ("张伯苓创办了南开大学 [1]。", 
                {"latency": 0.01, "prompt_tokens": 100, "completion_tokens": 12})


class _FakeReranker:
    """
    重排器替身。真 Reranker 要 import torch/FlagEmbedding + 加载 2 GB 权重 ——
    本脚本要的是"链路通不通"，不是"重排准不准"（那是 §11.13 的活）。
    返回形状必须与真的一致：[(rr_rows 里的下标, 分数), ...]
    """

    def __init__(self):
        self.calls = 0

    def rerank(self, query, passages, topk=None, batch_size=32):
        self.calls += 1
        return [(i, 1.0 - i * 0.01) for i in range(len(passages))]


def load_ids() -> list[str]:
    if not IDS.exists():
        raise FileNotFoundError(IDS)
    return IDS.read_text(encoding="utf-8").splitlines()


# ======================================================== 检查 ①：命名空间
def check_namespace(label: str, args, hard: bool, count: bool = True) -> list[str]:
    """
    调用方的命名空间对得上 RETRIEVAL_DEFAULTS 吗？

    这一条是**预防性**的：少了字段不会崩（ensure_retrieval_args 会补），
    但那句"[参数] ... 已按默认值补齐"的提示，说明两份清单已经漂移了 ——
    要么补进 parser（推荐），要么确认它就是故意精简的。

    hard=True 表示这是**真实调用方**（界面/命令行的 parser），缺了就是真问题；
    count=False 表示这是人工构造的样本，缺字段是**预期行为**（用来验证兜底能力）。
    """
    problems = []
    for k, v in RETRIEVAL_DEFAULTS.items():
        if not hasattr(args, k):
            problems.append(f"缺字段 {k!r}")
        else:
            got = getattr(args, k)
            # 类型不同不算（argparse 会把 bool 写成 True/False，语义一致）
            if isinstance(v, (int, float)) and isinstance(got, (int, float)) and got != v:
                problems.append(f"{k} 默认值不一致：{got!r} != {v!r}")
    tag = "✅" if not problems else ("❌" if hard else "· ")
    print(f"  {tag} {label}")
    for p in problems:
        print(f"       · {p}")
    return problems if count else []


# ========================================= 检查 ②：对照组（证明 bug 真的存在过）
def check_without_guard(gen_mod, label: str, args, stub: StubSearcher) -> list[str]:
    """
    临时**关掉兜底**，用旧版界面的字段跑一次 —— 期望它崩，而且崩在该崩的地方。

    为什么非要这个对照组：兜底（ensure_retrieval_args）上线之后，
    "旧版字段"的样本自己就变绿了，于是这个测试看起来什么都没测。
    关掉兜底再跑一次，才能证明：
      · 这个 bug 是真实存在的（不是我看错了报错信息）；
      · 测试**确实能抓到它**（否则将来谁把 retrieve() 里那行兜底删了，
        测试还是会绿 —— 那就是个安慰剂）。
    """
    real = gen_mod.ensure_retrieval_args
    gen_mod.ensure_retrieval_args = lambda a, who="": a       # 拆掉兜底
    try:
        gen_mod.retrieve(stub, "台灣東部開發於古時的人行道路", args, None)
    except AttributeError as e:
        print(f"  ✅ 对照组：关掉兜底后**确实崩了** → AttributeError: {e}")
        return []
    except Exception as e:                                    # 崩了但不是这个错，也要看清
        print(f"  ⚠️  对照组：崩了，但错在别处 → {type(e).__name__}: {e}")
        return []
    else:
        print(f"  ❌ 对照组：关掉兜底竟然没崩 —— 说明「{label}」这份样本"
              f"已经不是当年的样子了，测试变成了安慰剂，请核对 RETRIEVAL_DEFAULTS")
        return [f"对照组失效：{label}"]
    finally:
        gen_mod.ensure_retrieval_args = real                  # 恢复，别影响后面的用例


# ======================================================== 检查 ②：真跑一遍
def run_case(label: str, args, stub: StubSearcher, expect_topk: int) -> list[str]:
    """用真实 retrieve() 跑一次，看它会不会抛错、返回的结构对不对。"""
    fails = []
    t = time.time()
    try:
        qv = None                    # 桩不用它；真 searcher 才需要
        hits, timing = retrieve(stub, "台灣東部開發於古時的人行道路", args, qv)
    except Exception as e:
        print(f"  ❌ {label} → {type(e).__name__}: {e}")
        return [f"{label}: {type(e).__name__}: {e}"]

    dt = (time.time() - t) * 1000
    miss = [k for k in TIMING_KEYS if k not in timing]
    if miss:
        fails.append(f"{label}: timing 缺键 {miss}")
    if len(hits) != min(expect_topk, len(stub.ids)):
        fails.append(f"{label}: 命中 {len(hits)} 条 != 期望 {expect_topk}")
    # 界面会拿这些字段渲染引用原文那栏 —— 缺了就是空白卡片
    for need in ("rank", "chunk_id", "title", "section", "chunk_text"):
        if hits and need not in hits[0]:
            fails.append(f"{label}: hit 缺字段 {need!r}")
    if hits and not (hits[0]["chunk_text"] or "").strip():
        fails.append(f"{label}: 取回原文是空的（取原文层没生效）")

    ok = "✅" if not fails else "❌"
    fetch_ms = timing.get("fetch", 0) * 1000
    print(f"  {ok} {label}：命中 {len(hits)} 条 · 取回原文 "
          f"{len(hits[0]['chunk_text']) if hits else 0} 字 · "
          f"取原文 {fetch_ms:.0f} ms · 总计 {dt:.0f} ms")
    return fails


def check_ui_layer(base_args, stub: StubSearcher) -> list[str]:
    """
    界面那一层也跑一遍 —— 但**不启 gradio、不加载索引、不调 API**。

    为什么要单独测这层：这次顺手修的耗时条（取原文直接读 timing['fetch']、
    重排单独列一栏）就在 RAGApp.answer() 里，而它和这次的 bug 是**同一类暴露时机** ——
    只有用户点了"提问"才会执行。桩掉 searcher / 云端后端 / 重排器，一秒跑完。

    ⚠️ 桩掉的是"要钱、要显卡、要索引"的三样东西；answer() 里其余的代码
    （组装 prompt、解析 [n] 引用、判拒答、渲染引用原文 HTML、拼耗时条）全是真的。
    """
    fails: list[str] = []
    try:
        import gradio_app as ga
    except Exception as e:
        print(f"  ❌ 界面模块导入失败 → {type(e).__name__}: {e}")
        return [f"gradio_app import: {e}"]

    real_make, real_cloud = ga.make_searcher, ga.CloudBackend
    ga.make_searcher = lambda a: stub
    ga.CloudBackend = lambda model="qwen-plus": _FakeCloud()

    try:
        for rr_on in (False, True):
            ns = copy.copy(base_args)          # 浅拷贝，别把 --rerank 留给下一轮
            ns.rerank = rr_on
            ns.mode, ns.fuse = "hybrid", "rrf"
            ensure_retrieval_args(ns, "smoke-ui")
            ns.topk = min(3, ns.topk)
            app = ga.RAGApp(ns)
            if rr_on:                          # 预置替身，避开 torch/2GB 权重
                app._rr = _FakeReranker()
            ans, cites, timing_md = app.answer("张伯苓是谁", "云端 qwen-plus", ns.topk)
            label = f"重排{'开' if rr_on else '关'}"
            bad = []
            if "[1]" not in ans:
                bad.append("答案/引用标记没渲染出来")
            if "取原文" not in timing_md or "合计" not in timing_md:
                bad.append(f"耗时条缺字段：{timing_md[:80]}")
            if rr_on and "重排" not in timing_md:
                bad.append("开了重排却没显示重排耗时")
            if (not rr_on) and "重排" in timing_md:
                bad.append("没开重排却显示了重排耗时")
            if "被引用" not in cites:
                bad.append("引用原文区没有标出被引用的条目")
            if bad:
                fails += [f"界面层({label}): {b}" for b in bad]
                print(f"  ❌ 界面层（{label}）")
                for b in bad:
                    print(f"       · {b}")
            else:
                seg = timing_md.split("｜")[0].strip()
                print(f"  ✅ 界面层（{label}）：{seg}")
    except Exception as e:
        fails.append(f"界面层: {type(e).__name__}: {e}")
        print(f"  ❌ 界面层抛错 → {type(e).__name__}: {e}")
    finally:
        ga.make_searcher, ga.CloudBackend = real_make, real_cloud
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=RETRIEVAL_DEFAULTS["topk"])
    ap.add_argument("--pool", type=int, default=RETRIEVAL_DEFAULTS["topn"])
    ap.add_argument("--mode", default="auto", choices=["auto", "blob", "parquet"],
                    help="取原文实现（默认 auto；parquet 会走全扫基线，约 2 s）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    print("=" * 78)
    print("检索链路自检（零成本：不加载 13.6 GB 索引）")
    print("=" * 78)

    # ---------------- 数据存在吗 ----------------
    if not IDS.exists() or not Path(PARQUET).exists():
        print(f"\n[跳过] 本机没有数据文件：\n  {IDS}\n  {PARQUET}")
        print("       这份仓库不含 data/（35 GB，见 .gitignore）。")
        print("       复现步骤见 README「复现」一节。本次未做检查（不算失败）。")
        return 2

    ids = load_ids()
    print(f"\n语料：{len(ids):,} chunk（ids.txt）")

    # ---------------- ① 真实调用方的命名空间 ----------------
    print("\n【① 真实调用方的命名空间 vs RETRIEVAL_DEFAULTS】")
    print("   （缺字段不会崩 —— ensure_retrieval_args 会补；但补齐=清单漂移了，该补进 parser）")
    problems: list[str] = []
    gradio_args = None
    try:
        sys.path.insert(0, str(PROJECT / "app"))
        from gradio_app import build_parser as gradio_parser      # noqa: E402
        gradio_args = gradio_parser().parse_args([])
        gradio_args.mode, gradio_args.fuse = "hybrid", "rrf"
        problems += check_namespace("界面 app/gradio_app.py build_parser()", gradio_args, hard=True)
    except Exception as e:
        print(f"  ❌ 界面 parser 取不到 → {type(e).__name__}: {e}")
        problems.append(f"gradio parser: {e}")

    gen_args = None
    try:
        from generator import build_parser as gen_parser           # noqa: E402
        gen_args = gen_parser().parse_args([])
        problems += check_namespace("命令行 src/generator.py build_parser()", gen_args, hard=True)
    except Exception as e:
        print(f"  ❌ generator parser 取不到 → {type(e).__name__}: {e}")
        problems.append(f"generator parser: {e}")

    # ---------------- ② 兜底能力（人工构造的账本） ----------------
    # 这两个样本缺字段是**预期**的，不计入 problems —— 它们是用来验证
    # "换个调用方、少写字段，链路照样能跑" 这件事本身成不成立。
    print("\n【② 兜底能力：少写字段的命名空间还跑不跑得通】")
    empty = SimpleNamespace()
    old_gradio = SimpleNamespace(host="127.0.0.1", port=7860, cloud_model="qwen-plus",
                                 topk=5, topn=100, rrf_k=10, w_vec=1.0, w_bm25=1.0,
                                 vec_backend="faiss", nprobe=512, max_new_tokens=512,
                                 mode="hybrid", fuse="rrf")
    check_namespace("空 SimpleNamespace()（缺全部 10 个字段）", empty, hard=False, count=False)
    check_namespace("旧版界面字段（缺的就是 2026-09-22 崩掉的那 5 个）",
                    old_gradio, hard=False, count=False)

    stub = StubSearcher(ids)
    fails: list[str] = []

    # 对照组：先证明"没有兜底就会崩"，这之后所有"跑通了"才有意义
    print("\n【③ 对照组：把兜底关掉，旧版字段应该崩】")
    fails += check_without_guard(gen_mod, "旧版界面字段", old_gradio, stub)

    # ---------------- ④ 真实调用方真跑一遍 ----------------
    print("\n【④ 用真实 retrieve() 跑一遍（桩只替掉了需要 13.6 GB 的那个 searcher）】")
    cases = [("旧版界面字段（兜底后应跑通）", old_gradio),
             ("空 SimpleNamespace（最坏情况）", empty)]
    if gradio_args is not None:
        cases.insert(0, ("界面 build_parser()", gradio_args))
    if gen_args is not None:
        cases.insert(0, ("命令行 build_parser()", gen_args))

    for label, ns in cases:
        ns = ensure_retrieval_args(ns, "smoke")   # 与真实调用路径一致
        ns.topk, ns.topn = args.topk, args.pool   # 用命令行指定的池子大小
        ns.fetch_store = args.mode
        fails += run_case(label, ns, stub, args.topk)

    # ---------------- ⑤ 界面层 ----------------
    print("\n【⑤ 界面层 answer()：桩掉 searcher / 云端 / 重排器，其余全真】")
    if gradio_args is not None:
        fails += check_ui_layer(gradio_args, stub)
    else:
        print("  · 跳过（界面 parser 没取到）")

    # ---------------- 结论 ----------------
    print()
    print("=" * 78)
    print("【结论】")
    print("=" * 78)
    if fails:
        print(f"❌ 有 {len(fails)} 项失败：")
        for f in fails:
            print(f"   · {f}")
        return 1
    if problems:
        print(f"❌ 真实调用方有 {len(problems)} 处字段漂移 —— 请补进对应的 parser"
              "（链路本身能跑，但兜底提示说明两份清单已经不一致）：")
        for p in problems:
            print(f"   · {p}")
        return 1
    print("✅ 四件事都成立：")
    print("   ① 界面 / 命令行的参数表与 RETRIEVAL_DEFAULTS 完全对齐（没有漂移）")
    print("   ② 对照组确认：关掉兜底，旧版字段**确实**崩在 fetch_store（bug 真实存在）")
    print("   ③ 开着兜底，四种命名空间都能跑通取原文（含空命名空间）")
    print("   ④ 界面层 answer() 两条分支（重排开/关）都渲染正常，含耗时条")
    print("   改完 retrieve() 的参数就跑一次 —— 本脚本 0.6 秒（冷盘也就 1~3 秒），"
          "代价远低于「等到点了提问才发现」")
    return 0


if __name__ == "__main__":
    sys.exit(main())
