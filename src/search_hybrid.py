#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 8 步 · T7 混合检索（BM25 + 向量）· RRF 排名融合

为什么必须融合 —— 实测出来的，不是理论
----------------------------------------
第 8 步两条路各自跑完，用同一组 query 对照：

    Q「台灣東部開發於古時的人行道路」
      向量 top1 → 「東寧路 (臺南市)」      语义完全无关（模型没学好这个专名）
      BM25 top1 → 「蘇花古道」27.05 分     正文首句就是原话，1.33 倍领先第二名

    Q「這個地方氣候怎麼樣」（query 里没有任何地名）
      向量 top1 → 「高邮市 · 气候」         学**小节级语义**，这是向量的主场
      BM25 top1 → 「韩东君 · 評價」         被"這個/地方/怎麼樣"这些泛词带偏

→ 向量强在语义泛化，BM25 强在**字面精确**（专名、生僻词、代码符号）。
  没有任何一路能单独覆盖这两类问题，所以必须融合。

为什么是 RRF 而不是「分数加权」
--------------------------------
两路的分数**量纲根本不同**：向量是余弦 ∈ [-1, 1]，BM25 是 Σ IDF·tf_norm ∈ [0, 30+]。

- 直接相加 → BM25 完全压死向量（30 vs 0.6）
- min-max 归一化 → 对分数分布敏感，而每个 query 内部的分布又不一样，
  同一套权重换个问题就失灵

RRF（Reciprocal Rank Fusion, Cormack et al. 2009）**只用排名，不看分数**：

    score(d) = Σ_r  w_r / (k + rank_r(d))          k 默认 60

排名天然免疫量纲、免疫分数尺度漂移。这也是 Elasticsearch 8.x 的官方 RRF 实现。
代价：丢掉了"分差"信息（第 1 名 27 分还是 10 分在 RRF 里没区别）——
但对"多路结果需要合流"这个场景，稳健性远比信息量重要。

k 为什么默认取 10，而不是原论文的 60
--------------------------------------
k 不是无关紧要的常数，它决定「头部排名差异被放大多少」：

    第 1 名 vs 第 20 名：
      k=60 → 1/61 = 0.01639  vs  1/80 = 0.01250   （差 31%）
      k=10 → 1/11 = 0.09091  vs  1/30 = 0.03333   （差 173%）

k 越大、排名越"平"，**多路中游共识就越容易反超单路第一名**。实测（`src\\probe_fusion.py`，3 题）：

| 策略 | Q1 蘇花古道 | Q2 张克忠 | Q3 气候小节 | 命中 |
|---|---|---|---|---|
| rrf **k=60**（ES 默认） | ✅ | ❌ 被"两路中游"的朱汝瑾反超 | ❌ | 1/3 |
| rrf **k=10** | ✅ | ✅ | ❌ | **2/3** |
| cc（分数归一化加权） | ✅ | ❌ | ❌ | 1/3 |
| cc（z-score 版） | ✅ | ❌ | ❌ | 1/3 |
| `--fuse gate` | ✅ | ✅ | ✅ | **3/3** ⚠️ |

→ **k=60 是给"两路质量相当"的场景准备的默认值**；我们这里 BM25 质量波动极大
  （专名题能压过向量，泛词题全是噪声），所以要用更小的 k 让强信号说话。

Q3 为什么连 k=10 也救不了：纯 RRF **完全丢弃分数强度**，无法区分
「蘇花古道 27.05 分、领先 33%」（该信）和「韩东君 19.90 分、领先 2.8%」（纯噪声）——
两者都是 rank 1，拿一样的权重。

`--fuse gate` 用「**BM25 top1 领先次名的幅度**」当置信度信号，3/3 全过。
⚠️ **但那组阈值（10% / 0.3）是在这 3 条 query 上拟合出来的，样本量远不足以支撑泛化，
所以默认关闭。** 要用它，先标注几十条 query 做验证 —— 这一步没做完之前，3/3 只能当线索，不能当结论。

用法
----
    :: 三个问题，三模式对照
    "...python313\\python.exe" src\\search_hybrid.py

    :: 只跑混合
    "...python313\\python.exe" src\\search_hybrid.py --mode hybrid

    :: 调融合，看权重影响
    "...python313\\python.exe" src\\search_hybrid.py --topn 200 --rrf-k 60 --w-bm25 1.5

    :: 回到 ES 默认的 k=60 做对比
    "...python313\\python.exe" src\\search_hybrid.py --compare --rrf-k 60

    :: 打开实验性的领先幅度门控（阈值未经过充分验证，见上）
    "...python313\\python.exe" src\\search_hybrid.py --fuse gate
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from search_bm25 import build_idf, bm25_search, load_bm25  # noqa: E402
from search_vector import DEFAULT_QUERIES, fetch_texts, load_index, scan_topk  # noqa: E402

PROJECT = HERE.parent
VEC_INDEX = PROJECT / "data" / "index" / "bge-large-zh-v1.5"
BM25_INDEX = PROJECT / "data" / "index" / "bm25"
PARQUET = PROJECT / "data" / "processed" / "chunks.parquet"
FAISS_INDEX = PROJECT / "data" / "index" / "faiss_ivf_flat.index"
IDS_PATH = VEC_INDEX / "ids.txt"


# ---------------------------------------------------------------- 融合

def rrf_fuse(rank_lists, weights=None, k: int = 60, topn: int = 0):
    """
    RRF：把多路「排名列表」合成一个分数。

    参数
      rank_lists : [[row, row, ...], ...]   每路已按相关性降序排好的行号
      weights    : 每路权重，默认全 1
      k          : 平滑常数。k 越小，头部排名差异被放大得越厉害；
                   60 是原论文经验值，实践中 10~100 都能用。
      topn       : 只返回前 N（0 = 全返回）

    返回 [(row, fused_score), ...] 降序。
    """
    if weights is None:
        weights = [1.0] * len(rank_lists)
    acc = {}
    for ranks, w in zip(rank_lists, weights):
        if w == 0:
            continue
        for pos, row in enumerate(ranks, start=1):     # 排名从 1 开始
            row = int(row)
            acc[row] = acc.get(row, 0.0) + w / (k + pos)
    out = sorted(acc.items(), key=lambda x: (-x[1], x[0]))   # 同分时按行号，保证可复现
    return out[:topn] if topn else out


# ---------------------------------------------------------------- 检索器

class HybridSearcher:
    """
    一次性把三种检索都装进来，共享同一套索引与模型 —— 加载只做一遍。

    行号契约（贯穿全项目，别打破）：
        parquet 第 i 块 == emb.npy 第 i 行 == ids.txt 第 i 行 == BM25 postings 里的 doc_id i
    """

    def __init__(self, args):
        self.args = args
        print("=" * 76)
        print("第 8 步 · 混合检索（BM25 + 向量 · RRF）")
        print("=" * 76)

        t = time.time()
        self.ids, self.emb = load_index(VEC_INDEX)
        self.n = len(self.ids)
        print(f"向量索引  : {self.n:,} × {self.emb.shape[1]}  {self.emb.dtype}"
              f"（memmap，{time.time() - t:.1f} 秒）")

        t = time.time()
        # 归一化开关必须**成对**使用：索引侧和查询侧要么都开、要么都关。
        # 只关一边 = 静默漏词（不报错，只是某些词永远搜不到），所以
        # load_bm25 会在状态不符时直接拒绝启动，除非显式放行。
        # 这两个参数只服务于"量繁简归一化值多少"的对照实验，正常路径走默认值。
        bm25_dir = Path(getattr(args, "bm25_index", None) or BM25_INDEX)
        self.no_norm = bool(getattr(args, "no_norm", False))
        self.meta, self.vocab, self.indptr, self.doc_ids, self.tf_norm = load_bm25(
            bm25_dir, allow_norm_mismatch=self.no_norm)
        self.idf = build_idf(self.indptr, self.n)
        print(f"BM25 索引 : {self.meta['vocab_size']:,} 词 / {self.meta['postings']:,} postings"
              f"（memmap，{time.time() - t:.1f} 秒）"
              + ("  ⚠️ 归一化已关闭（对照实验）" if self.no_norm else ""))

        self.model = None                              # 惰性加载：只跑 bm25 模式时不必加载 BGE
        self.last_lead = None                          # 上一次检索的「BM25 top1 领先幅度」（查询级置信度）
        self.faiss_index = None
        self.vec_backend = args.vec_backend
        if args.mode in ("vector", "hybrid") and args.vec_backend == "faiss":
            import faiss
            if not FAISS_INDEX.exists():
                raise SystemExit(f"[错误] 缺 {FAISS_INDEX}，先跑 src\\build_faiss_index.py"
                                 f"（或用 --vec-backend scan）")
            t = time.time()
            # ⚠️ faiss 的 IO_FLAG_MMAP **只对连续数组（IndexFlat 等）有效**。
            #    IVF 类索引的倒排列表是一堆分离的 ArrayInvertedLists，没有连续内存布局可映射，
            #    强行传 IO_FLAG_MMAP 会抛：
            #      RuntimeError: could not load ArrayInvertedLists as 646f6c69 ("ilod")
            #    报错信息完全看不出"是 mmap 引起的"，所以这里显式回退到整体读入。
            try:
                self.faiss_index = faiss.read_index(str(FAISS_INDEX), faiss.IO_FLAG_MMAP)
                load_mode = "mmap"
            except RuntimeError:
                self.faiss_index = faiss.read_index(str(FAISS_INDEX))
                load_mode = "整体读入（IVF 不支持 mmap）"
            self.faiss_index.nprobe = args.nprobe
            print(f"faiss 索引: {self.faiss_index.ntotal:,} 条，nprobe={args.nprobe}"
                  f"（{load_mode}，{time.time() - t:.1f} 秒，"
                  f"约 {FAISS_INDEX.stat().st_size / 1e9:.2f} GB 常驻）")
        print(f"行号表    : {self.n:,} 行（全局契约）")

    # ---- 向量路 ----
    def _encode(self, queries):
        if self.model is None:
            t = time.time()
            from FlagEmbedding import FlagModel
            sys.path.insert(0, str(HERE))
            from vectorize import find_model
            self.model = FlagModel(str(find_model("large")), use_fp16=True)
            print(f"[模型加载] FlagModel bge-large-zh，{time.time() - t:.1f} 秒")
        return np.asarray(self.model.encode(queries, batch_size=8, max_length=512)).astype(np.float32)

    def search_vector(self, qv, topn):
        """返回 (rows, scores) —— 行号在前（与 search_bm25 保持同一约定）。"""
        if self.vec_backend == "faiss":
            _, I = self.faiss_index.search(qv[None, :], topn)
            return np.asarray(I[0], dtype=np.int64), None
        rows, scores = scan_topk(self.emb, qv, topn)
        return rows, scores

    # ---- BM25 路 ----
    def search_bm25(self, q, topn):
        rows, sc, detail = bm25_search(q, self.vocab, self.indptr, self.doc_ids,
                                       self.tf_norm, self.idf, self.n, topn,
                                       normalize=not self.no_norm)
        return np.asarray(rows, dtype=np.int64), sc, detail

    @staticmethod
    def _assert_rows(rows, n: int, tag: str):
        """
        防呆：两路都必须返回**行号**（int64 且落在 [0, n)），不能返回分数。

        加这一段的理由：曾经把 `scan_topk` 的 (scores, rows) 解包反了，
        分数 0.6 → int() → 0，100 个候选全塌成 row=0，RRF 里同一 key 被累加 100 次，
        伪造出 0.99 的假高分。**全程零报错**，只是结果看起来"命中了 row=0"。
        这类错误靠肉眼审不出来，只能靠断言。
        """
        if len(rows) == 0:
            return
        if rows.dtype.kind != "i":
            raise AssertionError(f"{tag} 返回的 dtype={rows.dtype}，不是整数行号 —— 多半是解包顺序反了")
        if rows.max() >= n or rows.min() < 0:
            raise AssertionError(f"{tag} 行号越界 [{rows.min()}, {rows.max()}] 超出 [0, {n:,})")

    # ---- 融合 ----
    def search(self, q, qv, mode, topk, topn, w_vec, w_bm25, rrf_k):
        """返回 (rows, scores, per_path)。per_path = {"vector": rows, "bm25": rows}"""
        per_path = {}
        t = time.time()
        if mode in ("vector", "hybrid"):
            v_rows, _ = self.search_vector(qv, topn)
            self._assert_rows(v_rows, self.n, "向量路")
            per_path["vector"] = v_rows
            t_vec = time.time() - t
        else:
            t_vec = 0.0

        t = time.time()
        if mode in ("bm25", "hybrid"):
            # 分数要留着 —— gate 策略靠「BM25 top1 领先次名多少」判断这题它是否可信
            b_rows, b_scores, detail = self.search_bm25(q, topn)
            self._assert_rows(b_rows, self.n, "BM25 路")
            per_path["bm25"] = b_rows
            t_bm = time.time() - t
        else:
            b_rows, b_scores, detail, t_bm = None, None, [], 0.0

        # ---- 查询级信号：BM25 top1 领先次名的幅度 ----
        # 「领先得越多 ⇒ 这一题 BM25 越是捞到了强匹配」。
        # 无论走不走 gate 都算一遍：评测侧要拿它做**查询路由**的分析
        # （11.11 证明了全局权重救不了，因为不同子集需要相反的权重）。
        self.last_lead = None
        if b_scores is not None and len(b_scores) > 1:
            s1, s2 = float(b_scores[0]), float(b_scores[1])
            self.last_lead = (s1 - s2) / max(abs(s2), 1e-9)

        t = time.time()
        if mode == "vector":
            rows = per_path["vector"][:topk]
            scores = None
        elif mode == "bm25":
            rows = per_path["bm25"][:topk]
            scores = None
        else:
            w_b = w_bm25
            if self.args.fuse == "gate" and self.last_lead is not None:
                lead = self.last_lead
                strong_w = getattr(self.args, "gate_strong_w", 1.0)
                # 批量评测时逐题打印会淹掉日志（104 题 = 104 行）→ 用 quiet 关掉
                say = (lambda *a: None) if getattr(self.args, "quiet", False) else print
                if lead < self.args.gate_margin:
                    # ⚠️⚠️ 这个"降权"分支已被实证为**负收益，不要用**（第 11 步 §11.12）：
                    #   匿名子集 R@5 0.781 → 0.719、释义改写 0.786 → 0.714、整体 0.875 → 0.846。
                    #   原因是"BM25 领先幅度小"并不等于"BM25 这题不可信" ——
                    #   匿名题（BM25 唯一强的子集）里也有领先幅度小的，被这条无差别降权打死。
                    #   正确方向是**反向**：领先幅度大时加权（见下面的 strong 分支）。
                    w_b = w_bm25 * self.args.gate_weak_w
                    say(f"       [gate] BM25 领先仅 {lead:.1%} < {self.args.gate_margin:.0%}"
                        f" → BM25 权重 ×{self.args.gate_weak_w} = {w_b}")
                else:
                    # 反向分支：置信时**加**权（默认 1.0 = 不加，保持原行为）
                    # 实测（门限 30%、强×2.0）：R@1 0.606→0.644、MRR 0.726→0.753，
                    # R@5 与基础事实题**一分没掉**（配对检验 4 胜 0 负，但 p=0.125 未达显著）。
                    w_b = w_bm25 * strong_w
                    say(f"       [gate] BM25 领先 {lead:.1%} ≥ {self.args.gate_margin:.0%}"
                        f" → BM25 权重 ×{strong_w} = {w_b}")
            fused = rrf_fuse([per_path["vector"], per_path["bm25"]],
                             weights=[w_vec, w_b], k=rrf_k, topn=topk)
            rows = np.asarray([r for r, _ in fused], dtype=np.int64)
            scores = np.asarray([s for _, s in fused], dtype=np.float32)
        t_fuse = time.time() - t
        return rows, scores, per_path, detail, (t_vec, t_bm, t_fuse)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="hybrid", choices=["vector", "bm25", "hybrid"])
    ap.add_argument("--query", action="append", default=None)
    ap.add_argument("--topk", type=int, default=5, help="最终返回几条")
    ap.add_argument("--topn", type=int, default=100,
                    help="每路先取多少候选再融合。太小则两路的好结果进不了融合池")
    ap.add_argument("--rrf-k", type=int, default=10,
                    help="RRF 平滑常数。**默认 10 而不是原论文的 60** —— 理由见下方说明")
    ap.add_argument("--fuse", default="rrf", choices=["rrf", "gate"],
                    help="rrf=纯排名融合；gate=再按「BM25 top1 领先幅度」调权（实验性，"
                         "阈值只在 3 条样本上拟合过，样本量严重不足）")
    ap.add_argument("--gate-margin", type=float, default=0.10,
                    help="gate 阈值：BM25 top1 领先次名低于此比例 → 认为它没捞到强匹配")
    ap.add_argument("--gate-weak-w", type=float, default=0.3,
                    help="gate 判定为「弱匹配」时，BM25 权重要乘的系数")
    ap.add_argument("--gate-strong-w", type=float, default=1.0,
                    help="gate 判定为「强匹配」（领先 ≥ margin）时 BM25 权重的系数。"
                         "1.0 = 不加权（默认，等于只有降权分支）")
    ap.add_argument("--w-vec", type=float, default=1.0)
    ap.add_argument("--w-bm25", type=float, default=1.0)
    ap.add_argument("--vec-backend", default="scan", choices=["scan", "faiss"],
                    help="scan=暴力精确（慢但无损）；faiss=IVF 近似（快，有召回损失）")
    ap.add_argument("--nprobe", type=int, default=64)
    ap.add_argument("--show", type=int, default=5)
    ap.add_argument("--compare", action="store_true",
                    help="三模式并排（每个 query 跑 3 遍，慢 3 倍）")
    args = ap.parse_args()
    queries = args.query or DEFAULT_QUERIES

    s = HybridSearcher(args)
    # 只跑 BM25 时不必加载 BGE（省 10 秒启动 + 1.3GB 显存）
    need_vec = args.compare or args.mode in ("vector", "hybrid")
    qv_all = s._encode(queries) if need_vec else [None] * len(queries)

    print(f"分块大小  : 200,000 行/块（峰值临时内存 ≈ 819 MB）")
    print(f"融合参数  : topn={args.topn}  RRF k={args.rrf_k}  "
          f"w_vec={args.w_vec}  w_bm25={args.w_bm25}")
    print(f"向量后端  : {args.vec_backend}"
          f"{' nprobe=' + str(args.nprobe) if args.vec_backend == 'faiss' else '（暴力精确）'}")

    modes = ["vector", "bm25", "hybrid"] if args.compare else [args.mode]
    id_list = s.ids

    for q, qv in zip(queries, qv_all):
        print()
        print("=" * 76)
        print(f"Q: {q}")
        print("=" * 76)

        # ---- 每个模式只跑一次，结果留着后面复用（向量路一跑就是数秒，不能重复调）----
        outcomes = {}
        for mode in modes:
            outcomes[mode] = s.search(q, qv, mode, args.topk, args.topn,
                                      args.w_vec, args.w_bm25, args.rrf_k)

        # ---- 打印各模式的 top-k ----
        for mode in modes:
            rows, scores, per_path, detail, (t_vec, t_bm, t_fuse) = outcomes[mode]
            lat = f"[向量 {t_vec:.2f}s · BM25 {t_bm * 1000:.0f}ms · 融合 {t_fuse * 1000:.1f}ms]"
            print(f"\n  ── {mode.upper()} ── {lat}")

            # 各路排名查表 —— 用来回答"是谁把这条拉上来的"
            pos_v = {int(r): i + 1 for i, r in enumerate(per_path.get("vector", []))}
            pos_b = {int(r): i + 1 for i, r in enumerate(per_path.get("bm25", []))}

            for rank, r in enumerate(rows, 1):
                if rank > args.show:
                    break
                r = int(r)
                if scores is not None:
                    tag = f"   (vec#{pos_v.get(r, '—')}  bm25#{pos_b.get(r, '—')})"
                    print(f"  {rank}. RRF {scores[rank - 1]:.5f}{tag}   row={r}")
                else:
                    tag = ""
                    if "vector" in per_path:
                        tag = f"   (vec#{pos_v.get(r, '—')})"
                    elif "bm25" in per_path:
                        tag = f"   (bm25#{pos_b.get(r, '—')})"
                    print(f"  {rank}.{tag}   row={r}")

        # ---- hybrid 的原文（信息量最大，单独展开）----
        if "hybrid" in outcomes:
            rows, scores, per_path, _, _ = outcomes["hybrid"]
            pos_v = {int(r): i + 1 for i, r in enumerate(per_path["vector"])}
            pos_b = {int(r): i + 1 for i, r in enumerate(per_path["bm25"])}
            cids = [id_list[int(r)] for r in rows]
            got = fetch_texts(PARQUET, cids)
            print("\n  ── 融合结果原文 ──")
            for rank, r in enumerate(rows, 1):
                r = int(r)
                rec = got.get(id_list[r])
                if rec is None:
                    print(f"  {rank}. [原文缺失] chunk_id={id_list[r]}")
                    continue
                sec = rec["section"] or "—"
                text = (rec["chunk_text"] or "").replace("\n", " ")[:74]
                print(f"  {rank}. {scores[rank - 1]:.5f}"
                      f"  (向量#{pos_v.get(r, '—')} BM25#{pos_b.get(r, '—')})")
                print(f"       {rec['title']}  ·  {sec}")
                print(f"       {text}")

    print()
    print("=" * 76)
    print("SEARCH_HYBRID_OK")
    print(f"  模式      : {', '.join(modes)}")
    print(f"  查询数    : {len(queries)}")
    print(f"  topk/topn : {args.topk} / {args.topn}")
    print(f"  RRF       : k={args.rrf_k}  w_vec={args.w_vec}  w_bm25={args.w_bm25}")
    print("=" * 76)
    print("怎么读结果：看 hybrid 那行的 (vec#N  bm25#M) 标记 ——")
    print("  某条只被一路召回到高位、另一路很靠后甚至没有，却最终排进 top-k，")
    print("  说明融合确实在做「取长补短」，而不是简单地把两路的共同结果重复一遍。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
