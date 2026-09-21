#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-encoder 重排（bge-reranker-v2-m3）。

============================================================================
为什么要它（第 8 步留下的结论）
============================================================================
第 8 步的混合检索用 RRF 融合两路**排名**，本质是"排名技巧"——
`gate` 策略在小样本上 3/3 命中，但阈值明显是拟合出来的，默认关闭。

reranker 和 RRF 的根本区别：
  - RRF：只看两路各自给的**名次**，模型从未读过 query 和文档的正文
  - reranker：把 (query, 文档) 拼成一个序列喂进模型，**逐对算相关性分数**

所以它比继续调融合参数根本得多。代价是慢：每个候选都要跑一次前向。

============================================================================
一个容易忽略的设计点：passage 用什么文本
============================================================================
用 `title｜section｜正文` 的拼接格式，**和向量化时完全一致**（vectorize.build_text）。

理由：评测集里的 hard_anon 题（实体匿名化）问的是
「有位创办了南开大学化工系的化学工程学家，他的籍贯是哪里」——
正文里根本没出现过"南开大学化工系"的完整表述时，**title 是唯一的强信号**。
只喂 chunk_text 会让 reranker 在这类题上退化成瞎猜。

用 build_text() 而不是自己拼 —— 与索引侧共用一个函数是第 9 步繁简事故的教训。
============================================================================
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))

# ModelScope 缓存根目录（与 vectorize.py / download_models.py 保持一致）
MODEL_ROOT = Path(r"E:\AI-learning\ms-cache\models")


def find_ms_model(pattern: str) -> Path:
    """
    在 ModelScope 缓存里按子串找模型目录。

    为什么不能复用 vectorize.find_model()：它把模式串**硬编码**成 `bge-{kind}-zh`，
    只认 bge 那套命名，传 "reranker-v2-m3" 会拼成 `bge-reranker-v2-m3-zh`，永远找不到。
    （同一个坏模式在第 9 步的 generator.py 里已经踩过一次，这里不重复踩。）
    """
    for p in sorted(MODEL_ROOT.glob(f"*{pattern}*")):
        for cand in (p / "snapshots" / "master", p):
            if (cand / "config.json").exists():
                return cand
    raise FileNotFoundError(
        f"找不到匹配 *{pattern}* 的本地模型（在 {MODEL_ROOT} 下）。\n"
        f"先跑：python src\\download_models.py --only bge-reranker-v2-m3")


class Reranker:
    """bge-reranker-v2-m3 的薄封装。惰性加载，首次调用才吃显存。"""

    def __init__(self, model_dir: str | Path | None = None, fp16: bool = True,
                 max_length: int = 512, verbose: bool = True):
        self.model_dir = Path(model_dir) if model_dir else find_ms_model("bge-reranker-v2-m3")
        self.fp16 = fp16
        self.max_length = max_length
        self.verbose = verbose
        self._m = None
        self.load_seconds = 0.0
        self.n_calls = 0
        self.total_pairs = 0
        self.total_seconds = 0.0

    def _load(self):
        if self._m is not None:
            return
        from FlagEmbedding import FlagReranker
        if self.verbose:
            print(f"[reranker] 加载 {self.model_dir.name}（fp16={self.fp16}）…", flush=True)
        t = time.time()
        # 顺手静音：transformers 5.x 加载权重时也会刷 `Loading weights` 进度条，
        # 而加载耗时我们自己会 print（进程没卡死这件事不需要靠进度条证明）。
        with contextlib.redirect_stderr(io.StringIO()):
            self._m = FlagReranker(str(self.model_dir), use_fp16=self.fp16)
        self.load_seconds = time.time() - t
        if self.verbose:
            print(f"[reranker] 就绪，耗时 {self.load_seconds:.1f} 秒", flush=True)

    def score(self, query: str, docs: list[str], batch_size: int = 16) -> list[float]:
        """
        逐对打分，返回与 docs 等长的分数列表（越大越相关）。

        ⚠️ compute_score 对**单条**输入会返回 float 而不是 list —— 必须归一化成 list，
           否则调用方按列表索引会炸。这是 FlagEmbedding 的老毛病（同名函数返回类型不一致）。
        """
        if not docs:
            return []
        self._load()
        pairs = [[query, d] for d in docs]
        t = time.time()
        # ⚠️ 别传 `show_progress_bar=False` —— 新版 FlagEmbedding 的签名是
        #    `compute_score(..., **kwargs)`，多余参数会被**照单全收再丢弃**：
        #    不报错、也不生效（2026-09-21 实测推翻旧写法）。
        #    真正的开关写死在源码里：`tqdm(..., disable=len(pairs) < batch_size)`，
        #    候选数 ≥ batch_size 时必定刷进度条（默认走 stderr）。
        #    所以只能从外面把 stderr 静音 —— 异常仍会正常抛出，
        #    只是不让 \r 进度条把日志搅成一团（评测日志 grep 会失效）。
        with contextlib.redirect_stderr(io.StringIO()):
            out = self._m.compute_score(pairs, batch_size=batch_size,
                                        max_length=self.max_length)
        self.total_seconds += time.time() - t
        self.n_calls += 1
        self.total_pairs += len(pairs)
        if isinstance(out, (int, float)):       # 单条时返回标量
            return [float(out)]
        return [float(x) for x in out]

    def truncation_flags(self, query: str, docs: list[str],
                         max_length: int | None = None) -> list[bool]:
        """
        每篇 passage 在 (query, passage) 对齐后是否**被截断**。

        为什么要这个：`max_length` 管的是**对**的总长，不是 passage 单独的长度。
        实测 1 token ≈ 1.38 字符（中文 + title/section 里的数字英文），
        512 只够约 370 字的 passage —— 而 gold passage 均值 413 字。
        被截掉的部分重排模型看不见，可能把 gold 排到后面去。

        分段长度按 HF 的拼接惯例算。⚠️ **特殊 token 是 4 个不是 3 个**：
        bge-reranker-v2-m3 用的 XLMRobertaTokenizer 模板是
        `<s> query </s></s> passage </s>`（**双 </s>**），
        所以总长 = qn + dn + 4。按 BERT 的 `[CLS] q [SEP] d [SEP]` 记成 +3
        会**恰好差 1 个 token** —— 实测 506 vs 真实 507（2026-09-21 核对）。
        差 1 看着无所谓，但会让"卡在边界上"的那一条题被误判为未截断。
        （这个 +1 是靠 tokenizer(q, d, truncation=False) 反推出来的，
          不是猜的 —— 换模型家族必须重新反推，别照抄。）
        """
        self._load()
        tok = self._m.tokenizer
        ml = max_length or self.max_length
        qn = len(tok.encode(query, add_special_tokens=False))
        avail = ml - qn - 4
        return [len(tok.encode(d, add_special_tokens=False)) > avail for d in docs]

    def rerank(self, query: str, docs: list[str], topk: int | None = None,
               batch_size: int = 16) -> list[tuple[int, float]]:
        """
        返回 [(原始下标, 分数), ...]，按分数降序。

        返回**原始下标**而不是文档本身 —— 调用方要拿它去回查 chunk_id，
        重排后位置全变了，只返回文本会丢掉这个映射（第 8 步 scan_topk
        把分数当行号用的那个 bug，就是丢了映射的后果）。
        """
        sc = self.score(query, docs, batch_size=batch_size)
        order = sorted(range(len(sc)), key=lambda i: (-sc[i], i))
        if topk is not None:
            order = order[:topk]
        return [(i, sc[i]) for i in order]

    def stats(self) -> dict:
        return {
            "model": self.model_dir.name,
            "fp16": self.fp16,
            "load_seconds": round(self.load_seconds, 2),
            "n_calls": self.n_calls,
            "total_pairs": self.total_pairs,
            "total_seconds": round(self.total_seconds, 2),
            "ms_per_pair": round(self.total_seconds / self.total_pairs * 1000, 2)
            if self.total_pairs else 0.0,
        }


# ============================================================ 冒烟自检
def smoke() -> int:
    """
    用**已知答案的样本**校准打分器。

    为什么必须做（第 9 步的教训）：验证工具本身会骗你。
    检查代码写错时，它给你的"通过"是假的。所以先用几组
    「明显相干 / 明显不相干」的样本确认分数方向是对的。
    """
    r = Reranker(verbose=True)

    CASES = [
        # (query, 应该高分的文档, 应该低分的文档)
        ("蘇花古道的起點和終點在哪裡",
         "蘇花古道｜路線｜蘇花古道北起宜蘭縣蘇澳鎮，南至花蓮縣花蓮市，全長約11公里。",
         "臺北市｜氣候｜臺北市位於臺灣北部，氣候為副熱帶季風氣候，夏季炎熱潮濕。"),
        ("哪位化学工程学家创办了南开大学化工系",
         "张克忠｜生平｜张克忠，天津人，中国化学工程学家，创办了南开大学化工系。",
         "苹果公司｜产品｜苹果公司于2007年发布了第一代iPhone。"),
        ("天目山主峰海拔多少米",
         "天目山｜地理｜主峰清凉峰位于杭州市临安区与安徽绩溪交界处，海拔1787米。",
         "中国人口｜统计｜2020年第七次全国人口普查显示全国人口共141178万人。"),
    ]

    print("=" * 70)
    print("reranker 冒烟：3 组正负样本，分数必须 正 > 负")
    print("=" * 70)

    all_ok = True
    for i, (q, good, bad) in enumerate(CASES, 1):
        s = r.score(q, [good, bad])
        ok = s[0] > s[1]
        all_ok &= ok
        print(f"\n[{i}] {'✅' if ok else '❌'}  {q}")
        print(f"      相关   {s[0]:+8.3f}   {good[:38]}…")
        print(f"      不相关 {s[1]:+8.3f}   {bad[:38]}…")

    # 再验一个边界：query 和两篇都不相关时，分数应该整体偏低且接近
    qn = "量子纠缠的贝尔不等式如何推导"
    s = r.score(qn, ["蘇花古道北起宜蘭縣蘇澳鎮。", "天目山主峰海拔1787米。"])
    print(f"\n[边界] 两篇都不相关：{s[0]:+.3f} / {s[1]:+.3f}"
          f"  （差值 {abs(s[0]-s[1]):.3f}，应该都低且接近）")
    print("       注意：这不是判对错，是看模型有没有'乱给高分'")

    print("\n" + "=" * 70)
    print(f"结果: {'✅ 全部通过 —— 分数方向正确，可以用于评测' if all_ok else '❌ 有样本判反，先别用'}")
    print(f"统计: {r.stats()}")
    return 0 if all_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="bge-reranker-v2-m3 重排")
    ap.add_argument("--smoke", action="store_true", help="跑已知答案自检")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--no-fp16", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        return smoke()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
