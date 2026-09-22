#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 9 步 · 生成与引用（M3）
==========================

第 8 步结束时，系统已经能"找到"正确的原文片段（top-5 命中）。
但那还不是 RAG —— 用户拿到的是一堆 chunk，不是答案。

这一步补上链路的最后一段：**把检索到的 chunk 变成带引用的答案**。

三件事决定这一步的质量（都是设计问题，不是调参）
--------------------------------------------------
1. **上下文编号 [1] [2] ...**
   每个 chunk 前面挂编号，并要求模型在答案里回标。
   没有编号，模型只能说"根据资料"——你无法验证它到底用了哪条，
   而"每个事实可追溯到原文"恰恰是 RAG 相对纯 LLM 唯一的硬优势。
   编号让"解释性"从一句口号变成可以自动打分的东西（M4 的引用准确率就靠它）。

2. **拒答条款（最关键的一条）**
   LLM 的默认行为是**尽力回答**。不给拒答指令，库外问题会被自信地胡说，
   而且胡说的内容外面包着一层"参考资料里说"的皮，比纯幻觉更危险。
   明确写"找不到依据时必须回答「根据已有资料无法回答」"，
   是幻觉率的最大单点改进 —— 成本一行 prompt，收益贯穿整个 M4。

3. **先结论后依据**
   让输出结构可预测，M4 做自动评测时才好抽取（不然每条答案格式都不一样）。

为什么是双路 LLM（云端 + 本地）
---------------------------------
技术方案定的：A 路 dashscope（质量基线）、B 路本地 Qwen2.5-1.5B-Instruct（零成本）。
两路对比本身就是 M4 的一个消融维度（Faithfulness / 引用准确率差多少）。
另外本地路还能证明一件事：**整条 RAG 链路可以完全离线运行**——
这点在面试里很好用（"数据不出内网"是企业真实诉求）。

还有第三路 `echo`：不调任何模型，只把组装好的 prompt 打印出来。
别小看它 —— prompt 里的问题（编号错位、上下文被截断、section 混进去一坨噪声）
在 echo 下一眼就能看出来，而调 LLM 时你只会看到"答案好像不太对"，很难定位。

用法
----
    :: 看 prompt 长什么样，零成本（强烈建议第一次先跑这个）
    "...python313\\python.exe" src\\generator.py --backend echo --limit 1

    :: 云端一路，3 条内置问题
    "...python313\\python.exe" src\\generator.py

    :: 指定问题 + 看完整上下文
    "...python313\\python.exe" src\\generator.py --query "蘇花古道全長多少公里" --show-context

    :: 本地一路（需先下好 Qwen2.5-1.5B-Instruct）
    "...python313\\python.exe" src\\generator.py --backend local

    :: 批量跑文件里的问题（每行一条，# 开头是注释）
    "...python313\\python.exe" src\\generator.py --file eval\\m3_in_kb.txt

    :: 换云端模型
    "...python313\\python.exe" src\\generator.py --cloud-model qwen-max

    :: 接上重排（第 11 步 §11.13 的生产配置）—— 端到端要和评测脚本同配置
    "...python313\\python.exe" src\\generator.py --rerank --testset eval\\qa_testset_v1.jsonl --tag testset_rr

    :: 跑评测集（结果里带 qid / gold_chunk_id / gold_rank，供生成侧指标用）
    "...python313\\python.exe" src\\generator.py --backend echo --testset eval\\qa_testset_v1.jsonl --limit 2

    :: 不加载 13.6GB 的 faiss，改用暴力精确扫（慢但省内存）
    "...python313\\python.exe" src\\generator.py --vec-backend scan
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

try:
    # line_buffering：第 8 步踩过的坑 —— 重定向到日志时 stdout 是块缓冲，
    # 脚本跑了半天日志还是 0 字节，看起来像卡死。逐行刷 + 命令行 -u 双保险。
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from search_hybrid import HybridSearcher, PARQUET          # noqa: E402
from fetch_store import get_fetcher                        # noqa: E402

PROJECT = HERE.parent
RESULTS = PROJECT / "eval" / "results"

DEFAULT_QUERIES = [
    "台灣東部開發於古時的人行道路",
    "中國的化學工程學家",
    "這個地方氣候怎麼樣",
]


# 云端模型的常量与错误翻译放在 src/cloud_models.py（**零依赖模块**）。
# 这里 import 进来，于是 `from generator import classify_api_error` 依然可用；
# 而只想调一次 API 的工具（check_backends / judge_faith）直接从 cloud_models 取 ——
# **不必为了一个常量把 faiss 拖进来**（判断模块边界的标准不是代码长不长，
# 而是谁该为谁的依赖买单）。
from cloud_models import (  # noqa: E402,F401
    BASE_URL,
    CLOUD_MODELS,
    DEFAULT_CLOUD_MODEL,
    api_error_kind,
    api_key_or_exit,
    classify_api_error,
    is_fatal_api_error,
)

# ==================================================================== Prompt
#
# 这三段是整个 M3 最值钱的部分，逐条说明为什么这么写。
#
# 【为什么要给编号，而不是让模型自己去认】
#   模型无法稳定地"引用一段它自己都分不清边界的文本"。编号把引用变成
#   一个**离散符号选择**任务，模型的错误率立刻降一个数量级。
#
# 【为什么拒答要说"必须直接回答这句话"，而不是"如果不知道就说不知道"】
#   后者给了模型自由裁量空间，实测它会把"知道一点点"也算作"知道"。
#   给一句**逐字的规定话术**，判定才能自动化（M4 的拒答准确率靠关键词匹配就行）。
#
# 【为什么禁止"根据参考资料"这类开头】
#   套话会挤掉真正有信息量的第一句话，而且让"先结论后依据"的格式失效。
#   顺带一个小收益：省 token。

SYSTEM_PROMPT = """你是一个严谨的中文知识库问答助手。你**只能**依据用户提供的【参考资料】回答问题。

必须遵守以下规则：
1. 答案中的每一个事实，都要在该事实后面标注来源编号，格式如 [1] 或 [2][3]。
2. 编号只能是【参考资料】中真实出现过的编号，禁止编造。
3. 如果【参考资料】中没有足够信息回答问题，你必须直接回答「根据已有资料无法回答」，
   不要使用你自己的知识补充，不要猜测，不要勉强作答。
4. 先给结论（一到两句话），再给依据。
5. 不要以「根据参考资料」「参考资料中提到」之类的话开头。
6. 用中文回答，简洁、直接。"""

USER_TEMPLATE = """【参考资料】
{context}

【问题】
{query}"""

# 拒答的标准话术 —— 也是 M4 判定"是否拒答"的匹配串
REFUSAL_TEXT = "根据已有资料无法回答"

# ⚠️ 判定拒答时必须同时列繁简两种写法。
# 语料是**繁体**中文维基，模型跟着语料的语体会用繁体回答 ——
# 第一版只写了简体「无法回答」，结果 Q3 明明拒答了，判定却是「拒答 0 条」。
# 这类错误的隐蔽性在于：程序不报错、指标也「正常」，只是数字是错的。
#
# ⚠️ 2026-09-21 又补了一次（§11.14）：只认逐字那句「无法回答」不够 ——
# 模型经常换措辞（"参考资料中没有提及…"）。实测 v1-20f55e 就这样：
#   「根据现有资料，【参考资料】中没有提及袁說友的朋友对他的评价[2]。」
# 它守规矩拒答了，却被判成"硬答"（最危险的那一类），只因为
# ① 标记表里写的是"资料中没有"，而原文多了个 `】`；
# ② 换措辞的说法没进表。
# ⇒ 匹配前统一**去掉括号与空白**，并把常见换说法补进表。
REFUSAL_MARKERS = [
    # 简体
    "无法回答", "没有任何相关", "资料中没有", "资料中未", "资料里没有", "参考资料中没有",
    "未提供相关信息", "无法从参考资料", "没有足够信息", "没有提及", "未提及",
    "没有说明", "未说明", "无法得知", "无从得知",
    # 繁体
    "無法回答", "沒有任何相關", "資料中沒有", "資料中未", "資料裡沒有", "參考資料中沒有",
    "未提供相關信息", "無法從參考資料", "沒有足夠信息", "沒有提及", "未提及",
    "沒有說明", "未說明", "無法得知", "無從得知",
    # 中英混排（模型偶尔夹英文）
    "无法确定", "無法確定",
]

# 去括号/空白后再匹配 —— 见上面 v1-20f55e 的教训
_BRACKET_RE = re.compile(r"[\[\]【】〔〕（）()「」『』\s]")

CITE_RE = re.compile(r"\[(\d{1,3})\]")


def build_context(hits):
    """
    hits: [{"title","section","chunk_text","row","chunk_id","vrank","brank","score"}, ...]
    返回拼好的上下文字符串。

    拼接格式延续第 6 步定的契约：`{title}｜{section}` 做头（全角竖线），
    section 为空时不留空段 —— 这和第 6 步切块时拼 chunk 的规则是同一条，
    不要在这里另起一套，否则检索时学到的语义和喂给 LLM 的文本会对不上。
    """
    blocks = []
    for i, h in enumerate(hits, 1):
        head = f"[{i}] {h['title']}"
        sec = (h.get("section") or "").strip()
        if sec:
            head += f"｜{sec}"
        body = (h.get("chunk_text") or "").strip()
        blocks.append(f"{head}\n{body}")
    return "\n\n".join(blocks)


def build_messages(query, hits):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            context=build_context(hits), query=query)},
    ]


# ==================================================================== 后端

MS_CACHE = Path(r"E:\AI-learning\ms-cache\models")


def find_local_model(name_substr: str):
    """
    在 ModelScope 缓存里按「名字片段」找模型目录。

    为什么不复用 `vectorize.find_model`：那个函数的模式串是**硬编码的**
    `bge-{kind}-zh`，只认 bge 系列（它的参数取值就只有 large / small）。
    传 "qwen-instruct" 进去会拼成 `bge-qwen-instruct-zh` —— 永远找不到，
    而且报错信息会误导你以为"模型没下载"。
    这里按名字片段 glob，通用于任何模型。
    """
    if not MS_CACHE.exists():
        return None
    for p in MS_CACHE.glob(f"*{name_substr}*"):
        for cand in (p / "snapshots" / "master", p):
            if (cand / "config.json").exists():
                return cand
    return None


class EchoBackend:
    """不调模型，只回显 prompt。用来审查上下文组装是否正确。"""

    name = "echo"
    is_model = False          # 见 run_one：echo 的"答案"其实是 prompt 本身，不能拿去做核查

    def generate(self, messages, max_new_tokens):
        return ("（echo 模式：未调用任何模型，下面是即将送进 LLM 的原文）\n\n"
                + "=" * 30 + " SYSTEM " + "=" * 30 + "\n"
                + messages[0]["content"] + "\n\n"
                + "=" * 30 + " USER " + "=" * 30 + "\n"
                + messages[1]["content"]), {}


class CloudBackend:
    """
    A 路：dashscope（阿里百炼），走 OpenAI 兼容端点。

    为什么用 openai SDK 而不是 dashscope SDK：兼容端点是标准协议，
    以后换任何一家（DeepSeek / 智谱 / vLLM 本地服务）都只改 base_url，
    业务代码一行不动。这是"可替换"的写法。
    """

    name = "dashscope"
    is_model = True

    def __init__(self, model=None, temperature=0.0, timeout=60, max_retries=2):
        from openai import OpenAI
        key = api_key_or_exit()          # 没设 key 就带着可执行提示退出
        self.model = model or DEFAULT_CLOUD_MODEL
        self.temperature = temperature
        # ⚠️ 不要自己写 for 循环重试 —— SDK 已经内置，而且做得比手写更对：
        #    它只重试**可重试**的类别（429 / 5xx / 连接错误），并尊重服务端的
        #    Retry-After 头；对 400/401/403 这类"重试一万次也一样"的错误不会浪费你时间。
        #    这里显式写出来，是为了让"我们考虑过重试"这件事留在代码里，
        #    而不是靠一个看不见的默认值（它默认就是 2，改掉默认没人会发现）。
        self.client = OpenAI(
            api_key=key,
            base_url=BASE_URL,           # 端点只写一处（cloud_models），体检脚本共用
            timeout=timeout,
            max_retries=max_retries,
        )

    def generate(self, messages, max_new_tokens):
        t = time.time()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_new_tokens,
        )
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
            "latency": time.time() - t,
        }
        return resp.choices[0].message.content, usage


class LocalBackend:
    """
    B 路：本地 Qwen2.5-1.5B-Instruct，4bit 量化。

    ⚠️ transformers 5.x 的两个坑（本机 5.17.0 实测）：
      - `from_pretrained` 的 `torch_dtype=` 已被 `dtype=` 取代（旧名会告警）
      - 4bit 加载本身不需要传 dtype，量化配置由 BitsAndBytesConfig 全权决定
    另外必须用 Instruct 版权重：base 版没有 chat template，
    `apply_chat_template` 会直接抛错，而且它也不会遵循引用/拒答这类指令。
    """

    name = "local"
    is_model = True

    def __init__(self, model_dir, quant="4bit", max_new_tokens=512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.model_dir = Path(model_dir)      # 落进结果文件用（见 run_one 的 gen_config）
        # 显示名：ModelScope 缓存是 <repo>/snapshots/master，直接用 .name 会只得到 "master"，
        # 看不出到底用的哪个模型。所以往上找一层。
        self.model_name = (self.model_dir.parent.parent.name
                           if self.model_dir.name == "master" else self.model_dir.name)
        self.quant = quant
        t = time.time()
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        kwargs = {"device_map": "auto"}
        if quant == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",          # nf4 比 fp4 在权重分布上更稳
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,     # 二次量化，再省 ~0.4GB 显存
            )
        else:
            # transformers 5.x：dtype 取代 torch_dtype
            kwargs["dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(str(model_dir), **kwargs)
        self.model.eval()
        print(f"[本地模型] 加载完成 {time.time() - t:.1f} 秒  "
              f"显存占用 {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB"
              if torch.cuda.is_available() else f"[本地模型] 加载完成 {time.time() - t:.1f} 秒（CPU）")

    def generate(self, messages, max_new_tokens):
        torch = self.torch
        text = self.tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        n_in = inputs["input_ids"].shape[1]
        t = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,                       # 贪心解码：评测要可复现
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
            )
        gen_ids = out[0][n_in:]
        answer = self.tok.decode(gen_ids, skip_special_tokens=True).strip()
        return answer, {"prompt_tokens": n_in, "completion_tokens": int(gen_ids.shape[0]),
                        "latency": time.time() - t}


# ==================================================================== 核查


def parse_citations(answer, k):
    """
    从答案里抽出 [n] 引用，返回 (用到的编号去重升序, 非法编号)。

    为什么要检查"非法编号"：模型偶尔会写 [7]，而上下文只有 5 条 ——
    这是**最难发现的一类幻觉**，因为答案读起来完全合理，编号看起来也很专业。
    不做这个检查，你会以为引用机制在工作，实际上它在编。
    """
    nums = [int(x) for x in CITE_RE.findall(answer or "")]
    used = sorted(set(nums))
    bad = [n for n in used if n < 1 or n > k]
    return used, bad


def is_refusal(answer):
    a = _BRACKET_RE.sub("", answer or "")
    return any(m in a for m in REFUSAL_MARKERS)


# ==================================================================== 取原文
#
# 取原文层已抽到 `fetch_store.py`。为什么值得单独一个模块：
#   它只依赖 pyarrow / numpy，**不依赖 faiss / torch** ——
#   于是 `diagnose_fetch.py` 能在不加载 13.6 GB 索引的情况下量它的耗时（1 分钟出结果）。
#
# 两条实现：
#   RowTextFetcher   按行号定位分片 + isin 过滤。基线，实测仍要 **1.7 s/次** ——
#                    它只解决了「去哪个分片找」，没解决「进了分片还要全扫 552 MB」。
#   BlobFetcher      侧车（JSON + int64 偏移数组）+ mmap，O(1) 点查。第 12 步新增。
#
# 由 --fetch-store 选择（auto：有侧车就用、没有就显式提示并退回基线 ——
# 静默退回是最坏的选择，你会以为优化生效了，其实没有）。


# ==================================================================== 检索


# ============================================ 检索链路参数：单一来源
# 为什么要有这一段（2026-09-22 界面报错后的复盘）：
# retrieve() 的参数全部来自 args。命令行版用 argparse 定义（自带默认值），
# 而 Gradio 界面**自己造了一个命名空间**再传给 retrieve() —— 参数清单于是有了两份。
# 第 12 步新增 args.fetch_store 时，命令行版一切正常，
# 界面版却是 `AttributeError: 'Namespace' object has no attribute 'fetch_store'`。
#
# 真正难受的是它**暴露的时机**：界面要把 13.6 GB 索引加载完、
# 用户输入问题点"提问"之后才炸 —— 改一行参数，代价是一次完整的启动 + 一次错误演示。
#
# 所以参数只在这里定义一次：
#   · argparse 用 RETRIEVAL_DEFAULTS[...] 当 default=  （不再是字面量）
#   · 任何自己造命名空间的调用方（界面 / 评测 / 自检）都走 ensure_retrieval_args()
# 缺哪个补哪个，**并且把补了哪些打印出来**。
# 不静默是关键：它把"两份清单漂移了"变成一行可见的提示，
# 而不是一个要等到用户点击才出现的 AttributeError。
RETRIEVAL_DEFAULTS = {
    "topk": 5,
    "topn": 100,
    "rrf_k": 10,
    "w_vec": 1.0,
    "w_bm25": 1.0,
    "rerank": False,
    "rerank_pool": 50,
    "rerank_max_length": 1024,
    "rerank_batch": 32,
    "fetch_store": "auto",
}

_FILLED_WHO: set[str] = set()


def ensure_retrieval_args(args, who: str = "调用方"):
    """
    给 args 补齐 retrieve() 需要的字段（就地补，并返回它）。

    只补"完全没有这个属性"的：已经传了值（哪怕是 False / 0）一律不动 —— 
    这里不做任何语义判断，否则会变成"偷偷改掉用户设的参数"。
    """
    missing = [k for k in RETRIEVAL_DEFAULTS if not hasattr(args, k)]
    for k in missing:
        setattr(args, k, RETRIEVAL_DEFAULTS[k])
    if missing and who not in _FILLED_WHO:
        _FILLED_WHO.add(who)
        print(f"[参数] {who} 的命名空间缺少 {len(missing)} 个检索参数，已按默认值补齐："
              f"{', '.join(missing)}")
        print("       （不是报错，但说明参数清单有两份 —— 见 generator.RETRIEVAL_DEFAULTS；"
              "建议调用方直接用 build_parser() 或显式声明这些字段）")
    return args


def make_searcher(args):
    """
    复用第 8 步的 HybridSearcher。

    注意它 __init__ 里会读 args.mode / args.vec_backend / args.nprobe / args.fuse，
    而 argparse 的 Namespace 恰好全都有这些字段 —— 但 fuse 在 generator 里
    不暴露给用户（gate 策略样本不足，默认关闭，第 8 步就说清楚了）。
    所以显式构造一个 SimpleNamespace，避免"命令行能传但没人验证过"的开关。
    """
    ns = SimpleNamespace(
        mode="hybrid",
        vec_backend=args.vec_backend,
        nprobe=args.nprobe,
        fuse="rrf",            # gate 策略在 3 条样本上拟合，不进生成链路
        gate_margin=0.10,
        gate_weak_w=0.3,
    )
    return HybridSearcher(ns)


def retrieve(searcher, query, args, qv, reranker=None):
    """
    返回 (hits, timing)。hits 按最终排名，带原文、两路排名与 chunk_id。

    重排的顺序很关键（照抄 evaluate.py 的口径，不要"优化"）：
      search(pool) → 取原文 → 只在池子前 rr_pool 个候选上重排 → 截 topk
    不能先截 topk 再重排 —— 那样重排只能在 5 个候选里重排，
    等于把这个模型最大的价值（把 gold 从第 30 名提到第 1 名）扔掉了。

    args 先过一遍 ensure_retrieval_args()：界面的命名空间少字段时**补齐**而不是崩
    （2026-09-22 的实际事故，见 RETRIEVAL_DEFAULTS 上面的注释）。
    """
    args = ensure_retrieval_args(args, "retrieve()")
    t0 = time.time()
    # ⚠️ 第 4 个参数（search 的 topk）是**融合输出条数** —— `rrf_fuse(..., topn=topk)`
    #    会把融合结果直接截到这里。所以必须传**池子大小**，不能传最终的 --topk。
    #    2026-09-21 实测踩到：传了 args.topk=5，融合只剩 5 条，重排就在这 5 条里重排 ——
    #    等于把这个模型最大的价值（把 gold 从第 30 名提到第 1 名）整个扔掉，
    #    而指标只表现为 "R@1 涨到 0.808 但 R@5 掉到 0.875"，**不报错、看着还挺合理**。
    #    抓到它的不是代码，是下面那句"检索侧口径应与评测脚本一致"的交叉核对。
    #    （评测脚本传的是 args.pool, args.pool —— 两边必须一致。）
    rows, scores, per_path, _, (t_vec, t_bm, t_fuse) = searcher.search(
        query, qv, "hybrid", args.topn, args.topn, args.w_vec, args.w_bm25, args.rrf_k)
    fused_n = len(rows)

    pos_v = {int(r): i + 1 for i, r in enumerate(per_path.get("vector", []))}
    pos_b = {int(r): i + 1 for i, r in enumerate(per_path.get("bm25", []))}
    # 按行号回查（快约 4 倍）—— rows 本来就是行号，
    # 没必要先换成随机哈希 chunk_id 再去 isin 全扫 4 个分片。
    t_f = time.time()
    got = get_fetcher(searcher.ids, PARQUET, args.fetch_store).fetch(rows)
    t_fetch = time.time() - t_f

    rr_scores = {}
    t_rr = 0.0
    if reranker is not None:
        from vectorize import build_text          # 与索引侧/评测侧同一个拼接函数
        rr_pool = args.rerank_pool or args.topn
        # 防呆断言：上面那个 topk 参数一旦被误设成最终条数，融合会被提前截断，
        # 重排池静默缩水而指标只是"看着还行"。这个断言的成本是几微秒。
        if fused_n < min(rr_pool, args.topn):
            raise SystemExit(
                f"[错误] 融合只给出 {fused_n} 条候选，却要重排 {rr_pool} 条 —— "
                f"search(..., topk=?) 的第 4 个参数必须是池子大小（--pool={args.topn}），"
                f"不能是最终的 --topk，否则重排池被静默截断（见 retrieve() 注释）")
        rr_rows = [int(r) for r in rows[:rr_pool]]
        passages = []
        for r in rr_rows:
            rec = got.get(searcher.ids[r])
            passages.append(build_text(rec["title"], rec["section"] or "", rec["chunk_text"] or "")
                            if rec else "")
        r0 = time.time()          # ⚠️ 必须用新变量，不能复用 t0 ——
        order = reranker.rerank(query, passages, topk=None, batch_size=args.rerank_batch)
        t_rr = time.time() - r0
        rows = [rr_rows[j] for j, _ in order]
        rr_scores = {rr_rows[j]: float(s) for j, s in order}

    # 截到 topk 之后再组装 —— 上面的重排用到了完整池子的分数
    rows = [int(r) for r in rows[:args.topk]]

    hits = []
    for rank, r in enumerate(rows, 1):
        rec = got.get(searcher.ids[r])
        if rec is None:
            continue
        hits.append({
            "rank": rank,
            "row": r,
            "chunk_id": rec["chunk_id"],
            "title": rec["title"],
            "section": rec["section"] or "",
            "chunk_text": rec["chunk_text"] or "",
            "score": rr_scores.get(r) if reranker is not None
                     else (float(scores[rank - 1]) if scores is not None else None),
            "vrank": pos_v.get(r),
            "brank": pos_b.get(r),
        })
    timing = {"vector": t_vec, "bm25": t_bm, "fuse": t_fuse, "fetch": t_fetch,
              "rerank": t_rr, "fused_n": fused_n, "rr_pool_n": len(rr_rows) if reranker else None,
              "retrieve_total": time.time() - t0}
    return hits, timing


# ==================================================================== 主流程


def run_one(searcher, reranker, backend, query, args, idx, meta=None):
    t_all = time.time()
    qv = searcher._encode([query])[0]
    hits, timing = retrieve(searcher, query, args, qv, reranker)

    if not hits:
        print("\n[警告] 检索无结果，跳过生成")
        return None

    messages = build_messages(query, hits)
    answer, usage = backend.generate(messages, args.max_new_tokens)
    total = time.time() - t_all

    if backend.is_model:
        used, bad = parse_citations(answer, len(hits))
        refused = is_refusal(answer)
    else:
        # echo 的"答案"就是 prompt 本身 —— 里面天然带 [1]...[k] 和那句拒答话术，
        # 拿它做核查会得到「引用了全部编号 + 判定拒答」的假结果。
        # 这个假结果的害处不小：它看起来像核查逻辑坏了，实际是喂错了输入。
        used, bad, refused = [], [], False

    # ---- 引用 → chunk_id 的映射（生成侧指标的核心） ----
    # 光知道"答案里出现了 [3]"没用，要能回答"第 3 条是不是 gold"。
    # 重排之后位置全变了，这个映射只能在这里做（hits 是唯一权威顺序）。
    hit_cids = [h["chunk_id"] for h in hits]
    cited_cids = [hit_cids[n - 1] for n in used if 1 <= n <= len(hit_cids)]

    meta = meta or {}
    gold_cid = meta.get("gold_chunk_id")
    gold_rank = next((i + 1 for i, c in enumerate(hit_cids) if c == gold_cid), None)

    # ---------------- 打印 ----------------
    print()
    print("=" * 76)
    print(f"[{idx}] Q: {query}"
          + (f"   （{meta.get('qid')} · {meta.get('kind')}）" if meta.get("qid") else ""))
    print("=" * 76)
    lat = (f"检索 {timing['retrieve_total'] * 1000:.0f} ms"
           f"（向量 {timing['vector'] * 1000:.0f} + BM25 {timing['bm25'] * 1000:.0f}"
           f" + 融合 {timing['fuse'] * 1000:.1f}"
           + (f" + 取原文 {timing['fetch'] * 1000:.0f} + 重排 {timing['rerank'] * 1000:.0f}"
              if args.rerank else "")
           + "）")
    print(f"【检索】{len(hits)} 条命中 · {lat}")
    for h in hits:
        sec = h["section"] or "—"
        text = h["chunk_text"].replace("\n", " ")[:60]
        score = f"{h['score']:.5f}" if h["score"] is not None else "—"
        print(f"   [{h['rank']}] {score}  (向量#{h['vrank'] or '—'} BM25#{h['brank'] or '—'})"
              f"  {h['title']} · {sec}")
        print(f"       {text}")

    if args.show_context:
        print("\n────── 送进模型的完整上下文 ──────")
        print(build_context(hits))
        print("────── 上下文结束 ──────")

    print(f"\n【答案】({usage.get('latency', 0):.2f}s"
          + (f" · 输入 {usage.get('prompt_tokens')} tok / 输出 {usage.get('completion_tokens')} tok"
             if usage.get("prompt_tokens") else "")
          + ")")
    for line in (answer or "").splitlines():
        print(f"   {line}")

    # ---------------- 自查 ----------------
    flags = []
    if refused:
        flags.append("拒答")
    if bad:
        flags.append(f"⚠️ 非法编号 {bad}（上下文只有 {len(hits)} 条）")
    if not used and not refused:
        flags.append("⚠️ 无任何引用标注")
    if backend.is_model:
        print(f"\n【核查】引用 {used if used else '无'} · "
              f"合法编号 {[n for n in used if n not in bad]} · "
              f"总耗时 {total:.2f}s"
              + ("   → " + " · ".join(flags) if flags else ""))
    else:
        print(f"\n【核查】echo 模式 —— 上面是 prompt 原文而非模型答案，不做引用/拒答判定")

    return {
        "idx": idx,
        "qid": meta.get("qid"),
        "kind": meta.get("kind"),
        "query": query,
        "backend": backend.name,
        "hits": [{k: h[k] for k in ("rank", "chunk_id", "title", "section", "score",
                                    "vrank", "brank")} for h in hits],
        # 原文正文必须落盘 —— 只存 title/chunk_id 的话，**离线算不出 faithfulness**：
        # 判"这句话有没有被它引用的块支撑"必须逐字比对那段文字。
        # 2026-09-21 实测踩到：拿旧结果文件跑指标脚本，覆盖率全是 0、
        # 数字违规 12/15 句 —— 判据没坏，是它没拿到要比对的文本。
        # 顺带一个好处：结果文件成了自证材料，别人拿这一个文件就能复核答案有没有依据。
        "contexts": [{"rank": h["rank"], "chunk_id": h["chunk_id"], "title": h["title"],
                      "section": h["section"], "text": h["chunk_text"]} for h in hits],
        "hit_chunk_ids": hit_cids,
        "gold_chunk_id": gold_cid,
        "gold_title": meta.get("gold_title"),
        "gold_answer": meta.get("answer"),
        "gold_rank": gold_rank,              # gold 在最终上下文里排第几（None = 没进上下文）
        "answerable_in_kb": meta.get("answerable_in_kb"),
        "context_chars": len(build_context(hits)),
        "answer": answer,
        "cited": used,
        "cited_chunk_ids": cited_cids,
        "bad_citations": bad,
        "refused": refused,
        "gen_config": {
            "topk": args.topk, "pool": args.topn, "rrf_k": args.rrf_k,
            "rerank": bool(args.rerank),
            "rerank_pool": (args.rerank_pool or args.topn) if args.rerank else None,
            "rerank_max_length": args.rerank_max_length if args.rerank else None,
            "temperature": getattr(backend, "temperature", None),
            # ⚠️ 不能直接写 getattr(backend, "model")：CloudBackend 的 .model 是**模型名**
            #    （字符串），而 LocalBackend 的 .model 是**模型对象** ——
            #    2026-09-21 实测：本地那次跑完全程 13.6 分钟，写文件时
            #    `TypeError: Object of type Qwen2ForCausalLM is not JSON serializable`，
            #    结果文件 0 字节，全部结果丢失。所以这里显式判类型。
            "cloud_model": (backend.model if isinstance(getattr(backend, "model", None), str)
                            else None),
            "local_model": getattr(backend, "model_name", None),
            "quant": getattr(backend, "quant", None),
        },
        "timing": timing,
        "usage": usage,
        "total_seconds": total,
    }


def load_items(args):
    """
    返回 [(question, meta), ...]。meta 空字典表示这条不是来自评测集。

    为什么要单独走 --testset 而不是让用户自己抽问题行：
    生成侧指标要回答的是「模型引用的那条是不是 gold」，没有 gold 就无从判起。
    评测集 jsonl 里本来就带 gold_chunk_id / answer / kind，
    拆成纯文本行再让评测脚本去猜着对齐，是把已有的信息扔掉再靠文件名拼回来。
    """
    items = [(q, {}) for q in (args.query or [])]

    for f in (args.file or []):
        p = Path(f)
        if not p.is_absolute():
            p = PROJECT / f
        if not p.exists():
            raise SystemExit(f"[错误] 问题文件不存在：{p}")
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                items.append((line, {}))

    if args.testset:
        p = Path(args.testset)
        if not p.is_absolute():
            p = PROJECT / args.testset
        if not p.exists():
            raise SystemExit(f"[错误] 评测集不存在：{p}")
        n = 0
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            it = json.loads(line)
            items.append((it["question"], it))
            n += 1
        print(f"[评测集] {p.name}：读入 {n} 条（带 qid / gold，可算引用准确率）")

    return items or [(q, {}) for q in DEFAULT_QUERIES]


def build_backend(args):
    if args.backend == "echo":
        return EchoBackend()
    if args.backend == "dashscope":
        b = CloudBackend(model=args.cloud_model, temperature=args.temperature)
        print(f"[云端模型] {args.cloud_model} @ dashscope 兼容端点（temperature={args.temperature}）")
        return b
    if args.backend == "local":
        d = Path(args.local_model) if args.local_model else find_local_model("Qwen2.5-1.5B-Instruct")
        if d is None or not Path(d).exists():
            raise SystemExit(
                "[错误] 找不到本地 Qwen2.5-1.5B-Instruct。先跑：\n"
                '  "...python313\\python.exe" src\\download_models.py --only qwen2.5-1.5b-instruct\n'
                "或用 --local-model 指定目录")
        print(f"[本地模型] {d}")
        return LocalBackend(d, quant=args.quant, max_new_tokens=args.max_new_tokens)
    raise SystemExit(f"[错误] 未知 backend: {args.backend}")


def build_parser() -> argparse.ArgumentParser:
    """
    参数定义只写一遍。

    抽成函数不是为了好看：**别的调用方要能拿到同一份清单**。
    界面（app/gradio_app.py）原来是手抄一份参数表，第 12 步加了 --fetch-store 之后
    它没跟上，于是用户点"提问"直接报 AttributeError（2026-09-22）。
    现在它可以直接 build_parser()，或者至少 ensure_retrieval_args() 兜底。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="dashscope",
                    choices=["echo", "dashscope", "local"],
                    help="echo=只回显 prompt（零成本，先跑这个验证上下文组装）")
    ap.add_argument("--cloud-model", default=DEFAULT_CLOUD_MODEL,
                    help=f"dashscope 模型名，默认 {DEFAULT_CLOUD_MODEL}（单一来源常量）。"
                         f"可选：{' / '.join(CLOUD_MODELS)}。"
                         "⚠️ qwen-plus 的免费额度已耗尽，要用得先在百炼控制台充值")
    ap.add_argument("--local-model", default=None, help="本地模型目录（默认自动找）")
    ap.add_argument("--quant", default="4bit", choices=["4bit", "fp16"])
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="默认 0，评测要可复现")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--query", action="append", default=None)
    ap.add_argument("--file", action="append", default=None,
                    help="问题文件，每行一条（# 注释）。**可多次传入**，按顺序拼接 —— "
                         "复测时要跑多组问题，一次加载索引比跑三次省 3 分钟")
    ap.add_argument("--testset", default=None,
                    help="评测集 jsonl（如 eval/qa_testset_v1.jsonl）。与 --file 的区别："
                         "它会**带上 qid / gold_chunk_id / gold_answer / kind** 落进结果文件 —— "
                         "生成侧指标（引用准确率、拒答归因）全靠这几个字段，"
                         "只给纯文本问题行的话，评测时无法与 gold 对齐。")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    # ⚠️ 以下 4 个默认值都来自 RETRIEVAL_DEFAULTS（单一来源），不要再写字面量
    ap.add_argument("--topk", type=int, default=RETRIEVAL_DEFAULTS["topk"])
    ap.add_argument("--topn", "--pool", dest="topn", type=int,
                    default=RETRIEVAL_DEFAULTS["topn"],
                    help="召回池：两路各取 topn 再 RRF 融合（默认 100）。"
                         "--pool 是同一参数的别名，与评测脚本口径一致")

    # ---------------- 重排（第 11 步 §11.13 的生产配置） ----------------
    # 加这一段的原因：generator / gradio 一直只跑 RRF 融合，**没接重排** ——
    # 也就是第 11 步辛苦量出来的 R@1 +0.019 从来没进过生成链路。
    # "评测脚本用了什么配置，端到端就该用什么配置"，否则测的是另一套系统。
    ap.add_argument("--rerank", action="store_true",
                    default=RETRIEVAL_DEFAULTS["rerank"],
                    help="开 cross-encoder 重排（bge-reranker-v2-m3）。生产配置建议开")
    ap.add_argument("--rerank-pool", type=int, default=RETRIEVAL_DEFAULTS["rerank_pool"],
                    help="喂给重排的候选数（在召回池基础上再截）。"
                         "§11.13 实测 100→50：重排延迟 -27%% 而指标不降；"
                         "⚠️ 不要用 --pool 去降，它还管 RRF 的融合输入，降了要掉 R@5")
    ap.add_argument("--rerank-max-length", type=int,
                    default=RETRIEVAL_DEFAULTS["rerank_max_length"],
                    help="重排 (query,passage) 对的截断长度。默认 1024 —— "
                         "512 会**静默**截掉 20.2%% 的 gold 且恰好伤在 R@1 上（§11.13）")
    ap.add_argument("--rerank-batch", type=int, default=RETRIEVAL_DEFAULTS["rerank_batch"])

    # ---------------- 取原文（第 12 步） ----------------
    # 为什么默认 auto 而不是直接写死侧车：侧车是**派生产物**，
    # 别人 clone 下来时还没有（2.9 GB，不入库）。写死会导致开箱即失败；
    # 而静默退回 parquet 又会让"优化生效了"变成错觉 —— 所以 auto 会打印一行提示。
    ap.add_argument("--fetch-store", default=RETRIEVAL_DEFAULTS["fetch_store"],
                    choices=["auto", "blob", "parquet"],
                    help="取原文实现：blob=侧车 mmap 随机访问（约 5 ms）；"
                         "parquet=基线全扫（约 1.7 s）；auto=有侧车就用。"
                         "建侧车：python src/fetch_store.py --build")
    ap.add_argument("--rrf-k", type=int, default=RETRIEVAL_DEFAULTS["rrf_k"])
    ap.add_argument("--w-vec", type=float, default=RETRIEVAL_DEFAULTS["w_vec"])
    ap.add_argument("--w-bm25", type=float, default=RETRIEVAL_DEFAULTS["w_bm25"])
    ap.add_argument("--vec-backend", default="faiss", choices=["faiss", "scan"],
                    help="faiss=249ms（13.6GB 常驻）；scan=暴力精确 8.2s（省内存）")
    ap.add_argument("--nprobe", type=int, default=512,
                    help="默认 512 —— 第 8 步实测此档 recall=1.000 且只用 249ms")
    ap.add_argument("--show-context", action="store_true", help="打印送进模型的完整上下文")
    ap.add_argument("--tag", default="", help="结果文件名后缀，如 in_kb / out_kb")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    # echo 的唯一用途就是审 prompt，自动打开全文 —— 否则回显的上下文被截成 60 字，
    # 等于白跑一趟（这个坑我第一版就踩了）。
    if args.backend == "echo":
        args.show_context = True

    items = load_items(args)
    if args.limit:
        items = items[:args.limit]

    print("=" * 76)
    print("第 9 步 · 生成与引用（M3）")
    print("=" * 76)
    print(f"后端      : {args.backend}")
    print(f"问题数    : {len(items)}")
    print(f"检索      : hybrid topk={args.topk} pool={args.topn} RRF k={args.rrf_k}")
    print(f"重排      : " + (f"开（候选 {args.rerank_pool or args.topn} · "
                             f"max_length={args.rerank_max_length} · "
                             f"batch={args.rerank_batch}）" if args.rerank else "关"))
    print(f"向量后端  : {args.vec_backend}"
          + (f" nprobe={args.nprobe}" if args.vec_backend == "faiss" else "（暴力精确）"))
    print()

    searcher = make_searcher(args)
    reranker = None
    if args.rerank:
        from rerank import Reranker
        reranker = Reranker(verbose=True, max_length=args.rerank_max_length)
    backend = build_backend(args)

    # ---------------- 边跑边写 ----------------
    # 为什么要这样（2026-09-21 的教训）：原来是跑完全部再一次性落盘，
    # 结果本地那次跑了 13.6 分钟后写文件时抛 TypeError，**结果文件 0 字节，全丢**。
    # 逐条写出 + flush 的三个好处：
    #   ① 崩了只丢一条，不是一整个实验；
    #   ② 跑批中途就能看进度（本项目第 8 步的纪律：进度看产物文件，别 grep 日志）；
    #   ③ 中途 Ctrl-C 也能保住已完成的部分（重跑时可用 --limit 接着来）。
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out = RESULTS / f"gen_{args.backend}{tag}.jsonl"

    records = []
    t0 = time.time()
    with out.open("w", encoding="utf-8", newline="\n") as fout:
        for i, (q, meta) in enumerate(items, 1):
            try:
                rec = run_one(searcher, reranker, backend, q, args, i, meta)
            except Exception as e:
                # 单条失败不该毁掉整批：跑到第 100 条才 403，前 99 条的价值不该丢。
                # 但「要不要继续」取决于错误性质 ——
                # 额度用尽 / Key 无效 / 模型名错，重试一万次也是同样的结果，
                # 继续跑只会白等 129 次，还把日志刷成 129 行一模一样的报错。
                reason, hint = classify_api_error(e)
                print(f"\n[错误] 第 {i}/{len(items)} 条失败：{reason}")
                if is_fatal_api_error(e):
                    print(f"\n[中止] {hint}")
                    print(f"       已跑完的 {len(records)} 条已落盘（逐条 flush）：{out}")
                    break
                print(f"       {hint}（跳过该条，继续）")
                continue
            if not rec:
                continue        # run_one 在「检索无结果」时返回 None
            try:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
            except TypeError as e:
                # 一条序列化不了不该拖垮整批；但必须**响亮地**报出来
                print(f"\n[错误] 第 {i} 条结果无法序列化（{e}）—— 该条已跳过，其余继续。"
                      f"多半是往记录里塞了非 JSON 对象（如模型实例）")
                continue
            records.append(rec)

    # ---------------- 汇总 ----------------
    n = len(records)
    refused = sum(1 for r in records if r["refused"])
    bad = sum(1 for r in records if r["bad_citations"])
    nocite = sum(1 for r in records if not r["cited"] and not r["refused"])
    # 检索总耗时按分项相加算 —— 不要用 timing["retrieve_total"]：
    # 那个字段曾经因为计时变量被复用而只记录了重排那一段（见 retrieve() 注释里的 r0）。
    ret = [sum(r["timing"][k] or 0 for k in ("vector", "bm25", "fuse", "fetch", "rerank"))
           for r in records]

    gen = [r["usage"].get("latency", 0) for r in records]

    print()
    print("=" * 76)
    print("GENERATOR_OK")
    print(f"  后端        : {args.backend}")
    print(f"  成功条数    : {n} / {len(items)}")
    print(f"  拒答        : {refused} 条")
    print(f"  非法编号    : {bad} 条")
    print(f"  无引用无拒答: {nocite} 条")
    with_gold = [r for r in records if r.get("gold_chunk_id")]
    if with_gold:
        in_ctx = sum(1 for r in with_gold if r["gold_rank"])
        cited_gold = sum(1 for r in with_gold
                         if r.get("gold_chunk_id") in (r.get("cited_chunk_ids") or []))
        print(f"  带 gold 条数: {len(with_gold)}")
        print(f"  其中 gold 进上下文: {in_ctx} 条（检索侧口径，应与评测脚本一致）")
        print(f"  其中 gold 被引用  : {cited_gold} 条（生成侧口径，M4 的引用准确率）")
    if ret:
        print(f"  检索耗时    : 中位 {sorted(ret)[len(ret) // 2] * 1000:.0f} ms")
    if gen:
        print(f"  生成耗时    : 中位 {sorted(gen)[len(gen) // 2]:.2f} s")
    print(f"  端到端耗时  : 中位 {sorted(r['total_seconds'] for r in records)[n // 2]:.2f} s")
    print(f"  总墙钟      : {time.time() - t0:.1f} s")
    print("=" * 76)

    if records:
        print(f"\n结果已存（边跑边写）：{out}")
        print(f"  文件行数应等于成功条数，可用：find /c /v \"\" \"{out}\"")

    print("\n怎么读结果：")
    print("  · 「拒答」应只出现在库外问题上；库里问题被拒答 = 误拒（M4 要量化的指标）")
    print("  · 「非法编号」必须为 0 —— 非 0 说明模型在编引用，这是最难肉眼发现的幻觉")
    print("  · 想看每条用了哪几号、对不对，翻上面的「【核查】」行")
    return 0


if __name__ == "__main__":
    sys.exit(main())
