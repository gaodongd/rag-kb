#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 11 步 §11.14 · 生成侧指标（M4 的后半段）
==========================================

第 11 步前面 13 节量的全是**检索**指标（R@1 / R@5 / MRR）。
但用户拿到的不是 chunk，是**答案**。检索 R@1 涨了 0.019，
到答案那一层还剩多少？——这就是本节要回答的问题。

四类指标（都是"能不能自动判"的硬指标，不做主观打分）
----------------------------------------------------
1. **拒答准确率**（只对库外题）
   语料里压根没有的信息，模型必须回答「根据已有资料无法回答」。
   没拒答 = 它在用世界知识硬答 —— 这是 RAG 里最危险的失败模式，
   因为答案外面裹着一层"参考资料"的皮，比裸 LLM 的幻觉更难发现。

2. **误拒率**（只对库内题）—— 但**必须分层**，否则这个数会骗人
   gold 进了上下文却拒答 → 真误拒（生成/指令跟随的问题，系统真正该挨的板子）
   gold 没进上下文而拒答 → **检索失败的传导**，拒答在这里反而是"正确的克制"
   两件事混在一个分母里，就只能得出"拒答率 6%"这种没法归因的数字。
   本项目的纪律：**归因靠切分，不靠均值**（§11.13 的方法论产出 2）。

3. **引用准确率**（严格按"引用的编号是不是 gold"判）
   答题时：
     · 引用编号里含 gold          → ✅ 引用正确
     · 有引用但没引到 gold        → ❌ 引用错误（引到了别的块，读者按引用回查会失望）
     · 一个引用都没有             → ⚠️ 无引用（结论无据可查，RAG 的唯一硬优势失效）
   另有两条硬红线（必须为 0 或接近 0）：
     · 非法编号（写了 [7] 而上下文只有 5 条）—— 编造引用
     · 上下文里没有 gold 却照样给引用 —— 详见下面 "ghost_cite" 的说明

4. **Faithfulness（忠实度）**：答案里的每一句话，能不能在被它引用的原文里找到？
   这一项最难自动判，所以本节做了**两级判据 + 人工校准**：
     · strict：句子自带引用，且被引用的块能支撑它
     · lenient：句子没带引用时，用**整条答案的引用集合**兜底（因为"先结论后依据"
       的写法里，结论句经常不重复标注）
     · 数字硬约束：句子里的阿拉伯数字/年份必须出现在被引用的块里 ——
       这一条几乎不会误判，是 faithfulness 判定里最可靠的那部分
   哪个判据更接近人，由 §11.14 的人工抽检来说话（**判据必须先校准**，
   本项目已为此栽过 4 次，见 §11.10）。

用法
----
    :: 单跑一份结果
    "...python313\\python.exe" src\\evaluate_gen.py --run 生产=eval\\results\\gen_dashscope_testset_rr.jsonl

    :: 多份对比（重排开/关、云端/本地）
    "...python313\\python.exe" src\\evaluate_gen.py ^
        --run 重排开=eval\\results\\gen_dashscope_testset_rr.jsonl ^
        --run 重排关=eval\\results\\gen_dashscope_testset_norr.jsonl

    :: 生成人工抽检表（默认 20 条，均匀覆盖拒答/引用对/引用错/无引用）
    "...python313\\python.exe" src\\evaluate_gen.py --run ... --audit 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RESULTS = PROJECT / "eval" / "results"
sys.path.insert(0, str(HERE))

from zh import norm_text        # noqa: E402  —— 与索引/检索侧同一个归一化函数

# 拒答话术。与 generator.py 的 REFUSAL_MARKERS 保持一致，但这里**分级**：
#   强标记 = prompt 里逐字规定的那句「无法回答」——命中即可判定
#   弱标记 = 模型自己换的说法（"没有提及""无从得知"）——算拒答，但单独计数
# 分级的原因（2026-09-21 实测）：v1-20f55e 的答案是
#   「根据现有资料，【参考资料】中没有提及袁說友的朋友对他的评价[2]。」
# 它明明拒答了，却因为标记串写的是"资料中没有"、而原文里多了个 `】`
# （"参考资料】中没有提及" 不含 "资料中没有"）**被判成硬答**。
# ⇒ 两个教训：① 匹配前必须**去掉括号等格式符号**；② 拒答的说法不止一种，
#   只认逐字那句会把"守规矩但换了措辞"的样本误判为幻觉。
#   2026-09-21 第二次补漏：**重排关**那一组里，v1-20f55e 的答案是
#   「根据提供的参考资料，没有关于袁說友的朋友对其评价的内容[1]。」
#   —— 它也是拒答，但说法是"没有关于"，前面那个标记表里没有，于是被列进
#   「库外题硬答（最危险的一类）」。**标记表漏词 = 把守规矩的样本报成幻觉**，
#   方向和上一个 bug 相反、后果一样坏。⇒ 拒答话术要按"意思"覆盖，不能只覆盖逐字句。
REFUSAL_STRONG = ["无法回答", "無法回答"]
REFUSAL_WEAK = [
    "无法据此回答", "無法據此回答", "没有提及", "未提及", "没有说明", "未说明",
    "没有任何相关", "没有相关", "资料中没有", "资料中未", "资料里没有",
    "未提供相关", "没有足够信息", "没有足够的信息", "无法得知", "无从得知",
    "无法确定", "無法確定", "无法从参考资料",
    # —— 2026-09-21 补：等价说法（"没有关于…的内容"这类）
    "没有关于", "沒有關於", "未涉及", "没有涉及", "没有记录", "未记录",
    "没有找到", "未找到", "无法找到", "没有查到", "并未提及", "并无提及",
    "没有介绍", "未介绍", "没有描述", "未描述", "没有给出", "未给出",
    "无法提供", "不能回答", "无法给出", "没有相关信息",
]

# 判官 verdict 的归一化。**必须做**：判官偶尔不按 JSON 里的 yes/no 回，改用中文
# "有依据/无依据"。它们语义完全等价，但原来的 judge_rate 只认 yes/no，
# 于是这两条被**静默踢出分母**（实测 20 个子句，涉及 2 条答案、其中一条是
# 最需要看的库外硬答题 v1-29bbff）。
# ⇒ 指标脚本不能"看不懂就丢"：丢的方向若是随机的，指标就不可比；
#   这里语义明确，归一化比丢弃更诚实。
VERDICT_YES = {"yes", "是", "有依据", "有依据的", "支持", "supported", "true"}
VERDICT_NO = {"no", "否", "无依据", "无依据的", "不支持", "unsupported", "false"}


def norm_verdict(v) -> str | None:
    """把判官的各种写法归一化成 'yes' / 'no'，认不出的返回 None（不计入分母）。"""
    if not isinstance(v, str):
        return None
    k = v.strip().lower()
    if k in VERDICT_YES:
        return "yes"
    if k in VERDICT_NO:
        return "no"
    return None

CITE_RE = re.compile(r"\[(\d{1,3})\]")

# ⚠️ 抽数字前必须**先剥掉引用标记**：`[2]` 里的 "2" 会被数字正则当成数字，
#    然后去原文里找 "2" —— 几乎必然找不到，于是每一句都判成"数字违规"。
#    2026-09-21 实测：这个 bug 让数字违规虚报成 126/175 句、
#    并把 faithfulness 打到 12%（因为数字硬约束一挂，strict 就全灭）。
#    判据自己错了时，它给出的数字看起来照样"像个指标"，这是最危险的一类错。
NUM_RE = re.compile(r"\d+(?:\.\d+)?%?")

# 句子切分：中文标点 + 换行。保留标点（后面算 n-gram 前会去掉）。
SENT_SPLIT_RE = re.compile(r"[。！？；\n]+|(?<=[.!?])\s+")

# 子句切分：逗号/分号/冒号。用于"按内容量加权"的 faithfulness 口径 ——
# 长句里的列举和解释会被拆开，各自判据，不会互相拖累。
CLAUSE_SPLIT_RE = re.compile(r"[，,；;：:]+|\s*[（(][^）)]{0,20}[）)]\s*")

# 客套/结构句，不承载事实，不该拖累 faithfulness
FILLER_PAT = re.compile(
    r"^(答案是|结论[:：]?|依据[:：]?|参考(资料)?[:：]?|来源[:：]?|说明[:：]?|注[:：]?|"
    r"注：|以上|综上|综上所述|以下是|主要依据|根据)")

PUNCT_RE = re.compile(r"[\s，。！？；：、（）()\[\]【】\"'“”‘’《》〈〉—\-…·,.;:!?]")
# 只去格式符号（保留标点），用于拒答匹配 —— 见上面 v1-20f55e 的例子
BRACKET_RE = re.compile(r"[\[\]【】〔〕（）()「」『』\s，。；：、,.!?！？]")


# ==================================================================== 读取


def load_records(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def measure_refusal(answer: str) -> dict:
    """
    从**答案原文**重新判拒答（不盲信生成侧落盘的 refused 标记）。

    为什么评测侧要自己再判一遍：那个标记是别的模块写的，
    判据藏在别的文件里 —— 一旦两边不同步（例如标记表补了新词，
    但结果文件是旧代码产的），指标会静默用错。所以这里重算，
    并把"与落盘标记不一致"的条数**当成诊断指标报出来**。
    """
    flat = BRACKET_RE.sub("", answer or "")
    strong = any(m in flat for m in REFUSAL_STRONG)
    weak = strong or any(m in flat for m in REFUSAL_WEAK)
    return {"strong": strong, "weak": weak}


def is_oob(r: dict) -> bool:
    """库外题：评测集里 answerable_in_kb=False（kind=oob）。"""
    if r.get("answerable_in_kb") is False:
        return True
    return r.get("kind") == "oob"


def has_gold(r: dict) -> bool:
    return bool(r.get("gold_chunk_id"))


def _flat(s: str) -> str:
    """归一化 + 去标点空白，只用于'答案里有没有出现 gold 答案'的词面比对。"""
    return PUNCT_RE.sub("", norm_text(s or ""))


def answer_hit(r: dict) -> bool:
    """
    答案是否包含 gold 答案（词面口径）。

    ⚠️ 这是**下界**，不是准确率。模型换个说法（gold「二百六十八」→ 答「268」）
    就判不中，所以它只会低不会高。把它当"上限"用会严重误导，
    正确用法是当**保守下界**，再靠人工抽检看它漏了多少（§11.14 校准表里有）。
    """
    ga = r.get("gold_answer")
    if not ga:
        return False
    return _flat(ga) in _flat(r.get("answer") or "")


def e2e_pass(r: dict) -> bool:
    """
    单题端到端成败 —— 把检索和生成串成一个数：
      库外题：正确拒答 = 通过
      库内题：没拒答 **且** 答案里出现 gold 答案 = 通过
    这个口径比"检索命中率"更接近用户体感，也是 §11.14 做配对检验用的量。
    拒答用评测侧重判的结果（`_refused`），拿不到时退回落盘标记。
    """
    refused = bool(r.get("_refused", r.get("refused")))
    if is_oob(r):
        return refused
    return (not refused) and answer_hit(r)


# ==================================================================== 判据


def split_sentences(answer: str) -> list[str]:
    out = []
    for s in SENT_SPLIT_RE.split(answer or ""):
        s = s.strip()
        if len(s) >= 4:               # 太短的碎片（"如下："）不单独判
            out.append(s)
    return out


def split_clauses(sent: str) -> list[str]:
    return [c.strip() for c in CLAUSE_SPLIT_RE.split(sent or "") if c.strip()]


def ngrams(text: str, n: int = 2) -> set[str]:
    """
    2-gram 集合。**先做繁简归一化**（与索引/检索侧同一个 norm_text）。

    ⚠️ 2026-09-21 实测踩到：答案是简体（"邓小平在1982年会见撒切尔夫人"），
    原文是繁体（"鄧小平會見撒切尔夫人"）—— 不做归一化，2-gram 直接对不上，
    覆盖率虚低到 0.5，一条**答案正确、引用也正确**的样本被判成"无依据"。
    本项目的语料是繁简混排的中文维基，这条对**任何**中文文本比对都成立，
    不只是检索侧的事。
    """
    t = PUNCT_RE.sub("", norm_text(text or ""))
    return {t[i:i + n] for i in range(len(t) - n + 1)} if len(t) >= n else ({t} if t else set())


def coverage(sentence: str, chunk: str) -> float:
    """句子有多少比例的 2-gram 能在被引用的块里找到。0~1。（v1 判据，保留做对照）"""
    a = ngrams(sentence)
    if not a:
        return 0.0
    b = ngrams(chunk)
    return len(a & b) / len(a)


# ---------------------------------------------------------------------------
# v3 判据：实词覆盖率
# ---------------------------------------------------------------------------
# 为什么 2-gram 不够（2026-09-21 人工核对了 12 条被判"无依据"的子句）：
#   · 「白沙江的源头叫中和河」vs 原文「源头称中和河」→ 2-gram 0.50 ❌
#     （就差一个"叫/称"，可 2-gram 一半都带上了这个字）
#   · 「是吉林省长春市人」vs 原文「刘烨，吉林长春人」→ 0.33 ❌
#   · 「该艺术场馆的名字是为了纪念亨利·詹姆斯·西蒙」vs「畫廊得名於亨利·詹姆斯·西蒙的名字命名，
#      以紀念他…」→ 0.55 ❌
#   这三条**肉眼一看就是有依据**，但 2-gram 对"同义改写""多一两个字"极其敏感，
#   短句尤其吃亏（分母太小，一两个 2-gram 的差异就是 0.2~0.3）。
#
# 实词覆盖率换了个问法：**这句话的关键内容词，在原文里出现了几个？**
#   · 停用词、单字、结构词先滤掉（它们不承载事实，命中与否都不说明问题）
#   · 剩下的实词只要在原文里**作为子串出现**就算命中（不要求连续、不要求顺序）
# 它对人怎么写更宽容，但对"凭空多出来的实体/数字"依然敏感 ——
# 正是 faithfulness 该有的取舍。
try:
    import jieba
    jieba.setLogLevel(60)          # 别让它把 "Building prefix dict" 刷进指标输出
    HAVE_JIEBA = True
except Exception:                  # pragma: no cover
    HAVE_JIEBA = False

# 结构词 / 元话语：不承载事实
STOPWORDS = set("""
的 了 在 是 有 和 与 及 也 就 都 而 但 并 或 对 从 到 为 以 之 其 该 这 那 这些 那些
他 她 它 他们 我们 你们 个 把 被 由 于 中 上 下 时 后 前 会 能 可 要 将 等 其中 以及
一个 一种 进行 通过 对于 关于 根据 依据 参考 参考资料 资料 原文 上文 上述 如下
明确 指出 记载 说明 表示 显示 认为 因此 所以 但是 不过 并且 而且 或者 如果 虽然
主要 重要 相关 方面 情况 内容 部分 时候 之后 之前 已经 这个 那个 什么 怎么 怎样
此外 同时 另外 然而 不过 总体 综上 首先 其次 最后 目前 当时 后来 最终 首次 之一
期间 之间 以来 同年 同年月 此时 此时 届时 同期 此後 此后 原先 原本 初期 末期
""".split())

PUNC_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)


def content_tokens(text: str) -> list[str]:
    """
    抽"实词"：长度 ≥2、不是停用词、不是纯标点。
    用 jieba 分词（与 BM25 侧同一个分词器）；繁简先归一到简体，
    否则繁体原文切出来的词和简体答案对不上（同一个坑，第三次踩了 —— 见 ngrams 注释）。
    """
    t = norm_text(text or "")
    if not HAVE_JIEBA:
        # 没装 jieba 时的兜底：按 2-gram 当词用（仍比整句 2-gram 判据宽松）
        return list(ngrams(t))
    out = []
    for w in jieba.cut(t):
        w = w.strip()
        if len(w) < 2 or w in STOPWORDS or PUNC_ONLY_RE.match(w):
            continue
        out.append(w)
    return out


def content_coverage(sentence: str, chunk: str) -> float:
    """
    句子实词里，有多少比例能在原文中找到（子串即可）。
    返回 (覆盖率, 缺失的实词)。
    """
    toks = content_tokens(sentence)
    if not toks:
        return 1.0, []                 # 没有实词 = 没有事实主张，不该判负
    c = norm_text(chunk or "")
    miss = [w for w in toks if w not in c]
    return (len(toks) - len(miss)) / len(toks), miss[:8]


# 结构句/元话语：整句都是"引用的脚手架"，不承载任何事实。
# 实测（§11.14 扫全量）：被判"无依据"的子句里，绝大多数是这一类 ——
#   「依据是参考资料[1]明确指出」「资料[1]和[3]也一致记载此事」
#   「在【参考资料】[1]中明確指出」「依据来自【参考资料】[1]」
#   「方位描述以[1][2]綜合判斷無矛盾」（模型对引用方式的自述）
# 它们既不该判 ✅ 也不该判 ❌ —— 判了就是把噪声灌进分母，
# 让 faithfulness 看起来比实际低一大截（实测：这一类占了低覆盖子句的 ~90%）。
META_WORDS = set("""
资料 参考资料 參考資料 原文 上文 文中 依据 根據 根据 记载 記載 说明 說明 指出 明确指出
明确 表示 表明 表明 表明 提到 提及 所述 上述 以上 以下 如下 如下 佐证 證實 证实 印证
一致 综合 綜合 判断 判斷 得出 据此 據此 由此 因此 可见 可見 得知 可知 得知 显示 顯示
是 也 亦 都 均 皆 就 便 即 并 並 且 该 該 此 此事 这 這 其 其中 所 者 的 了 在 中 里 裡
来自 來自 源自 出自 参见 參見 详见 詳見 见 見 如上 如下 时间点 時間點 这一 這一 那
无矛盾 無矛盾 矛盾 方位 描述 部分 内容 內容 情况 情況 方式 方面 说法 說法 版本
此外 同时 同時 另外 然而 不过 總體 总体 综上 首先 其次 最后 最後 目前 当时 當時
后来 後來 最终 最終 首次 之一 期间 期間 之间 之間 以来 以來 同年 此时 此時 届时 屆時
同期 原先 原本 初期 末期 以及 并且 並且 而且 或者 一时 一时间 这件事 此事
""".split())


def is_struct_clause(text: str) -> bool:
    """
    是否"不承载事实"的子句（元话语 / 切分残渣）。

    判法：把引用标记与标点去掉，再看**实词是不是清一色元话语词**。
    比正则穷举稳健 —— 模型的套话写法五花八门（"依据是参考资料1明确指出"
    这种连正则都难写），但它们有一个共同点：**除了"我在引用"这件事之外，
    没有任何信息**。所以就按这个特点判。
    """
    t = PUNCT_RE.sub("", CITE_RE.sub("", text or ""))
    if not t:
        return True
    toks = content_tokens(t)
    if not toks:
        return True
    return all(w in META_WORDS for w in toks)


def digits_of(text: str) -> set[str]:
    """
    阿拉伯数字系。去掉末尾的 % 便于比对（'2.6%' 与 '2.6' 视为同一个数）。
    ⚠️ 先剥引用标记 —— 否则 `[2]` 的 "2" 会被当成数字（见上面 NUM_RE 的注释）。
    中文数字（二百六十八）不参与硬判：它在普通词里也出现（"十"、"百"），
    拿它做硬判据必然误报（本项目"报警/提示两级"的惯例）。
    """
    t = CITE_RE.sub(" ", text or "")
    return {m.rstrip("%") for m in NUM_RE.findall(t)}


def digits_supported(sentence: str, chunk: str) -> tuple[bool, list[str]]:
    """
    句子里的数字是否都能在被引用的块里找到。
    返回 (是否全部支持, 缺失的数字)。句子没有数字时视为通过。
    """
    d = digits_of(sentence)
    if not d:
        return True, []
    c = digits_of(chunk)
    missing = sorted(x for x in d if x not in c)
    return (not missing), missing


def judge_sentence(sent: str, cited_chunks: list[str], all_chunks: list[str],
                   thresh: float, cthresh: float = 0.6) -> dict:
    """
    单句判定。同时给出两套判据，方便用人工标签对比（本项目对判据的一贯做法）：
      supported_*        v1/v2：2-gram 覆盖率 ≥ thresh
      supported_c_*      v3：**实词覆盖率 ≥ cthresh**（主判据，理由见 content_coverage）
    返回 dict + 覆盖率（cov / ccov）+ 数字硬违规（num_missing）
    """
    nums = [int(x) for x in CITE_RE.findall(sent)]
    own = [c for n, c in zip(range(1, len(cited_chunks) + 1), cited_chunks) if n in nums]

    has_own = bool(nums)
    src = own if has_own else all_chunks

    cov = max((coverage(sent, c) for c in src), default=0.0)
    ccov = max((content_coverage(sent, c)[0] for c in src), default=0.0)
    miss_tok = []
    if src:
        miss_tok = min((content_coverage(sent, c)[1] for c in src), key=len)

    ok_num, missing = True, []
    if has_own:
        # 数字硬约束只在"句子自带引用"时生效 —— 无引用的句子它自己都没声明来源，
        # 用全答案的块去判它等于放宽了标准，反而会掩盖问题。
        if own:
            allm = [digits_supported(sent, c) for c in own]
            ok_num = any(x[0] for x in allm)
            if not ok_num:
                miss_sets = [set(x[1]) for x in allm]
                missing = sorted(set.intersection(*miss_sets)) if miss_sets else []
                if not missing:
                    missing = sorted(set.union(*miss_sets))

    return {
        "sentence": sent,
        "cites": nums,
        "has_own_cite": has_own,
        "coverage": round(cov, 3),
        "ccoverage": round(ccov, 3),
        "missing_tokens": miss_tok,
        "num_missing": missing,
        "supported_strict": bool(has_own and cov >= thresh and ok_num),
        "supported_lenient": bool(cov >= thresh and ok_num),
        "supported_c_strict": bool(has_own and ccov >= cthresh and ok_num),
        "supported_c_lenient": bool(ccov >= cthresh and ok_num),
    }


def context_texts(r: dict) -> list[str]:
    """
    取"模型当时看到的原文"，拼成与 prompt 里一致的 `title｜section\\n正文`。

    ⚠️ 只有 `contexts` 字段（2026-09-21 起落盘）里才有正文；
    旧的 `hits` 字段有意只存了元信息（title/section/score），**没有正文**。
    拿旧文件跑 faithfulness 会得到"覆盖率全 0、数字违规遍地"的假结果 ——
    判据没坏，是没拿到要比对的文本。所以这里返回空列表，由调用方报 "—"，
    **绝不能退化成 0**：0 看起来像个正常指标值，会把人骗过去。
    """
    out = []
    for c in (r.get("contexts") or []):
        head = c.get("title") or ""
        if c.get("section"):
            head += f"｜{c['section']}"
        out.append(head + "\n" + (c.get("text") or ""))
    if out:
        return out
    # 兼容：进程内直接调用（未落盘）时 hits 里还带着 chunk_text
    for h in (r.get("hits") or []):
        if h.get("chunk_text"):
            head = h.get("title") or ""
            if h.get("section"):
                head += f"｜{h['section']}"
            out.append(head + "\n" + h["chunk_text"])
    return out


def cited_chunk_texts(r: dict) -> list[str]:
    """按引用编号取出对应的原文（rank 与 contexts 对齐）。"""
    ctx = {c["rank"]: (c.get("text") or "") for c in (r.get("contexts") or [])}
    return [ctx[n] for n in (r.get("cited") or []) if n in ctx]


def gold_answer_in_cited(r: dict) -> bool:
    """
    宽松引用口径：**gold 答案本身出现在被引用的原文里**。

    为什么需要它（2026-09-21 实测）：strict 口径只认 gold chunk，
    但中文维基里同一件事常写在多个条目/多个块里。实测 v1-1f1e6e：
    问「邓小平在哪一年会见撒切尔夫人时谈到一国两制」→ 答「1982年」，
    引用 [1]（一国两制｜重新提出）的原文里明明白白写着
    "1982年9月…鄧小平會見撒切尔夫人時稱…" —— 但它被 strict 判成"引用错误"，
    因为 gold chunk 是另一条同样讲这事的块 [2]。
    **那是判据错了，不是答案错了。**

    这个口径直接回答一个更实在的问题：**读者按引用回查，能不能看到答案？**
    它是 strict 的上界意义上的"不冤枉"版本，两个都报，差值本身就是
    "同一事实在语料里重复出现"的规模。
    """
    ga = r.get("gold_answer")
    if not ga:
        return False
    return any(_flat(ga) in _flat(t) for t in cited_chunk_texts(r))


def faithfulness(r: dict, thresh: float, cthresh: float = 0.6) -> dict:
    """
    整条答案的 faithfulness。两级判据**并列报**（哪个更接近人，由人工抽检定）：

    ① 句子级（v1）：句子 2-gram 覆盖率 ≥ 阈值 且 数字全支持
       ⚠️ 实测问题：模型的长句里常带自己的解释（"…显示其具有政治野心"），
       整句 2-gram 覆盖率天然被拉低（实测 0.35~0.5），
       于是"事实有依据但多说了一句评价"的句子被整句判负。

    ② 子句级 + 按字数加权（v2，主判据）：
       先按逗号/分号/冒号切成子句，逐个判支撑，再按**子句字数加权**聚合。
       语义是「答案里有多大比例的内容能在被引用的原文里找到」——
       这比数句子更接近人看答案时在做的事：**逐条事实核对，而不是整段比对**。
       长句里有依据的部分照样算数，不会因为多了一句评论而全灭。

    返回 dict（available=False 表示拿不到原文，调用方必须显示 "—"）
    """
    answer = r.get("answer") or ""
    chunk_texts = context_texts(r)
    if not chunk_texts:
        return {"n_sent": 0, "n_strict": 0, "n_lenient": 0,
                "ratio_strict": None, "ratio_lenient": None,
                "clause_ratio_strict": None, "clause_ratio_lenient": None,
                "n_clause": 0, "n_clause_strict": 0, "n_clause_lenient": 0,
                "num_violations": 0, "has_cite": bool(r.get("cited")),
                "available": False, "sentences": [], "clauses": []}

    sents = split_sentences(answer)
    details, clauses = [], []
    for s in sents:
        if FILLER_PAT.match(s) and not CITE_RE.search(s):
            continue                             # 结构句不计入（不承载事实）
        details.append(judge_sentence(s, chunk_texts, chunk_texts, thresh, cthresh))

    # 整条答案的引用集合（供 lenient 兜底）
    all_cites = sorted({int(x) for x in CITE_RE.findall(answer)})
    all_cited = [c for n, c in enumerate(chunk_texts, 1) if n in all_cites]

    for s in sents:
        if is_struct_clause(s):                  # 剔掉"根据资料[2]"式噪声，别灌进分母
            continue
        for cl in split_clauses(s):
            if is_struct_clause(cl):
                continue
            d = judge_sentence(cl, chunk_texts, all_cited or chunk_texts, thresh, cthresh)
            d["len"] = len(PUNCT_RE.sub("", cl))
            clauses.append(d)

    n = len(details)
    ns = sum(1 for d in details if d["supported_strict"])
    nl = sum(1 for d in details if d["supported_lenient"])
    ncs = sum(1 for d in details if d["supported_c_strict"])
    ncl = sum(1 for d in details if d["supported_c_lenient"])

    wsum = sum(c["len"] for c in clauses)
    ws = sum(c["len"] for c in clauses if c["supported_strict"])
    wl = sum(c["len"] for c in clauses if c["supported_lenient"])
    wcs = sum(c["len"] for c in clauses if c["supported_c_strict"])
    wcl = sum(c["len"] for c in clauses if c["supported_c_lenient"])

    return {
        "n_sent": n, "n_strict": ns, "n_lenient": nl,
        "ratio_strict": round(ns / n, 4) if n else None,
        "ratio_lenient": round(nl / n, 4) if n else None,
        "ratio_c_strict": round(ncs / n, 4) if n else None,
        "ratio_c_lenient": round(ncl / n, 4) if n else None,
        "n_clause": len(clauses),
        "n_clause_strict": sum(1 for c in clauses if c["supported_strict"]),
        "n_clause_lenient": sum(1 for c in clauses if c["supported_lenient"]),
        "n_clause_c_strict": sum(1 for c in clauses if c["supported_c_strict"]),
        "n_clause_c_lenient": sum(1 for c in clauses if c["supported_c_lenient"]),
        "clause_ratio_strict": round(ws / wsum, 4) if wsum else None,
        "clause_ratio_lenient": round(wl / wsum, 4) if wsum else None,
        "clause_ratio_c_strict": round(wcs / wsum, 4) if wsum else None,
        "clause_ratio_c_lenient": round(wcl / wsum, 4) if wsum else None,
        "num_violations": sum(1 for d in details if d["num_missing"]),
        "has_cite": bool(r.get("cited")),
        "available": True,
        "sentences": details,
        "clauses": clauses,
    }


# ==================================================================== 汇总


def med(v):
    v = sorted(v)
    return v[len(v) // 2] if v else None


def load_judge(path: Path) -> dict:
    """
    读 LLM 判官结果（src/judge_faith.py 产出），返回 {qid: [ {clause, verdict, why} ]}。

    为什么要它：词面判据在本项目的人工校准里只有 61~72% 的一致率，而且**漏检方向很致命**
    （"用词全对、关系说反"判不出来）。所以 faithfulness 的主口径换成判官，
    词面判据降级为下界。判官本身也是过了校准才被允许上场的（16/16，含 4 条合成对照）。
    """
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["qid"]] = rec.get("clauses") or []
    return out


def judge_rate(clauses: list[dict]) -> tuple[int, int]:
    """(判为有依据的条数, 已判条数)。

    verdict 先归一化（yes/是/有依据 都算 yes），认不出的才排除在分母之外。
    归一化前实测有 20 个子句因为"判官答了中文"而被静默丢弃，见 VERDICT_YES 处的注释。
    """
    vals = [v for v in (norm_verdict(c.get("verdict")) for c in clauses) if v]
    return sum(1 for v in vals if v == "yes"), len(vals)


def retrieve_total_ms(r: dict) -> float | None:
    """
    检索总耗时（毫秒）。**从分项相加**，不信落盘的 `retrieve_total`。

    为什么：generator.py 里那个总计时变量曾经被重排那段复用（`t0 = time.time()`
    写在了重排前面），于是"重排开"的结果文件里 `retrieve_total` 只剩重排时间
    （实测 1359 ms，而真实总耗时约 3.5 s）—— 数字看着完全正常，方向还正好相反
    （重排开看起来比关还快）。源码已修（`r0`），但**已经落盘的数据不改**，
    所以评测侧一律按分项重算，并把这个不一致当诊断指标报出来。
    """
    t = r.get("timing") or {}
    parts = [t.get(k) or 0.0 for k in ("vector", "bm25", "fuse", "fetch", "rerank")]
    s = sum(parts)
    if s > 0:
        return s * 1000
    return (t["retrieve_total"] * 1000) if t.get("retrieve_total") else None


def evaluate(rows: list[dict], thresh: float, cthresh: float = 0.6,
             judge: dict | None = None) -> dict:
    # ---- 0. 拒答：评测侧重判（不盲信落盘标记），并统计两边不一致的条数 ----
    for r in rows:
        ref = measure_refusal(r.get("answer") or "")
        r["_ref_strong"], r["_ref_weak"] = ref["strong"], ref["weak"]
        r["_refused"] = ref["weak"]          # 主口径：强+弱（弱标记也认，理由见 REFUSAL_WEAK）
        r["_refused_stored"] = bool(r.get("refused"))
    n_mismatch = sum(1 for r in rows if r["_refused"] != r["_refused_stored"])

    oob = [r for r in rows if is_oob(r)]
    inkb = [r for r in rows if not is_oob(r)]

    # ---- 1. 拒答 ----
    oob_refused = [r for r in oob if r["_refused"]]
    oob_answered = [r for r in oob if not r["_refused"]]
    inkb_refused = [r for r in inkb if r["_refused"]]

    # 误拒分层：gold 到底有没有进上下文
    gold_in_ctx = [r for r in inkb if r.get("gold_rank")]
    gold_missing = [r for r in inkb if has_gold(r) and not r.get("gold_rank")]
    refuse_when_gold_in_ctx = [r for r in gold_in_ctx if r["_refused"]]
    refuse_when_gold_missing = [r for r in gold_missing if r["_refused"]]

    # ---- 2. 引用 ----
    # 只在"真的给出了答案"的条目上算引用指标 —— 拒答条目本来就不该有引用
    answered = [r for r in rows if not r["_refused"]]
    answered_inkb = [r for r in answered if not is_oob(r)]

    no_cite = [r for r in answered if not r.get("cited")]
    bad_num = [r for r in answered if r.get("bad_citations")]
    invalid_cid = [r for r in answered
                   if any(c not in (r.get("hit_chunk_ids") or []) for c in (r.get("cited_chunk_ids") or []))]

    ans_gold_in_ctx = [r for r in answered_inkb if r.get("gold_rank")]
    cite_hit = [r for r in ans_gold_in_ctx if r.get("gold_chunk_id") in (r.get("cited_chunk_ids") or [])]
    cite_miss = [r for r in ans_gold_in_ctx if r.get("gold_chunk_id") not in (r.get("cited_chunk_ids") or [])]
    cite_miss_with_cite = [r for r in cite_miss if r.get("cited")]
    cite_miss_no_cite = [r for r in cite_miss if not r.get("cited")]
    # 宽松口径：gold 答案出现在被引用的原文里（分母 = 有 gold 答案且有引用的条目）
    cited_inkb = [r for r in answered_inkb if r.get("gold_answer") and r.get("cited")]
    cite_answer_in = [r for r in cited_inkb if gold_answer_in_cited(r)]

    # ghost_cite：上下文里根本没有 gold，答案却给出引用。
    # 不能直接判错 —— 它可能引用了别的块来支持一个"部分相关"的回答。
    # 但它是最需要人看一眼的一类，所以单独计数、逐条列出。
    ghost_cite = [r for r in answered_inkb
                  if has_gold(r) and not r.get("gold_rank") and r.get("cited")]

    # ---- 3. faithfulness ----
    for r in answered:
        r["_fa"] = faithfulness(r, thresh, cthresh)
    fa_rows = [r for r in answered if r["_fa"]["available"] and r["_fa"]["n_sent"] > 0]
    fa_null = [r for r in answered if not r["_fa"]["available"]]
    fa_inkb = [r for r in fa_rows if not is_oob(r)]
    fa_oob = [r for r in fa_rows if is_oob(r)]

    def avg(rs, key):
        vals = [r["_fa"][key] for r in rs if r["_fa"][key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    # ---- 3.5 LLM 判官口径（主口径，前提是判官已通过校准） ----
    j_yes = j_tot = 0
    j_by_kind: dict[str, list[int]] = {}
    j_rows = 0
    if judge:
        for r in answered:
            cl = judge.get(r.get("qid"))
            if not cl:
                continue
            y, n = judge_rate(cl)
            if n == 0:
                continue
            j_rows += 1
            j_yes += y
            j_tot += n
            k = "oob" if is_oob(r) else r.get("kind") or "?"
            j_by_kind.setdefault(k, []).append(y / n)

    n_ans_inkb = len(answered_inkb)
    m = {
        "n_total": len(rows),
        "n_oob": len(oob),
        "n_inkb": len(inkb),
        # 1 拒答
        "oob_refusal_rate": round(len(oob_refused) / len(oob), 4) if oob else None,
        "oob_refusal_rate_strong": (
            round(sum(1 for r in oob if r["_ref_strong"]) / len(oob), 4) if oob else None),
        "refusal_flag_mismatch_n": n_mismatch,
        "oob_answered_n": len(oob_answered),
        "oob_answered_qids": [{"qid": r.get("qid"), "question": r["query"],
                               "answer": (r.get("answer") or "")[:120]}
                              for r in oob_answered],
        "inkb_false_refusal_rate": round(len(inkb_refused) / len(inkb), 4) if inkb else None,
        "inkb_refused_n": len(inkb_refused),
        "gold_in_ctx_n": len(gold_in_ctx),
        "gold_missing_n": len(gold_missing),
        "refuse_gold_in_ctx_n": len(refuse_when_gold_in_ctx),
        "refuse_gold_missing_n": len(refuse_when_gold_missing),
        "refuse_gold_in_ctx_rate": (round(len(refuse_when_gold_in_ctx) / len(gold_in_ctx), 4)
                                    if gold_in_ctx else None),
        "refuse_gold_missing_rate": (round(len(refuse_when_gold_missing) / len(gold_missing), 4)
                                     if gold_missing else None),
        "refuse_gold_in_ctx_qids": [r.get("qid") for r in refuse_when_gold_in_ctx],
        # 2 引用
        "answered_n": len(answered),
        "no_cite_n": len(no_cite),
        "no_cite_rate": round(len(no_cite) / len(answered), 4) if answered else None,
        "bad_citation_n": len(bad_num),
        "invalid_cid_n": len(invalid_cid),
        "ans_gold_in_ctx_n": len(ans_gold_in_ctx),
        "cite_hit_n": len(cite_hit),
        "cite_miss_n": len(cite_miss),
        "cite_miss_with_cite_n": len(cite_miss_with_cite),
        "cite_miss_no_cite_n": len(cite_miss_no_cite),
        "cite_precision": (round(len(cite_hit) / len(ans_gold_in_ctx), 4)
                           if ans_gold_in_ctx else None),
        # 宽松口径（按引用回查能不能看到答案）
        "cite_answer_in_n": len(cite_answer_in),
        "cite_answer_in_denom": len(cited_inkb),
        "cite_answer_in_rate": (round(len(cite_answer_in) / len(cited_inkb), 4)
                               if cited_inkb else None),
        "ghost_cite_n": len(ghost_cite),
        "ghost_cite_qids": [r.get("qid") for r in ghost_cite],
        "cite_len_avg": round(sum(len(r.get("cited") or []) for r in answered) / len(answered), 2)
                        if answered else None,
        # 2.5 端到端单题成败（把检索 + 生成串成一个数）
        "answer_hit_n": sum(1 for r in inkb if answer_hit(r)),
        "answer_hit_rate": (round(sum(1 for r in inkb if answer_hit(r)) / len(inkb), 4)
                            if inkb else None),
        "e2e_pass_n": sum(1 for r in rows if e2e_pass(r)),
        "e2e_pass_rate": round(sum(1 for r in rows if e2e_pass(r)) / len(rows), 4) if rows else None,
        "e2e_pass_inkb_n": sum(1 for r in inkb if e2e_pass(r)),
        "e2e_pass_oob_n": sum(1 for r in oob if e2e_pass(r)),
        # 3 faithfulness（句子级，按题目平均，再按题聚合的平均）
        "fa_n_answers": len(fa_rows),
        "fa_unavailable_n": len(fa_null),
        "fa_strict_inkb": avg(fa_inkb, "ratio_strict"),
        "fa_lenient_inkb": avg(fa_inkb, "ratio_lenient"),
        "fa_strict_oob": avg(fa_oob, "ratio_strict"),
        "fa_lenient_oob": avg(fa_oob, "ratio_lenient"),
        "fa_strict_all": avg(fa_rows, "ratio_strict"),
        "fa_lenient_all": avg(fa_rows, "ratio_lenient"),
        # v2：子句级 + 按字数加权（主判据，理由见 faithfulness() 文档）
        "fac_strict_inkb": avg(fa_inkb, "clause_ratio_strict"),
        "fac_lenient_inkb": avg(fa_inkb, "clause_ratio_lenient"),
        "fac_strict_oob": avg(fa_oob, "clause_ratio_strict"),
        "fac_strict_all": avg(fa_rows, "clause_ratio_strict"),
        "fac_lenient_all": avg(fa_rows, "clause_ratio_lenient"),
        # v3：子句级 + 按字数加权，覆盖度用**实词**算（词面主判据，理由见 content_coverage）
        "fac_c_strict_inkb": avg(fa_inkb, "clause_ratio_c_strict"),
        "fac_c_lenient_inkb": avg(fa_inkb, "clause_ratio_c_lenient"),
        "fac_c_strict_oob": avg(fa_oob, "clause_ratio_c_strict"),
        "fac_c_strict_all": avg(fa_rows, "clause_ratio_c_strict"),
        "fac_c_lenient_all": avg(fa_rows, "clause_ratio_c_lenient"),
        "fac_clause_n": sum(r["_fa"]["n_clause"] for r in fa_rows),
        "fac_clause_strict_n": sum(r["_fa"]["n_clause_strict"] for r in fa_rows),
        "fac_clause_lenient_n": sum(r["_fa"]["n_clause_lenient"] for r in fa_rows),
        "fac_c_clause_strict_n": sum(r["_fa"]["n_clause_c_strict"] for r in fa_rows),
        "fac_c_clause_lenient_n": sum(r["_fa"]["n_clause_c_lenient"] for r in fa_rows),
        # 3.5 LLM 判官（faithfulness 的主口径）
        "judge_available": bool(judge),
        "judge_rows": j_rows,
        "judge_yes_n": j_yes,
        "judge_clause_n": j_tot,
        "judge_rate": round(j_yes / j_tot, 4) if j_tot else None,
        "judge_rate_by_kind": {k: round(sum(v) / len(v), 4) for k, v in j_by_kind.items()},
        # 判官与词面判据的差 —— 差值就是"词面判据漏掉的改写/概括"
        "judge_minus_lexical": (round(j_yes / j_tot - (avg(fa_rows, "clause_ratio_c_lenient") or 0), 4)
                                if j_tot and avg(fa_rows, "clause_ratio_c_lenient") is not None
                                else None),
        "fa_num_violation_sent_n": sum(r["_fa"]["num_violations"] for r in fa_rows),
        "fa_sent_total": sum(r["_fa"]["n_sent"] for r in fa_rows),
        # 4 成本/延迟（检索总耗时按分项重算，理由见 retrieve_total_ms）
        "tok_in": sum(r["usage"].get("prompt_tokens") or 0 for r in rows),
        "tok_out": sum(r["usage"].get("completion_tokens") or 0 for r in rows),
        "lat_retrieve_ms": med([x for x in (retrieve_total_ms(r) for r in rows) if x]),
        "lat_rerank_ms": med([(r["timing"].get("rerank") or 0) * 1000 for r in rows
                              if r.get("timing")]) if any(
                                  (r["timing"].get("rerank") or 0) > 0 for r in rows) else 0,
        "lat_fetch_ms": med([(r["timing"].get("fetch") or 0) * 1000 for r in rows
                             if r.get("timing")]),
        # 落盘值与分项和是否对得上（对不上说明生成侧计时曾经写错）
        "lat_total_mismatch_n": sum(
            1 for r in rows if (r.get("timing") or {}).get("retrieve_total")
            and abs(sum((r["timing"].get(k) or 0) for k in ("vector", "bm25", "fuse", "fetch", "rerank"))
                    - r["timing"]["retrieve_total"]) > 0.2),
        "lat_gen_s": med([r["usage"].get("latency") for r in rows if r.get("usage")]),
        "lat_total_s": med([r["total_seconds"] for r in rows]),
    }
    m["_refuse_gold_in_ctx_rows"] = refuse_when_gold_in_ctx
    m["_cite_miss_rows"] = cite_miss
    m["_oob_answered_rows"] = oob_answered
    m["_ghost_rows"] = ghost_cite
    return m


# ==================================================================== 打印


def fmt(x, pct=True, nd=4):
    if x is None:
        return "—"
    return f"{x * 100:.1f}%" if pct else f"{x}"


def _w(s: str) -> int:
    """字符串的**显示宽度**：中日韩全角字符占 2 列。用它对齐中文表头，
    否则 f"{x:>12}" 按字符数补齐，中文列会整体右移、表头与数据错位。"""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad(s: str, width: int, align: str = ">") -> str:
    s = str(s)
    n = width - _w(s)
    if n <= 0:
        return s
    return (" " * n + s) if align == ">" else (s + " " * n)


def print_single(label: str, m: dict, rows: list[dict]) -> None:
    print()
    print("=" * 78)
    print(f"生成侧指标 · {label}")
    print("=" * 78)
    print(f"总条数 {m['n_total']}（库内 {m['n_inkb']} / 库外 {m['n_oob']}）")

    print("\n【1】拒答行为")
    print(f"  库外题拒答准确率        : {fmt(m['oob_refusal_rate'])}"
          f"  ({int((m['oob_refusal_rate'] or 0) * m['n_oob'])}/{m['n_oob']})"
          f"   [只认逐字话术的严格口径: {fmt(m['oob_refusal_rate_strong'])}]")
    if m["refusal_flag_mismatch_n"]:
        print(f"  ⚠️ 与生成侧落盘标记不一致 {m['refusal_flag_mismatch_n']} 条"
              f" —— 说明两处判据已经不同步（本指标以评测侧重判为准）")
    print(f"  库外题**未拒答**（硬答）: {m['oob_answered_n']} 条  ← 幻觉风险最高，需逐条看")
    print(f"  库内题误拒率            : {fmt(m['inkb_false_refusal_rate'])}"
          f"  ({m['inkb_refused_n']}/{m['n_inkb']})")
    print(f"    ├ gold 进了上下文却拒答 : {m['refuse_gold_in_ctx_n']}/{m['gold_in_ctx_n']}"
          f"  = {fmt(m['refuse_gold_in_ctx_rate'])}   ← 真误拒，生成侧的错")
    print(f"    └ gold 没进上下文而拒答 : {m['refuse_gold_missing_n']}/{m['gold_missing_n']}"
          f"  = {fmt(m['refuse_gold_missing_rate'])}   ← 检索失败的传导，拒答是克制")

    print("\n【2】引用（只在给出答案的条目上算，拒答条目不参与）")
    print(f"  给出答案                : {m['answered_n']} 条")
    print(f"  无任何引用              : {m['no_cite_n']} 条  ({fmt(m['no_cite_rate'])})")
    print(f"  非法编号（编造引用）    : {m['bad_citation_n']} 条  ← 必须为 0")
    print(f"  编号↔chunk 映射错位     : {m['invalid_cid_n']} 条  ← 内部一致性检查，必须为 0")
    print(f"  引用准确率（gold 在上下文时，引用是否含 gold）: "
          f"{fmt(m['cite_precision'])}  ({m['cite_hit_n']}/{m['ans_gold_in_ctx_n']})")
    print(f"  引用准确率·宽松（gold 答案出现在被引用原文里）: "
          f"{fmt(m['cite_answer_in_rate'])}  ({m['cite_answer_in_n']}/{m['cite_answer_in_denom']})"
          f"  ← 读者按引用回查能看到答案的比例")
    print(f"    ├ 有引用但没引到 gold   : {m['cite_miss_with_cite_n']} 条  ← 引用错误")
    print(f"    └ 干脆没有引用          : {m['cite_miss_no_cite_n']} 条")
    print(f"  平均引用条数            : {m['cite_len_avg']}")
    print(f"  ghost_cite（gold 不在上下文却仍给引用）: {m['ghost_cite_n']} 条  ← 需人工过一眼")

    print("\n【2.5】端到端单题成败（检索 + 生成串起来算）")
    print(f"  库内题 gold 答案词面命中率 : {fmt(m['answer_hit_rate'])}"
          f"  ({m['answer_hit_n']}/{m['n_inkb']})  ← 下界，换个说法就判不中")
    print(f"  端到端通过率               : {fmt(m['e2e_pass_rate'])}"
          f"  ({m['e2e_pass_n']}/{m['n_total']})")
    print(f"    ├ 库内通过（未拒答且命中）: {m['e2e_pass_inkb_n']}/{m['n_inkb']}")
    print(f"    └ 库外通过（正确拒答）    : {m['e2e_pass_oob_n']}/{m['n_oob']}")

    print("\n【3】Faithfulness（句子级，逐题平均）")
    if m["fa_unavailable_n"]:
        print(f"  ⚠️ {m['fa_unavailable_n']} 条**拿不到原文**（结果文件里没有 contexts 字段）——")
        print("     下面的「—」是**不可算**，不是「0 分」——旧格式结果文件请重跑生成，别照抄数字。")
    print(f"  库内题 strict / lenient : {fmt(m['fa_strict_inkb'])} / {fmt(m['fa_lenient_inkb'])}")
    print(f"  库外题 strict / lenient : {fmt(m['fa_strict_oob'])} / {fmt(m['fa_lenient_oob'])}")
    print(f"  全量   strict / lenient : {fmt(m['fa_strict_all'])} / {fmt(m['fa_lenient_all'])}")
    print(f"  ▸ 子句级·按字数加权（词面主判据=实词覆盖率）: strict "
          f"{fmt(m['fac_c_strict_all'])} / lenient {fmt(m['fac_c_lenient_all'])}")
    print(f"    子句级·2-gram 版本（对照）              : strict "
          f"{fmt(m['fac_strict_all'])} / lenient {fmt(m['fac_lenient_all'])}")
    print(f"    子句数 {m['fac_clause_n']}"
          f"（实词判支撑 strict {m['fac_c_clause_strict_n']} · lenient {m['fac_c_clause_lenient_n']}）")
    print(f"  可算条数                : {m['fa_n_answers']}"
          + (f"（另有 {m['fa_unavailable_n']} 条不可算）" if m['fa_unavailable_n'] else ""))
    print(f"  数字硬违规句数          : {m['fa_num_violation_sent_n']} / {m['fa_sent_total']} 句")
    if m.get("judge_available"):
        print(f"  ★ LLM 判官（主口径，已过校准） : {fmt(m['judge_rate'])}"
              f"  ({m['judge_yes_n']}/{m['judge_clause_n']} 子句，{m['judge_rows']} 条答案)")
        if m.get("judge_minus_lexical") is not None:
            print(f"    与词面判据(实词·lenient)之差 : {m['judge_minus_lexical'] * 100:+.1f} 个百分点"
                  f"  ← 差值 = 词面判据漏掉的改写/概括")
        bk = m.get("judge_rate_by_kind") or {}
        if bk:
            print("    分题型 : " + " · ".join(f"{k} {v:.1%}" for k, v in sorted(bk.items())))

    print("\n【4】成本与延迟")
    print(f"  输入 / 输出 token       : {m['tok_in']:,} / {m['tok_out']:,}")
    print(f"  检索中位（分项重算）    : {m['lat_retrieve_ms']:.0f} ms"
          f"（其中 取原文 {m['lat_fetch_ms']:.0f} + 重排 {m['lat_rerank_ms']:.0f}）")
    if m.get("lat_total_mismatch_n"):
        print(f"  ⚠️ 有 {m['lat_total_mismatch_n']} 条的落盘 retrieve_total 与分项和对不上"
              f" —— 该文件是修计时 bug 之前生成的，一律以分项重算为准")
    print(f"  生成中位 / 端到端中位   : {m['lat_gen_s']:.2f} s / {m['lat_total_s']:.2f} s")


def print_compare(metrics: list[tuple[str, dict]]) -> None:
    if len(metrics) < 2:
        return
    cols = ["拒答准确率", "误拒率", "真误拒", "引用准确率", "引用宽松", "答案命中",
            "端到端通过", "判官FA", "词面FA", "检索ms", "端到端s"]
    print()
    print("=" * 78)
    print("横向对比")
    print("=" * 78)
    head = pad("配置", 14, "<") + "".join(pad(c, 11) for c in cols)
    print(head)
    print("-" * _w(head))

    def row(label, m):
        vals = [
            fmt(m["oob_refusal_rate"]), fmt(m["inkb_false_refusal_rate"]),
            fmt(m["refuse_gold_in_ctx_rate"]), fmt(m["cite_precision"]),
            fmt(m["cite_answer_in_rate"]), fmt(m["answer_hit_rate"]),
            fmt(m["e2e_pass_rate"]), fmt(m.get("judge_rate")),
            fmt(m["fac_c_lenient_all"]),
            f"{m['lat_retrieve_ms']:.0f}", f"{m['lat_total_s']:.2f}",
        ]
        print(pad(label, 14, "<") + "".join(pad(v, 11) for v in vals))

    for label, m in metrics:
        row(label, m)


# ==================================================================== 配对检验


def sign_test(pairs: list[tuple[bool, bool]]) -> dict:
    """
    配对符号检验（精确二项，双尾）。
    拿它来回答"重排开比关好，是真提升还是抖动"——
    只看两个百分数的大小会把噪声当结论（§11.11 已经吃过一次这个亏）。
    """
    n_plus = sum(1 for a, b in pairs if b and not a)      # A 错 → B 对
    n_minus = sum(1 for a, b in pairs if a and not b)     # A 对 → B 错
    n = n_plus + n_minus
    if n == 0:
        return {"n_plus": 0, "n_minus": 0, "n": 0, "p": 1.0}
    import math
    k = min(n_plus, n_minus)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"n_plus": n_plus, "n_minus": n_minus, "n": n, "p": round(min(1.0, 2 * tail), 4)}


def paired_compare(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """按 qid 对齐两份结果，逐题比对「通过/不通过」。"""
    ma = {r["qid"]: r for r in rows_a if r.get("qid")}
    mb = {r["qid"]: r for r in rows_b if r.get("qid")}
    common = [q for q in ma if q in mb]
    out = {"n_aligned": len(common),
           "n_only_a": len([q for q in ma if q not in mb]),
           "n_only_b": len([q for q in mb if q not in ma])}
    for name, fn in [("e2e_pass", e2e_pass),
                     ("refused", lambda r: bool(r.get("refused"))),
                     ("answer_hit", answer_hit),
                     ("cite_gold", lambda r: bool(r.get("gold_chunk_id"))
                      and r.get("gold_chunk_id") in (r.get("cited_chunk_ids") or []))]:
        out[name] = sign_test([(fn(ma[q]), fn(mb[q])) for q in common])
        out[name]["a_n"] = sum(1 for q in common if fn(ma[q]))
        out[name]["b_n"] = sum(1 for q in common if fn(mb[q]))
    return out


def write_markdown(metrics: list[tuple[str, dict]], pairs: dict | None, out: Path) -> None:
    lines = [
        "# 生成侧指标表（§11.14）",
        "",
        "指标口径见 `src/evaluate_gen.py` 文件头。",
        "",
        "- **判官FA** = LLM 判官（`src/judge_faith.py`）判为「有依据」的子句比例，**主口径**；",
        "  判官先用人工标签校准（16/16，含 4 条合成对照）才上场。",
        "- **词面FA** = 实词覆盖率（阈值 0.6）按字数加权（子句级），**下界**；",
        "  人工校准显示它的一致率只有 61~72%，且对「用词全对、关系说反」完全漏检 ——",
        "  它与判官FA 的差值，就是「概括/改写」被打掉的量。",
        "- **答案命中** = 答案里出现 gold 答案（词面，**下界**；换个说法就判不中）。",
        "",
        "| 配置 | 库外拒答准确率 | 库内误拒率 | 真误拒(gold在) | 引用准确率(严) "
        "| 引用准确率(宽) | 答案命中(下界) | 端到端通过率 | 判官FA | 词面FA | 检索中位 | 端到端中位 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, m in metrics:
        lines.append(
            f"| {label} | {fmt(m['oob_refusal_rate'])} | {fmt(m['inkb_false_refusal_rate'])} | "
            f"{fmt(m['refuse_gold_in_ctx_rate'])} | {fmt(m['cite_precision'])} | "
            f"{fmt(m['cite_answer_in_rate'])} | {fmt(m['answer_hit_rate'])} | "
            f"{fmt(m['e2e_pass_rate'])} | {fmt(m.get('judge_rate'))} | {fmt(m['fac_c_lenient_all'])} | "
            f"{m['lat_retrieve_ms']:.0f} ms | {m['lat_total_s']:.2f} s |")


    if pairs:
        lines += ["", "## 配对检验（A→B 逐题符号检验，精确二项双尾）", ""]
        lines.append(f"对齐 {pairs['n_aligned']} 条（仅 A {pairs['n_only_a']} / 仅 B {pairs['n_only_b']}）。")
        lines += ["", "| 判定 | A 通过 | B 通过 | A错→B对 | A对→B错 | p |", "|---|---|---|---|---|---|"]
        for k in ("e2e_pass", "refused", "answer_hit", "cite_gold"):
            s = pairs.get(k)
            if not s:
                continue
            lines.append(f"| {k} | {s['a_n']} | {s['b_n']} | {s['n_plus']} | {s['n_minus']} | {s['p']} |")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def print_problem_list(m: dict, limit: int = 12) -> None:
    """把必须人看一眼的三类条目列出来 —— 光有率没有例子，归因无从谈起。"""
    for key, title, note in [
        ("_oob_answered_rows", "库外题硬答（未拒答）—— 最危险的一类",
         "模型用世界知识作答而非拒答，答案外部还包着『参考资料』的样子"),
        ("_cite_miss_rows", "引用错误（gold 就在上下文里，答案却没引到它）",
         "读者按引用回查会失望；也可能是答案本身答错了"),
        ("_ghost_rows", "ghost_cite（gold 不在上下文却仍给引用）",
         "可能引了别的块支撑一个部分正确的回答 —— 只作提示，不算错"),
    ]:
        rs = m.get(key) or []
        if not rs:
            continue
        print()
        print(f"── {title} —— 共 {len(rs)} 条" + (f"，列前 {min(limit, len(rs))} 条" if len(rs) > limit else ""))
        print(f"   （{note}）")
        for r in rs[:limit]:
            ans = (r.get("answer") or "").replace("\n", " ")
            print(f"   · {r.get('qid')} [{r.get('kind')}] {r['query']}")
            print(f"     gold={r.get('gold_answer')!r}  gold_rank={r.get('gold_rank')}"
                  f"  引用={r.get('cited')}")
            print(f"     ans: {ans[:150]}")


# ==================================================================== 抽检表


def write_audit(label: str, rows: list[dict], n: int, thresh: float, cthresh: float = 0.6) -> Path:
    """
    写人工抽检表。抽样按"最容易判错/最必须看"的四类分层抽，
    不随机抽 —— 随机会抽出一堆 easy 题的完美答案，校准不到判据的边界。
    """
    answered = [r for r in rows if not r.get("_refused", r.get("refused"))]
    for r in answered:
        if "_fa" not in r:
            r["_fa"] = faithfulness(r, thresh, cthresh)

    buckets = {
        "A 库外未拒答（硬答）": [r for r in rows if is_oob(r) and not r.get("_refused")],
        "B 库内拒答（误拒）": [r for r in rows if not is_oob(r) and r.get("_refused")],
        "C 引用错误（gold 在上下文却没引到）": [r for r in answered
                                                if not is_oob(r) and r.get("gold_rank")
                                                and r.get("gold_chunk_id") not in (r.get("cited_chunk_ids") or [])],
        "D 引用正确（gold 被引用）": [r for r in answered
                                      if r.get("gold_chunk_id") in (r.get("cited_chunk_ids") or [])],
        "E 无引用但有答案": [r for r in answered if not r.get("cited")],
    }
    picked, seen = [], set()
    per = max(1, n // len(buckets))
    for name, rs in buckets.items():
        for r in rs[:per]:
            if id(r) not in seen:
                seen.add(id(r))
                picked.append((name, r))

    out = RESULTS / f"gen_audit_{label}.md"
    lines = [
        f"# 生成侧判据人工校准表 · {label}",
        "",
        "> 目的：验证 `evaluate_gen.py` 的自动判据（引用是否正确 / 句子是否有据）"
        "与人眼判定的一致率。",
        "> **判据必须先校准** —— 本项目已为此栽过 4 次（§11.10）。",
        "",
        f"判定阈值：2-gram 覆盖率 ≥ {thresh}；数字硬约束：句内阿拉伯数字必须在被引用的块里出现。",
        "",
        "请逐条在「人工判定」列写：`对` / `错` / `部分`，并在「备注」写不一致的原因。",
        "",
        "---",
        "",
    ]
    for i, (bucket, r) in enumerate(picked, 1):
        fa = r["_fa"]
        lines += [
            f"## {i}. [{bucket}] {r.get('qid')} · {r.get('kind')}",
            "",
            f"**问题**：{r['query']}",
            "",
            f"**gold 答案**：{r.get('gold_answer') if r.get('gold_answer') else '（库外题，应为拒答）'}",
            "",
            f"**gold 在上下文里的位置**：{('第 %d 条' % r['gold_rank']) if r.get('gold_rank') else '未进上下文'}",
            "",
            f"**模型答案**（原文，未截断）：",
            "",
            "```",
            (r.get("answer") or "").strip(),
            "```",
            "",
            f"**答案引用的编号**：{r.get('cited') or '（无）'}",
            "",
            "**引用的编号 ↔ 实际块**：",
            "",
        ]
        hit_cids = r.get("hit_chunk_ids") or []
        hits = r.get("hits") or []
        for h in hits:
            mark = ""
            if h["chunk_id"] == r.get("gold_chunk_id"):
                mark = "  ← **gold**"
            lines.append(f"- `[{h['rank']}]` {h.get('title')}｜{h.get('section') or '—'}{mark}")
        lines += [
            "",
            "### 原文（判 faithfulness 必须看这个，别只看答案）",
            "",
        ]
        for c in (r.get("contexts") or []):
            mark = "  ← **gold**" if c.get("chunk_id") == r.get("gold_chunk_id") else ""
            body = (c.get("text") or "").replace("\n", " ")
            lines += [
                f"**`[{c['rank']}]` {c.get('title')}｜{c.get('section') or '—'}**{mark}",
                "",
                "```",
                body[:700] + ("…（截断）" if len(body) > 700 else ""),
                "```",
                "",
            ]
        lines += [
            "",
            f"**自动判据（句子级）**：strict {fa['ratio_strict']} / lenient {fa['ratio_lenient']}"
            f"（{fa['n_strict']}/{fa['n_sent']} 句 strict 通过；数字违规 {fa['num_violations']} 句）",
            "",
            f"**自动判据（子句级·按字数加权，主判据）**：strict {fa['clause_ratio_strict']}"
            f" / lenient {fa['clause_ratio_lenient']}",
            "",
            "**子句级判定**（人工重点看这里：判'real 依据'还是'模型自己加的'）：",
            "",
            "| # | 子句 | 引用 | 覆盖率 | strict | lenient | 是否有据(人工) |",
            "|---|---|---|---|---|---|---|",
        ]
        for j, d in enumerate(fa.get("clauses") or [], 1):
            s = d["sentence"].replace("|", "/")[:44]
            lines.append(
                f"| {j} | {s} | {d['cites'] or '—'} | {d['coverage']} | "
                f"{'✅' if d['supported_strict'] else '❌'} | "
                f"{'✅' if d['supported_lenient'] else '❌'} |  |")
        lines += [
            "",
            "**句子级判定**：",
            "",
            "| # | 句子（截断） | 引用 | 覆盖率 | strict | lenient | 数字缺失 |",
            "|---|---|---|---|---|---|---|",
        ]

        for j, d in enumerate(fa["sentences"], 1):
            s = d["sentence"].replace("|", "/")[:48]
            lines.append(
                f"| {j} | {s} | {d['cites'] or '—'} | {d['coverage']} | "
                f"{'✅' if d['supported_strict'] else '❌'} | "
                f"{'✅' if d['supported_lenient'] else '❌'} | {d['num_missing'] or '—'} |")
        lines += [
            "",
            "**人工判定**：",
            "",
            "- 引用是否正确：",
            "- 答案是否忠于原文（有无幻觉）：",
            "- 备注：",
            "",
            "---",
            "",
        ]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return out


# ==================================================================== main


def main() -> int:
    ap = argparse.ArgumentParser(description="生成侧指标（拒答 / 引用 / Faithfulness）")
    ap.add_argument("--run", action="append", required=True,
                    help="标签=结果文件路径（可多次）。第一个作为主报告")
    ap.add_argument("--judge", action="append", default=None,
                    help="LLM 判官结果文件（judge_faith.py 产出），**按出现顺序与 --run 一一对应**。"
                         "给了就用判官口径作为 faithfulness 主指标；不给则只报词面下界。")
    ap.add_argument("--thresh", type=float, default=0.6,
                    help="v1 判据的句子 2-gram 覆盖率阈值，默认 0.6（保留做对照）")
    ap.add_argument("--cthresh", type=float, default=0.6,
                    help="v3 判据（主判据）的实词覆盖率阈值，默认 0.6 —— "
                         "由 §11.14 用 12 条人工标注的子句校准得出")
    ap.add_argument("--audit", type=int, default=0, help="额外产出 N 条人工抽检表")
    ap.add_argument("--out", default=None, help="指标 json 输出路径（默认 eval/results/gen_metrics.json）")
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        if "=" in spec:
            label, path = spec.split("=", 1)
        else:
            path = spec
            label = Path(spec).stem
        p = Path(path)
        if not p.is_absolute():
            p = PROJECT / path
        if not p.exists():
            raise SystemExit(f"[错误] 结果文件不存在：{p}")
        runs.append((label, p))

    print("=" * 78)
    print("生成侧指标 · §11.14")
    print("=" * 78)
    print(f"阈值 thresh={args.thresh}")
    for label, p in runs:
        print(f"  {label:<12} {p.name}")

    metrics = []
    all_metrics = {}
    rows_by_label = {}
    judges = list(args.judge or [])
    if judges and len(judges) != len(runs):
        raise SystemExit(f"[错误] --judge 给了 {len(judges)} 个，--run 有 {len(runs)} 个 —— "
                         "必须一一对应（否则会把 A 的判定安到 B 头上，指标全错且不报错）")
    for idx, (label, p) in enumerate(runs):
        rows = load_records(p)
        rows_by_label[label] = rows
        jd = None
        if judges:
            jp = Path(judges[idx])
            if not jp.is_absolute():
                jp = PROJECT / judges[idx]
            if not jp.exists():
                raise SystemExit(f"[错误] 判官文件不存在：{jp}")
            jd = load_judge(jp)
            print(f"[判官] {label} ← {jp.name}（{len(jd)} 条答案的判定）")
        m = evaluate(rows, args.thresh, args.cthresh, jd)
        cfg = next((r.get("gen_config") for r in rows if r.get("gen_config")), {})
        m["_config"] = cfg
        m["_file"] = p.name
        metrics.append((label, m))
        all_metrics[label] = {k: v for k, v in m.items() if not k.startswith("_")}

        print_single(label, m, rows)
        print_problem_list(m)
        if args.audit:
            f = write_audit(label, rows, args.audit, args.thresh, args.cthresh)
            print(f"\n[人工抽检表] {f}")

    print_compare(metrics)

    pairs = None
    if len(runs) == 2:
        la, lb = runs[0][0], runs[1][0]
        pairs = paired_compare(rows_by_label[la], rows_by_label[lb])
        print()
        print("=" * 78)
        print(f"配对检验：{la} → {lb}（逐题符号检验，精确二项双尾）")
        print("=" * 78)
        print(f"对齐 {pairs['n_aligned']} 条"
              f"（仅 {la} {pairs['n_only_a']} / 仅 {lb} {pairs['n_only_b']}）")
        for k, name in [("e2e_pass", "端到端通过"), ("refused", "拒答"),
                        ("answer_hit", "答案命中"), ("cite_gold", "引用到 gold")]:
            s = pairs[k]
            print(f"  {name:<10} {la} {s['a_n']:>3} → {lb} {s['b_n']:>3}"
                  f"   （错→对 {s['n_plus']} · 对→错 {s['n_minus']}）p={s['p']}")

    md = RESULTS / "gen_tables.md"
    write_markdown(metrics, pairs, md)
    print(f"\n[表格已存] {md}")

    out = Path(args.out) if args.out else (RESULTS / "gen_metrics.json")
    if not out.is_absolute():
        out = PROJECT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"threshold": args.thresh, "runs": all_metrics},
        ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(f"\n[指标已存] {out}")
    print("\nGEN_EVAL_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
