#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 10 步 · 评测集构建（M4 的第一块砖）

为什么评测集必须排在最前面
--------------------------
M4 要产出的是一张「5 组消融对比表」，而表里**每一格的数字都依赖同一件东西：gold 标注**。
没有它，reranker 加不加、融合策略选哪个，都只能凭感觉说"好像好一点" ——
写不进简历，也经不起面试官追问一句"好多少？怎么量的？"。

方案原话：「RAG 评测集没有现成的，企业里也是自己造：用 LLM 从语料生成候选 QA
（标注 gold 来源 chunk_id）+ 人工逐条校对。自动生成的 100 条里约三成是废的，
校对这一步不能省。」

本脚本负责「生成 + 自动预筛」，把废题率压下来，剩下的交给人工抽检。

采样设计 —— 这才是本脚本真正的技术含量
--------------------------------------
调 LLM 生成问答只是体力活。**难的是让评测集有区分度**。

反面教材：如果 100 条全是「问题约等于原文子串」的题，那么 5 组消融会全部打平 ——
vector / bm25 / hybrid / reranker 开与不开，Recall@5 都是 1.000，
一张表拉出来全是等号，**等于白测**。这是评测集设计最常见的死法。

所以人为构造难度梯度：

| 题型 | 构造方式 | 预期谁能答对 | 用来回答什么问题 |
|---|---|---|---|
| `easy` | 同字形提问，含实体名 | BM25 / 向量都行 | 基线：能不能搜到 |
| `hard_anon` | **实体匿名化**：禁止出现任何专有名词 | 只有语义检索能 | **混合检索到底值不值** |
| `hard_para` | 同义改写，换掉关键词 | 语义检索占优 | 词面匹配的脆弱性 |
| `t2s` | 繁体提问 → 简体 gold | 依赖归一化 | 繁简归一化值多少 |
| `s2t` | 简体提问 → 繁体 gold | 依赖归一化 | 同上，反方向 |
| `oob` | 库内实体 + 库外属性 | 谁都不该答对 | 拒答能力（防幻觉） |

`hard_anon` 是最有价值的一类。把「张克忠」换成
「有位天津出生、创办了南开大学化工系的化学工程学家」，问题里一个专有名词都没有 ——
BM25 的倒排索引此时毫无办法（分词后全是常用词），
**只有向量检索能靠语义召回**。它直接决定了「混合检索值不值」这个结论能不能成立。

反过来也要防：不能全是 `hard_anon`，否则基线全丢、差距被夸大。
所以按梯度配比，而不是清一色出难题。

三条硬约束（都是踩过的坑）
--------------------------
1. **过滤噪声段**：`section` 高频值里有 `外部連結`(5052) / `参见`(4436) 这类，
   内容是一串网址或条目名列表，**没有可问的事实**。不过滤就会生成
   "这篇条目有哪些外部链接" 这种废题。
2. **一个条目只出一题**：实测 441,362 个条目 / 829,100 chunk，平均每条目 1.9 个 chunk。
   若从同一篇文档抽多题，检索命中一个 chunk 就算全中 —— 指标虚高。
3. **库外题必须自动校验**：第 9 步踩过一次 —— 我用「SpaceX 星舰第五次试飞」
   当库外题，结果语料里真有，模型答得又快又准。**不能靠人猜"这个应该不在库里"**，
   要用检索 + LLM 判定来筛。

用法
----
  # 采样预览（不调 API，只看选出哪些 chunk —— 先看这个再花钱）
  python src\\build_testset.py --dry-run

  # 正式生成
  python src\\build_testset.py

  # 自定义配比
  python src\\build_testset.py --n-easy 45 --n-anon 35 --n-para 15 --n-cross 15 --n-oob 30
"""

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from zh import looks_fanti, norm_text  # noqa: E402

PROJECT = HERE.parent
PARQUET = PROJECT / "data" / "processed" / "chunks.parquet"
OUT_DIR = PROJECT / "eval"

SEED = 42

# 没有可问事实的段 —— 内容是链接列表 / 条目导航，不是叙述
NOISE_SECTIONS = {
    "外部連結", "外部链接", "外部鏈接", "外部链结", "參見", "参见",
    "參考資料", "参考资料", "參考文獻", "参考文献", "注釋", "注释",
    "延伸閱讀", "延伸阅读", "相關條目", "相关条目", "来源", "來源",
    "連結", "链接", "参考", "參考", "外部資源", "外部资源",
}

MIN_LEN, MAX_LEN = 150, 800

# 库外属性：维基百科**通常不记载**的人物属性。
# 选它的理由：这类属性是"有人会真的好奇、但百科不写"，最接近真实的用户幻觉触发场景。
# 三条筛选纪律（都是踩过坑才加的）：
#   ① 只配给**人物** —— 早期版本给任意实体配属性，生成出「2017年颶風馬克斯的血型」
#      「末昂语的身份证号码」这种荒谬题，一眼假，测不出真实拒答能力。
#   ② 不用隐私/违法类（银行卡号、身份证号）—— 模型拒答可能因为"这是隐私不能说"
#      而不是"资料里没有"，两种拒答混在一起，指标就脏了。
#   ③ 属性必须是"百科偶尔写、多数不写"的中性事实 —— 这样才需要靠检索校验来筛，
#      而不是靠人猜"这个肯定不在库里"。
OOB_ATTRS = [
    "血型", "身高和体重", "本名（原名）", "性格特点",
    "个人爱好", "饮食习惯", "座右铭", "朋友对他的评价",
    "出生的具体医院", "有没有兄弟姐妹",
]

# 人物判据：标题像人名（2~5 字、不以事物后缀结尾），且正文开头有生卒年或人物特征词
_BIRTH_RE = re.compile(r"[（(]\s*\d{3,4}\s*年")
_NON_PERSON_TAIL = tuple(
    "站語場年路線山河湖市縣區鎮村島國會黨軍戰歌書影劇學院系科門綱目"
    "瀑橋塔港灣峰嶺泉寺廟宮觀堂社城園嶼洲漠峽街巷里弄"
    "叶葉草花樹木竹藤蕨苔菌魚鳥蟲獸龍馬牛羊犬貓鼠蛇蛙蝶蜂"      # 动植物
    "器機車船機炮槍彈刀劍鐘錶鏡燈傘鞋帽衣褲裙襪杯碗盤瓶罐"    # 人造物件
)


def is_person_like(title, text):
    """
    粗判这个条目是不是「人物」。不追求精确 —— 判错的会被后面的检索校验筛掉。

    ⚠️ 2026-09-20 实测踩的坑：早期版本把「生于」也当人物特征词，
    结果**植物条目全部中招** —— 植物志的固定句式是「生于山坡草地/海拔 800 米」，
    于是生成了「中华淡竹叶有血型吗？」这种荒谬题。
    → 教训：跨领域的关键词匹配必然误伤，`出生于` 比 `生于` 精确得多，
      一个字之差就是"人物"和"植物"的分界。
    """
    if not (2 <= len(title) <= 5):
        return False
    if title.endswith(_NON_PERSON_TAIL):
        return False
    head = (text or "")[:200]
    if _BIRTH_RE.search(head):
        return True
    # 只用「出生于/出生於」，不用「生于」—— 后者是植物志/地理志的高频词
    return any(k in head for k in ("出生于", "出生於", "逝世", "卒於", "卒于", "漢族", "汉族"))


def is_listy(t, bullet_thresh=0.35, short_thresh=0.6):
    """
    是不是「列表 / 导航 / 表格」型内容 —— 这类内容**没有可问的事实**。

    为什么光过滤 section 名不够（2026-09-20 实测）：
      section 是「車站構造」这种看着很正常的名字，但正文其实是

          * 東京巨蛋城
          ** 東京巨蛋
          ** 東京巨蛋城遊樂園
          ...

      还有「後樂園」条目，64 个字全是 '* 慈眼堂 / * 御野島 / * 中之島 …'。
      从这种文本只能问出「这个条目里列了哪些地方」，是废题。
      **问题出在内容格式，不在 section 名上。**

    两条判据（命中任一即判为列表）：
      ① 以 * / : / | 等符号开头的行占比 >= 35%
      ② 行数 >= 5 且「12 字以内的短行」占比 >= 60%（碎片化罗列）
    """
    lines = [l.strip() for l in (t or "").split("\n") if l.strip()]
    if not lines:
        return True
    bullet = sum(1 for l in lines
                 if l.startswith(("*", "**", "***", ":", "：", "·", "|", "－", "-", "#", "◆", "■")))
    if bullet / len(lines) >= bullet_thresh:
        return True
    if len(lines) >= 5:
        short = sum(1 for l in lines if len(l) <= 12)
        if short / len(lines) >= short_thresh:
            return True
    return False


# ==================================================================== 采样


def _iter_chunks(shards=None):
    """按分片流式读，避免一次性把 3.3M 行拉进内存。"""
    import pyarrow.dataset as ds
    d = ds.dataset(str(PARQUET), format="parquet")
    files = sorted(d.files)
    if shards:
        files = [f for i, f in enumerate(files) if i in shards]
    for f in files:
        tbl = ds.dataset(str(f), format="parquet").to_table(
            columns=["chunk_id", "title", "section", "chunk_text"])
        cids = tbl["chunk_id"].to_pylist()
        tis = tbl["title"].to_pylist()
        ses = tbl["section"].to_pylist()
        txs = tbl["chunk_text"].to_pylist()
        for row in zip(cids, tis, ses, txs):
            yield row


def pick_candidates(target_per_bucket, shards=None, verbose=True):
    """
    扫描语料，每个条目(title)只留一个 chunk（选最长的那条，内容最完整），
    再按「字形 × 内容特征」分成两个桶，供后续按题型抽样。

    返回 (buckets, persons)：
      buckets = {"simp": [...], "fanti": [...]}
      persons = 疑似人物条目的 title 列表（库外题专用 —— 属性只配给人物）
    """
    best = {}   # title -> chunk（同条目保留最长）
    n_raw = n_title = n_noise = n_len = n_list = 0
    for cid, ti, se, tx in _iter_chunks(shards):
        n_raw += 1
        ti = (ti or "").strip()
        se = (se or "").strip()
        tx = tx or ""
        if not ti:
            n_title += 1
            continue
        if se in NOISE_SECTIONS:
            n_noise += 1
            continue
        if not (MIN_LEN <= len(tx) <= MAX_LEN):
            n_len += 1
            continue
        if is_listy(tx):
            n_list += 1
            continue
        cur = best.get(ti)
        if cur is None or len(tx) > len(cur["chunk_text"]):
            best[ti] = {"chunk_id": cid, "title": ti, "section": se,
                        "chunk_text": tx}

    buckets = defaultdict(list)
    persons = []
    for ti, c in best.items():
        # 用 chunk_text 判字形；标题也要看（有些条目名是繁体而正文是简体）
        probe = ti + c["chunk_text"]
        kind = "fanti" if looks_fanti(probe) else "simp"
        c["script"] = kind
        buckets[kind].append(c)
        if is_person_like(ti, c["chunk_text"]):
            persons.append(ti)

    if verbose:
        print(f"[采样] 扫描 {n_raw:,} 条 chunk")
        print(f"       剔除 空标题 {n_title:,} · 噪声段 {n_noise:,} · "
              f"长度不合规 {n_len:,} · **列表型 {n_list:,}**")
        print(f"       条目去重后可用 {len(best):,} 个")
        print(f"       简体条目 {len(buckets['simp']):,} · 繁体条目 {len(buckets['fanti']):,}")
        tot = len(buckets["simp"]) + len(buckets["fanti"])
        if tot:
            print(f"       繁体条目占比 {len(buckets['fanti'])/tot*100:.1f}%")
            # ⚠️ 这个数**不能**和「47.3% 的 chunk 含繁体特征字」直接比 —— 两个口径不同：
            #    47.3% 是早期探查用的粗筛（只查 30 个繁体特征字，要求命中 ≥3 个）；
            #    这里是 zhconv 全字表 + 2% 字符阈值，检测更全、阈值更低，所以必然更高。
            #    写错口径会得出"采样偏了"的假结论。
        print(f"       疑似人物条目 {len(persons):,} 个（库外题只从这里取）")
    return buckets, persons


# ==================================================================== 生成


SYS = "你是中文检索评测集的出题人。你出的题会被用来衡量一个检索系统的能力，所以题目质量比数量重要。"

BASE_RULES = """【硬性要求】
1. 问题必须**只依据上面的原文就能回答**，标准答案要能在原文里直接找到依据。
2. 问法要**自然、口语化**，像一个普通人在搜索框里打字。
   禁止"根据材料"、"文中提到"、"上述内容"这类考试腔。
3. 标准答案是**一句短话**，只写事实本身（如「天津」「1942年」），不要解释、不要复述原文。
4. 只输出 JSON，不要任何多余文字。

【输出格式】
{"question": "问题", "answer": "标准答案"}"""


def build_prompt(c, kind):
    """按题型拼 prompt —— 差异全在 extra 那几句。"""
    extra = {
        "easy": "5. 问题里可以自然地提到主体名称。",
        "hard_anon": (
            f"5. **绝对禁止**在问题里出现标题「{c['title']}」中的任何字词，"
            f"也禁止出现原文里的专有名词（人名、地名、机构名、作品名、年份例外）。\n"
            f"   必须**用特征描述来指代它**。例如不要问「张克忠的籍贯」，"
            f"而要问「有位创办了南开大学化工系的化学工程学家，他的籍贯是哪里」。\n"
            f"   目标是：读者靠这些描述能猜到问的是谁，但问题里一个专名都没有。"
        ),
        "hard_para": (
            "5. 请**用近义词替换**原文里的关键术语"
            "（例如「籍贯」→「老家」，「创办」→「建立」，「任职」→「做过什么工作」），"
            "但不要改变问题所指的事实。"
        ),
        # ⚠️ 2026-09-20 修正：这两个键原来写反了。
        # t2s 的字面含义是 Traditional→Simplified = **繁体提问 → 简体 gold**，
        # 所以它的指令必须是"原文（=gold）是简体，请用繁体提问"。
        # 原来写成"原文繁体、请用简体提问"，实现的是 s2t 的语义 —— 而 plan 那边
        # 也是按这个反语义配的池子，于是 prompt 和 plan 自洽、**只有标签名和定义相反**。
        # 后果：导出的 kind 字段方向标反（数据本身没错，q_fanti/c_fanti 是实测的），
        # 按 kind 分组的报告会把"繁问简答"和"简问繁答"两行数字互换。
        "t2s": "5. 原文是简体中文，请**用繁体中文**提问。",
        "s2t": "5. 原文是繁体中文，请**用简体中文**提问。",
    }[kind]

    user = (f"【原文】\n标题：{c['title']}\n章节：{c['section'] or '（无）'}\n"
            f"正文：\n{c['chunk_text']}\n\n{BASE_RULES}\n{extra}")
    return [{"role": "system", "content": SYS}, {"role": "user", "content": user}]


def gen_one(backend, c, kind, retries=2):
    """生成一条 QA，返回 dict 或 None。"""
    for attempt in range(retries + 1):
        try:
            txt, _ = backend.generate(build_prompt(c, kind), max_new_tokens=400)
            m = re.search(r"\{.*\}", txt, re.S)
            if not m:
                raise ValueError(f"无 JSON：{txt[:120]}")
            obj = json.loads(m.group(0))
            q = (obj.get("question") or "").strip()
            a = (obj.get("answer") or "").strip()
            if not q or not a:
                raise ValueError("空字段")
            return {"question": q, "answer": a}
        except Exception as e:
            if attempt == retries:
                print(f"  [失败] {c['chunk_id'][:8]} {kind}: {type(e).__name__} {e}")
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


# ==================================================================== 预筛

# 「问多个量」的措辞。**只收强标记**（各是／各为／分别…），
# 不用「和／与／、」—— 实测误报太多：像
#   「一种体细长侧扁、银色体色带深色纵带…的鱼，它最长能长到多少厘米」
# 里就有「、」和「多少」，但它只问一个量（11 公分）。
# 判据宁可窄，也不要制造噪音 —— 见 11.10 的教训。
MULTI_ASK = re.compile(r"各是|各为|各自|各多少|各有多|分别是|分别多少|分别有多")
NUM_VAL = re.compile(r"\d+(?:\.\d+)?")


def multi_ask_one_answer(question, answer, text):
    """
    问题在问「多个量」，答案却只给了一个 —— **字符覆盖率抓不到这类缺陷**。

    2026-09-21 实测例（v1-d27c79）：
      问「喙宽度和厚度各是多少毫米」→ 答「2.6毫米」
      原文：「喙宽度约2.6毫米，喙厚度约2.6毫米」—— 两个值**恰好相同**，
      出题的 LLM 把它们合并成了一个。

    为什么 `answer_coverage` 拦不住：那个判据问的是
    「答案里的字有多少能在原文找到」，答案是 1.0（「2.6毫米」确实在原文里）。
    它查的是**每个字有没有出处**，不查**该答的量有没有答全** ——
    这是判据的**能力边界**，不是阈值调得不对。

    判据设计：题干有强多问标记 + 答案是单个数值 + 原文含 ≥2 个数值。
    三条同时成立才报，只针对数字型（非数字型的「答漏」见不了底，不硬凑）。
    """
    if not MULTI_ASK.search(norm_text(question)):
        return None
    nums_a = NUM_VAL.findall(answer)
    if len(nums_a) != 1:
        return None                    # 0 个（列表型答案）或多个（已答全）都不报
    nums_t = NUM_VAL.findall(text)
    if len(nums_t) < 2:
        return None                    # 原文只有一个量，那答案给一个是对的
    return (f"疑似「问多个量、只答一个」：题干问多个量，答案是单个数值 "
            f"「{answer}」，而原文含 {len(nums_t)} 个数值")


def answer_coverage(answer, text):
    """
    标准答案里有多少比例的字（去重后）能在 gold 原文中找到。

    为什么不用「连续子串匹配」（2026-09-20 实测踩的坑）：
      子串判据假设答案是原文的**原文照抄**，但 LLM 提取答案时是**重组/概括**的，
      而且这恰恰是好的出题方式。实测误报例：

        答「一只巨大野熊」→ 原文「一隻…（修饰语）…巨大的野熊」，中间隔着定语 → 判失败
        答「Windows Phone、Android、IOS、Amazon」→ 原文用「和」「还有」连接 → 判失败

      9/140 条警告里绝大多数是这类误报，会让校对清单充满噪音。

    字符覆盖率对这类改写宽容，但**真幻觉仍然抓得住**：
    如果答案里的实体/数字原文根本没有，覆盖率会明显掉下来。
    """
    a = set(re.sub(r"[，。、（）()「」“”‘’\s·：；！？\[\]〈〉《》]", "", norm_text(answer)))
    if not a:
        return 1.0
    t = set(norm_text(text))
    return len(a & t) / len(a)


def leak_ratio(question, text):
    """
    问题有多大比例是直接从原文抄的。
    用「问题里的 4-gram 有多少出现在原文中」衡量 ——
    太高说明这是抄题不是出题（检索会变成字符串匹配，白送分）。
    """
    q = norm_text(question)
    t = norm_text(text)
    grams = [q[i:i + 4] for i in range(max(0, len(q) - 3))]
    if not grams:
        return 0.0
    hit = sum(1 for g in grams if g in t)
    return hit / len(grams)


def check_item(it, c, kind):
    """三条廉价本地校验，返回 (是否通过, 问题列表)。"""
    problems = []
    q, a = it["question"], it["answer"]

    # 1) 抄题检测：easy 题允许一定重合（本来就该像），难题不允许
    lr = leak_ratio(q, c["chunk_text"])
    it["leak_ratio"] = round(lr, 3)
    if kind == "easy" and lr > 0.85:
        problems.append(f"easy 题泄漏率过高 {lr:.2f}（几乎是原文子串）")
    if kind in ("hard_anon", "hard_para") and lr > 0.55:
        problems.append(f"{kind} 泄漏率过高 {lr:.2f}（改写没做够）")

    # 2) 答案必须真在原文里有依据 —— 用字符覆盖率（见 answer_coverage 的说明）
    cov = answer_coverage(a, c["chunk_text"])
    it["answer_coverage"] = round(cov, 3)
    if cov < 0.85:
        problems.append(f"答案与原文重合度过低 {cov:.2f}（疑似幻觉答案）")

    # 3) 匿名化必须真的匿名
    if kind == "hard_anon":
        qn = norm_text(q)
        leaked = [w for w in re.findall(r"[\u4e00-\u9fa5]{2,}", norm_text(c["title"]))
                  if w in qn]
        if leaked:
            problems.append(f"匿名化失败：问题里出现了标题词 {leaked}")

    # 4) 「问多个量、只答一个」—— 字符覆盖率的盲区（见 multi_ask_one_answer）
    ma = multi_ask_one_answer(q, a, c["chunk_text"])
    if ma:
        problems.append(ma)

    return (not problems), problems


def confirm_persons(backend, titles, batch=45):
    """
    把「疑似人物」的条目名交给 LLM 确认 —— 启发式筛不准，这一步是兜底。

    为什么必须有（2026-09-20 实测）：
      启发式把「置將法」（北宋的一项法规）和「曹州」（地名）都判成了人物，
      生成出「那个北宋的置將法，它有血型吗？」「曹州是在哪家医院出生的呀？」
      这种荒谬题。**跨领域的关键词匹配天然不可靠**，改字表是打地鼠。
      → 便宜的修法：一次给 LLM 45 个条目名让它挑真人，成本比逐条判低两个数量级。

    为什么只判条目名就够：中文维基的条目名区分度很高，
    「葛登名」「业喜海顺」是人名，「曹州」「置將法」一看就不是。不需要正文。
    """
    out = []
    for i in range(0, len(titles), batch):
        chunk = titles[i:i + batch]
        listing = "\n".join(f"{j+1}. {t}" for j, t in enumerate(chunk))
        prompt = [
            {"role": "system", "content": "你是严谨的资料分类员。"},
            {"role": "user", "content": (
                "下面是从中文维基百科抽出的条目名。请挑出**关于真实人物（真人）**的那些。\n"
                "注意：地名、行政区、机构、法规制度、事件、动植物、作品、职位都不算人物。\n\n"
                f"{listing}\n\n"
                '只输出 JSON：{"persons": [编号,...]}'
            )},
        ]
        try:
            txt, _ = backend.generate(prompt, max_new_tokens=600)
            m = re.search(r"\{.*\}", txt, re.S)
            idxs = json.loads(m.group(0)).get("persons", [])
            out += [chunk[int(j) - 1] for j in idxs
                    if isinstance(j, (int, str)) and str(j).isdigit()
                    and 1 <= int(j) <= len(chunk)]
            print(f"          LLM 确认：本批 {len(chunk)} 个 → 人物 {len(idxs)} 个")
        except Exception as e:
            print(f"          [警告] 人物确认失败（{type(e).__name__}），本批原样保留")
            out += chunk
    return out


def gen_oob(backend, entity, attr, retries=2):
    """生成一条库外问题：库内实体 + 库外属性。"""
    prompt = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": (
            f"请围绕「{entity}」这个实体，出一句话的自然提问，"
            f"提问的落点必须是**该实体的「{attr}」**。\n\n"
            f"【硬性要求】\n"
            f"1. 读起来要像真人真的会问的（允许有一点点口语气）。\n"
            f"2. 只输出 JSON。\n\n"
            f'【输出格式】\n{{"question": "问题"}}'
        )},
    ]
    for attempt in range(retries + 1):
        try:
            txt, _ = backend.generate(prompt, max_new_tokens=200)
            m = re.search(r"\{.*\}", txt, re.S)
            obj = json.loads(m.group(0))
            q = (obj.get("question") or "").strip()
            if q:
                return q
            raise ValueError("空问题")
        except Exception as e:
            if attempt == retries:
                print(f"  [失败] oob {entity}: {type(e).__name__} {e}")
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def verify_oob(backend, question, hits):
    """
    库外题的自动校验 —— 第 9 步的教训。

    判据不是"检索有没有命中"（库内实体 + 库外属性，命中实体条目是必然的），
    而是：**把命中的原文给裁判模型看，问它能不能回答这个问题**。
    能答 → 说明这个属性其实在库里 → 不是合格的库外题，剔除。
    """
    ctx = "\n\n".join(
        f"[{i+1}] {h.get('title','')}｜{h.get('section','')}\n{(h.get('chunk_text') or '')[:600]}"
        for i, h in enumerate(hits[:5])
    )
    prompt = [
        {"role": "system", "content": "你是严格的资料审核员。"},
        {"role": "user", "content": (
            f"【问题】{question}\n\n【资料】\n{ctx}\n\n"
            f"问题：仅凭以上资料，能否回答上面这个问题？\n"
            f"只输出 JSON：{{\"answerable\": true 或 false, \"why\": \"一句话理由\"}}"
        )},
    ]
    try:
        txt, _ = backend.generate(prompt, max_new_tokens=200)
        m = re.search(r"\{.*\}", txt, re.S)
        return bool(json.loads(m.group(0)).get("answerable"))
    except Exception:
        # 校验失败时保守处理：当作"可能可答"，让人工在校对时定夺
        return None


# ==================================================================== 输出


def write_review(items, path):
    """写出人可读的校对清单 —— 人工校对不该被逼着读 jsonl。"""
    lines = [
        "# 评测集候选 · 人工校对清单",
        "",
        f"共 {len(items)} 条。**请逐条看三件事**：",
        "",
        "1. 问题是否自然（像真人会问的，不是考试题）",
        "2. 标准答案是否真的能从下面那段原文里找到",
        "3. gold 条目是否正确（问题问的确实是这个条目）",
        "",
        "校对后在末尾「结论」里写要删的编号即可。",
        "",
        "---",
        "",
    ]
    for i, it in enumerate(items, 1):
        kind = it["kind"]
        tag = {"easy": "易", "hard_anon": "难·匿名", "hard_para": "难·改写",
               "t2s": "繁问简", "s2t": "简问繁", "oob": "库外"}.get(kind, kind)
        lines.append(f"## {i}. [{tag}] {it['question']}")
        lines.append("")
        lines.append(f"- **标准答案**：{it.get('answer') or '（库外题，应拒答）'}")
        if kind == "oob":
            lines.append(f"- **说明**：库外题 —— 正确答案是「根据已有资料无法回答」")
        else:
            lines.append(f"- **gold**：`{it['title']}` ｜ 章节 `{it['section'] or '—'}` ｜ "
                         f"chunk `{it['chunk_id'][:12]}…` ｜ 采样判定 {it.get('script','')}")
            if it.get("script_cross"):
                lines.append(
                    f"- 🔤 **实测字形交叉**：问题{'繁体' if it['q_fanti'] else '简体'}"
                    f" / gold{'繁体' if it['c_fanti'] else '简体'}"
                    f" —— 这道题实际测的是繁简归一化")
            if it.get("leak_ratio") is not None:
                # ⚠️ 2026-09-21 更正：原来这里写的是「easy 通常 0.6~0.9，匿名题应 <0.3」，
                #    那是**动手前拍的猜测值**，定稿后实测完全不是这样：
                #      easy 45 题 均值 0.260 / 中位 0.250（18/104 题甚至是 0.000）
                #      hard_anon 0.244 · hard_para 0.123 · 全部 104 题均值 0.240
                #    猜测值把读者（包括我自己做抽检时）引偏了，所以改成实测区间。
                lines.append(f"- **泄漏率**：{it['leak_ratio']}（问题 4-gram 有多少出现在"
                             f"原文里。实测中位 ≈0.25，≥0.6 才算抄题）")
            if it.get("answer_coverage") is not None:
                lines.append(f"- **答案覆盖率**：{it['answer_coverage']}"
                             f"（答案的字有多少能在原文里找到，<0.85 才可疑）")
            # ⚠️ chunk_text 内部含换行（一条 chunk 常是多句话、多段）。
            #    直接拼进一行 markdown 会**把行拆开**，且 Windows 上
            #    write_text 会把那些 \n 翻译成 \r\n，在文件里留下游离的 \r
            #    （2026-09-21 实测：review_v1.md 里积了 110 个，某些查看器
            #      渲染时会把行首覆盖掉）。换成可见分隔符，一条题就占一行。
            raw_text = (it.get('chunk_text') or '')[:260].replace("\r", "").replace("\n", " ／ ")
            lines.append(f"- **原文**：{raw_text}…")
        if it.get("warn"):
            lines.append(f"- ⚠️ **自动预筛提示**：{it['warn']}")
        lines.append("")
        lines.append("- [ ] 问题自然　- [ ] 答案在原文中　- [ ] gold 正确")
        lines.append("")
        lines.append("---")
        lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append("要删除的编号：")
    lines.append("")
    # newline="\n" 必须显式写 —— 否则 Windows 上 write_text 会把每个 \n
    # 翻译成 \r\n（newline=None 时写 os.linesep），产出的行尾随平台漂移。
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def recheck(src_path, tag="recheck"):
    """
    对已生成的候选集**重跑预筛**并重写校对清单 —— 不调 API，不花钱。

    为什么要有这个模式：预筛规则是可以改进的（实测「答案必须是原文连续子串」
    这条误报率太高，改成了字符覆盖率）。规则一改，就该能立刻重判已有候选，
    而不是重新花钱生成一遍。

    这也是把「生成」和「校验」分成两段的好处 —— **只有校验是廉价的、可反复迭代的**。
    """
    rows = [json.loads(l) for l in src_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    n_before = sum(1 for r in rows if r.get("pass_local"))
    n_changed = 0
    for r in rows:
        old = r.get("pass_local")
        if r.get("kind") == "oob":
            # 库外题的"通过"= 校验确认它确实不在库内
            r["pass_local"] = (r.get("answerable_in_kb") is False)
        else:
            c = {"chunk_id": r.get("chunk_id", ""), "title": r.get("title", ""),
                 "section": r.get("section") or "", "chunk_text": r.get("chunk_text") or ""}
            ok, probs = check_item(r, c, r["kind"])
            r["pass_local"] = ok
            r["warn"] = " ／ ".join(probs) if probs else ""
        if r.get("pass_local") != old and old is not None:
            n_changed += 1

    n_after = sum(1 for r in rows if r.get("pass_local"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cand = OUT_DIR / f"qa_testset_candidates_{tag}.jsonl"
    with cand.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    order = {"easy": 0, "hard_anon": 1, "hard_para": 2, "t2s": 3, "s2t": 4, "oob": 5}
    review = OUT_DIR / f"qa_testset_review_{tag}.md"
    write_review(sorted(rows, key=lambda x: order.get(x.get("kind"), 9)), review)

    print(f"[重判] {src_path.name} → {len(rows)} 条")
    print(f"       预筛通过 {n_before} → **{n_after}**（{n_after - n_before:+d} 条）")
    print(f"       落盘 {cand}")
    print(f"       校对清单 {review}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-easy", type=int, default=45, help="简单题（同字形 + 含实体名）")
    ap.add_argument("--n-anon", type=int, default=35, help="匿名化难题（无专有名词）")
    ap.add_argument("--n-para", type=int, default=15, help="同义改写难题")
    ap.add_argument("--n-cross", type=int, default=15, help="跨字形题（繁问简/简问繁，各半）")
    ap.add_argument("--n-oob", type=int, default=30, help="库外题（多生成一些，校验会筛掉一部分）")
    ap.add_argument("--shards", default=None, help="限定分片，如 0,1（默认全部）")
    ap.add_argument("--workers", type=int, default=8, help="并发数")
    ap.add_argument("--dry-run", action="store_true", help="只采样，不调 API")
    ap.add_argument("--recheck", default=None,
                    help="只对已有候选集重跑预筛（传 jsonl 路径），不调 API")
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()

    if args.recheck:
        return recheck(Path(args.recheck), tag=args.tag)

    rnd = random.Random(SEED)
    shards = [int(x) for x in args.shards.split(",")] if args.shards else None

    buckets, persons = pick_candidates(None, shards)
    simp, fanti = buckets["simp"], buckets["fanti"]
    rnd.shuffle(simp)
    rnd.shuffle(fanti)

    # 分桶取用：easy 用简体与繁体各半（同字形提问），跨字形题反向配对
    plan = []
    half_easy = args.n_easy // 2
    plan += [("easy", c) for c in simp[:half_easy]]
    plan += [("easy", c) for c in fanti[:args.n_easy - half_easy]]
    plan += [("hard_anon", c) for c in simp[half_easy:half_easy + args.n_anon // 2]]
    plan += [("hard_anon", c) for c in fanti[half_easy:][:args.n_anon - args.n_anon // 2]]

    n_para_s = args.n_para // 2
    plan += [("hard_para", c) for c in simp[half_easy + args.n_anon // 2:][:n_para_s]]
    plan += [("hard_para", c) for c in fanti[half_easy + args.n_anon - args.n_anon // 2:][:args.n_para - n_para_s]]

    # 跨字形：繁体 gold 配**简体**提问（s2t）、简体 gold 配**繁体**提问（t2s）
    # 键名与上面 prompt 的修正是配套的，两处必须同时改。
    half_cross = args.n_cross // 2 + args.n_cross % 2
    off_s = half_easy + args.n_anon // 2 + n_para_s
    off_f = half_easy + (args.n_anon - args.n_anon // 2) + (args.n_para - n_para_s)
    plan += [("s2t", c) for c in fanti[off_f:][:half_cross]]
    plan += [("t2s", c) for c in simp[off_s:][:args.n_cross - half_cross]]

    print()
    print(f"[题型配比] easy {args.n_easy} · anon {args.n_anon} · para {args.n_para} "
          f"· cross {args.n_cross} · oob {args.n_oob}")
    print(f"           库内合计 {len(plan)} 条待生成")
    # 库外题实体池：启发式粗筛（dry-run 看到的就是这批），
    # 正式跑时再过一遍 LLM 确认 —— 见 confirm_persons 里记的那个坑
    rnd.shuffle(persons)
    oob_pool = persons[:max(args.n_oob * 2, args.n_oob)]
    oob_pairs = []
    print()

    if args.dry_run:
        print("--- 采样预览（前 8 条，未调 API）---")
        for kind, c in plan[:8]:
            print(f"  [{kind:9s}] {c['title'][:20]:22s} {c['section'][:10]:12s} "
                  f"{len(c['chunk_text']):4d}字  {c['script']}")
            print(f"      {c['chunk_text'][:70]}…")
        print()
        print("--- 库外题实体预览（启发式粗筛；正式跑时还会过一遍 LLM 确认）---")
        for e in oob_pool[:8]:
            print(f"  {e[:26]:28s} × {rnd.choice(OOB_ATTRS)}")
        print()
        print("（--dry-run 结束。确认采样合理后去掉该参数正式生成）")
        return 0

    from generator import CloudBackend
    backend = CloudBackend(model="qwen-plus", temperature=0.7, timeout=90)

    # ---- 库外题实体：启发式粗筛 → LLM 确认（实测启发式会把法规/地名当人）----
    if args.n_oob and oob_pool:
        print(f"[库外] 候选实体 {len(oob_pool)} 个，交 LLM 确认是否真人 …")
        confirmed = confirm_persons(backend, oob_pool)
        print(f"       确认为人物 {len(confirmed)}/{len(oob_pool)} 个")
        oob_pairs = [(e, rnd.choice(OOB_ATTRS)) for e in confirmed[:args.n_oob]]
        if len(oob_pairs) < args.n_oob:
            print(f"[警告] 确认后只够 {len(oob_pairs)} 条库外题（目标 {args.n_oob}）")

    items = []
    t0 = time.time()
    print(f"[生成] {len(plan)} 条库内题，并发 {args.workers} …")
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(gen_one, backend, c, kind): (kind, c) for kind, c in plan}
        for n, fut in enumerate(as_completed(futs), 1):
            kind, c = futs[fut]
            r = fut.result()
            if not r:
                continue
            r.update(chunk_id=c["chunk_id"], title=c["title"], section=c["section"],
                     chunk_text=c["chunk_text"], script=c["script"], kind=kind)
            ok, probs = check_item(r, c, kind)
            r["pass_local"] = ok
            r["warn"] = " ／ ".join(probs) if probs else ""

            # 用**实测**字形替代采样时的预测 —— 见 actual_script_cross 的说明
            r["q_fanti"] = looks_fanti(r["question"], 0.02)
            r["c_fanti"] = looks_fanti(c["chunk_text"], 0.05)
            r["script_cross"] = bool(r["q_fanti"] != r["c_fanti"])

            items.append(r)
            if n % 20 == 0:
                print(f"       {n}/{len(plan)} …")

    n_pass = sum(1 for it in items if it["pass_local"])
    n_cross = sum(1 for it in items if it["script_cross"])
    print(f"[生成] 完成 {len(items)} 条，本地预筛通过 {n_pass} 条"
          f"（{len(items)-n_pass} 条带警告，仍保留供人工判断）")
    print(f"       实测字形交叉 {n_cross} 条（问题字形 ≠ gold 字形）"
          f"—— 这批是衡量繁简归一化价值的关键样本")

    if oob_pairs:
        print(f"[生成] {len(oob_pairs)} 条库外题，并发校验 …")
        def do_oob(pair):
            e, a = pair
            q = gen_oob(backend, e, a)
            if not q:
                return None
            return {"question": q, "answer": None, "kind": "oob",
                    "title": e, "section": a, "chunk_id": None,
                    "chunk_text": None, "entity": e, "oob_attr": a}
        with ThreadPoolExecutor(args.workers) as ex:
            oob_items = [r for r in ex.map(do_oob, oob_pairs) if r]

        # 自动校验：把检索 top-5 交给裁判模型判"能否回答"
        print(f"       校验 {len(oob_items)} 条（查是否其实在库内）…")
        from generator import retrieve, make_searcher

        class _A:
            pass
        a_stub = _A()
        a_stub.vec_backend, a_stub.nprobe = "faiss", 512
        a_stub.topk, a_stub.topn = 5, 100
        a_stub.fuse, a_stub.rrf_k = "rrf", 10
        a_stub.w_vec, a_stub.w_bm25 = 1.0, 1.0
        searcher = make_searcher(a_stub)

        # 🔴 必须先预热一次 —— FlagModel 的**首次加载不是线程安全的**。
        # 2026-09-20 试跑实测：judge 用 4 线程并发时，4 个线程同时触发首次
        # encode，日志里出现 3 次「[模型加载] FlagModel bge-large-zh」，
        # 其中 3 条报 `AttributeError: 'list' object has no attribute 'keys'`。
        # 串行跑同样的查询则 100% 正常 —— 典型的一次性竞态。
        # 先单线程跑一次把模型加载完，之后再进并发就安全了。
        print("       预热向量模型（FlagModel 首次加载非线程安全）…")
        searcher._encode(["预热"])

        def judge(it):
            try:
                # qv 要自己算 —— retrieve 不会替我们编码查询
                qv = searcher._encode([it["question"]])[0]
                hits, _ = retrieve(searcher, it["question"], a_stub, qv)
                it["answerable_in_kb"] = verify_oob(backend, it["question"], hits)
                it["top5_titles"] = [h.get("title", "")[:20] for h in hits[:5]]
            except Exception as e:
                it["answerable_in_kb"] = None
                it["judge_error"] = f"{type(e).__name__}: {e}"
            return it

        with ThreadPoolExecutor(4) as ex:
            oob_items = list(ex.map(judge, oob_items))

        n_ok = sum(1 for it in oob_items if it["answerable_in_kb"] is False)
        n_bad = sum(1 for it in oob_items if it["answerable_in_kb"] is True)
        print(f"       校验结果：确认库外 {n_ok} 条 · **其实在库内 {n_bad} 条（已标为不合格）**")
        items += oob_items

    # ---- 落盘 ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cand = OUT_DIR / f"qa_testset_candidates_{args.tag}.jsonl"
    with cand.open("w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # 校对清单：库内题按题型排，库外题放最后
    order = {"easy": 0, "hard_anon": 1, "hard_para": 2, "t2s": 3, "s2t": 4, "oob": 5}
    items_sorted = sorted(items, key=lambda x: order.get(x["kind"], 9))
    review = OUT_DIR / f"qa_testset_review_{args.tag}.md"
    write_review(items_sorted, review)

    print()
    print(f"[落盘] 候选集 {cand}")
    print(f"       校对清单 {review}")
    print(f"[耗时] {time.time()-t0:.0f} 秒")
    print()
    print("下一步：打开校对清单人工过一遍（重点看硬难题和库外题），")
    print("       把要删的编号记下来，然后定稿 100 条。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
