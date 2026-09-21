"""Spark 数据清洗管道：JSONL → 干净 chunk → Parquet。

这是整个项目最核心的一段（M1），也是简历上"数据工程"能力的证据来源。

用法：
    # 本地模式小样本试跑（先跑这个）
    python src/build_pipeline.py --input data/raw/wiki.jsonl --limit-docs 2000

    # 正式跑
    python src/build_pipeline.py --input data/raw/wiki.jsonl

    # 推到集群（M1b，把 master 换掉即可）
    python src/build_pipeline.py --master spark://192.168.192.129:7077

本机实测坑（都是真踩过的，别绕）：
  * JAVA_HOME 环境变量在这台机器上是空的 → Spark 启动直接失败。本脚本会自动
    从 PATH 里的 java 反推 JAVA_HOME 兜底，但建议你还是手动设成永久变量。
  * **Windows 写 Parquet 需要 winutils.exe** → 否则报 "HADOOP_HOME and
    hadoop.home.dir are unset"。读 jsonl 不需要它，所以前 6 步全绿、第 7 步才炸。
    脚本会自动找 <项目>/hadoop/bin/winutils.exe。
  * **Windows 上别用 local[*]** → 同时拉太多 Python worker，会随机崩一个，报
    java.net.SocketException: Connection reset by peer: socket write error，
    而且看不到任何 Python 异常。默认已改成 local[4]。
  * Spark worker 用的 Python 必须和 driver 一致，否则报
    "Python in worker has different version"。脚本里已强制设 PYSPARK_PYTHON。
  * Windows 上 Arrow 加速偶发崩，脚本里已关闭。
  * 输出目录已存在会直接报错，脚本用 mode('overwrite') 处理。
  * 别用 spark.read.json 自动推断 schema —— samplingRatio 默认 1.0，
    会把整个 9.4GB 文件先解析一遍猜类型。脚本已改成显式 schema。
  * Spark 溢写目录默认在 C 盘 Temp，本机 C 盘紧，脚本已挪到 data/_spark_tmp。
  * 如果报其他 winutils.exe 相关错误 → 见 操作手册.md 的"易错点 J"。
"""
import argparse
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

# ------------------------------------------------------------------
# 环境兜底：必须在 import pyspark 之前做完
# ------------------------------------------------------------------
def ensure_java_home():
    """本机 JAVA_HOME 是空的，Spark 会启动失败。从 PATH 里的 java 反推。"""
    jh = os.environ.get("JAVA_HOME", "")
    if jh and Path(jh, "bin", "java.exe").exists():
        return jh
    java_exe = shutil.which("java")
    if java_exe:
        home = str(Path(java_exe).resolve().parent.parent)
        os.environ["JAVA_HOME"] = home
        return home
    return None


def ensure_hadoop_home():
    """Windows 上**写**文件（Parquet）必须有 winutils.exe，否则报
        java.io.FileNotFoundException: HADOOP_HOME and hadoop.home.dir are unset

    阴险的地方：**读** jsonl 不需要它，所以前 6 步全绿，只在第 7 步落盘时炸。
    hadoop.dll 必须能被找到，所以还要把 <HADOOP_HOME>\\bin 塞进 PATH。
    """
    here = Path(__file__).resolve().parent.parent
    candidates = []
    if os.environ.get("HADOOP_HOME"):
        candidates.append(Path(os.environ["HADOOP_HOME"]))
    candidates += [
        here / "hadoop",            # 项目内（推荐）
        Path("E:/AI-learning/hadoop"),
        Path("C:/hadoop"),
    ]
    for c in candidates:
        if (c / "bin" / "winutils.exe").exists():
            os.environ["HADOOP_HOME"] = str(c)
            os.environ["hadoop.home.dir"] = str(c)
            os.environ["PATH"] = str(c / "bin") + os.pathsep + os.environ.get("PATH", "")
            return c
    return None


JAVA_HOME = ensure_java_home()
HADOOP_HOME = ensure_hadoop_home()
# worker 进程必须用同一个 Python 解释器
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, StructField, StructType


# JSONL 的字段是固定的（wiki_extract.py 产出），显式声明 schema。
# 别用 spark.read.json 的自动推断：它对 JSON 的 samplingRatio 默认是 1.0，
# 意味着会把 **整个文件** 先解析一遍来猜类型 —— 9.4GB 白读一次，十几分钟。
JSON_SCHEMA = StructType([
    StructField("doc_id", StringType(), True),
    StructField("title", StringType(), True),
    StructField("text", StringType(), True),
])


# ------------------------------------------------------------------
# 清洗规则（wikitext → 纯文本）
# ------------------------------------------------------------------
RE_COMMENT = re.compile(r"<!--.*?-->", re.S)
RE_REF = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", re.S | re.I)

# ⚠️ 表格这条**故意**不用缓释点号，别"顺手优化"（2026-09-17 实测教训）：
#     真实维基里有嵌套表格  {|  外层  ...  {| 内层 |}  ...  |}
#     * `.*?\|\}`（现用）从最外层 {| 一路吃到第一个 |} —— 整块端掉，正确。
#     * `(?!\{\||\|\}).`（缓释）一遇到内层 {| 就退缩，只删得掉内层，
#       留下一个空的外层表壳 `{|\n|\n|}` —— 全量实测 {| 残留反而涨 8 倍。
#   结论：表格的语义是"整块吃"，不是"以定界符为界"。快慢不同，别混用。
RE_TABLE = re.compile(r"\{\|.*?\|\}", re.S)

# 命名空间链接 —— `[[Category:X]]` / `[[:Category:X]]`。
#
# ⚠️ 这不是"链接"，是**元数据**，必须整块删掉，不能像普通内链那样只剥方括号。
# 旧版没有这条规则，于是 RE_LINK_BARE 把 `[[Category:杭州步行街]]`
# 变成纯文本 `Category:杭州步行街` 当正文留下 —— 结果：
#   全量实测（2026-09-17，3,926,261 块）：
#     含 `Category:` 的块            492,872 条   12.553%
#     整块只有分类标签、零正文的块    143,683 条    3.660%  ← 纯噪声，检索时只会污染召回
#   实测样例（这一条整块就是这个，一个字正文都没有）：
#     中国丝绸城步行街
#     Category:杭州步行街 / Category:拱墅区建筑物 / Category:丝绸 / Category:杭州市场
#
# 排在 RE_LINK_* 之前跑：一旦被剥成纯文本就再也认不出来了。
RE_CATEGORY = re.compile(
    r"\[\[\s*:?\s*(?:Category|分類|分类|CATEGORY)\s*:[^\[\]]*\]\]", re.I)

# <gallery> ... </gallery> —— 整块删掉，不能只删标签。
#
# ⚠️ 这是 `File:` 残留的**第一个来源**（第 6 步审计实测，2026-09-17）。
# gallery 里的文件行**本来就没有方括号**，是独立的 gallery 语法：
#     <gallery>
#     File:Delphi amphitheater from above dsc06297.jpg|位於[[希臘]]的古代劇場
#     File:Theater Orange.jpg|位於[[法國]]的羅馬劇場
#     </gallery>
# 旧版只有 RE_HTML 删掉 `<gallery>` `<gallery>` 这两个标签本身，
# 里面的 `File:xxx.jpg|说明` 全部变成正文留下。
# 实测：20,000 篇样本里 2,552 篇（12.8%）含 `File:` / `Image:` 残留文本。
RE_GALLERY = re.compile(r"<gallery[^>]*>.*?</gallery\s*>", re.S | re.I)

# 缓释点号 (tempered dot) —— 只把 **真正的定界符** 排除在外，单个括号放行。
# 旧写法 `[^\[\]]+` 遇到"链接文字里带一个方括号"就永久失配，迭代一万轮也没用：
#     [[SMTOWN LIVE 2025|SMTOWN LIVE 2025 [THE CULTURE, THE FUTURE]]]
#                                    ↑ 就是这个单个 [ ] 卡死
RE_TEMPLATE = re.compile(r"\{\{(?:(?!\{\{|\}\}).)*?\}\}", re.S)
RE_FILE = re.compile(r"\[\[(?:File|Image|文件|图像|圖片)\s*:(?:(?!\[\[|\]\]).)*?\]\]",
                     re.I | re.S)
# ⚠️ 内链正则必须**排除命名空间前缀**，否则会跟 RE_FILE / RE_CATEGORY 抢活。
# 第 6 步审计实测的 `File:` 残留，第二个来源就是这个"抢活"：
#   [[File:maya.svg|thumb|200px|right|[[玛雅文明#数学|玛雅]]数字，黑點代表一…]]
#     ① RE_FILE 因内层 [[ 失配
#     ② RE_LINK_PIPE 把内层 [[玛雅文明#数学|玛雅]] 剥成 `玛雅`
#     ③ 外层此刻变"干净"了 → **RE_LINK_BARE 抢先把它当普通链接剥成
#        `File:maya.svg|thumb|200px|right|玛雅数字…`**，整块内容留成了正文
# 修法：加命名空间负向前瞻 —— 带 File:/Image:/Category: 前缀的一概不碰，
# 交给专门的规则处理。这样嵌套 File 会在下一轮迭代里被 RE_FILE 正确整块删掉。
_RE_NS = r"(?!\s*(?:File|Image|文件|图像|圖片|Category|分類|分类)\s*:)"
RE_LINK_PIPE = re.compile(r"\[\[" + _RE_NS + r"((?:(?!\[\[|\]\]).)+?)\|((?:(?!\[\[|\]\]).)+?)\]\]", re.S)
RE_LINK_BARE = re.compile(r"\[\[" + _RE_NS + r"((?:(?!\[\[|\]\]).)+?)\]\]", re.S)
RE_EXTLINK = re.compile(r"\[https?://\S+\s*([^\]]*)\]")
RE_HTML = re.compile(r"<[^>]{1,200}>")
RE_QUOTE = re.compile(r"'''''|'''|''")
RE_MAGIC = re.compile(r"__[A-Z]+__")
RE_SPACES = re.compile(r"[ \t]{2,}")
RE_MULTI_NL = re.compile(r"\n{3,}")
RE_CN = re.compile(r"[\u4e00-\u9fff]")

# 语言变体标记 -{...}- ：中文维基里到处都是，是"残留标记"的头号来源。
#   -{zh-cn:域;zh-tw:體}-        → 域
#   -{zh-hans:信息; zh-hant:資訊;}-  → 信息
#   -{于}-                        → 于   （没有 key，原样保留内容）
#   -{}-                          → 空
#
# ⚠️ 这里有**两套**正则，别合并成一套（2026-09-17 实测血泪）：
#
#   RE_LANGVAR   严格版：内容不含任何 {} 。**第一遍用**。
#   RE_LANGVAR_ANY 缓释版：内容只排除 `-{` / `}-` 这对真定界符。**最后一遍用**。
#
# 为什么必须先严格后缓释：缓释版会吃掉这种情况
#     -{zh-hant:[[File:A.svg|thumb|...]]; zh-hans:[[File:B.svg|thumb|...]]}-
# 但它此刻跑在"剥链接"之前，于是 _pick_langvar 把 zh-hans 变体**内联进正文** ——
# 里面还带着完整的 [[File:...]] 标记。等于"删掉的东西又被搬回来了"，
# 实测就是:旧版没有的 [[ 残留，新版反而多出来。第二遍跑在链接剥完之后，
# 那时的 -{...}- 内部已经只剩纯文本，再缓释也不可能内联出标记。
RE_LANGVAR = re.compile(r"-\{([^{}]*)\}-")
RE_LANGVAR_ANY = re.compile(r"-\{((?:(?!-\{|\}-).)*)\}-", re.S)

# {{lang|en|Foo}} / {{lang-en|Foo}} —— 这类模板的真实内容是有意义的，
# 整块删掉会丢信息（"本姓 {{lang|en|O'Neal}}" 会变成"本姓 "）。先抽内容再删模板。
RE_LANG_TPL = re.compile(r"\{\{\s*lang(?:ue)?\s*[-|]\s*[a-zA-Z-]+\s*\|\s*([^{}|]+?)\s*\}\}",
                         re.I)

# ============================================================================
# 「删过头」留下的空壳 —— 第 6 步数据审计才发现的，前两轮**完全没检测这一类**
# ============================================================================
# 问题来源：{{lang-en|Kïrïl Pavlov}} 这类模板被 RE_TEMPLATE 整段删掉，
# 但它外围的括号是独立字符，于是正文里留下一个光秃秃的 `（）`。
#
# 全量实测（2026-09-17，3,934,758 块）：
#     空括号 （）    310,941 块   7.902%
#     连续标点 ，。    76,587 块   1.946%
#     空书名号 《》    49,986 块   1.270%
#     空引号 「」『』   14,532 块   0.369%
#   对照：当时残留标记最多的一项（{{ ）只有 702 块 = 0.018%。
#   **「删过头」比「没删干净」严重 440 倍**，而我前两轮一直在盯后者。
#
# 实测样例：
#   基里尔·帕夫洛夫（）是一名哈萨克斯坦举重运动员。
#   文之里站（）是位於日本大阪府大阪市阿倍野區…
#   C·J·克蕾格在第一季第18集《》中對嘴演唱的《》…
#
# 为什么不能靠"扩展模板名单"解决：这类模板有 {{lang-xx}} / {{IPA}} / {{jpn}} /
# {{nihongo}} / {{transl}} / {{kor}} … 几十种还带地区变体，穷举必漏。
# **兜底删空壳才是正解** —— 不管哪个模板删空的，空壳一概不要。
#
# ⚠️ 第四轮补丁（2026-09-17，人工抽样的第 4 条样本逼出来的）：
# 上面这版只认「括号里**什么都没有**」，于是漏掉了一整类 ——
#     `山田宗昌（やまだ むねなさ，1562年—1636年）` 模板删完 → `山田宗昌（，）號匡得`
# 逗号还在括号里，`[（(]\s*[）)]` 匹配失败，**验收脚本报 EMPTYPAREN=0，给了我假绿**。
# 全量实测 89,143 块 = 2.685%（比 EMPTYPAREN 声称的 0 严重得多）。
# 形态高度集中：`（，）` 63%、`（；）` 24%、`（、）` 4%、`（-）` 1%，合计 92%。
#
# 所以括号内的字符集从 `\s*` 扩成"空白 + 中文标点 + ASCII 标点 + 连接号"。
# **刻意不收 `…` 和 `.`** —— `（...）` 是合法的省略用法，删了就是误伤。
# （`*` 允许零长度，所以原来的空括号 （） 仍然被这条覆盖。）
SHELL_INNER = r"[\s，。；：、,;:\-–—]*"
RE_EMPTY_SHELL = re.compile(
    r"[（(]" + SHELL_INNER + r"[）)]"          # （） （，） （；） （、） （-）
    r"|[《]" + SHELL_INNER + r"[》]"           # 《》 《，》
    r"|[「『]" + SHELL_INNER + r"[」』]"        # 「」 『』
    r"|[【]" + SHELL_INNER + r"[】]"           # 【】
    r"|[“]" + SHELL_INNER + r"[”]"             # “”
)

# 连续标点 —— 同样是"删掉中间内容"的产物。
# 实测样例：
#     [[淮阳]]、[[登封]]、、[[郾城]]     ← 中间某个链接被整个删掉，留出 `、、`
#     并增加了12个功能键。。|电子科技大学研发…
#
# ⚠️ 第一版只压「同一个标点重复」（`([，。；、])\1+`），结果 A/B 显示
# 该指标**从 301 涨到 1,493** —— 因为真正多的是「不同标点相邻」（`。、` / `，。`），
# 它们来自"原本被链接隔开、清洗后接到一起"。**统计口径和修复口径必须一致**，
# 否则指标会骗你。改成统一压 `[，。；、]` 的任意连续组合。
#
# 保留规则：组合里只要有句号就留句号（句号是断句，优先级最高），否则留第一个。
# 不收录 `？！` —— "真的吗？！" 是正常中文，压掉就是误伤。
# `……`(U+2026) 和 `——`(U+2014) 不在字符集里，天然不受影响。
RE_DUP_PUNCT = re.compile(r"[，。；、]{2,}")


def _squash_punct(m):
    s = m.group(0)
    return "。" if "。" in s else s[0]


def _dirty(s: str) -> bool:
    """内容里还残留 wiki 标记？这种内容一律不许内联回正文。"""
    return ("[[" in s) or ("]]" in s) or ("{{" in s) or ("}}" in s)


def _pick_langvar(m):
    """从 -{...}- 里挑一个中文变体，挑不到就退回原内容。"""
    body = m.group(1).strip()
    if not body:
        return ""
    # 优先级：简体 > 通用中文 > 繁体（本项目是简体语料）
    for key in ("zh-cn", "zh-hans", "zh", "zh-hant", "zh-tw", "zh-hk"):
        mm = re.search(re.escape(key) + r"\s*:\s*([^;}]+)", body, re.I)
        if mm:
            v = mm.group(1).strip()
            # 安全阀：挑出来的变体若还带标记（说明链接/模板还没剥），
            # 宁可丢空也绝不内联回正文 —— 否则等于把删掉的标记搬回来。
            if _dirty(v):
                return ""
            return v
    if ":" not in body:
        return "" if _dirty(body) else body     # -{于}- / -{’}-
    fallback = body.split(":", 1)[1].split(";")[0].strip()
    return "" if _dirty(fallback) else fallback


def clean_wikitext(text: str) -> str:
    """把 wikitext 标记洗成人能读的纯文本。顺序很重要，别调换。

    两个"顺序型"坑（2026-09-16 实测，全量跑完抽样才发现）：
      1. 语言变体 -{...}- 必须最先剥。它自带 {} ，留着会让 RE_TEMPLATE 的
         [^{}]* 跨不过去，于是 {{lang|en|O-{’}-Neal}} 整块残留。
      2. File/内链剥离必须**迭代到不动点**。File 说明文字里常嵌普通内链：
             [[File:X.jpg|thumb|[[保羅·凱恩]]畫的圖]]
         RE_FILE 的 [^\\[\\]]* 跨不过嵌套，第一次必然失配；等 RE_LINK_BARE
         把内层拆掉之后，外层才变得可匹配 —— 但那时 RE_FILE 已经跑过了。
         只跑一轮的话，这 11.5 万个带图条目就永久残留了 [[File:...]]。
      3. 语言变体 -{...}- 必须**跑两轮**：一轮在最前（剥掉简单变体，免得它的 {} 挡住
         RE_TEMPLATE），一轮在所有链接/模板都剥完之后（此时 -{zh-hant:<图片>;zh-hans:</图片>}-
         已经退化成 -{zh-hant:;zh-hans:}-，终于能被判空）。只跑前面那轮，全量会残留
         数百个 -{...}- 空壳。
      4. 光加迭代次数**救不了**"字符类排除错"的问题。旧写法 `[^\[\]]+` 遇到
         "链接文字里含单个方括号" 永远失配 —— 迭代一万轮也是零。正解是缓释点号
         `(?:(?!\[\[|\]\]).)+?`：只排 [[ / ]]，单个括号放行。
         反例警告：缓释**不能**用在表格上（见 RE_TABLE 上方注释），会适得其反。
    """
    if not text:
        return ""
    t = RE_COMMENT.sub("", text)
    t = RE_REF.sub("", t)
    t = RE_TABLE.sub("", t)
    # 命名空间链接必须赶在 RE_LINK_* 之前删干净 —— 被剥成纯文本后就认不出来了
    t = RE_CATEGORY.sub("", t)
    # gallery 整块删，必须赶在 RE_HTML 之前（RE_HTML 会先吃掉 <gallery> 标签，
    # 之后 RE_GALLERY 就匹配不到了，里面的 File: 行会全部留下）
    t = RE_GALLERY.sub("", t)

    # 第 1 步：语言变体标记（必须在模板之前）
    t = RE_LANGVAR.sub(_pick_langvar, t)
    # 第 2 步：{{lang|xx|}} 抽内容（必须在删模板之前）
    t = RE_LANG_TPL.sub(r"\1", t)
    # 第 3 步：删模板，嵌套可能多层，跑到不动点
    # 上限给到 10：中文维基里 {{clade}}/{{Infobox}} 套 6~8 层很常见
    for _ in range(10):
        t, n = RE_TEMPLATE.subn("", t)
        if n == 0:
            break

    # 第 4 步：文件/内链——反复跑到不动点，让嵌套括号逐层解开
    for _ in range(8):
        before = t
        t = RE_FILE.sub("", t)
        t = RE_LINK_PIPE.sub(r"\2", t)      # [[A|B]] -> B
        t = RE_LINK_BARE.sub(r"\1", t)      # [[A]]   -> A
        if t == before:
            break

    t = RE_EXTLINK.sub(r"\1", t)
    t = RE_HTML.sub("", t)
    t = RE_QUOTE.sub("", t)
    t = RE_MAGIC.sub("", t)

    # 第 5 步：语言变体的**第二遍**（见 docstring 第 3 条）。
    # 此刻链接/模板/HTML 都剥干净了，裸的 -{zh-hant:;zh-hans:}- 终于能识别并丢空。
    # 跑 2 轮：_pick_langvar 可能产出 "-{a}-{b}-" 这种新相邻结构。
    for _ in range(2):
        t, n = RE_LANGVAR_ANY.subn(_pick_langvar, t)
        if n == 0:
            break

    # 第 6 步：清掉"删过头"留下的空壳（见 RE_EMPTY_SHELL 上方的实测数据）。
    # 跑 3 轮：删掉内层空壳后外层才可能变空（（（））→ （） → 空），一轮不够。
    for _ in range(3):
        t, n = RE_EMPTY_SHELL.subn("", t)
        if n == 0:
            break
    t = RE_DUP_PUNCT.sub(_squash_punct, t)

    t = RE_SPACES.sub(" ", t)
    t = RE_MULTI_NL.sub("\n\n", t)
    return t.strip()


def cn_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(RE_CN.findall(text)) / len(text)


def max_char_share(text: str) -> float:
    """出现次数最多的字符占比。超过 0.4 基本是乱码或刷字符。"""
    if not text:
        return 1.0
    return max(Counter(text).values()) / len(text)


RE_HEADING_LINE = re.compile(r"^={2,6}\s*(.+?)\s*={2,6}$", re.M)
RE_PARA_SPLIT = re.compile(r"\n\s*\n")


def split_chunks(text: str, max_chars: int = 800, overlap: int = 100, min_chars: int = 50):
    """标题感知切块：先按 == 标题 == 切 section，再在 section 内按空行聚段。"""
    out = []
    # 1. 切成 (section_title, body)
    sections, last, cur_title = [], 0, ""
    for m in RE_HEADING_LINE.finditer(text):
        if m.start() > last:
            sections.append((cur_title, text[last:m.start()]))
        cur_title = m.group(1).strip()
        last = m.end()
    sections.append((cur_title, text[last:]))

    # 2. section 内聚段
    for sec_title, body in sections:
        paras = [p.strip() for p in RE_PARA_SPLIT.split(body.strip()) if p.strip()]
        buf = ""
        for p in paras:
            if len(buf) + len(p) + 1 <= max_chars:
                buf = f"{buf}\n{p}" if buf else p
                continue
            if len(buf) >= min_chars:
                out.append((sec_title, buf))
            # 单段超长：硬切 + 重叠
            while len(p) > max_chars:
                out.append((sec_title, p[:max_chars]))
                p = p[max_chars - overlap:]
            buf = p
        if len(buf) >= min_chars:
            out.append((sec_title, buf))
    return out


# ------------------------------------------------------------------
# 包成 Spark UDF 用的纯函数（返回简单类型，避免复杂 schema 序列化问题）
# ------------------------------------------------------------------
def udf_clean(text):
    return clean_wikitext(text) if text else None


def udf_chunk(text):
    """返回 ['章节标题\u0001正文', ...]，\u0001 做分隔符，Spark 侧再拆开。"""
    if not text:
        return []
    return [f"{sec}\u0001{body}" for sec, body in split_chunks(text)]


def udf_cn_ratio(text):
    return float(cn_ratio(text)) if text else 0.0


def udf_max_char_share(text):
    return float(max_char_share(text)) if text else 1.0


def main():
    # 重定向到文件时（python x.py > run.log），stdout 默认是**块缓冲**的：
    # print 的内容不会实时落盘，要等进程结束或缓冲区满 —— 于是你完全看不到进度，
    # 一个跑一小时的管道瞎跑到最后才知道错了。改成行缓冲，tail 就能监工。
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    here = Path(__file__).resolve().parent.parent
    # Windows 上默认不能用 local[*]：核多 = 同时拉起一堆 Python worker，
    # 每个 worker 都要把整批（几十 KB × N 行）数据反序列化到自己的内存里，
    # 会随机崩掉一个 worker，症状是
    #   java.net.SocketException: Connection reset by peer: socket write error
    # 且**看不到任何 Python 异常**（worker 是被硬杀的）。
    # 本机实测：local[*] 必崩，local[4] 稳。
    default_master = "local[4]" if os.name == "nt" else "local[*]"
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(here / "data/raw/wiki.jsonl"))
    ap.add_argument("--output", default=str(here / "data/processed/chunks.parquet"))
    ap.add_argument("--master", default=default_master,
                    help="local[4] 单机（Windows 别用 local[*]，见上）；集群填 spark://192.168.192.129:7077")
    ap.add_argument("--driver-memory", default="4g")
    ap.add_argument("--tmp-dir", default="",
                    help="Spark 磁盘溢写目录。默认指向 C 盘 Temp，会吃紧；建议指到 E 盘")
    ap.add_argument("--limit-docs", type=int, default=0, help="只取前 N 篇，0=全部")
    ap.add_argument("--min-chunk-chars", type=int, default=50)
    ap.add_argument("--min-cn-ratio", type=float, default=0.30)
    ap.add_argument("--max-char-share", type=float, default=0.40)
    args = ap.parse_args()

    if sys.version_info >= (3, 12):
        print("=" * 64)
        print("[错误] 这个 Python 版本跑不了 PySpark 3.5.8")
        print("=" * 64)
        print(f"当前版本：{sys.version.split()[0]}")
        print()
        print("原因：Python 3.12+ 改了 socket 内部行为，PySpark 3.5 用")
        print("BufferedRWPair 包装 socket 跟 worker 通信时会崩，症状是")
        print("Spark 能启动、能建 DataFrame，但一执行 count() 就报")
        print("  OSError: [WinError 10038] 在一个非套接字上尝试了一个操作")
        print("  SparkException: Python worker exited unexpectedly (crashed)")
        print()
        print("解法：用 Python 3.11 跑这个脚本：")
        print(r'   "C:\Users\搞懂\AppData\Local\Programs\Python\Python311\python.exe" src\build_pipeline.py')
        print("=" * 64)
        return 1

    if not JAVA_HOME:
        print("[错误] 找不到 Java。装 JDK 或把 JAVA_HOME 设对再跑。")
        return 1
    if os.name == "nt" and not HADOOP_HOME:
        print("=" * 64)
        print("[错误] Windows 上找不到 winutils.exe")
        print("=" * 64)
        print("读 jsonl 不需要它，但写 Parquet 需要，所以前 6 步会全绿、第 7 步才炸。")
        print("修复：下载 Hadoop 3.3.x 的 winutils.exe + hadoop.dll，放到")
        print(f"    {here / 'hadoop' / 'bin'}")
        print("命令：")
        print("  curl -sL -o hadoop/bin/winutils.exe \\")
        print("    https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-3.3.5/bin/winutils.exe")
        print("  curl -sL -o hadoop/bin/hadoop.dll \\")
        print("    https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-3.3.5/bin/hadoop.dll")
        print("=" * 64)
        return 1
    src = Path(args.input)
    if not src.exists():
        print(f"[错误] 找不到输入文件：{src}")
        print("先跑：python src/wiki_extract.py --limit 5000")
        return 1

    # Spark 的磁盘溢写目录默认是 C 盘 Temp。跑大数据量时会写掉好几个 GB，
    # 本机 C 盘本来就紧 → 默认挪到项目目录下。
    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else (here / "data" / "_spark_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print("=" * 64)
    print("Spark 数据清洗管道")
    print(f"  Java    : {JAVA_HOME}")
    print(f"  Hadoop  : {HADOOP_HOME or '（非 Windows，不需要）'}")
    print(f"  Python  : {sys.executable}")
    print(f"  master  : {args.master}")
    print(f"  输入    : {src}")
    print(f"  输出    : {args.output}")
    print(f"  溢写目录: {tmp_dir}")
    print("=" * 64)

    spark = (
        SparkSession.builder
        .appName("wiki-clean-pipeline")
        .master(args.master)
        .config("spark.driver.memory", args.driver_memory)
        .config("spark.sql.shuffle.partitions", "8")     # 本地跑，别用默认 200
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")  # Windows 上更稳
        .config("spark.local.dir", str(tmp_dir))          # 溢写别写 C 盘
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # ---- 1+2. 读入 + 清洗（合并成"一次读取、一次缓存"）----
    #
    # 为什么合并：Spark 是惰性的，只有 count()/write() 这类 action 才触发计算。
    # 如果像原来那样先 df.cache() 缓存**原始**数据，等于把 9.4GB 的 jsonl 文本
    # 原封不动塞进内存/磁盘（本机实测：MemoryStore 存不下，全部 spill 到磁盘，
    # 白写白读 9.4GB）。
    #
    # 正确做法：文件只读一遍；清洗结果只缓存一份；用一次 agg 同时拿到
    # "原始条数"和"清洗后条数"。整条管道里 clean_wikitext 只执行 1 次。
    f_clean = F.udf(udf_clean, StringType())
    df = (spark.read.schema(JSON_SCHEMA).json(str(src))
          .withColumn("clean_text", f_clean(F.col("text")))
          .drop("text"))
    if args.limit_docs:
        df = df.limit(args.limit_docs)

    # 保留原始条数：先用一列标记"是否通过长度过滤"，过滤前先缓存
    df = df.withColumn("_keep", (F.length("clean_text") >= 200).cast("int"))
    df.cache()
    row = df.agg(F.count(F.lit(1)).alias("raw"),
                 F.sum("_keep").alias("keep")).first()
    n_docs_raw, n_docs_clean = int(row["raw"]), int(row["keep"] or 0)

    print(f"\n[1/7] 读入文档            : {n_docs_raw:,}")
    if n_docs_raw == 0:
        print("[致命错误] 读入 0 条 —— 输入文件为空或 schema 不匹配，停止。")
        return 1
    print(f"[2/7] 清洗后（长度>=200） : {n_docs_clean:,}  (剔掉 {n_docs_raw - n_docs_clean:,})")

    df = df.filter(F.col("_keep") == 1).drop("_keep")

    # ---- 3. 文档级精确去重 ----
    df = df.withColumn("doc_hash", F.md5(F.col("clean_text")))
    df = df.dropDuplicates(["doc_hash"])
    n_docs_dedup = df.count()
    print(f"[3/7] 文档去重后          : {n_docs_dedup:,}  "
          f"(重复率 {(1 - n_docs_dedup / max(n_docs_clean, 1)) * 100:.1f}%)")

    # ---- 4. 切块 ----
    f_chunk = F.udf(udf_chunk, ArrayType(StringType()))
    df = df.withColumn("chunks", f_chunk(F.col("clean_text")))
    df = df.withColumn("chunk", F.explode("chunks")).drop("chunks")
    df = df.withColumn("section", F.split(F.col("chunk"), "\u0001")[0])
    df = df.withColumn("chunk_text", F.split(F.col("chunk"), "\u0001")[1]).drop("chunk")
    n_chunks = df.count()
    print(f"[4/7] 切块后              : {n_chunks:,} chunks "
          f"(平均 {n_chunks / max(n_docs_dedup, 1):.1f} 块/篇)")

    # ---- 5. 质量过滤 ----
    f_cn = F.udf(udf_cn_ratio, "double")
    f_rep = F.udf(udf_max_char_share, "double")
    df = (df
          .withColumn("cn_ratio", f_cn(F.col("chunk_text")))
          .withColumn("char_share", f_rep(F.col("chunk_text")))
          .withColumn("char_len", F.length("chunk_text")))

    before = df.count()
    df = df.filter(
        (F.col("char_len") >= args.min_chunk_chars)
        & (F.col("cn_ratio") >= args.min_cn_ratio)
        & (F.col("char_share") <= args.max_char_share)
    )
    n_after = df.count()
    print(f"[5/7] 质量过滤后          : {n_after:,}  (剔掉 {before - n_after:,}, "
          f"{(1 - n_after / max(before, 1)) * 100:.1f}%)")

    # ---- 6. chunk 级去重 + 加 ID ----
    df = df.withColumn("chunk_hash", F.md5(F.col("chunk_text")))
    df = df.dropDuplicates(["chunk_hash"])
    n_final = df.count()
    print(f"[6/7] chunk 去重后        : {n_final:,}  (剔掉 {n_after - n_final:,})")

    df = df.withColumn("chunk_id", F.col("chunk_hash"))
    out_cols = ["chunk_id", "doc_id", "title", "section", "chunk_text",
                "char_len", "cn_ratio", "doc_hash", "source"]
    if "source" not in df.columns:
        df = df.withColumn("source", F.lit("zhwiki"))

    out = df.select(*out_cols)

    # ---- 7. 落盘 ----
    out_path = Path(args.output)
    print(f"[7/7] 写出 Parquet → {out_path}")
    out.repartition(4).write.mode("overwrite").parquet(str(out_path))

    # ---- 统计 ----
    elapsed = time.time() - t_start
    stats = {
        "raw_docs": n_docs_raw,
        "clean_docs": n_docs_clean,
        "dedup_docs": n_docs_dedup,
        "chunks_before_filter": before,
        "chunks_final": n_final,
        "avg_chunks_per_doc": round(n_final / max(n_docs_dedup, 1), 2),
        "doc_dedup_rate": round(1 - n_docs_dedup / max(n_docs_clean, 1), 4),
        "chunk_filter_rate": round(1 - n_final / max(before, 1), 4),
        "elapsed_sec": round(elapsed, 1),
        "throughput_docs_per_sec": round(n_docs_raw / max(elapsed, 1e-6), 1),
    }

    stats_dir = here / "eval" / "results"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_file = stats_dir / f"m1_stats_{time.strftime('%Y%m%d_%H%M%S')}.json"
    import json
    stats_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 64)
    print("统计（这三个数写进简历）")
    for k, v in stats.items():
        print(f"  {k:26s}: {v}")
    print(f"  stats 已存 → {stats_file}")
    print("=" * 64)

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
