"""从维基百科 dump（bz2 压缩的 XML）流式提取条目 → JSONL。

两种解析后端：
  --engine line （默认，推荐）逐行状态机解析，不经过 XML 解析器
  --engine xml  iterparse 解析（保留用于交叉校验）

────────────────────────────────────────────────────────────────────
🔥 为什么默认不用 iterparse（2026-09-16 实测，血泪教训）
────────────────────────────────────────────────────────────────────
用 `ET.iterparse` 跑全量时，每次都在**完全相同**的位置崩：

    xml.etree.ElementTree.ParseError:
    out of memory: line 28421794, column 44
    （328,850 页 / 已读 1.35 G字符 / 只产出 153,123 条，预期 79 万）

排查过程（每一步都是实测，不是猜）：
  1. 进程 RSS 全程只有 25 → 60 MB，机器有 18GB 空闲 → **不是真的内存不足**
  2. 崩溃点所在内容（第 2700 万~3000 万行）单独切出来跑 → **完全正常**
  3. 把切片重复 8 份（1.46GB）→ 也不崩
  4. 本机 `expat_2.8.2`，暴露了 `SetAllocTrackerActivationThreshold` /
     `SetAllocTrackerMaximumAmplification` —— 这是 Expat 2.6+ 新增的
     **「动态内存异常放大保护」**，默认激活阈值 64 MiB、最大放大倍数 100.0。
     它的实现方式就是**让分配返回 NULL**，于是对外报 `XML_ERROR_NO_MEMORY`
     （"out of memory"），而进程其实没吃多少内存 —— 症状完全吻合。
  5. `ET.XMLParser` 的内部 expat 对象在 C 层，Python 侧**拿不到**（无 `.parser`
     属性），所以没法把限制放开。

结论：**换成逐行状态机解析，彻底不经过 expat。** 副作用是好的 ——
媒体维基 dump 的排版严格稳定（`<page>` / `<title>` / `<ns>` / `<id>` /
`<text bytes=...>`），行解析既有速度又能顺带拿 `bytes` 属性做完整性校验。

────────────────────────────────────────────────────────────────────
⚠️ 另修一个更危险的 bug
────────────────────────────────────────────────────────────────────
旧版把**任何** `ET.ParseError` 都当成"文件被截断，属正常现象"打印出来，
然后**照常报成功**。于是 79 万条里丢了 63 万条，脚本还说"完成"。
**绿色的谎言比红色报错危险得多。** 现在：
  * 只有真正的流末尾（EOFError）才算正常结束
  * 其他任何异常 → 打印 `[致命错误]` + 退出码 1，绝不报成功

────────────────────────────────────────────────────────────────────
排版事实（实测，写解析器前必须确认过）
────────────────────────────────────────────────────────────────────
      <page>                                    ← 缩进 2
        <title>浙江省</title>                     ← 缩进 4
        <ns>0</ns>
        <id>12345</id>
        <revision>
          <id>82009188</id>
          ...
          <text bytes="1081" sha1="..." xml:space="preserve">{{存档页}}   ← 正文紧跟在开标签后
    &lt;ul&gt;&lt;li&gt;...                        ← 正文行无缩进，< 全被转义成 &lt;
    [[Category:...]]</text>                     ← ⚠️ 结束标签接在正文最后一行末尾，不单独成行
          <sha1>...</sha1>
        </revision>
      </page>

用法：
    # 正式跑（全量）
    python src/wiki_extract.py

    # 小样本试跑（先跑通再放开）
    python src/wiki_extract.py --limit 5000

    # 交叉校验：两种引擎跑同一段，逐条比对（证明解析器没写错）
    python src/wiki_extract.py --engine xml --limit 20000 --output data/raw/_x.xml.jsonl
"""
import argparse
import bz2
import html
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

# ── 不想要的命名空间（0 = 正文条目）────────────────────────────────
SKIP_TITLE_PREFIX = (
    "Wikipedia:", "Template:", "Category:", "File:", "Help:", "Portal:",
    "MediaWiki:", "Module:", "Draft:", "Talk:", "User:", "User talk:",
    "Wikipedia talk:", "Template talk:", "Category talk:", "File talk:",
    "模块:", "模板:", "分类:", "文件:", "帮助:", "维基百科:", "用户:",
)

MIN_TEXT_LEN = 200          # 正文短于这个长度视为残页，丢弃

RE_TITLE = re.compile(r"<title>(.*?)</title>", re.S)
RE_NS = re.compile(r"<ns>(.*?)</ns>", re.S)
RE_ID = re.compile(r"<id>(\d+)</id>")
RE_BYTES = re.compile(r'\bbytes="(\d+)"')
REDIRECT_RE = re.compile(r"^\s*#(REDIRECT|重定向|重新導向)\s*\[\[", re.I)


class Stats:
    def __init__(self):
        self.page = 0            # 读到的 <page> 总数
        self.accepted = 0        # 采纳
        self.skip_ns = 0         # 非正文命名空间
        self.skip_empty = 0      # 无标题或空正文
        self.skip_prefix = 0     # 标题前缀命中
        self.skip_redirect = 0   # 重定向
        self.skip_short = 0      # 正文太短
        self.bad_bytes = 0       # 声明的 bytes 与解出字节数不一致
        self.byte_checked = 0    # 参与 bytes 校验的条数
        self.replacement = 0     # 正文里出现 U+FFFD（解码替换符）
        self.chars = 0           # 读入的总字符数
        self.text_chars = 0      # 采纳条目的正文字符数
        self.max_text = 0
        self.max_title = ""

    def brief(self):
        return (f"page={self.page:,} 采纳={self.accepted:,} "
                f"非正文={self.skip_ns:,} 空={self.skip_empty:,} "
                f"前缀={self.skip_prefix:,} 重定向={self.skip_redirect:,} "
                f"过短={self.skip_short:,}")


# ══════════════════════════════════════════════════════════════════
# 引擎一：逐行状态机（默认）
# ══════════════════════════════════════════════════════════════════
def iter_pages_line(path: Path, st: Stats, limit: int = 0):
    """逐行解析。不经过任何 XML 解析器 —— 因此不受 expat 反滥用限流影响。"""
    in_page = False
    in_text = False
    title = ""
    ns = ""
    page_id = ""
    declared = None
    buf = []

    with bz2.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            st.chars += len(line)

            # ── 正文内部：只找 </text>，其余原样收集 ──
            if in_text:
                idx = line.find("</text>")
                if idx >= 0:
                    buf.append(line[:idx])
                    in_text = False
                else:
                    buf.append(line)
                continue

            s = line.lstrip()

            if not in_page:
                if s.startswith("<page>"):
                    in_page = True
                    st.page += 1
                    title = ns = page_id = ""
                    declared = None
                    buf = []
                continue

            # ── page 内部 ──
            if s.startswith("<title>"):
                m = RE_TITLE.search(line)
                title = html.unescape(m.group(1)).strip() if m else ""
            elif s.startswith("<ns>"):
                m = RE_NS.search(line)
                ns = m.group(1).strip() if m else ""
            elif s.startswith("<id>") and not page_id:
                m = RE_ID.search(line)
                page_id = m.group(1) if m else ""
            elif s.startswith("<text"):
                if line.rstrip().endswith("/>"):
                    pass                                  # 自闭合 = 空正文
                else:
                    gt = line.find(">")
                    m = RE_BYTES.search(line[:gt])
                    declared = int(m.group(1)) if m else None
                    rest = line[gt + 1:]
                    idx = rest.find("</text>")
                    if idx >= 0:
                        buf.append(rest[:idx])            # 单行内闭合
                    else:
                        buf.append(rest)
                        in_text = True
            elif s.startswith("</page>"):
                in_page = False

                # ── 过滤 ──
                if ns != "0":
                    st.skip_ns += 1
                    continue
                if not title or not buf:
                    st.skip_empty += 1
                    continue
                if title.startswith(SKIP_TITLE_PREFIX):
                    st.skip_prefix += 1
                    continue

                text = html.unescape("".join(buf))
                if REDIRECT_RE.match(text):
                    st.skip_redirect += 1
                    continue
                if len(text) < MIN_TEXT_LEN:
                    st.skip_short += 1
                    continue

                # ── 完整性自检：用 <text bytes="N"> 核对 ──
                if declared is not None:
                    st.byte_checked += 1
                    if len(text.encode("utf-8")) != declared:
                        st.bad_bytes += 1

                if "\ufffd" in text:
                    st.replacement += 1

                st.accepted += 1
                st.text_chars += len(text)
                if len(text) > st.max_text:
                    st.max_text = len(text)
                    st.max_title = title

                yield {"doc_id": page_id, "title": title, "text": text}

                if limit and st.accepted >= limit:
                    return


# ══════════════════════════════════════════════════════════════════
# 引擎二：iterparse（保留做交叉校验；⚠️ 全量会触发 expat 限制）
# ══════════════════════════════════════════════════════════════════
def _local(tag):
    if isinstance(tag, str) and tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def iter_pages_xml(path: Path, st: Stats, limit: int = 0):
    with bz2.open(path, "rt", encoding="utf-8", errors="replace") as f:
        ctx = ET.iterparse(f, events=("end",))
        root = None
        for _ev, elem in ctx:
            if _local(elem.tag) != "page":
                continue
            st.page += 1
            if root is None:
                root = getattr(ctx, "root", None)

            title = ns = rev_text = page_id = ""
            for child in elem:
                name = _local(child.tag)
                if name == "title":
                    title = child.text or ""
                elif name == "ns":
                    ns = (child.text or "").strip()
                elif name == "id" and not page_id:
                    page_id = (child.text or "").strip()
                elif name == "revision":
                    for sub in child:
                        if _local(sub.tag) == "text":
                            rev_text = sub.text or ""
                            break
            elem.clear()
            if root is not None:
                root.clear()

            if ns != "0":
                st.skip_ns += 1
                continue
            if not title or not rev_text:
                st.skip_empty += 1
                continue
            if title.startswith(SKIP_TITLE_PREFIX):
                st.skip_prefix += 1
                continue
            if REDIRECT_RE.match(rev_text):
                st.skip_redirect += 1
                continue
            if len(rev_text) < MIN_TEXT_LEN:
                st.skip_short += 1
                continue

            st.accepted += 1
            st.text_chars += len(rev_text)
            yield {"doc_id": page_id, "title": title, "text": rev_text}

            if limit and st.accepted >= limit:
                return


# ══════════════════════════════════════════════════════════════════
def main():
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",
                    default="E:/AI-learning/data/zhwiki-latest-pages-articles.xml.bz2",
                    help="维基 dump 的 bz2 路径（保留官方原名，方便溯源到具体 dump）")
    ap.add_argument("--output", default=str(here / "data/raw/wiki.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="最多提取多少条，0 = 不限")
    ap.add_argument("--engine", choices=["line", "xml"], default="line",
                    help="line=逐行状态机（默认，推荐）；xml=iterparse（仅小样本交叉校验用）")
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        print(f"[错误] 找不到输入文件：{src}")
        print("先下载：见 操作手册.md 第 3 步")
        return 1

    if args.engine == "xml" and not args.limit:
        print("[警告] xml 引擎在全量数据上会触发 expat 的反滥用保护并报 "
              "'out of memory'（见本文件顶部说明）。")
        print("        全量请用 --engine line；xml 只用于小样本交叉校验。")
        print("        5 秒后继续，Ctrl+C 取消 ...")
        time.sleep(5)

    size_mb = src.stat().st_size / 1024 ** 2
    print(f"输入：{src}  ({size_mb:,.1f} MB)")
    print(f"输出：{dst}")
    print(f"引擎：{args.engine}    上限：{args.limit if args.limit else '不限'}")
    print("-" * 68)

    st = Stats()
    t0 = time.time()
    engine = iter_pages_line if args.engine == "line" else iter_pages_xml
    last_report = 0

    try:
        with dst.open("w", encoding="utf-8") as out:
            for rec in engine(src, st, args.limit):
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if st.accepted - last_report >= 20000:
                    last_report = st.accepted
                    el = time.time() - t0
                    print(f"  采纳 {st.accepted:,} 条 | page {st.page:,} | "
                          f"{st.chars / 1024**2 / max(el, 1e-6):.0f} M字符/s", end="\r")
    except EOFError:
        print(f"\n[提示] bz2 流到末尾正常结束。")
        print("       若你下的是**截断文件**，这是预期行为；")
        print("       若用的是完整 dump，请核对文件大小是否等于远端的 Content-Length。")
    except Exception as e:
        # 🔥 关键修复：任何真实错误都不许伪装成"正常截断"
        print(f"\n[致命错误] {type(e).__name__}: {e}")
        print(f"  已处理：{st.brief()}")
        print("  输出文件**不完整，不要使用**。")
        import traceback
        traceback.print_exc()
        return 1

    dt = time.time() - t0
    print(f"\n完成：{st.accepted:,} 条 → {dst}")
    print(f"耗时：{dt:.1f} 秒（{st.accepted / max(dt, 1e-6):.0f} 条/秒，"
          f"{st.chars / 1024**2 / max(dt, 1e-6):.1f} M字符/秒）")
    if dst.exists():
        print(f"文件大小：{dst.stat().st_size / 1024 ** 2:,.1f} MB")

    print("\n── 过滤明细 ──")
    print(f"  读入 <page>        {st.page:>10,}")
    print(f"  采纳（正文条目）    {st.accepted:>10,}")
    print(f"    ├ 非正文命名空间  {st.skip_ns:>10,}")
    print(f"    ├ 空标题/空正文   {st.skip_empty:>10,}")
    print(f"    ├ 标题前缀命中    {st.skip_prefix:>10,}")
    print(f"    ├ 重定向          {st.skip_redirect:>10,}")
    print(f"    └ 正文过短(<{MIN_TEXT_LEN}) {st.skip_short:>6,}")

    print("\n── 数据自检 ──")
    print(f"  正文总字符         {st.text_chars / 1e8:>10.3f} 亿")
    print(f"  平均每条           {st.text_chars / max(st.accepted, 1):>10,.0f} 字符")
    print(f"  最长条目           {st.max_text:>10,} 字符  {st.max_title[:30]}")
    if st.byte_checked:
        ok = (st.byte_checked - st.bad_bytes) / st.byte_checked * 100
        print(f"  bytes 属性核对      {ok:>9.2f}% 一致（{st.byte_checked - st.bad_bytes:,}"
              f"/{st.byte_checked:,}）")
        if st.bad_bytes:
            print(f"    ⚠️ {st.bad_bytes:,} 条不一致 —— 解析可能漏了行，值得查")
    print(f"  含替换符 U+FFFD     {st.replacement:>10,} 条（应为 0，非 0 说明源文件有坏字节）")

    print("\n下一步：抽查几条")
    print(f'  head -c 1200 "{dst}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
