# -*- coding: utf-8 -*-
"""
清洗器第二轮 A/B —— 验证「缓释点号 + 语言变体跑两遍」是否真的降残留。

这一版对比的是 **改之前的清洗器** vs **build_pipeline 里现在的清洗器**，
所以结论直接等于上线效果，不是纸面推演。

两处改动（都在 2026-09-17 全量验收发现问题后做的）：
  1. 缓释点号 (tempered dot)：`[^\\[\\]]+` → `(?:(?!\\[\\[|\\]\\]).)+?`
     旧写法遇到"链接文字含单个方括号" 永久失配，加迭代次数救不了。
     注意：**表格 RE_TABLE 故意不缓释**，嵌套表格反而要 `.*?` 整块吃。
  2. 语言变体 -{...}- 跑两遍：第一遍在模板之前，第二遍在所有链接剥完之后。
     因为 -{zh-hant:[[File:X.svg|thumb|...]]; zh-hans:[[File:Y.svg|...]]}- 里的
     File 链接会挡住 `[^{}]*`，只有等链接剥完才退化成 -{zh-hant:;zh-hans:}-。

用法：
    python src/diag_clean2.py --n 20000
"""
import argparse
import json
import re
import sys
from pathlib import Path

here = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(here / "src"))
import build_pipeline as bp  # noqa: E402


# ---------------------------------------------------------------------------
# v1 = 改之前的清洗器（照抄旧代码，只用于对照，别再用）
# ---------------------------------------------------------------------------
V1_REF = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", re.S | re.I)
V1_TABLE = re.compile(r"\{\|.*?\|\}", re.S)
V1_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
V1_FILE = re.compile(r"\[\[(?:File|Image|文件|图像|圖片)\s*:[^\[\]]*\]\]", re.I)
V1_LINK_PIPE = re.compile(r"\[\[([^\[\]|]+)\|([^\[\]]+)\]\]")
V1_LINK_BARE = re.compile(r"\[\[([^\[\]]+)\]\]")
V1_LANGVAR = re.compile(r"-\{([^{}]*)\}-")
V1_EXTLINK = re.compile(r"\[https?://\S+\s*([^\]]*)\]")
V1_HTML = re.compile(r"<[^>]{1,200}>")
V1_QUOTE = re.compile(r"'''''|'''|''")
V1_MAGIC = re.compile(r"__[A-Z]+__")
V1_SPACES = re.compile(r"[ \t]{2,}")
V1_MULTI_NL = re.compile(r"\n{3,}")


def clean_v1(text: str) -> str:
    if not text:
        return ""
    t = bp.RE_COMMENT.sub("", text)
    t = V1_REF.sub("", t)
    t = V1_TABLE.sub("", t)
    t = V1_LANGVAR.sub(bp._pick_langvar, t)
    t = bp.RE_LANG_TPL.sub(r"\1", t)
    for _ in range(6):
        t, n = V1_TEMPLATE.subn("", t)
        if n == 0:
            break
    for _ in range(4):
        before = t
        t = V1_FILE.sub("", t)
        t = V1_LINK_PIPE.sub(r"\2", t)
        t = V1_LINK_BARE.sub(r"\1", t)
        if t == before:
            break
    t = V1_EXTLINK.sub(r"\1", t)
    t = V1_HTML.sub("", t)
    t = V1_QUOTE.sub("", t)
    t = V1_MAGIC.sub("", t)
    t = V1_SPACES.sub(" ", t)
    t = V1_MULTI_NL.sub("\n\n", t)
    return t.strip()


clean_v2 = bp.clean_wikitext

PATTERNS = ["[[", "{{", "<ref", "-{", "{|", "<div", "]]", "}}", "}-"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(here / "data/raw/wiki.jsonl"))
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--show", type=int, default=4)
    args = ap.parse_args()

    n = 0
    a1 = {k: 0 for k in PATTERNS}
    a2 = {k: 0 for k in PATTERNS}
    regress = []   # 新版残留、旧版干净的（回归！必须为零）
    fixed = []     # 旧版残留、新版干净的（修复样本）

    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if n >= args.n:
                break
            n += 1
            try:
                doc = json.loads(line)
            except Exception:
                continue
            raw = doc.get("text") or ""
            c1, c2 = clean_v1(raw), clean_v2(raw)
            for k in PATTERNS:
                r1, r2 = k in c1, k in c2
                a1[k] += r1
                a2[k] += r2
                if r1 and not r2 and len(fixed) < args.show:
                    i = c1.find(k)
                    fixed.append((k, doc.get("title", ""), c1[i - 40:i + 110]))
                if r2 and not r1 and len(regress) < args.show:
                    i = c2.find(k)
                    regress.append((k, doc.get("title", ""), c2[i - 40:i + 110]))

    print("=" * 74)
    print(f"清洗器 A/B（v1=改之前  v2=现在的 build_pipeline）  样本 {n:,} 篇")
    print("=" * 74)
    print(f"{'标记':<7}{'v1 残留':>20}{'v2 残留':>20}{'降幅':>12}")
    for k in PATTERNS:
        x, y = a1[k], a2[k]
        r = f"{x / y:.1f}x" if y else ("全部清零" if x else "-")
        flag = "  ⚠️回归" if y > x else ""
        print(f"{k:<7}{x:>13,} ({x / n * 100:5.2f}%){y:>13,} ({y / n * 100:5.2f}%){r:>12}{flag}")
    print("=" * 74)

    if regress:
        print("\n⚠️ 回归样例（v2 残留、v1 干净）—— 有的话说明改坏了：")
        for k, t, s in regress:
            print(f"  [{k}] {t}: ...{s}")

    if fixed:
        print("\n✅ 修复样例（v1 残留、v2 干净）：")
        for k, t, s in fixed:
            print(f"  [{k}] {t}: ...{s}")


if __name__ == "__main__":
    main()
