"""清洗规则 A/B 对比 + 文档淘汰原因分解。

为什么单独写这个脚本：
    全量管道要跑 12 分钟，"改一版正则就跑一遍全量"是最贵的调试方式。
    正确做法是**拿真实的中段样本先离线 A/B**，确认改进方向和幅度，再上全量。

用法：
    python src/diag_clean.py --n 20000
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pipeline import (  # noqa: E402
    clean_wikitext, split_chunks, cn_ratio, max_char_share,
    RE_COMMENT, RE_REF, RE_TABLE, RE_FILE, RE_LINK_PIPE, RE_LINK_BARE,
    RE_EXTLINK, RE_HTML, RE_QUOTE, RE_MAGIC, RE_SPACES, RE_MULTI_NL,
)

# 旧版 clean（保留在这里只为了做 A/B 对照，不要在管道里用它）
OLD_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")


def clean_wikitext_old(text: str) -> str:
    if not text:
        return ""
    t = RE_COMMENT.sub("", text)
    t = RE_REF.sub("", t)
    t = RE_TABLE.sub("", t)
    for _ in range(4):
        t, n = OLD_TEMPLATE.subn("", t)
        if n == 0:
            break
    t = RE_FILE.sub("", t)
    t = RE_LINK_PIPE.sub(r"\2", t)
    t = RE_LINK_BARE.sub(r"\1", t)
    t = RE_EXTLINK.sub(r"\1", t)
    t = RE_HTML.sub("", t)
    t = RE_QUOTE.sub("", t)
    t = RE_MAGIC.sub("", t)
    t = RE_SPACES.sub(" ", t)
    t = RE_MULTI_NL.sub("\n\n", t)
    return t.strip()


RE_MARKS = {
    "双开方括号 [[": re.compile(r"\[\["),
    "双开大括号 {{": re.compile(r"\{\{"),
    "<ref": re.compile(r"<ref", re.I),
    "表格 {|": re.compile(r"\{\|"),
    "-{ 语言变体": re.compile(r"-\{"),
}


def sample_middle(path: Path, n: int, seed: int = 42):
    """从文件中段取样本 —— 文件开头的条目偏大、偏完整，不具代表性。"""
    size = path.stat().st_size
    rows = []
    with path.open("r", encoding="utf-8") as f:
        f.seek(size // 2)
        f.readline()                     # 丢掉可能被截断的半行
        for _ in range(n):
            line = f.readline()
            if not line:
                break
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/wiki.jsonl")
    ap.add_argument("--n", type=int, default=20000)
    args = ap.parse_args()
    path = Path(args.input)

    rows = sample_middle(path, args.n)
    print(f"样本：{len(rows):,} 条（取自文件 50% 位置，比开头更有代表性）")
    if not rows:
        return 1

    stat = {k: Counter() for k in ("old", "new")}
    drop_reason = Counter()
    len_before, len_after = [], []
    kept_chunks = {"old": 0, "new": 0}

    for r in rows:
        text = r.get("text") or ""
        old, new = clean_wikitext_old(text), clean_wikitext(text)
        len_before.append(len(text))
        len_after.append(len(new))

        for tag, c in (("old", old), ("new", new)):
            for name, pat in RE_MARKS.items():
                if pat.search(c):
                    stat[tag][name] += 1
            if len(c) >= 200:
                kept_chunks[tag] += sum(1 for _ in split_chunks(c))

        # 淘汰原因（按新版算）
        if len(new) < 200:
            if len(text) < 400:
                drop_reason["原文本身就很短（<400 字符）"] += 1
            elif len(old) < 200:
                drop_reason["旧版也 <200（清洗前就没什么正文）"] += 1
            else:
                drop_reason["旧版够长但新版被削短"] += 1

    n = len(rows)
    print()
    print("=" * 70)
    print(f"{'指标':<24}{'旧版':>14}{'新版':>14}")
    print("-" * 70)
    for name in RE_MARKS:
        o, w = stat["old"][name], stat["new"][name]
        print(f"{name:<24}{o:>8} ({o/n*100:5.2f}%){w:>8} ({w/n*100:5.2f}%)")
    print("-" * 70)
    print(f"{'清洗后 <200 字符的条数':<24}{sum(1 for c in []):>0}", end="")
    print()
    n_drop_old = sum(1 for r in rows
                     if len(clean_wikitext_old(r.get("text") or "")) < 200)
    n_drop_new = sum(1 for r in rows if len(clean_wikitext(r.get("text") or "")) < 200)
    print(f"{'淘汰条数 (<200 字符)':<24}{n_drop_old:>8} ({n_drop_old/n*100:5.2f}%)"
          f"{n_drop_new:>8} ({n_drop_new/n*100:5.2f}%)")
    print(f"{'产出 chunk 数':<24}{kept_chunks['old']:>8}      {kept_chunks['new']:>8}")
    print(f"{'平均原文长度':<24}{sum(len_before)/n:>14.0f}")
    print(f"{'平均清洗后长度':<24}{sum(len_after)/n:>14.0f}")
    print("=" * 70)
    print("淘汰原因分解（按新版）：")
    for k, v in drop_reason.most_common():
        print(f"  {v:>7,}  ({v/n*100:5.2f}%)  {k}")

    print()
    print("样例（新版清洗后）:")
    for r in random.Random(7).sample(rows, 3):
        c = clean_wikitext(r.get("text") or "")
        print(f"  [{r.get('title')}] len={len(c)}")
        print(f"    {c[:140]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
