#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
云端模型体检（零成本：**不加载索引**，几秒出结果）。

为什么需要它（2026-09-22）：
    用户在界面上问问题，收到一整屏 403 原始 JSON —— 免费额度用尽。
    但「哪个模型还能用」这件事，**当时没有任何地方能一眼看到**：
    只能一个个手试，或者去控制台翻账单。这个脚本把这件事变成一条命令。

它做三件事：
    ① 列出待测模型（清单来自 generator.CLOUD_MODELS，**单一来源**）
    ② 各发一个 max_tokens=1 的请求，报告可用性 + 往返延迟
    ③ 给出结论：推荐哪个、失败的具体原因、下一步怎么做

⚠️ 设计取舍（值得记的一条）：**它只回答「能不能用」，不回答「哪个答得好」。**
   后者要靠评测集（见 §11.14）—— 拿一个 "hi" 的回复质量去排模型是自欺欺人。
   探测器的价值在于**便宜、快、可信**，所以每个模型只花 1 个输出 token。

用法：
    python src\\check_backends.py                    # 测内置清单
    python src\\check_backends.py --models qwen-turbo,qwen-max
    python src\\check_backends.py --selftest          # 只测本地分类逻辑，**不发网络请求**
"""

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

# ⚠️ 从 **cloud_models** 取，不是从 generator —— generator 会拖进 faiss，
#    那这个"几秒出结果"的体检工具就名不副实了（见 cloud_models 的文档字符串）。
from cloud_models import (  # noqa: E402
    BASE_URL,
    CLOUD_MODELS,
    DEFAULT_CLOUD_MODEL,
    api_error_kind,
    classify_api_error,
    is_fatal_api_error,
)


# ------------------------------------------------------------------ 排版小工具
def _w(s: str) -> int:
    """按终端显示宽度算长度：CJK 与全角标点占 2 列。"""
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def _pad(s: str, n: int) -> str:
    return s + " " * max(0, n - _w(s))


def _plain(s: str) -> str:
    """剥掉给界面看的 markdown 标记 —— 终端里 `**` 和反引号只是噪音。"""
    return s.replace("**", "").replace("`", "")


# ------------------------------------------------------------------ 探测
def probe(model: str, timeout: float = 30.0, max_tokens: int = 1) -> dict:
    """
    给单个模型发一个最小请求，返回 {"ok", "ms", "reason", "hint", "kind"}。

    两个刻意的选择：
      ① `max_tokens=1` —— 探测器只关心"通不通"，输出越少越便宜越快。
      ② `max_retries=0` —— **与业务代码相反**。业务里让 SDK 自动重试（见 CloudBackend），
         但探测器必须**如实反映此刻能不能用**：重试会把失败包住、还会把等待拖长，
         于是"体检报告"和用户实际感受不一致（报告说好、用户点下去还是报错）。
         探测器的职责是**快速失败并说清原因**，不是替服务端掩盖抖动。
    """
    from openai import OpenAI

    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        return {"ok": False, "ms": None, "kind": "auth",
                "reason": "环境变量 DASHSCOPE_API_KEY 未设置",
                "hint": "设置后要重开终端才生效"}

    client = OpenAI(api_key=key, base_url=BASE_URL, timeout=timeout, max_retries=0)
    t = time.time()
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=max_tokens,
        )
    except Exception as e:
        reason, hint = classify_api_error(e)
        return {"ok": False, "ms": (time.time() - t) * 1000,
                "kind": api_error_kind(e), "reason": _plain(reason), "hint": _plain(hint)}
    return {"ok": True, "ms": (time.time() - t) * 1000,
            "kind": "ok", "reason": "可用", "hint": ""}


def report(models: list[str], results: list[dict]) -> int:
    """打印表格 + 结论，返回退出码（有可用模型 = 0，全挂 = 1）。"""
    print()
    print(f"  {_pad('模型', 18)}{_pad('结果', 12)}{_pad('往返', 10)}说明")
    print("  " + "-" * 72)
    for m, r in zip(models, results):
        mark = "✅ 可用" if r["ok"] else "❌ " + _plain(_SHORT.get(r["kind"], "失败"))
        ms = f"{r['ms']:.0f} ms" if r["ms"] is not None else "—"
        note = "" if r["ok"] else r["reason"]
        print(f"  {_pad(m, 18)}{_pad(mark, 12)}{_pad(ms, 10)}{note}")

    ok = [m for m, r in zip(models, results) if r["ok"]]
    bad = [(m, r) for m, r in zip(models, results) if not r["ok"]]

    print()
    print("=" * 78)
    print("【结论】")
    print("=" * 78)
    if ok:
        # 推荐规则：默认模型可用就推荐它（因为评测/文档都以它为准），否则推荐最快的
        if DEFAULT_CLOUD_MODEL in ok:
            rec = DEFAULT_CLOUD_MODEL
            why = "默认值"
        else:
            times = {m: r["ms"] for m, r in zip(models, results) if r["ok"] and r["ms"]}
            rec = min(times, key=times.get) if times else ok[0]
            why = "可用里最快"
        print(f"  ✅ {len(ok)}/{len(models)} 个可用。建议用 {rec}（{why}）："
              f"`--cloud-model {rec}`")
        print("     界面：下拉「生成模型」里直接选。")
    else:
        print("  ❌ 一个都不能用。先看下面的原因，通常不是代码问题。")

    if bad:
        print()
        print(f"  ❌ 不可用的 {len(bad)} 个：")
        for m, r in bad:
            print(f"     · {m}：{r['reason']}")
            if r["hint"]:
                print(f"       → {r['hint']}")
        # 按错误类型归纳一句可执行的总结，而不是让用户自己从 5 行里找规律
        kinds = {r["kind"] for _, r in bad}
        if kinds == {"quota"}:
            print()
            print("  不可用的都是「免费额度用尽」—— 代码没坏。三条路：")
            print("    ① 换成上面可用的模型（最快，零成本）")
            print("    ② 去阿里百炼控制台充值，或关掉「仅使用免费额度」模式")
            print("    ③ 用本地模型（界面下拉选「本地 Qwen2.5-1.5B」，不联网、不花钱）")
    print()
    print("  ⚠️ 本次体检只消耗约 %d 个输出 token（每个模型 1 个），成本可忽略。" % len(models))
    return 0 if ok else 1


_SHORT = {"quota": "额度用尽", "auth": "鉴权失败", "ratelimit": "被限流",
          "timeout": "超时", "conn": "连不上", "model": "模型不可用", "unknown": "失败"}


# ------------------------------------------------------------------ 自检
class _FakeError(Exception):
    pass


# (报错原文, 期望归类, 期望"是否致命")
SELFTEST_CASES = [
    ("Error code: 403 - {'error': {'message': 'Free quota exhausted. To continue "
     "accessing the model on a paid basis...', 'type': 'AllocationQuota.FreeTierOnly', "
     "'code': 'AllocationQuota.FreeTierOnly'}}", "quota", True),
    ("Error code: 401 - {'error': {'message': 'Invalid API-key provided.'}}", "auth", True),
    ("Error code: 429 - Request rate increased too quickly.", "ratelimit", False),
    ("Request timed out.", "timeout", False),
    ("Error code: 400 - Model not exist.", "model", True),
    ("Some brand-new failure nobody has seen before", "unknown", False),
]


def selftest() -> int:
    """
    不发网络请求，只验证「错误归类」这张表本身是对的。

    为什么这条自检值得单独存在：分类函数的输入是**别人服务端返回的字符串**，
    我们只能靠关键词猜。而猜错的后果是"给用户指错路"（把额度问题说成网络问题，
    让用户去查网络 —— 这是最坏的一种错误提示）。
    所以用**真实抓到的报错原文**做回归样本（第一条就是 2026-09-22 截图里那条），
    以后改关键词表时能立刻发现有没有把已有场景覆盖掉。
    """
    print("【自检】错误归类（不需要网络）")
    bad = 0
    for msg, want_kind, want_fatal in SELFTEST_CASES:
        e = _FakeError(msg)
        kind = api_error_kind(e)
        fatal = is_fatal_api_error(e)
        reason, hint = classify_api_error(e)
        ok = (kind == want_kind and fatal == want_fatal and reason and hint)
        if not ok:
            bad += 1
        print(f"  {'✅' if ok else '❌'} {kind:<10} fatal={str(fatal):<5} "
              f"{_plain(reason)[:44]}")
        if not ok:
            print(f"       期望 kind={want_kind} fatal={want_fatal}，"
                  f"实得 kind={kind} fatal={fatal}")
    print()
    print(f"  {len(SELFTEST_CASES) - bad}/{len(SELFTEST_CASES)} 通过"
          + ("（全绿）" if not bad else " ← 有回归，去改 cloud_models 的关键词表"))
    return 1 if bad else 0


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(
        description="云端模型体检：列出哪些 dashscope 模型现在能用（零成本，不加载索引）")
    ap.add_argument("--models", default="",
                    help=f"逗号分隔的模型名（默认用 generator.CLOUD_MODELS："
                         f"{','.join(CLOUD_MODELS)}）")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--selftest", action="store_true",
                    help="只验证错误归类逻辑，不发任何网络请求")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    models = ([m.strip() for m in args.models.split(",") if m.strip()]
              or list(CLOUD_MODELS))
    key = os.environ.get("DASHSCOPE_API_KEY")

    print("=" * 78)
    print("云端模型体检 · dashscope 兼容端点")
    print("=" * 78)
    print(f"  Key      : {'已设置（' + key[:6] + '***）' if key else '❌ 未设置'}"
          f"   ← 只打前 6 位，不泄全文")
    print(f"  待测     : {len(models)} 个")
    print(f"  探针     : prompt='hi'，max_tokens=1，max_retries=0（快速失败，不掩盖抖动）")
    print()
    print("  探测中 …")

    results = [probe(m, timeout=args.timeout) for m in models]
    return report(models, results)


if __name__ == "__main__":
    sys.exit(main())
