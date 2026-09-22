#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
云端模型的常量与错误翻译（**零依赖：只用标准库**）。

为什么单独一个文件，而不是留在 generator.py 里：
    generator.py 顶层 `from search_hybrid import ...`，那会拖进 faiss。
    于是任何"只想调一次 API"的工具（`check_backends.py` 的模型体检、
    `judge_faith.py` 的判官）只要 import generator，就得先付出加载
    faiss 的代价 —— 而这类工具的卖点恰恰是「零成本、几秒出结果」。

    拆出来之后，这一层只依赖标准库：import 它就是瞬间的事。
    这是"按依赖边界切模块"的一个具体例子 ——
    判断标准不是"代码长不长"，而是**谁需要为谁的依赖买单**。
"""

import os

# dashscope（阿里百炼）的 OpenAI 兼容端点。
# 用兼容端点而不是 dashscope 私有 SDK：换任何一家（DeepSeek / 智谱 / vLLM 本地服务）
# 业务代码都只改这一个 base_url。
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ============================================================ 云端模型清单
#
# ⚠️ 这两项是**单一来源**：命令行、界面下拉、体检脚本、判官都从这里取。
#    原因见操作手册 BM 坑 —— 同一个东西在两处定义，就一定会漂移，
#    而漂移的暴露时机会被「加载 13.6 GB 索引」放大成一次演示事故（2026-09-22）。
#
# 【为什么默认不是 qwen-plus】
#   2026-09-22 用户实测：qwen-plus 免费额度耗尽，接口直接 403
#   `AllocationQuota.FreeTierOnly`。此前评测（§11.14）用的正是 qwen-plus ——
#   那次是**跑在额度还没用完的时候**，结论本身没问题，
#   但默认值不能停在"已经不能用的模型"上。
#   ⇒ 默认换成 qwen-turbo：免费额度可用、速度快、且在 §11.14 里当过判官、已过 16/16 校准。
#   要恢复 qwen-plus：去阿里百炼控制台充值，或关掉「仅使用免费额度」模式。
DEFAULT_CLOUD_MODEL = "qwen-turbo"
CLOUD_MODELS = ["qwen-turbo", "qwen-plus", "qwen-max", "qwen3-max", "qwen-flash"]

# 重试一万次也还是这个结果 → 应当**立即中止整批**，别白跑 129 条。
# 反例：ratelimit / timeout / conn 是瞬时的，SDK 已退避重试过，继续下一条是对的。
FATAL_API_ERROR_KINDS = {"quota", "auth", "model"}

_API_ERROR_HINTS = {
    "quota": ("云端模型的**免费额度已用尽**（阿里百炼 FreeTierOnly）",
              "换一个还有额度的模型（界面「生成模型」下拉里选），"
              "或去百炼控制台充值 / 关掉「仅使用免费额度」模式。"
              "命令行跑 `src\\check_backends.py` 可以列出现在哪些模型能用。"),
    "auth": ("API Key 无效或未设置",
             "检查环境变量 DASHSCOPE_API_KEY；改完要**重开终端/黑窗口**才生效。"),
    "ratelimit": ("请求被限流（429）",
                  "等几秒再试；持续出现就换模型（qwen-turbo 更快更宽松）。"),
    "timeout": ("请求超时", "网络或服务端慢：重试一次，或换成更快的 qwen-turbo。"),
    "conn": ("连不上 dashscope 端点",
             "检查网络/代理；`src\\check_backends.py` 能快速验证连通性。"),
    "model": ("模型名不存在，或该账号无权访问这个模型",
              "确认模型名拼写；用 `src\\check_backends.py` 看哪些模型可用。"),
}


def api_error_kind(e: BaseException) -> str:
    """
    把异常归类成一个短标签。**判定逻辑只写在这一处** ——
    classify_api_error()（给人看的原因）和 is_fatal_api_error()（要不要立即中止）
    都从它取。这正是 2026-09-22 那次事故的教训：同一条判定写两遍，
    两处就一定会不一致，而且不一致的时候没人会发现。
    """
    msg = str(e)
    low = msg.lower()
    name = type(e).__name__

    if "freetieronly" in low or "free quota" in low:
        return "quota"
    if "401" in msg or "invalid api key" in low or "authentication" in low:
        return "auth"
    if "429" in msg or "rate limit" in low or "throttl" in low:
        return "ratelimit"
    if "timeout" in low or "timed out" in low or name in ("APITimeoutError", "TimeoutError"):
        return "timeout"
    if "connection" in low or name == "APIConnectionError":
        return "conn"
    if "model not exist" in low or "invalidparameter" in low or "model_not_found" in low:
        return "model"
    return "unknown"


def is_fatal_api_error(e: BaseException) -> bool:
    """这个错误重试也没用吗？（额度 / 鉴权 / 模型名）"""
    return api_error_kind(e) in FATAL_API_ERROR_KINDS


def classify_api_error(e: BaseException) -> tuple[str, str]:
    """
    把云端 API 的异常翻译成「一句话原因 + 一条能照着做的动作」。

    为什么必须做这件事（2026-09-22 用户截图就是这么来的）：
      SDK 抛出的异常里裹着**服务端返回的原始 JSON**（截图上是满满一大坨
      `{'error': {'message': 'Free quota exhausted...', 'type': 'AllocationQuota.FreeTierOnly'...}}`）。
      它对开发者有用，对使用者是纯噪音 —— 用户只看到"坏了"，不知道
      "是免费额度用完了，换个模型或去充值就行"。

    ⚠️ 原则：**错误提示里必须给出下一步动作**，否则等于没提示。
       "生成失败"这四个字的信息量是零。

    返回 (原因, 怎么办)；未知异常走兜底，保留截断后的原文（排查未知错误要用）。
    """
    kind = api_error_kind(e)
    if kind in _API_ERROR_HINTS:
        return _API_ERROR_HINTS[kind]
    # 兜底：先说人话，再附截断的原文 —— 未知错误没有原文就没法查
    return (f"调用云端模型失败（{type(e).__name__}）",
            str(e).strip().replace("\n", " ")[:300])


def api_key_or_exit() -> str:
    """取 DASHSCOPE_API_KEY；没设就带着可执行提示退出。"""
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise SystemExit(
            "[错误] 环境变量 DASHSCOPE_API_KEY 未设置。设置后要**重开终端/黑窗口**才生效。")
    return key
