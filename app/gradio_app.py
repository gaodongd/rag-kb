#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第 9 步 · Gradio 演示界面（M3 的交付物）

这个文件只做**界面**，不重新实现任何检索/生成逻辑 —— 全部 import 自
`generator.py` 和 `search_hybrid.py`。

为什么这点很重要：演示界面最常见的腐化方式是"为了快点出效果，先在这里抄一份
检索代码，回头再统一"。然后两份代码开始各自演化，某天你发现 demo 的表现和
评测脚本对不上，却查不出哪边是对的。**界面必须是薄薄一层。**

界面显示四样东西（都不是装饰，各自有明确用途）：
  1. 答案 —— 带 [n] 引用标注
  2. 引用原文（可折叠）—— 让答案**可验证**，这是 RAG 相对纯 LLM 的硬优势
  3. 耗时分解 —— 检索/向量/BM25/融合/取原文/生成，哪一段慢一目了然
  4. 引用了哪几条 —— 模型说用了 [2]，你得能点开 [2] 看看它到底说了什么

⚠️ 本机 gradio 是 6.27.0，两个已知破坏性变更：
   · `gr.Interface(allow_flagging=...)` 已被删除 → 必须用 `flagging_mode="never"`
   · 本文件用 Blocks 而不是 Interface，天然不涉及；但如果你要改回 Interface，记得这条
   · 启动必须绑 127.0.0.1，不要绑 0.0.0.0

用法
----
    :: 启动（默认云端 qwen-plus，检索用 faiss nprobe=512）
    "...python313\\python.exe" src\\gradio_app.py

    :: 换端口 / 不自动开浏览器
    "...python313\\python.exe" src\\gradio_app.py --port 7861 --no-browser
"""

import argparse
import html
import sys
import time
from pathlib import Path
from types import SimpleNamespace

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

HERE = Path(__file__).resolve().parent          # ...\rag-kb\app
PROJECT = HERE.parent
# ⚠️ 要插的是 src/，不是本文件所在的 app/。
#    插错的话 import generator 会直接 ModuleNotFoundError —— 这个坑我第一版就踩了。
sys.path.insert(0, str(PROJECT / "src"))

from generator import (  # noqa: E402
    CLOUD_MODELS,
    DEFAULT_CLOUD_MODEL,
    RETRIEVAL_DEFAULTS,
    CloudBackend,
    LocalBackend,
    build_context,
    build_messages,
    classify_api_error,
    ensure_retrieval_args,
    find_local_model,
    is_fatal_api_error,
    is_refusal,
    make_searcher,
    parse_citations,
    retrieve,
)


def model_of_backend_label(label: str) -> str:
    """
    下拉标签 → 后端要用的模型名。

    标签是写给人看的（"云端 qwen-turbo"），模型名是给 API 的。
    中间这层映射必须有且只有一处 —— 否则改个显示文案就会连带崩掉调用
    （同 BM 坑：同一条信息在两处定义，就一定会漂移）。
    """
    return label.split(" ", 1)[1].strip() if " " in label else label


class RAGApp:
    """把检索器 + 生成后端包成一个状态对象：索引只加载一次，多次问答复用。"""

    def __init__(self, args):
        self.args = args
        # 兜底：万一 args 是别处手造的（少字段），在这里补齐而不是等 retrieve() 崩。
        # 2026-09-22 的教训：界面自己抄了一份参数表，generator 加了 --fetch-store
        # 它没跟上，于是**索引加载完、用户点了"提问"之后**才抛 AttributeError。
        ensure_retrieval_args(args, "gradio RAGApp")
        self.searcher = make_searcher(args)
        self._clouds: dict[str, CloudBackend] = {}
        # 启动即建一个（构造时会检查 DASHSCOPE_API_KEY，把"key 没设"提前暴露，
        # 而不是等用户点完"提问"才报）。构造本身不产生 API 调用、不花钱。
        self.cloud(args.cloud_model)
        self._local = None                      # 懒加载：不选本地就不占显存
        self._rr = None                         # 重排器也懒加载：不 --rerank 就不占内存
        print(f"[就绪] 云端模型：{args.cloud_model}（下拉可切换 · 默认再启动一次仍用它）")

    def cloud(self, model: str) -> CloudBackend:
        """
        按模型名缓存云端后端 —— 界面下拉可以切模型（2026-09-22 加）。

        为什么缓存而不是每次 new：OpenAI 客户端内部维护连接池，
        每次重建等于每次重新握手；而下拉里切来切去是常态。

        为什么默认值是 qwen-turbo：qwen-plus 免费额度已耗尽（2026-09-22 实测 403），
        能用什么见 generator.CLOUD_MODELS 与 `src/check_backends.py`。
        """
        if model not in self._clouds:
            self._clouds[model] = CloudBackend(model=model)
            print(f"[云端] 新建后端：{model}")
        return self._clouds[model]

    def reranker(self):
        """
        懒加载 cross-encoder（bge-reranker-v2-m3）。

        为什么懒加载：它要额外吃 2 GB 上下内存 + 几十秒加载，
        而**大部分提问根本不需要它**（比如第一次跑通验证）。
        只在 args.rerank 为真、且第一次真正要检索时才加载。
        """
        if self._rr is None:
            from rerank import Reranker
            print("[重排] 首次使用，加载 bge-reranker-v2-m3 …")
            self._rr = Reranker(verbose=True, max_length=self.args.rerank_max_length)
        return self._rr

    def local(self):
        if self._local is None:
            print("[本地] 首次使用，加载 Qwen2.5-1.5B-Instruct（4bit，约 30 秒）…")
            d = find_local_model("Qwen2.5-1.5B-Instruct")
            if d is None:
                raise RuntimeError(
                    "找不到本地 Qwen2.5-1.5B-Instruct —— 先跑 "
                    '"...python313\\python.exe" src\\download_models.py --only qwen2.5-1.5b-instruct')
            self._local = LocalBackend(d, quant="4bit", max_new_tokens=self.args.max_new_tokens)
        return self._local

    def answer(self, query, backend_name, topk):
        query = (query or "").strip()
        if not query:
            return "（请输入问题）", "", ""

        t0 = time.time()
        qv = self.searcher._encode([query])[0]

        # ---- 检索（复用第 8 步的混合检索，参数与命令行版完全一致）----
        # 重排器只在 --rerank 时懒加载一次；没开就传 None，与命令行版同一分支
        hits, timing = retrieve(self.searcher, query, self.args, qv,
                                self.reranker() if self.args.rerank else None)
        if not hits:
            return "检索无结果。", "", ""

        # ---- 生成 ----
        messages = build_messages(query, hits)
        backend = (self.local() if backend_name.startswith("本地")
                   else self.cloud(model_of_backend_label(backend_name)))
        try:
            answer, usage = backend.generate(messages, self.args.max_new_tokens)
        except Exception as e:
            # ⚠️ 别把 SDK 的原始 JSON 糊给用户 —— 2026-09-22 的截图就是那样：
            # 满屏 `{'error': {'message': 'Free quota exhausted...', 'type':
            # 'AllocationQuota.FreeTierOnly', 'param': None, 'code': ...}}`，
            # 用户看完只知道"坏了"，不知道"换个模型就行"。
            # classify_api_error 把常见错误翻成「原因 + 一条能照着做的动作」。
            reason, hint = classify_api_error(e)
            if is_fatal_api_error(e):
                extra = ("\n\n> 换个模型重试最快：上面「生成模型」下拉里选别的"
                         "（qwen-turbo 通常还有免费额度）。")
            else:
                extra = "\n\n> 这类是临时故障，直接再点一次「提问」通常就行。"
            return f"### ❌ {reason}\n\n{hint}{extra}", "", ""

        total = time.time() - t0
        used, bad = parse_citations(answer, len(hits))
        refused = is_refusal(answer)

        # ---- 顶部状态行 ----
        flags = []
        if refused:
            flags.append("🟡 拒答")
        if bad:
            flags.append(f"🔴 非法编号 {bad}（上下文只有 {len(hits)} 条）")
        if not used and not refused:
            flags.append("🟠 无引用标注")
        head = ""
        if flags:
            head = "> " + " · ".join(flags) + "\n\n"

        # ---- 耗时条 ----
        # 取原文用 timing['fetch'] 直接取，**不要用减法**：
        # 减法把"重排"也算了进去，一旦开了重排（--rerank），这一栏显示的就是错的、
        # 而且看不出错（数字还在合理的量级上）。
        parts = [f"向量 {timing['vector'] * 1000:.0f}",
                 f"BM25 {timing['bm25'] * 1000:.0f}",
                 f"融合 {timing['fuse'] * 1000:.1f}",
                 f"取原文 {timing['fetch'] * 1000:.0f}"]
        if timing.get("rerank"):
            parts.append(f"重排 {timing['rerank'] * 1000:.0f}")
        timing_md = (
            f"**检索 {timing['retrieve_total'] * 1000:.0f} ms**"
            f"（{' · '.join(parts)}）"
            f" ｜ **生成 {usage.get('latency', 0):.2f} s**"
            + (f"（输入 {usage.get('prompt_tokens')} tok / 输出 {usage.get('completion_tokens')} tok"
               if usage.get("prompt_tokens") else "")
            + f" ｜ 合计 **{total:.2f} s**"
            + f"\n\n引用编号：`{used if used else '无'}`　上下文 {len(build_context(hits))} 字"
        )

        # ---- 引用原文（HTML，可折叠）----
        rows = []
        for h in hits:
            mark = ""
            if h["rank"] in used:
                mark = " ← 被引用"
            sec = html.escape(h["section"] or "—")
            title = html.escape(h["title"])
            body = html.escape(h["chunk_text"])
            score = f"{h['score']:.5f}" if h["score"] is not None else "—"
            rows.append(f"""
<div style="border:1px solid #e3e3e3;border-radius:8px;padding:10px 12px;margin:8px 0;background:#fafafa">
  <div style="font-weight:600;color:#1f2937">
    [{h['rank']}] {title}<span style="color:#6b7280;font-weight:400"> · {sec}</span>
    <span style="color:#dc2626;font-weight:500">{mark}</span>
  </div>
  <div style="color:#6b7280;font-size:12px;margin:4px 0 6px">
    RRF {score}　向量排名 #{h['vrank'] or '—'}　BM25 排名 #{h['brank'] or '—'}　chunk_id={h['chunk_id']}
  </div>
  <div style="color:#374151;font-size:13px;line-height:1.7;white-space:pre-wrap">{body}</div>
</div>""")
        cites_html = f'<div style="font-family:system-ui">{head_html("引用原文", len(hits))}{"".join(rows)}</div>'
        return head + answer, cites_html, timing_md


def head_html(title, n):
    return (f'<div style="font-size:13px;color:#6b7280;margin-bottom:4px">'
            f'{title}（{n} 条）—— 「← 被引用」标出答案里 [n] 真正用到的那几条</div>')


def build_ui(app: RAGApp):
    import gradio as gr

    with gr.Blocks(title="中文知识库 RAG 问答") as demo:
        gr.Markdown(
            "# 中文知识库 RAG 问答\n"
            "语料：中文维基百科清洗管道产出 **331.6 万** chunk ｜ "
            "检索：BM25 + 向量（bge-large-zh）RRF 融合 ｜ "
            "生成：dashscope / 本地 Qwen2.5-1.5B"
        )

        with gr.Row():
            q = gr.Textbox(label="问题", placeholder="例如：台灣東部開發於古時的人行道路",
                           lines=1, scale=5, autofocus=True)
            # 下拉内容从 generator.CLOUD_MODELS 生成（单一来源，别再硬编码）。
            # 原来这里写死了 "云端 qwen-plus" —— 而它的免费额度已耗尽，
            # 用户点"提问"直接吃 403（2026-09-22 的截图）。
            cloud_choices = list(CLOUD_MODELS)
            if app.args.cloud_model not in cloud_choices:      # --cloud-model 传了自定义的
                cloud_choices.insert(0, app.args.cloud_model)
            choices = [f"云端 {m}" for m in cloud_choices] + ["本地 Qwen2.5-1.5B"]
            backend = gr.Dropdown(choices, value=f"云端 {app.args.cloud_model}",
                                  label="生成模型", scale=2)
            topk = gr.Slider(3, 10, value=5, step=1, label="返回条数", scale=2)
        with gr.Row():
            btn = gr.Button("提问", variant="primary", scale=1)
            clr = gr.Button("清空", scale=1)

        ans = gr.Markdown(label="答案")
        timing = gr.Markdown()
        with gr.Accordion("引用原文（展开可逐条核对）", open=True):
            cites = gr.HTML()

        def _run(query, backend_name, k):
            app.args.topk = int(k)
            return app.answer(query, backend_name, int(k))

        btn.click(_run, inputs=[q, backend, topk], outputs=[ans, cites, timing])
        q.submit(_run, inputs=[q, backend, topk], outputs=[ans, cites, timing])
        clr.click(lambda: ("", "", "", ""), outputs=[q, ans, cites, timing])

        gr.Markdown(
            "> 提示：模型是**基于检索到的原文**作答的。答案里的 `[n]` 对应下方引用的编号，"
            "展开即可核对 —— 如果某条 `[n]` 的内容和答案对不上，那就是检索错了或模型编了。\n>\n"
            "> 拒答是**设计行为**：上下文里没有依据时，模型会回答「根据已有资料无法回答」，"
            "而不是用自己的知识编一个。\n>\n"
            "> ⚠️ **模型额度**：`qwen-plus` 的免费额度已耗尽（选了会报 403），"
            "换 `qwen-turbo`（默认）或 `qwen-max` 即可。"
            "跑 `src\\check_backends.py` 可以列出当前哪些模型可用。"
        )
    return demo


def build_parser() -> argparse.ArgumentParser:
    """
    界面自己的参数表。

    ⚠️ 检索相关的参数一律从 generator.RETRIEVAL_DEFAULTS 取默认值，
    **不要再写字面量** —— 写死就又会变成"两份清单"，
    而这里的漏项在上一次（--fetch-store）表现为：索引加载完、点了"提问"才报 AttributeError。
    自检脚本 src/smoke_retrieve.py 会拿这个 parser 和 retrieve() 对一遍。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="绑 127.0.0.1，不要绑 0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--cloud-model", default=DEFAULT_CLOUD_MODEL,
                    help=f"云端模型名，默认 {DEFAULT_CLOUD_MODEL}（与命令行版同一个常量）。"
                         f"可选：{' / '.join(CLOUD_MODELS)}")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--vec-backend", default="faiss", choices=["faiss", "scan"])
    ap.add_argument("--nprobe", type=int, default=512)

    # ---------------- 检索链路（与命令行版同一份默认值） ----------------
    ap.add_argument("--topk", type=int, default=RETRIEVAL_DEFAULTS["topk"])
    ap.add_argument("--topn", "--pool", dest="topn", type=int,
                    default=RETRIEVAL_DEFAULTS["topn"])
    ap.add_argument("--rrf-k", type=int, default=RETRIEVAL_DEFAULTS["rrf_k"])
    ap.add_argument("--w-vec", type=float, default=RETRIEVAL_DEFAULTS["w_vec"])
    ap.add_argument("--w-bm25", type=float, default=RETRIEVAL_DEFAULTS["w_bm25"])
    ap.add_argument("--fetch-store", default=RETRIEVAL_DEFAULTS["fetch_store"],
                    choices=["auto", "blob", "parquet"],
                    help="取原文实现：auto=有侧车就用（默认）/ blob / parquet")

    # 重排：界面默认**关**（省内存、启动快），但生产配置是开 ——
    # 加了它，界面才能和评测脚本跑同一套配置（否则演示的是另一套系统）。
    ap.add_argument("--rerank", action="store_true",
                    default=RETRIEVAL_DEFAULTS["rerank"],
                    help="开 cross-encoder 重排（bge-reranker-v2-m3）。"
                         "§11.13 实测 R@1 0.613→0.885，代价是每次查询多约 2.8 s")
    ap.add_argument("--rerank-pool", type=int, default=RETRIEVAL_DEFAULTS["rerank_pool"])
    ap.add_argument("--rerank-max-length", type=int,
                    default=RETRIEVAL_DEFAULTS["rerank_max_length"])
    ap.add_argument("--rerank-batch", type=int, default=RETRIEVAL_DEFAULTS["rerank_batch"])
    return ap


def main() -> int:
    args = build_parser().parse_args()

    # make_searcher() 读 vec_backend / nprobe，其余字段它也用（mode/fuse 等），
    # 这里补齐成 SimpleNamespace 兼容的形状。
    args.mode = "hybrid"
    args.fuse = "rrf"
    ensure_retrieval_args(args, "gradio main()")

    print("=" * 76)
    print("第 9 步 · RAG 演示界面")
    print(f"检索      : hybrid topk={args.topk} pool={args.topn} RRF k={args.rrf_k}")
    print(f"取原文    : {args.fetch_store}（auto = 有侧车就用侧车；没侧车会打印一行退回 parquet）")
    print("重排      : " + (f"开（候选 {args.rerank_pool or args.topn} · "
                             f"max_length={args.rerank_max_length}）"
                             if args.rerank else
                             "关 —— 生产配置是开（加 --rerank：R@1 0.613→0.885，每次查询多约 2.8 s）"))
    print("=" * 76)
    app = RAGApp(args)
    demo = build_ui(app)
    print(f"\n打开浏览器：http://{args.host}:{args.port}\n")
    demo.launch(server_name=args.host, server_port=args.port,
                inbrowser=not args.no_browser, show_error=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
