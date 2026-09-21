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
    CloudBackend,
    LocalBackend,
    build_context,
    build_messages,
    find_local_model,
    is_refusal,
    make_searcher,
    parse_citations,
    retrieve,
)


class RAGApp:
    """把检索器 + 生成后端包成一个状态对象：索引只加载一次，多次问答复用。"""

    def __init__(self, args):
        self.args = args
        self.searcher = make_searcher(args)
        self.cloud = CloudBackend(model=args.cloud_model)
        self._local = None                      # 懒加载：不选本地就不占显存
        print(f"[就绪] 云端后端：{args.cloud_model}")

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
        hits, timing = retrieve(self.searcher, query, self.args, qv)
        if not hits:
            return "检索无结果。", "", ""

        # ---- 生成 ----
        messages = build_messages(query, hits)
        backend = self.local() if backend_name.startswith("本地") else self.cloud
        try:
            answer, usage = backend.generate(messages, self.args.max_new_tokens)
        except Exception as e:
            return f"**生成失败**：`{type(e).__name__}: {e}`", "", ""

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
        timing_md = (
            f"**检索 {timing['retrieve_total'] * 1000:.0f} ms**"
            f"（向量 {timing['vector'] * 1000:.0f} · BM25 {timing['bm25'] * 1000:.0f}"
            f" · 融合 {timing['fuse'] * 1000:.1f} · "
            f"取原文 {(timing['retrieve_total'] - timing['vector'] - timing['bm25'] - timing['fuse']) * 1000:.0f}）"
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
            backend = gr.Dropdown(["云端 qwen-plus", "本地 Qwen2.5-1.5B"],
                                  value="云端 qwen-plus", label="生成模型", scale=2)
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
            "而不是用自己的知识编一个。"
        )
    return demo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="绑 127.0.0.1，不要绑 0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--cloud-model", default="qwen-plus")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--topn", type=int, default=100)
    ap.add_argument("--rrf-k", type=int, default=10)
    ap.add_argument("--w-vec", type=float, default=1.0)
    ap.add_argument("--w-bm25", type=float, default=1.0)
    ap.add_argument("--vec-backend", default="faiss", choices=["faiss", "scan"])
    ap.add_argument("--nprobe", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    # make_searcher() 读 vec_backend / nprobe，其余字段它也用（mode/fuse 等），
    # 这里补齐成 SimpleNamespace 兼容的形状。
    args.mode = "hybrid"
    args.fuse = "rrf"

    print("=" * 76)
    print("第 9 步 · RAG 演示界面")
    print("=" * 76)
    app = RAGApp(args)
    demo = build_ui(app)
    print(f"\n打开浏览器：http://{args.host}:{args.port}\n")
    demo.launch(server_name=args.host, server_port=args.port,
                inbrowser=not args.no_browser, show_error=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
