"""实测向量化吞吐 —— 决定「要索引多少数据」的那个数字。

为什么必须先测这个：
  整条 RAG 项目的规模决策（索引多少条目、要不要采样）都建立在
  "4060 上 bge-large-zh 每秒能处理多少 chunk" 之上。
  这个数不能估 —— 估错 10 倍，你要么白等 30 小时，要么知识库小得没意义。
  跑一次 2 分钟，把预算从猜测变成测量。

用法（CMD）：
  "C:\\Users\\搞懂\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" src\\bench_embed.py
  :: 指定测试规模 / 序列长度
  "...python.exe" src\\bench_embed.py --n 4000 --seq-len 512
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# 生产库用的是 800 字符切块（build_pipeline.py 的默认值）
CHUNK_CHARS = 800
OVERLAP = 100


def find_model():
    """在 ms-cache 里找 bge-large-zh-v1.5 的本地目录。"""
    cache = Path(r"E:\AI-learning\ms-cache")
    if not cache.exists():
        return None
    for p in cache.rglob("model.safetensors"):
        if "bge-large-zh" in str(p.parent):
            return p.parent
    # 有些版本不落 model.safetensors，退一步找 config
    for p in cache.rglob("config.json"):
        if "bge-large-zh" in str(p.parent) and (p.parent / "vocab.txt").exists():
            return p.parent
    return None


def load_chunks(n, jsonl):
    """从真实语料里切出 n 个 chunk，让测出来的长度分布接近生产环境。"""
    if not Path(jsonl).exists():
        raise FileNotFoundError(f"找不到样本语料 {jsonl}，先跑 wiki_extract.py")
    chunks = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"]
            # 去掉开头的信息框/模板噪音（模拟清洗后的输入）
            i = 0
            while i < len(text) and len(chunks) < n:
                chunks.append(text[i:i + CHUNK_CHARS])
                i += CHUNK_CHARS - OVERLAP
            if len(chunks) >= n:
                break
    return chunks[:n]


def load_model(path):
    """优先 FlagEmbedding；不行退 sentence-transformers；再不行退裸 transformers。"""
    try:
        from FlagEmbedding import FlagModel
        m = FlagModel(str(path), query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                      use_fp16=True)
        return ("FlagEmbedding", lambda texts, b: m.encode(texts, batch_size=b, max_length=512))
    except Exception as e:
        print(f"  [info] FlagEmbedding 不可用（{type(e).__name__}: {str(e)[:80]}），尝试 sentence-transformers")

    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(str(path), device="cuda")
        return ("sentence-transformers",
                lambda texts, b: m.encode(texts, batch_size=b, normalize_embeddings=True))
    except Exception as e:
        print(f"  [info] sentence-transformers 不可用（{type(e).__name__}），退回裸 transformers")

    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(path))
    model = AutoModel.from_pretrained(str(path)).cuda().half().eval()

    def run(texts, batch):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                enc = tok(texts[i:i + batch], padding=True, truncation=True,
                          max_length=512, return_tensors="pt").to("cuda")
                h = model(**enc).last_hidden_state[:, 0]        # BGE 用 CLS 池化
                out.append(torch.nn.functional.normalize(h, dim=-1))
        return torch.cat(out).cpu().numpy()

    return ("transformers(CLS池化)", run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="测多少个 chunk")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--jsonl", default=r"E:\AI-learning\projects\rag-kb\data\raw\wiki_sample.jsonl")
    ap.add_argument("--full-docs", type=int, default=790_000,
                    help="全量条目数（用于折算总耗时；默认 79 万，来自 432MB 样本外推）")
    ap.add_argument("--full-chars", type=float, default=63.8e8,
                    help="全量正文字符数（默认 63.8 亿，来自 432MB 样本外推）")
    args = ap.parse_args()

    print("=" * 64)
    print("向量化吞吐实测")
    print("=" * 64)

    import torch
    print(f"torch {torch.__version__} | CUDA {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name}  {p.total_memory / 1024 ** 3:.1f}GB")

    path = find_model()
    if not path:
        print("\n[错误] 缓存里找不到 bge-large-zh-v1.5。先跑：")
        print(r'  "...python.exe" src\download_models.py --only bge-large-zh-v1.5')
        return 1
    print(f"模型：{path}")

    print(f"\n从真实语料切 {args.n} 个 chunk（{CHUNK_CHARS} 字符 / 重叠 {OVERLAP}）...")
    chunks = load_chunks(args.n, args.jsonl)
    lens = [len(c) for c in chunks]
    print(f"  平均 {statistics.mean(lens):.0f} 字符 / 中位 {statistics.median(lens):.0f}")

    backend, run = load_model(path)
    print(f"加载方式：{backend}")

    print("\n预热（CUDA 首次要编译 kernel，别把预热算进结果）...")
    run(chunks[:64], args.batch)

    print(f"正式测：{len(chunks)} 个 chunk，batch={args.batch}")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    run(chunks, args.batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t0

    tps = len(chunks) / dt
    print(f"\n耗时 {dt:.1f}s  →  {tps:.0f} chunk/秒")

    # 折算总耗时
    est_chunks_raw = args.full_chars / (CHUNK_CHARS - OVERLAP)
    print("\n" + "=" * 64)
    print("折算到全量语料（注意：这是清洗【前】字符数，清洗后耗时会更短）")
    print("=" * 64)
    print(f"全量条目      : {args.full_docs:,.0f}")
    print(f"全量正文      : {args.full_chars / 1e8:.1f} 亿字符")
    print(f"切块数（估）  : {est_chunks_raw / 1e4:.0f} 万 chunk")
    print(f"→ 向量化耗时  : {est_chunks_raw / tps / 3600:.1f} 小时")
    print("\n清洗会去掉大量模板/链接标记，实际字符数通常降到 5–7 成，")
    print("所以真实耗时按上面的 50%–70% 估更准。")
    print("=" * 64)

    # 反推"3 小时能索引多少"
    budget = tps * 3 * 3600
    print(f"\n反推：想控制在 3 小时内，chunk 数上限 ≈ {budget / 1e4:.0f} 万")
    print(f"      对应正文字符 ≈ {budget * (CHUNK_CHARS - OVERLAP) / 1e8:.1f} 亿"
          f"（约占全量的 {budget * (CHUNK_CHARS - OVERLAP) / args.full_chars * 100:.0f}%）")
    print(f"      对应条目 ≈ {args.full_docs * budget * (CHUNK_CHARS - OVERLAP) / args.full_chars / 1e4:.0f} 万")
    return 0


if __name__ == "__main__":
    sys.exit(main())
