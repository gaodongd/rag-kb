"""从 ModelScope 下载项目所需的模型。

用法：
    python src/download_models.py                     # 只下必装项
    python src/download_models.py --all               # 连消融实验要用的也下了
    python src/download_models.py --only bge-reranker-v2-m3

为什么不用 huggingface：
  本机 huggingface.co 完全不通（HTTP 000），hf-mirror.com 经 huggingface_hub
  请求会拿到 0 字节空文件（它的 User-Agent 被镜像反爬拦了）。
  试过改客户端工厂、改 build_hf_headers，都没用。别在这上面耗时间。

⚠️ 为什么必须给 allow_patterns（实测查过仓库清单，不是想当然）：
  bge-large-zh-v1.5 仓库里同时放了这两份**内容完全等价的权重**：
      model.safetensors    1,302,138,752 字节
      pytorch_model.bin    1,302,220,525 字节
  不限定的话 snapshot_download 会把两份都拖下来 —— 白下 1.21GB。
  所以下面每个模型都显式给 allow_patterns，只要 safetensors 那一份。
  另外 1_Pooling/ 是 sentence-transformers 的池化配置，FlagEmbedding 会读，
  必须一起下（漏了会报找不到池化层）。
"""
import argparse
import sys
import time
from pathlib import Path

from modelscope import snapshot_download

CACHE = r"E:\AI-learning\ms-cache"

# 权重只要 safetensors（bin 是等价副本）；1_Pooling 是池化配置，别漏
BASE_PATTERNS = ["model.safetensors", "*.json", "*.txt", "1_Pooling/*"]

# key: (模型 ID, 说明, 大致体积, 是否必装)
MODELS = {
    "bge-large-zh-v1.5": (
        "AI-ModelScope/bge-large-zh-v1.5", "向量化主力模型", "1.21GB", True),
    "bge-small-zh-v1.5": (
        "AI-ModelScope/bge-small-zh-v1.5", "消融对照，验证模型大小对检索的影响", "95MB", False),
    "bge-reranker-v2-m3": (
        "AI-ModelScope/bge-reranker-v2-m3", "重排序模型（M2 混合检索用）", "2.3GB", False),
    # ⚠️ 必须用 Instruct 版，不是 base 版。
    # M3 的 B 路要求模型「按 [n] 标注引用」「找不到依据就拒答」——
    # 这些是**指令跟随**能力，base 版只会顺着上下文往下续写，不会拒绝，也不会标引用。
    # 而且 Instruct 版有现成的 chat template，base 版没有，得自己拼 <|im_start|>。
    "qwen2.5-1.5b-instruct": (
        "Qwen/Qwen2.5-1.5B-Instruct", "本地生成模型（M3 的 B 路），和云端 API 做对照", "3.1GB", False),
}


def fetch(targets):
    ok, fail = [], []
    for i, key in enumerate(targets, 1):
        model_id, desc, size, _ = MODELS[key]
        print(f"\n[{i}/{len(targets)}] {key}  ({size})")
        print(f"        {model_id} — {desc}")
        t0 = time.time()
        try:
            path = snapshot_download(model_id, cache_dir=CACHE,
                                     allow_patterns=BASE_PATTERNS)
            dt = time.time() - t0
            # 实测落盘体积 —— 用来验证 allow_patterns 真的生效了（应该 ≈ 1.21GB 而非 2.4GB）
            real = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
            print(f"        OK  {dt:.0f}s  实际落盘 {real / 1024 ** 2:.0f} MB")
            print(f"        {path}")
            ok.append(key)
        except Exception as e:
            print(f"        失败: {type(e).__name__}: {e}")
            fail.append((key, str(e)))
    return ok, fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="连可选项一起下载")
    ap.add_argument("--only", help="只下指定 key，如 bge-large-zh-v1.5")
    args = ap.parse_args()

    if args.only:
        if args.only not in MODELS:
            print(f"[错误] 未知的 key: {args.only}")
            print("可选：", ", ".join(MODELS))
            return 1
        targets = [args.only]
    else:
        targets = [k for k, v in MODELS.items() if v[3] or args.all]

    Path(CACHE).mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print(f"从 ModelScope 下载 {len(targets)} 个模型")
    print(f"缓存目录：{CACHE}")
    print(f"过滤规则：{BASE_PATTERNS}")
    print("=" * 64)
    print("\n提示：中断后重跑同一条命令即可续传，不会从头开始。")
    print("开始下载...（Ctrl+C 可随时中断）")

    ok, fail = fetch(targets)

    print("\n" + "=" * 64)
    print(f"成功 {len(ok)} 个，失败 {len(fail)} 个")
    for m, e in fail:
        print(f"  失败：{m}\n        {e}")
    if fail:
        print("\n失败的重跑一次通常就好（ModelScope 偶尔抽风）。")
    else:
        print("全部就绪。下一步：跑 src\\bench_embed.py 实测向量化吞吐。")
    print("=" * 64)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
