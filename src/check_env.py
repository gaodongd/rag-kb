"""环境自检 —— 跑这个，全绿才继续。

用法：
    :: 主力自检（Python 3.13）
    "C:\\Users\\搞懂\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" src\\check_env.py

    :: 加跑 Spark 冒烟测试（会真的起一个 Spark 会话，约 40 秒）
    "C:\\Users\\搞懂\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" src\\check_env.py --spark-test

任何一行 [FAIL] 都别往下走，先解决它。
[info] 只是告诉你情况，不影响判断。
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

OK = "  [OK]  "
BAD = "  [FAIL]"
WARN = "  [WARN]"
INFO = "  [info]"

PY313 = r"C:\Users\搞懂\AppData\Local\Programs\Python\Python313\python.exe"
PY311 = r"C:\Users\搞懂\AppData\Local\Programs\Python\Python311\python.exe"

results = {"ok": 0, "bad": 0, "warn": 0}


def report(status, name, detail):
    print(f"{status} {name}: {detail}")
    if status == OK:
        results["ok"] += 1
    elif status == BAD:
        results["bad"] += 1
    elif status == WARN:
        results["warn"] += 1


def check_python():
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if (3, 9) <= v < (3, 12):
        report(OK, "当前解释器", f"{ver} —— Spark 专用，正确")
    elif v >= (3, 12):
        report(INFO, "当前解释器", f"{ver} —— 主力环境（向量化/检索/生成）；"
                                    f"Spark 不能用它跑，见下面 pyspark 一项")
    else:
        report(BAD, "当前解释器", f"{ver} 过低，需要 3.9+")


def check_java():
    """Spark 靠 Java 跑。JAVA_HOME 没设是这台机器的老毛病。"""
    jh = os.environ.get("JAVA_HOME", "")
    if jh and Path(jh, "bin", "java.exe").exists():
        report(OK, "JAVA_HOME", jh)
    else:
        report(BAD, "JAVA_HOME", f"未设置或无效（当前值：{jh or '空'}）")

    java_exe = shutil.which("java")
    if java_exe:
        try:
            r = subprocess.run([java_exe, "-version"], capture_output=True,
                               text=True, timeout=20, encoding="utf-8", errors="replace")
            first = (r.stderr or "").strip().splitlines()[0]
            report(OK, "java 命令", first)
        except Exception as e:
            report(WARN, "java 命令", f"存在但执行失败：{e}")
    else:
        report(BAD, "java 命令", "PATH 里找不到 java")


def decode_any(raw: bytes) -> str:
    """Windows 上子进程输出可能是 utf-8 也可能是 GBK，逐个试。"""
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def probe(interpreter, code, timeout=90):
    """用另一个解释器跑一段代码，返回 (成功, 完整输出文本)。"""
    if not Path(interpreter).exists():
        return False, "解释器不存在"
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    try:
        r = subprocess.run([interpreter, "-c", code], capture_output=True,
                           timeout=timeout, env=env)
        text = decode_any((r.stdout or b"") + (r.stderr or b"")).strip()
        return r.returncode == 0, text
    except subprocess.TimeoutExpired:
        return False, f"超时（>{timeout}s）"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_pyspark_situation():
    """这台机器最容易踩的坑：PySpark 只能跑在 3.11 上。

    为什么不能只 import 一下就算过：
      import pyspark 在 3.13 上也能成功（它就是个普通 Python 包），
      但一执行 count() 就会因为 3.12+ 改了 socket 行为而崩：
        OSError: [WinError 10038] ... Python worker exited unexpectedly
      所以必须去 3.11 那边实测。
    """
    if sys.version_info >= (3, 12):
        print(f"{INFO} pyspark: 不装在当前解释器（3.12+ 与 PySpark 3.5.x 不兼容，已实测）")
        ok, out = probe(PY311, "import pyspark; print(pyspark.__version__)")
        if ok:
            report(OK, "pyspark（在 Python 3.11 里）", f"版本 {out}")
            report(INFO, "Spark 专用解释器", PY311)
        else:
            report(BAD, "pyspark（在 Python 3.11 里）", f"{out}")
            print(f'       修："{PY311}" -m pip install pyspark==3.5.8')
    else:
        try:
            m = __import__("pyspark")
            report(OK, "pyspark", f"版本 {getattr(m, '__version__', '?')}")
        except Exception as e:
            report(BAD, "pyspark", f"{type(e).__name__}: {e}")


SPARK_SMOKE = r"""
import os, shutil, sys
from pathlib import Path
# 注意：不能用 setdefault —— 环境里 JAVA_HOME 可能是空字符串（键存在但值是空的），
# setdefault 遇到已存在的键（哪怕是空串）不会替换，Spark 就直接找不到 Java。
if not os.environ.get('JAVA_HOME'):
    j = shutil.which('java')
    if j:
        os.environ['JAVA_HOME'] = str(Path(j).resolve().parent.parent)
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
from pyspark.sql import SparkSession
s = (SparkSession.builder.master('local[2]').appName('check_env')
     .config('spark.ui.showConsoleProgress', 'false')
     .config('spark.sql.shuffle.partitions', '4').getOrCreate())
s.sparkContext.setLogLevel('ERROR')
df = s.createDataFrame([(1, 'a'), (2, 'b'), (3, 'c')], ['id', 'v'])
assert df.count() == 3, 'count 不对'
assert df.groupBy().sum('id').collect()[0][0] == 6, '聚合不对'
s.stop()
print('SPARK_SMOKE_OK')
"""


def check_spark_smoke():
    print("  ... 正在起 Spark（约 40 秒，第一次更慢，别以为卡死了）")
    ok, out = probe(PY311, SPARK_SMOKE, timeout=300)
    if "SPARK_SMOKE_OK" in out:
        report(OK, "Spark 冒烟测试", "起会话 / count / 聚合 全部通过")
        return
    report(BAD, "Spark 冒烟测试", "没有输出成功标记")
    # 只挑有诊断价值的行，别把 JVM 那一大堆日志倒给用户
    keys = ("Error", "error", "Exception", "Traceback", "Cannot", "cannot",
            "failed", "Failed", "not found", "WinError", "Caused by")
    lines = [l for l in out.splitlines() if any(k in l for k in keys)]
    for l in (lines[-6:] or out.splitlines()[-6:]):
        print(f"       | {l.strip()[:150]}")
    print(f"       退出码非 0。排查：确认 JAVA_HOME 已设且**重开过终端**；"
          f"再不行把上面几行发我。")


def check_import(module, label, required=True):
    try:
        m = __import__(module)
        report(OK, label, f"版本 {getattr(m, '__version__', '?')}")
    except Exception as e:
        report(BAD if required else WARN, label, f"{type(e).__name__}: {e}")


def check_torch():
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            report(OK, "torch / CUDA",
                   f"{torch.__version__} | {p.name} ({p.total_memory / 1024 ** 3:.0f}GB) | CUDA {torch.version.cuda}")
        else:
            report(WARN, "torch / CUDA", f"{torch.__version__} 但 CUDA 不可用，向量化会慢几十倍")
    except Exception as e:
        report(BAD, "torch / CUDA", f"{type(e).__name__}: {e}")


def check_disk():
    for drive in ("E:\\", "C:\\", "D:\\"):
        try:
            free_gb = shutil.disk_usage(drive).free / 1024 ** 3
            if drive == "E:\\":
                report(OK if free_gb > 40 else WARN, f"磁盘 {drive}",
                       f"剩余 {free_gb:.1f} GB" + ("" if free_gb > 40 else " 偏少"))
            else:
                print(f"{INFO} 磁盘 {drive}: 剩余 {free_gb:.1f} GB")
        except Exception:
            pass


def check_dirs():
    root = Path(__file__).resolve().parent.parent
    need = ["data/raw", "data/interim", "data/processed", "src", "eval/results", "app"]
    missing = [d for d in need if not (root / d).exists()]
    if missing:
        report(WARN, "项目目录", f"缺：{missing}")
    else:
        report(OK, "项目目录", f"{root}")


def check_data():
    d = Path(r"E:\AI-learning\data\zhwiki-latest-pages-articles.xml.bz2")
    part = Path(str(d) + ".part")
    chunks = Path(r"E:\AI-learning\data\_chunks")
    if d.exists():
        report(OK, "维基数据", f"已就绪 {d.stat().st_size / 1024 ** 3:.2f} GB")
    else:
        n = 0
        if chunks.exists():
            n = sum(p.stat().st_size for p in chunks.glob("*.bin"))
        if n or part.exists():
            extra = part.stat().st_size if part.exists() else 0
            report(INFO, "维基数据", f"下载中（已下 {(n + extra) / 1024 ** 3:.2f} GB）"
                                     f"—— 重跑 download_wiki_mt.py 自动续传")
        else:
            report(INFO, "维基数据", "还没下 —— 第 3 步：src\\download_wiki_mt.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spark-test", action="store_true", help="加跑 Spark 冒烟测试（约 40 秒）")
    args = ap.parse_args()

    print("=" * 64)
    print("RAG 知识库项目 · 环境自检")
    print("=" * 64)

    check_python()
    check_dirs()
    check_data()
    print("-" * 64)
    print("Java / Spark（M1 数据管道用，跑在 Python 3.11 上）")
    check_java()
    check_pyspark_situation()
    if args.spark_test:
        check_spark_smoke()
    print("-" * 64)
    print("解析 / 文本处理")
    check_import("jieba", "jieba（中文分词）")
    check_import("pypdf", "pypdf（PDF 解析）")
    check_import("docx", "python-docx（Word 解析）")
    print("-" * 64)
    print("模型 / 向量化（M2 之后用）")
    check_torch()
    check_import("transformers", "transformers")
    check_import("FlagEmbedding", "FlagEmbedding")
    check_import("chromadb", "chromadb")
    check_import("rank_bm25", "rank_bm25")
    check_import("modelscope", "modelscope（模型下载）")
    print("-" * 64)
    print("服务（M5 用）")
    check_import("fastapi", "fastapi")
    check_import("uvicorn", "uvicorn")
    print("-" * 64)
    print("磁盘与缓存")
    check_disk()
    ms_cache = Path("E:/AI-learning/ms-cache")
    if ms_cache.exists():
        subs = [p.name for p in ms_cache.iterdir()]
        print(f"{INFO} 模型缓存 {ms_cache}: 存在，{len(subs)} 个子目录")
    else:
        print(f"{INFO} 模型缓存 {ms_cache}: 尚未创建")

    print("=" * 64)
    print(f"结果：{results['ok']} 项通过 / {results['bad']} 项失败 / {results['warn']} 项警告")
    print("环境就绪，可以进下一步。" if not results["bad"] else "有 [FAIL]，先修完再继续。")
    print("=" * 64)
    return 1 if results["bad"] else 0


if __name__ == "__main__":
    sys.exit(main())
