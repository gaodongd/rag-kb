#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
取原文层 —— 从「为了 100 行去解压 2.1 GB」换成「mmap 随机访问」。

============================================================================
这一层在解决什么问题（数字先摆出来）
============================================================================
检索链路的分项延迟（2026-09-21 实测，129 题中位数）：

    取原文 1896 ms  >  重排 1359 ms  ≈  生成 1231 ms  ≫  向量检索 220 ms

**取原文比整个重排还慢，是端到端的 40%。** 这不是"顺带优化"，是最大的一项。

============================================================================
为什么会这么慢（三层原因，逐层拆）
============================================================================
① parquet 是为"扫全表"设计的，不是为"点查"设计的。
   它是**列式 + 分块 + 压缩**存储：要拿到某一行的数据，必须先定位 row group、
   把那个 row group 里的整列**解压**出来。一个 row group 有 20 万行 / 179 MB，
   而我们只要其中 100 行 —— 剩下的 99.95% 是白解压的。
   实测：4 个分片各 552 MB，1 个分片 5 个 row group，每 group 约 179 MB（未压缩）。

② `chunk_id.isin(...)` 这种过滤方式，**剪不掉任何枝**。
   parquet 的行组统计（min/max）本来能用来跳过无关行组，但 chunk_id 是
   随机哈希（`b534134734df4a711535059fce4a87c4`），任何行组的 min/max 都覆盖
   全域 —— 于是 4 个分片全都要打开。
   对比：如果按**时间/自增 ID/行号**这类有序键查，parquet 能直接跳过 95% 的行组。
   **"过滤字段是不是有序的"，决定了谓词下推有没有用。**

③ 所以 `RowTextFetcher`（按行号定位分片）只解决了 1/4：
   它知道该去哪个分片找，但进了分片还是要全扫。实测仍然 1.9 s。

============================================================================
设计：侧车（sidecar）随机访问存储
============================================================================
把 4 列原样导出一份**为点查优化**的副本：

    chunks_text.jsonl   每行一条 JSON（id/title/section/text），UTF-8
    chunks_text.off.npy int64 偏移数组，长度 N+1

取第 i 行 = `mmap[off[i] : off[i+1]]` —— **O(1)，不解压、不扫描、不依赖文件系统缓存**。
外加 mmap 的两个好处：① 不需要把 2.1 GB 读进内存（虚拟内存 + 页缓存，进程 RSS 几乎不涨）；
② 多次查询之间自动复用页缓存，且**冷启动也只有几十毫秒**。

⚠️ 体积没变小（还是约 2.1 GB），**变的是布局**。
   这是本步最该记住的一句话：
   **同样的字节数，"按行放"和"按列放"的取数代价能差两个数量级。**
   用 2.1 GB 磁盘换 1.9 s 的确定性延迟，这笔账在本地服务里非常划算。

⚠️ 一致性契约：侧车是**派生产物**，parquet 才是唯一真相。
   所以必须有 `--verify`：随机抽 N 行，逐字段与 parquet 比对。
   派生数据不做校验 = 埋一颗"输出对了但数据是旧的"的雷。

用法
----
    # 建（约 2~5 分钟，读 2.1 GB + 写 2.1 GB；跑之前关掉 Gradio）
    python src/fetch_store.py --build
    # 自检：随机抽 300 行，与 parquet 逐字段比对
    python src/fetch_store.py --verify 300
    # 看体积与行数契约
    python src/fetch_store.py --info
"""
from __future__ import annotations

import argparse
import json
import mmap
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))

# 与 search_vector.fetch_texts 取的列保持一致 —— 少一列都能让上层静默拿不到字段
COLS = ["chunk_id", "title", "section", "chunk_text"]

STORE = PROJECT / "data" / "index" / "fetch_store"
BLOB = STORE / "chunks_text.jsonl"
OFFS = STORE / "chunks_text.off.npy"


def list_parts(parquet: Path) -> list[Path]:
    """
    列出 parquet 数据集的所有分片，**按文件名排序**。

    顺序有语义：行号契约（ids.txt 第 i 行 == 第 i 行数据）依赖它。
    排序规则一变，行号全错 —— 所以下面 RowTextFetcher 每次取回都要断言。
    """
    import pyarrow.dataset as ds_mod

    return [Path(f) for f in sorted(ds_mod.dataset(str(parquet), format="parquet").files)]


# ==================================================================== 基线：parquet


class RowTextFetcher:
    """
    基线实现：按**行号**回查原文 —— 比按 chunk_id 查快约 4 倍。

    为什么能快：parquet 的 4 个分片是按行序**连续切分**的
    （2026-09-19 实测：829,100 / 829,099 / 829,098 / 829,098，区间首尾相接），
    所以给定全局行号就能唯一定位分片，只需扫 1/4 数据。

    为什么原来的写法慢：`fetch_texts` 用 `chunk_id.isin(...)` 过滤，
    而 chunk_id 是随机哈希 —— 任何分片的 min/max 统计都剪不掉枝，
    只能把 2.17 GB 的 4 个分片全扫一遍（实测 1.2~2.9 s，**比向量检索还慢**）。

    ⚠️ 它**没有**解决"分片内部还是要全扫"这件事，所以实测仍有 1.9 s。
       保留它有两个用途：① 侧车建好前/损坏时的兜底；② 做 A/B 的对照。

    ⚠️ 这个优化建立在「分片按行序连续」这个**假设**上。假设一旦不成立
       （比如换了数据集、重切了分片），结果会**静默错位**。
       所以每批取回后都断言 chunk_id 与行号对得上 ——
       断言的成本是几微秒，换来的是"错了会响"（第 8 步 `_assert_rows` 的同一条纪律）。
    """

    kind = "parquet"

    def __init__(self, parquet: Path, id_list):
        import pyarrow.parquet as pq

        self.id_list = id_list
        self.parts = []
        cum = 0
        for f in list_parts(parquet):
            n = pq.ParquetFile(f).metadata.num_rows
            self.parts.append((Path(f), cum, cum + n))
            cum += n
        self.total = cum
        self.n = cum
        if cum != len(id_list):
            raise SystemExit(
                f"[错误] parquet 共 {cum:,} 行，但 ids.txt 有 {len(id_list):,} 行 —— "
                f"行号契约已破，不能按行号回查（退回 chunk_id 查询或重建索引）")

    def _locate(self, row: int) -> Path:
        for f, lo, hi in self.parts:
            if lo <= row < hi:
                return f
        raise IndexError(f"行号 {row:,} 超出 [0, {self.total:,})")

    def summary(self) -> str:
        return " / ".join(f"{f.name[5:10]}:{hi - lo:,}" for f, lo, hi in self.parts)

    def fetch(self, rows):
        # 延迟导入：让本模块在没有 faiss/torch 的环境里也能跑（诊断脚本要用）
        from search_vector import fetch_texts

        by_part = {}
        for r in rows:
            by_part.setdefault(self._locate(int(r)), []).append(int(r))

        out = {}
        for f, rs in by_part.items():
            cids = [self.id_list[r] for r in rs]
            got = fetch_texts(f, cids)
            for r in rs:                        # ← 错位断言，见类文档
                cid = self.id_list[r]
                rec = got.get(cid)
                if rec is not None and rec["chunk_id"] != cid:
                    raise AssertionError(
                        f"按行号回查错位：row={r:,} 期望 chunk_id={cid}，"
                        f"实际拿到 {rec['chunk_id']} —— 分片不是按行序连续切分的？")
            out.update(got)
        return out


# ==================================================================== 侧车：mmap


class BlobFetcher:
    """
    侧车实现：`mmap` + 偏移数组，**O(1) 点查**，不解压、不扫描。

    取第 i 行的代价与 i 的位置无关（不是"越靠后越慢"）——
    这是"随机访问"和"顺序扫描"的本质区别，也是它比 parquet 快两个数量级的原因。

    ⚠️ 不做"越界就返回 None"这种事：行号越界一定是上游算错了，
       静默返回 None 会让错误一路飘到 prompt 里（少一条上下文，答案歪了，
       而日志里只看到"检索命中 4 条"）。直接抛 IndexError。
    """

    kind = "blob"

    def __init__(self, blob: Path = BLOB, offs: Path = OFFS, id_list=None):
        if not blob.exists() or not offs.exists():
            raise SystemExit(
                f"[错误] 侧车不存在：{blob.name} / {offs.name}\n"
                f"       先建：python src/fetch_store.py --build")
        self.blob_path = Path(blob)
        self.off = np.load(offs, mmap_mode="r")
        self.n = int(len(self.off) - 1)
        self.total = self.n
        self.id_list = id_list
        self._fh = open(blob, "rb")
        self.mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        if id_list is not None and len(id_list) != self.n:
            raise SystemExit(
                f"[错误] 侧车 {self.n:,} 行，但 ids.txt 有 {len(id_list):,} 行 —— "
                f"侧车与索引版本不一致，重建侧车")

    def summary(self) -> str:
        return f"{self.blob_path.name} {self.n:,} 行 / {self.blob_path.stat().st_size / 1e9:.2f} GB"

    def raw(self, row: int) -> bytes:
        """不带解析地拿一行原始字节（诊断脚本要用它量"纯 IO"耗时）。"""
        i = int(row)
        if i < 0 or i >= self.n:
            raise IndexError(f"行号 {i:,} 超出 [0, {self.n:,})")
        return self.mm[int(self.off[i]):int(self.off[i + 1])]

    def payload(self, row: int) -> dict:
        return json.loads(self.raw(row))

    def fetch(self, rows):
        out = {}
        mm = self.mm
        off = self.off
        loads = json.loads
        for r in rows:
            # 错位断言：与 RowTextFetcher 同一条纪律 —— 契约破了要响，不能静默错位
            p = loads(mm[int(off[int(r)]):int(off[int(r) + 1])])
            if self.id_list is not None and p["id"] != self.id_list[int(r)]:
                raise AssertionError(
                    f"侧车错位：row={int(r):,} 期望 chunk_id={self.id_list[int(r)]}，"
                    f"实际拿到 {p['id']} —— 侧车与 ids.txt 不是同一次构建的？")
            out[p["id"]] = {"chunk_id": p["id"], "title": p["title"],
                            "section": p["section"], "chunk_text": p["text"]}
        return out

    def close(self):
        try:
            self.mm.close()
            self._fh.close()
        except Exception:
            pass


# ==================================================================== 构建与自检


def build(parquet: Path, blob: Path = BLOB, offs: Path = OFFS,
          batch_rows: int = 100_000, id_list=None) -> None:
    """
    流式导出侧车 —— 一次顺序扫 parquet，边读边写（**绝不攒到最后一次性写**）。

    为什么必须边读边写：本仓库有过一次教训（本地生成 13.6 分钟、129 条答案
    因为最后一次性落盘时崩掉而全丢）。长跑任务的 IO 必须增量。
    """
    import pyarrow.parquet as pq

    parts = list_parts(parquet)
    blob.parent.mkdir(parents=True, exist_ok=True)
    tmp = blob.with_suffix(".jsonl.tmp")
    off_acc = np.empty(1, dtype=np.int64)
    pos = 0
    rows_done = 0
    t0 = time.time()

    with open(tmp, "wb", buffering=1 << 22) as fo:
        offs_list = [0]
        for f in parts:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=batch_rows, columns=COLS):
                d = batch.to_pydict()
                n = len(d["chunk_id"])
                buf = bytearray()
                for i in range(n):
                    line = json.dumps(
                        {"id": d["chunk_id"][i], "title": d["title"][i],
                         "section": d["section"][i], "text": d["chunk_text"][i]},
                        ensure_ascii=False, separators=(",", ":"))
                    b = line.encode("utf-8") + b"\n"
                    buf += b
                    pos += len(b)
                    offs_list.append(pos)
                fo.write(buf)
                rows_done += n
                if rows_done % 500_000 < batch_rows:
                    el = time.time() - t0
                    print(f"[建侧车] {rows_done:,} 行 / {pos / 1e9:.2f} GB "
                          f"/ {el:.0f}s（{rows_done / max(el, 1e-9) / 1000:.0f}k 行/s）",
                          flush=True)
            print(f"[建侧车] 分片 {f.name[:14]} 完成（累计 {rows_done:,} 行）", flush=True)

    off_acc = np.asarray(offs_list, dtype=np.int64)
    del offs_list

    # 行数契约：侧车行数必须等于 ids.txt 行数，否则一切按行号的取数都是错的
    if id_list is not None and rows_done != len(id_list):
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"[错误] 侧车导出 {rows_done:,} 行，但 ids.txt 有 {len(id_list):,} 行 —— "
            f"先别用，检查 parquet 与索引是不是同一批数据")

    # 原子替换：先写 .tmp，校验通过再改名 —— 中途失败不会留下"半截但看起来能用"的侧车
    #
    # ⚠️ 侧车是**一对文件**（内容 + 偏移），两次 replace 之间仍有极小的窗口。
    #    没有引入"构建号配对"，因为兜底已经够用：
    #      · 行数契约（开侧车时 n 必须等于 ids.txt 行数）
    #      · 每次取回的错位断言（chunk_id 必须与行号对得上）
    #    残留风险被**响亮地**检测到，而不是静默错位 —— 这才是底线。
    #
    # ⚠️ Windows 特有陷阱：mmap 打开的文件**不能被改名/删除**（占用锁）。
    #    所以重建侧车时必须先关掉正在用它的进程（Gradio 是最常忘的那个）。
    tmp_off = offs.with_suffix(".off.tmp.npy")
    np.save(tmp_off, off_acc)
    try:
        tmp.replace(blob)
        tmp_off.replace(offs)
    except PermissionError as e:
        raise SystemExit(
            f"[错误] 侧车正被其它进程占用，无法替换：{e}\n"
            f"       最常见的原因：**Gradio 还开着**（它 mmap 了侧车）。\n"
            f"       关掉它再跑。注意此时侧车可能处于「半新半旧」状态，\n"
            f"       重跑一次 --build，或先 --verify 300 确认再使用。") from None
    el = time.time() - t0
    print(f"[建侧车] 完成：{rows_done:,} 行 / 偏移 {off_acc.nbytes / 1e6:.1f} MB / "
          f"{blob.stat().st_size / 1e9:.2f} GB / 耗时 {el:.0f}s")
    print(f"[建侧车] 下一步必须自检：python src/fetch_store.py --verify 300")


def verify(parquet: Path, ids_path: Path, n: int = 300, seed: int = 0) -> int:
    """
    随机抽 n 行，两条路逐字段比对（**这是侧车唯一的验收方式**）。

    比什么：不只是"能不能取到"，而是 title / section / chunk_text 逐字节相等。
    只比 chunk_id 是不够的 —— 错位一行也能取到一个合法的 chunk_id。
    """
    id_list = read_ids(ids_path)
    a = RowTextFetcher(parquet, id_list)
    b = BlobFetcher(id_list=id_list)
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, b.n, size=n).tolist()

    ta = time.time()
    ga = a.fetch(rows)
    t_a = time.time() - ta
    tb = time.time()
    gb = b.fetch(rows)
    t_b = time.time() - tb

    bad = 0
    for r in rows:
        cid = id_list[int(r)]
        ra, rb = ga.get(cid), gb.get(cid)
        if ra is None or rb is None:
            print(f"  ✗ row={r:,} 一侧取不到：parquet={ra is not None} blob={rb is not None}")
            bad += 1
            continue
        for k in ("chunk_id", "title", "section", "chunk_text"):
            va, vb = ra.get(k), rb.get(k)
            if (va or "") != (vb or ""):
                print(f"  ✗ row={r:,} 字段 {k} 不一致："
                      f"parquet={str(va)[:60]!r} vs blob={str(vb)[:60]!r}")
                bad += 1
                break
    print(f"[自检] 抽样 {n} 行 -> 不一致 {bad} 行")
    print(f"[自检] 同批耗时：parquet {t_a * 1000:.0f} ms  vs  侧车 {t_b * 1000:.1f} ms"
          f"  （{t_a / max(t_b, 1e-9):.0f}×）")
    return 1 if bad else 0


def read_ids(ids_path: Path) -> list[str]:
    """读 ids.txt（约 331 万行）。行号契约的另一半，必须和 parquet 一起验。"""
    with open(ids_path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


# ==================================================================== 工厂


_FETCHER = None


def open_fetcher(parquet: Path, id_list, mode: str = "auto"):
    """
    mode: auto / blob / parquet

    auto 的取舍：侧车在就用侧车（快两个数量级），不在就退回 parquet 并**显式提示**
    怎么建 —— 静默退回是最坏的选择：你会以为优化生效了，其实没有。
    """
    if mode == "parquet":
        return RowTextFetcher(parquet, id_list)
    if mode == "blob":
        f = BlobFetcher(id_list=id_list)
        print(f"[取原文] 侧车随机访问：{f.summary()}")
        return f
    if BLOB.exists() and OFFS.exists():
        f = BlobFetcher(id_list=id_list)
        print(f"[取原文] 侧车随机访问：{f.summary()}")
        return f
    print("[取原文] ⚠️ 未找到侧车，退回 parquet 全扫（约 1.9 s/次）。"
          "建议先跑：python src/fetch_store.py --build")
    return RowTextFetcher(parquet, id_list)


def get_fetcher(id_list, parquet: Path, mode: str = "auto"):
    """全局缓存一个 fetcher（分片行数/偏移数组只需算一次）。"""
    global _FETCHER
    if _FETCHER is None:
        _FETCHER = open_fetcher(parquet, id_list, mode)
        if _FETCHER.kind == "parquet":
            print(f"[取原文] parquet 按行号定位分片：{_FETCHER.summary()}"
                  f"（共 {_FETCHER.total:,} 行，只需扫 1/{len(_FETCHER.parts)}）")
    return _FETCHER


# ==================================================================== CLI


def main() -> int:
    ap = argparse.ArgumentParser(description="取原文侧车：构建 / 自检 / 信息")
    ap.add_argument("--parquet", default=str(PROJECT / "data" / "processed" / "chunks.parquet"))
    ap.add_argument("--ids", default=str(PROJECT / "data" / "index" / "bge-large-zh-v1.5" / "ids.txt"))
    ap.add_argument("--blob", default=str(BLOB))
    ap.add_argument("--offs", default=str(OFFS))
    ap.add_argument("--build", action="store_true", help="构建侧车（约 2~5 分钟）")
    ap.add_argument("--verify", type=int, default=0, metavar="N", help="抽样 N 行与 parquet 比对")
    ap.add_argument("--info", action="store_true", help="打印侧车体积与行数契约")
    ap.add_argument("--batch-rows", type=int, default=100_000)
    args = ap.parse_args()

    parquet, ids_path = Path(args.parquet), Path(args.ids)
    blob, offs = Path(args.blob), Path(args.offs)

    if args.info:
        if not blob.exists():
            print(f"[信息] 侧车不存在：{blob}")
            return 1
        id_list = read_ids(ids_path)
        off = np.load(offs, mmap_mode="r")
        print(f"[信息] 侧车 {blob}  {blob.stat().st_size / 1e9:.2f} GB")
        print(f"[信息] 偏移 {offs}  {off.nbytes / 1e6:.1f} MB（{len(off) - 1:,} 行 + 1 个哨兵）")
        print(f"[信息] ids.txt {len(id_list):,} 行 -> "
              f"{'✅ 行数契约一致' if len(off) - 1 == len(id_list) else '❌ 行数不一致，需重建'}")
        return 0

    if args.build:
        id_list = read_ids(ids_path)
        print(f"[建侧车] 源：{parquet}（{len(id_list):,} 行契约）")
        build(parquet, blob, offs, args.batch_rows, id_list)
        return 0

    if args.verify:
        return verify(parquet, ids_path, args.verify)

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
