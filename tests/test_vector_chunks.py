"""
向量库文本块查看器：展示已存入 Chroma（chunks 集合）的具体文本块内容与元数据。

三种用法：
    python tests/test_vector_chunks.py                  # 读取并展示向量库里全部 chunk
    python tests/test_vector_chunks.py --doc DOC_ID     # 只看某个文档（doc_id）的 chunk
    python tests/test_vector_chunks.py --preview [路径]  # 对本地文档跑切块流程预览（不依赖已存库/embedding 模型）

说明：
    - 主模式连接项目 data/chroma_db，逐条打印每个 chunk 的
      id / 类别(category) / 章节 / 小节 / 位置(location) / 文本全文。
    - 若库为空（count=0），会提示先入库，并建议用 --preview 对本地文档预览切块结果。
    - 这样即可看到「最终写进向量库的文本块」长什么样，包括表格被转成的 Markdown。
"""

import os
import sys
import argparse

# 让脚本能 import backend 包（与 tests/test_intent.py 一致）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.services.vector.vector_store import VectorStore
from backend.services.document.file_process import FileParser, DocumentChunker

LINE = "─" * 70


def dump_stored_chunks(doc_id: str = None) -> None:
    """连接真实向量库，把 chunks 集合里的文本块全部打印出来。"""
    vs = VectorStore()
    total = vs.chunks_col.count()
    print(f"== 向量库 chunks 集合总条数：{total} ==")
    if total == 0:
        print("（库为空，请先通过管理端上传文档，或运行 python tests/smoke_admin.py 入库）")
        print("    也可直接预览切块：python tests/test_vector_chunks.py --preview tests/data/test.txt")
        return

    # 只取文档与元数据，不取 embedding 向量（太大没必要打印）
    where = {"doc_id": doc_id} if doc_id else None
    data = vs.chunks_col.get(include=["documents", "metadatas"], where=where)
    ids = data.get("ids") or []
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []

    print(f"== 展示 {len(ids)} 条 chunk" + (f"（doc_id={doc_id}）" if doc_id else "）") + " ==")
    for i, (cid, text, meta) in enumerate(zip(ids, docs, metas), start=1):
        cat = meta.get("category", "?")
        chap = meta.get("chapter_title", "")
        sub = meta.get("sub_title", "")
        loc = meta.get("location", "")
        print(LINE)
        print(f"[{i}/{len(ids)}] id={cid}")
        print(f"    类别：{cat}  | 章节：{chap}  | 小节：{sub}")
        print(f"    位置：{loc}")
        print(f"    文本块内容（{len(text)} 字符）：")
        print(text)
    print(LINE)
    print(f"共展示 {len(ids)} 条 chunk")


def preview_document(path: str) -> None:
    """对一个本地文档跑 FileParser + DocumentChunker，预览切块结果（不入库、不调模型）。"""
    abspath = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.exists(abspath):
        print(f"[ERROR] 文件不存在：{abspath}")
        return

    doc_id = os.path.splitext(os.path.basename(abspath))[0]
    doc_name = os.path.basename(abspath)
    print(f"== 预览切块（不入库、不调用 embedding 模型）== 文件：{abspath}")
    fragments = FileParser.parse(abspath)
    print(f"    原始片段数：{len(fragments)}（段落/表格按文档原顺序）")

    chunker = DocumentChunker(chunk_max_size=500, chunk_overlap=100)
    chunks = chunker.chunk_document(fragments, doc_id, doc_name)
    print(f"    切块后 chunk 数：{len(chunks)}")
    for i, c in enumerate(chunks, start=1):
        t = c["text"]
        meta = c["meta"]
        print(LINE)
        print(f"[{i}/{len(chunks)}] 章节：{meta.get('chapter_title', '')}  | "
              f"位置：{meta.get('location', '')}")
        print(f"    内容（{len(t)} 字符）：")
        print(t)
    print(LINE)
    print(f"共预览 {len(chunks)} 个 chunk")


def main() -> None:
    p = argparse.ArgumentParser(description="向量库文本块查看器")
    p.add_argument("--doc", help="只看某个 doc_id 的 chunk（主模式下的过滤）")
    p.add_argument(
        "--preview", nargs="?", const=None,
        help="对本地文档跑切块预览（不依赖已存库）。可跟文件路径，缺省用 tests/data/test.txt",
    )
    args = p.parse_args()

    if args.preview is not None:
        # 传了 --preview 但没给路径 → 用默认示例文档
        path = args.preview if args.preview else os.path.join(ROOT, "tests", "data", "test.txt")
        preview_document(path)
    else:
        dump_stored_chunks(args.doc)


if __name__ == "__main__":
    main()
