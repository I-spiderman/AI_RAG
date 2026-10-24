"""
VectorStore：两层 ChromaDB 向量库
- categories 层：类别描述向量，用于入库自动分类与查询自动路由
- chunks 层：分块文本 + 向量 + 元数据，每条带 category 字段
"""

import logging
import os
from typing import List, Dict, Callable

import chromadb

from services.vector.embedding_service import ChromaEmbeddingFunction

logger = logging.getLogger(__name__)


class VectorStore:
    """
    两层向量库（基于 ChromaDB）：

    第一层 categories（类别层）：
        每个类别一条向量，由类别的描述文字 embedding 而来，
        用于入库时自动分类、查询时自动路由。

    第二层 chunks（文本块层）：
        所有分块文本 + 向量 + 元数据，每条都带 category 字段。
    """
    def __init__(self, embed_function: Callable[[List[str]], List[List[float]]] = None):
        # 数据库持久化目录：项目根 data/chroma_db（文件在 services/vector/ 下，上三级为项目根）
        persist_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "chroma_db",
        )
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embed_function = embed_function

        # 第二层：文本块集合，用 cosine 距离（因为embedding是归一化的后的向量）
        self.chunks_col = self.client.get_or_create_collection(
            name="chunks",
            metadata={"hnsw:space": "cosine"},
            embedding_function=ChromaEmbeddingFunction()
        )
        # 第一层：类别集合
        self.categories_col = self.client.get_or_create_collection(
            name="categories",
            metadata={"hnsw:space": "cosine"},
            embedding_function=ChromaEmbeddingFunction()
        )
        logger.info(
            f"向量库初始化完成: chunks={self.chunks_col.count()} 条, "
            f"categories={self.categories_col.count()} 个类别"
        )

    def seed_categories(self, categories: Dict[str, str]) -> None:
        """
        初始化或刷新类别层。把预定义类别的描述文字转成向量存起来。
        只在首次启动、或修改了类别清单时调用一次。

        :param categories: {"人事行政": "考勤、请假、报销等制度", ...}
        """
        if self.embed_function is None:
            raise ValueError("初始化类别需要传入 embed_function（embedding_service.embed_texts）")

        names = list(categories.keys())
        descs = list(categories.values())
        vectors = self.embed_function(descs)

        self.categories_col.upsert(
            ids=names,
            documents=descs,
            metadatas=[{"category": name} for name in names],
            embeddings=vectors
        )
        logger.info(f"已初始化{len(names)}个类别：{names}")

    def classify(self, embedding: List[float]) -> str:
        """
        自动将文档归类
        内部方法：给一个 chunk 的向量，返回最匹配的类别名。
        原理：拿这个向量去 categories 集合做最近邻搜索。
        """
        if self.categories_col.count() == 0:
            return "未分类"
        results = self.categories_col.query(
            query_embeddings=[embedding],
            n_results=1,
        )
        if results["ids"] and results["ids"][0]:
            return results["ids"][0][0]

        return "未分类"

    def add_chunks(self, chunks: List[Dict], category: str = None) -> None:
        """
        把分块向量写入第二层。

        :param chunks: generate_embedding 输出的列表，每项含 text / meta / embedding
        :param category:
            - 传入字符串：强制所有 chunk 归入该类别
            - 不传（None）：每个 chunk 自动根据向量相似度归类
        """
        if not chunks:
            logger.warning("add_chunks: 输入的 chunks 列表为空")
            return

        ids = []
        documents = []
        metadatas = []
        embeddings = []
        auto_count = {}

        for idx, chunk in enumerate(chunks):
            doc_id = chunk["meta"].get("doc_id", "unknown")
            ids.append(f"{doc_id}_chunk_{idx}")
            documents.append(chunk["text"])
            embeddings.append(chunk["embedding"])
            meta = dict(chunk["meta"])
            cat = category or self.classify(chunk["embedding"])
            meta["category"] = cat
            metadatas.append(meta)

            auto_count[cat] = auto_count.get(cat, 0) + 1

        # upsert：相同 id 自动覆盖，不会重复报错
        self.chunks_col.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        logger.info(f"已写入{len(chunks)}条分块，自动分类{auto_count}")

    def query(self, query_embedding: List[float], top_k: int = 5, category: str = None) -> List[Dict]:
        """
        根据问题向量检索最相关的文本块。

        :param query_embedding: 用户问题的向量
        :param top_k: 返回前几条
        :param category:
            - 指定：在该类别下检索
            - 不指定：先自动判断问题属于哪个类别，再在该类别下检索
        :return: [{"text":..., "meta":..., "distance":...}, ...]
        """
        # 没指定类别就先自动路由
        if category is None:
            category = self.classify(query_embedding)
            if category and category != "未分类":
                logger.info(f"自动判断问题属于类别：{category}")

        # 构造 chroma 的过滤条件
        where = {"category": category} if category and category != "未分类" else None

        results = self.chunks_col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )

        # 整理返回结果
        hits = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                hits.append({
                    "text": results["documents"][0][i],
                    "meta": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if results["distances"] else None,
                })
        return hits

    def list_categories(self) -> List[str]:
        """列出当前所有已注册的类别名"""
        data = self.categories_col.get()
        return [m["category"] for m in data["metadatas"]]

    def delete_by_doc(self, doc_id: str) -> None:
        """按 doc_id 删除某个文档的所有 chunk（重新上传时先调这个）"""
        self.chunks_col.delete(where={"doc_id": doc_id})
        logger.info(f"已删除文档 {doc_id} 的所有 chunk")

