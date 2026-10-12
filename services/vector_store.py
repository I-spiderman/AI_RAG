import os
import logging
from typing import List, Dict, Callable
import chromadb
logger = logging.getLogger(__name__)

from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
class _ChromaEmbeddingFunction(EmbeddingFunction):

    def __call__(self, input: Documents) -> Embeddings:
        from services.embedding_service import embed_texts
        return embed_texts(input)

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
        # 数据库持久化目录：和本文件同级的 chroma_db 文件夹
        persist_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "chroma_db"
        )
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embed_function = embed_function

        # 第二层：文本块集合，用 cosine 距离（因为embedding是归一化的后的向量）
        self.chunks_col = self.client.get_or_create_collection(
            name="chunks",
            metadata={"hnsw:space": "cosine"},
            embedding_function=_ChromaEmbeddingFunction()
        )
        # 第一层：类别集合
        self.categories_col = self.client.get_or_create_collection(
            name="categories",
            metadata={"hnsw:space": "cosine"},
            embedding_function=_ChromaEmbeddingFunction()
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

    def _classify(self, embedding: List[float]) -> str:
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
            cat = category or self._classify(chunk["embedding"])
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
            category = self._classify(query_embedding)
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


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    from embedding_service import embed_texts

    vs = VectorStore(embed_function=embed_texts)
    vs.seed_categories({
        "人事行政": "公司考勤制度、请假流程、年假病假、报销标准、出差审批、工资福利、入职离职、人事行政规定",
        "技术运维": "服务器部署、系统配置、故障排查、命令行操作、重启维护、环境搭建、数据库管理、网络安全",
        "产品FAQ": "产品功能介绍、客户常见问题、收费标准、使用教程、售后服务、账号管理、功能更新",
    })

    # 用真实 embedding，不要用 [0.1]*1024
    test_texts = [
        "年假需提前三天在OA系统提交申请，病假需提供医院开具的诊断证明，部门负责人审批后交人事部备案。",
        "服务器出现故障时使用sudo reboot命令重启系统，重启前需通知运维团队，重启后检查Nginx和数据库服务是否正常启动。",
    ]

    vectors = embed_texts(test_texts)  # 真实向量

    test_chunks = [
        {"text": test_texts[0],
         "meta": {"doc_id": "test_001", "doc_name": "考勤制度.txt"},
         "embedding": vectors[0]},
        {"text": test_texts[1],
         "meta": {"doc_id": "test_002", "doc_name": "运维手册.txt"},
         "embedding": vectors[1]},
    ]

    vs.add_chunks(test_chunks)
    print("所有类别:", vs.list_categories())

