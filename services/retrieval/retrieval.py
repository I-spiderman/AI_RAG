"""
混合检索 + 重排模块：
1. 粗排：向量检索（语义）+ 关键词检索（BM25）两路召回，RRF 融合排序
2. 精排：召回池交 Reranker（交叉编码器）打分过滤，返回最相关的 final_top_k 条
3. build_context：把检索结果拼成 prompt 参考资料文本（带相关度分数）
"""

import logging
import time
from typing import List, Dict

from services.vector.embedding_service import embed_texts
from services.retrieval.reranker import Reranker
from services.vector.vector_store import VectorStore

logger = logging.getLogger(__name__)

class Retriever:
    """混合检索：语义向量检索 + 关键词检索（BM25），两路通过RRF融合排序"""

    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store
        self.reranker = Reranker()

    def retrieve(self, question: str, recall_k: int = 15, final_top_k: int = 5,
                 category: str = None, max_distance: float = 0.8) -> List[Dict]:
        """
        混合检索 + 重排
        :param question: 用户问题
        :param recall_k: 召回池大小：每条检索路的召回量，RRF 融合后保留给 Reranker 的候选条数
        :param final_top_k: 最终返回条数（Reranker 精排后给 LLM 的）
        :param category: 指定类别，不指定则自动路由
        :param max_distance: 距离过滤阈值（仅对未命中关键词的向量结果生效）
        """
        start_time = time.time()
        logger.info(f"收到问题：{question}")
        query_vec = embed_texts([question])[0]

        # 未指定类别，自动识别类别
        if category is None:
            category = self.vs.classify(query_vec)
            if category and category != "未分类":
                logger.info(f"自动识别到类别：{category}")

        where = {"category": category} if category and category != "未分类" else None

        # 语义向量检索
        vector_hits = self.vs.chunks_col.query(
            query_embeddings=[query_vec],
            n_results=recall_k,
            where=where
        )

        # 关键词检索
        # chromaDB 自动创建的索引，用于BM25检索
        text_hits = self.vs.chunks_col.query(
            query_texts=[question],
            n_results=recall_k * 2,
            where=where
        )

        # 融合排序：RRF 加权排序
        # RRF 公式：score = sum(1 / (K + rank)) ,  K 一般取60
        rrf_k = 60
        scores = {}  # id -> {score, data}

        for rank, (doc, dist, meta) in enumerate(zip(
            vector_hits["documents"][0],
            vector_hits["distances"][0],
            vector_hits["metadatas"][0]
        )):
            chunk_id = vector_hits["ids"][0][rank]
            scores[chunk_id] = {
                "score": 1 / (rrf_k + rank + 1),
                "text": doc, "meta": meta, "dist": dist,
                "vector_rank": rank + 1, "text_rank": None
            }

        for rank, (doc, dist, meta) in enumerate(zip(
                text_hits["documents"][0],
                text_hits["distances"][0],
                text_hits["metadatas"][0]
        )):
            chunk_id = text_hits["ids"][0][rank]
            if chunk_id in scores:
                scores[chunk_id]["score"] += 1 / (rrf_k + rank + 1)
                scores[chunk_id]["text_rank"] = rank + 1
            else:
                scores[chunk_id] = {
                    "score": 1 / (rrf_k + rank + 1),
                    "text": doc, "meta": meta, "dist": dist,
                    "vector_rank": None, "text_rank": rank + 1
                }

        # 按融合分数排序
        merged = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        candidates = []
        for m in merged:
            if len(candidates) >= recall_k:
                break
            # 距离过滤：仅对未命中关键词的向量结果生效
            if m["text_rank"] is None and m["dist"] is not None and m["dist"] > max_distance:
                continue
            candidates.append({
                "text": m["text"],
                "meta": m["meta"],
                "distance": m["dist"],
                "rrf_score": m["score"]
            })
        logger.info(f"混合检索召回池 {len(candidates)} 条，耗时 {time.time() - start_time:.4f}秒")

        # ---------- 精排：Reranker ----------
        if not candidates:
            return []
        try:
            hits = self.reranker.rerank(question, candidates, final_top_k)
        except Exception as e:
            # 模型缺失/加载失败等：降级为召回原序（不重排），保证系统可用
            logger.warning(f"重排失败，降级返回召回池前 {final_top_k} 条：{e}")
            hits = candidates[:final_top_k]

        for h in hits:
            s = f"{h['rerank_score']:.4f}" if "rerank_score" in h else "N/A"
            logger.info(f"  精排：相关度{s} | {h['text'][:40]}")
        logger.info(f"重排完成：返回 {len(hits)} 条，检索+重排总耗时 {time.time() - start_time:.4f}秒")
        return hits

    def build_context(self, hits: List[Dict]) -> str:
        """把检索结果拼成 prompt 参考资料文本（带相关度，供 LLM 参考可靠性）"""
        if not hits:
            return ""
        parts = []
        for i, h in enumerate(hits, 1):
            meta = h["meta"]
            source = meta.get("doc_name", "未知文档")
            chapter = meta.get("chapter_title", "未知章节")
            score = h.get("rerank_score")
            score_text = f"，相关度{score:.2f}" if score is not None else ""
            parts.append(f"[参考资料{i}]（来源：{source} {chapter}{score_text}）\n{h['text']}")
        return "\n\n".join(parts)
