"""
混合检索 + 重排模块：
1. 粗排：向量检索（语义）+ 关键词检索（BM25）两路召回，RRF 融合排序
2. 精排：召回池交 Reranker（交叉编码器）打分过滤，返回最相关的 final_top_k 条
3. build_context：把检索结果拼成 prompt 参考资料文本（带相关度分数）
"""

import logging
import time
from typing import List, Dict
import re

from backend.services.vector.embedding_service import embed_texts
from backend.services.retrieval.reranker import Reranker
from backend.services.vector.vector_store import VectorStore
from rank_bm25 import BM25Okapi
import jieba

logger = logging.getLogger(__name__)

class Retriever:
    """混合检索：语义向量检索 + 关键词检索（BM25），两路通过RRF融合排序"""

    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store
        self.reranker = Reranker()
        # BM25 索引缓存：集合条数/内容变化时自动失效重建
        self._bm25_cache = {"sig":None, "index":None, "rows":None}

    # ---- BM25 关键词召回（词法匹配，与语义向量互补）----
    @staticmethod
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """中文走 jieba 分词，英文/数字保留原词；统一小写、去空白与纯标点。"""
        text = re.sub(r"\s+", " ", text or "")
        toks = []
        if jieba:
            for w in jieba.cut(text):
                w = w.strip().lower()
                if w and not re.fullmatch(r"[\W_]+", w):
                    toks.append(w)
        else:
            toks = [w.strip().lower() for w in re.split(r"[\s\W_]+", text) if w]
        return toks

    def _bm25_search(self, question: str, where: dict, recall_k: int) -> List[Dict]:
        """从当前类别范围(where)内取语料，用 BM25Okapi 做词法打分召回。"""
        # 注意：Chroma 的 get() 不允许 include 里写 "ids"（会抛 ValueError），ids 总是自动返回
        data = self.vs.chunks_col.get(where=where, include=["documents", "metadatas"])
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        if len(ids) <= 1:
            return []  # 语料太少，BM25 无意义
        # 集合内容变化（新增/重灌文档）时重建索引；否则复用进程内缓存
        sig = (len(ids), hash("".join(ids) + "".join(docs)))
        if self._bm25_cache["sig"] != sig:
            tokenized = [self._tokenize(d) for d in docs]
            self._bm25_cache = {
                "sig": sig,
                "index": BM25Okapi(tokenized),
                "ids": ids, "docs": docs, "metas": metas,
            }
        scores = self._bm25_cache["index"].get_scores(self._tokenize(question))
        ranked = sorted(
            zip(self._bm25_cache["ids"], self._bm25_cache["docs"],
                self._bm25_cache["metas"], scores),
            key=lambda x: x[3], reverse=True,
        )
        return [
            {"id": r[0], "text": r[1], "meta": r[2], "score": float(r[3])}
            for r in ranked[:recall_k]
        ]

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

        # 未指定类别，自动识别类别（相似度不足时由云端 LLM 兜底，失败降级本地）
        if category is None:
            cls = self.vs.classify(query_vec, text=question)
            cat = cls["category"]
            top2 = cls["top2"]
            if cat and cat != "未分类":
                logger.info(f"自动识别到类别：{cat}（{cls['method']}）")
            elif top2:
                logger.info(f"分类不确定，按相似度前二扩大检索：{top2}")
        else:
            cat = category
            top2 = None

        # 构造过滤条件：确定单一类别则精确过滤；否则前二类扩大范围
        if cat and cat != "未分类":
            where = {"category": cat}
        elif top2:
            where = {"category": {"$in": top2}}
        else:
            where = None

        # 语义向量检索
        vector_hits = self.vs.chunks_col.query(
            query_embeddings=[query_vec],
            n_results=recall_k,
            where=where
        )

        # 关键词检索（真·BM25 词法召回，与语义路互补，且与向量路共享 where 范围）
        bm25_hits = self._bm25_search(question, where, recall_k)

        # ---------- RRF 融合（向量 + BM25 两路）----------
        rrf_k = 60
        scores: Dict[str, dict] = {}

        # 向量结果整理成统一结构
        vector_list = []
        for rank in range(len(vector_hits["ids"][0])):
            vector_list.append({
                "id": vector_hits["ids"][0][rank],
                "text": vector_hits["documents"][0][rank],
                "meta": vector_hits["metadatas"][0][rank],
                "dist": vector_hits["distances"][0][rank],
            })

        def _add_path(hits, rank_key):
            for rank, hit in enumerate(hits):
                cid = hit["id"]
                s = scores.get(cid)
                if s is None:
                    s = {"score": 0.0, "text": hit["text"], "meta": hit["meta"],
                         "dist": hit.get("dist"), "vector_rank": None, "bm25_rank": None}
                    scores[cid] = s
                s["score"] += 1 / (rrf_k + rank + 1)
                s[rank_key] = rank + 1

        _add_path(vector_list, "vector_rank")
        _add_path(bm25_hits, "bm25_rank")

        # 按融合分数排序
        merged = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        candidates = []
        for m in merged:
            if len(candidates) >= recall_k:
                break
            # 距离过滤：仅对「未命中关键词」的纯向量结果生效
            if m["bm25_rank"] is None and m["dist"] is not None and m["dist"] > max_distance:
                continue
            candidates.append({
                "text": m["text"],
                "meta": m["meta"],
                "distance": m["dist"],
                "rrf_score": m["score"]
            })
        logger.info(
            f"混合检索召回池 {len(candidates)} 条"
            f"（向量{len(vector_list)} + BM25{len(bm25_hits)}），"
            f"耗时 {time.time() - start_time:.4f}秒"
        )

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
