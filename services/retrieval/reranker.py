"""
Reranker 重排模块：交叉编码器精排

位置：混合检索（粗排：向量+BM25+RRF）之后、Prompt 组装之前。
作用：把 (问题, 候选片段) 逐对拼起来喂进模型，输出真实相关性分数，
      把最相关的顶到前面，过滤低相关片段——弥补粗排只看排名不看内容的缺陷。

模型：Qwen3-Reranker-0.6B（本地，与 Qwen3-Embedding-0.6B 同系列同量级）
分数：CrossEncoder 输出 logits → sigmoid 转成 0~1 概率，越高越相关
阈值：默认 0.3 —— 重排后最高分仍低于阈值的候选整体视为不相关，返回空
降级：模型加载失败由 Retriever 捕获，降级返回召回原序（系统不中断）
"""

import logging
import math
import os
import torch
from typing import List, Dict

logger = logging.getLogger(__name__)

def _sigmoid(x: float) -> float:
    """logits → 0~1 概率"""
    return 1.0 / (1.0 + math.exp(-x))

class Reranker:
    """交叉编码器重排器（模型懒加载：第一次 rerank 时才加载，不拖慢启动）"""
    def __init__(self, model_path: str = None, device: str = None, threshold: float = 0.3):
        # 模型目录在项目根 models/（文件在 services/retrieval/ 下，上三级为项目根）
        self.model_path = model_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "models/Qwen3-Reranker-0.6B",
        )
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_path, device=self.device)
            logger.info(f"已加载Reranker模型：{self.model_path}，设备：{self.device}")
        return self._model

    def rerank(self, query: str, candidates: List[Dict], final_top_k: int = 5) -> List[Dict]:
        """
        对候选池精排：
        1. 每对 (query, 候选text) 过交叉编码器 → logits → sigmoid 概率
        2. 过滤低于 threshold 的候选（整体不相关直接返回空）
        3. 按分数降序取前 final_top_k 条，每项写回 rerank_score 字段
        """
        if not candidates:
            return []
        model = self._get_model()

        pairs = [(query, c["text"]) for c in candidates]
        logits = model.predict(pairs, batch_size=32)

        for cand, logit in zip(candidates, logits):
            cand["rerank_score"] = round(_sigmoid(float(logit)), 4)

        kept = [c for c in candidates if c["rerank_score"] >= self.threshold]
        if not kept:
            best = max(c["rerank_score"] for c in candidates)
            logger.warning(
                f"重排后无候选达到阈值 {self.threshold}（最高 {best:.4f}），返回空"
            )
            return []
        kept.sort(key=lambda x: x["rerank_score"], reverse=True)
        return kept[:final_top_k]
