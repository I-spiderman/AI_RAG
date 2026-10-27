"""应用依赖：RAGService 全局单例（FastAPI lifespan 启动时初始化一次）

RAGService 初始化要加载 embedding 模型、连 MySQL/Redis，耗时数秒，
不能每个请求都初始化，所以用 lifespan 做一次性懒加载。
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.services.core.rag_service import RAGService

logger = logging.getLogger(__name__)

_rag: RAGService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期：启动时初始化 RAGService（加载向量模型，可能需要数秒）"""
    global _rag
    logger.info("初始化 RAGService（加载向量库/模型，可能需要数秒）...")
    _rag = RAGService()
    logger.info("RAGService 就绪")
    yield
    _rag = None


def get_rag() -> RAGService:
    """获取全局 RAGService 实例（lifespan 启动完成后非 None）"""
    return _rag
