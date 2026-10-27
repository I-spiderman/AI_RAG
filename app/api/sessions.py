"""会话路由：多会话登记 / 列表 / 历史还原

- POST /api/sessions            新建会话登记（前端生成 session_id 传入）
- GET  /api/sessions            当前用户的会话列表（最新更新在前）
- GET  /api/sessions/{sid}/messages  历史消息（从 Redis 短期记忆还原：
                                    最近 K 轮完整原文 + 滚动摘要）
全部走 Bearer token 解析 user_id，会话按用户隔离。
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app.api.auth import _resolve_user
from app.db import create_session, list_sessions, session_belongs_to
from app.deps import get_rag

router = APIRouter(prefix="/api/sessions", tags=["sessions"])
_bearer = HTTPBearer(auto_error=False)
logger = logging.getLogger(__name__)


class SessionCreate(BaseModel):
    session_id: str


@router.post("")
def create(sreq: SessionCreate, cred: HTTPAuthorizationCredentials = Depends(_bearer)):
    """登记一个新会话（重复登记幂等）"""
    user_id, _ = _resolve_user(cred)
    create_session(user_id, sreq.session_id)
    return {"session_id": sreq.session_id, "title": "新对话"}


@router.get("")
def list_sess(cred: HTTPAuthorizationCredentials = Depends(_bearer)):
    """当前用户的会话列表"""
    user_id, _ = _resolve_user(cred)
    return list_sessions(user_id)


@router.get("/{session_id}/messages")
def history(session_id: str, cred: HTTPAuthorizationCredentials = Depends(_bearer)):
    """按 session_id 从 Redis 还原对话历史（摘要 + 最近 K 轮原文）"""
    user_id, _ = _resolve_user(cred)
    if not session_belongs_to(user_id, session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    rag = get_rag()
    if rag is None:
        raise HTTPException(status_code=503, detail="服务初始化中，请稍后重试")
    ctx = rag.memory.get_context(session_id)
    return {
        "session_id": session_id,
        "summary": ctx["summary"],
        "messages": ctx["window"],   # [(role, content), ...]
    }
