"""聊天路由：POST /api/chat → SSE 流式问答

- 认证：Bearer token（复用 auth 的 token 解析，取真实 user_id）
- 四路分流（知识问答/闲聊/工单/转人工）由 RAGService.ask_stream 内部完成
- 响应为 text/event-stream，data 帧格式：
    data: {"type":"token","content":"..."}  逐块文本
    data: {"type":"done"}                   正常结束
    data: {"type":"error","content":"..."}  异常
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app.api.auth import _resolve_user
from app.db import update_title_if_default
from app.deps import get_rag

router = APIRouter(prefix="/api/chat", tags=["chat"])
_bearer = HTTPBearer(auto_error=False)
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: str
    session_id: str


@router.post("")
def chat(req: ChatRequest, cred: HTTPAuthorizationCredentials = Depends(_bearer)):
    """SSE 流式问答：token 解析 user_id → RAGService.ask_stream → 逐块推送"""
    user_id, username = _resolve_user(cred)
    rag = get_rag()
    if rag is None:
        raise HTTPException(status_code=503, detail="服务初始化中，请稍后重试")

    # 新会话第一条消息 → 自动生成会话标题（失败不影响问答主流程）
    try:
        update_title_if_default(req.session_id, req.question)
    except Exception:
        logger.warning("会话标题更新失败（忽略）", exc_info=True)

    def event_stream():
        yield _frame({"type": "status", "content": "正在思考..."})
        try:
            for piece in rag.ask_stream(req.question, req.session_id, user_id):
                yield _frame({"type": "token", "content": piece})
            yield _frame({"type": "done"})
        except Exception as e:
            logger.exception(f"问答流异常：{e}")
            yield _frame({"type": "error", "content": "（服务异常，请稍后重试）"})

    return StreamingResponse(event_stream(),
                             media_type="text/event-stream; charset=utf-8",
                             headers={
                                 "Cache-Control": "no-cache",          # 禁止中间缓存
                                 "X-Accel-Buffering": "no",            # 关键：让 nginx 不缓冲 SSE
                                 "Connection": "keep-alive",           # 保持长连接（修正原 Contention 拼写错误）
                             })

def _frame(obj: dict) -> str:
    """SSE data 帧：data: {json}\n\n"""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"
