"""管理员路由：文档上传入库 / 入库历史 / 热门工单 / 删除文档

全部接口要求 Bearer token 且用户 role == 'admin'，否则 403。
"""

import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app import db
from app.api.auth import _resolve_user
from app.deps import get_rag

router = APIRouter(prefix="/api/admin", tags=["admin"])
_bearer = HTTPBearer(auto_error=False)
logger = logging.getLogger(__name__)


def _require_admin(cred: HTTPAuthorizationCredentials):
    """解析 token + 校验管理员身份；非管理员抛 403"""
    user_id, username = _resolve_user(cred)
    user = db.get_user_by_username(username)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user_id, username


def _extract_field(text: str, label: str) -> str:
    """从工单文本里按标签取值，取不到返回空串"""
    for line in text.splitlines():
        if line.startswith(f"{label}："):
            return line[len(label) + 1:].strip()
    return ""


@router.post("/documents")
def upload_doc(file: UploadFile = File(...), cred: HTTPAuthorizationCredentials = Depends(_bearer)):
    """上传文档入库：保存临时 → ingest_file（覆盖旧版本）→ 记录入库日志"""
    user_id, username = _require_admin(cred)
    rag = get_rag()
    if rag is None:
        raise HTTPException(status_code=503, detail="服务初始化中，请稍后重试")

    file_name = file.filename or "upload.txt"
    doc_id = Path(file_name).stem  # 文件名去扩展名作为 doc_id

    suffix = Path(file_name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".txt")
    try:
        tmp.write(file.file.read())
        tmp.close()
        # 同 doc_id 覆盖：先清旧 chunk，再入库（避免旧 chunk 残留）
        rag.vs.delete_by_doc(doc_id)
        n = rag.ingest_file(tmp.name, doc_id=doc_id, doc_name=file_name)
        if n == 0:
            raise HTTPException(status_code=400, detail="文档解析为空，请检查文件内容")
        db.log_ingest(doc_id, file_name, n, username)
        logger.info(f"[管理员 {username}] 文档入库：{file_name}（{n} 块）")
        return {"doc_id": doc_id, "doc_name": file_name, "chunk_count": n}
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


@router.get("/documents")
def list_docs(cred: HTTPAuthorizationCredentials = Depends(_bearer)):
    """历史入库情况：Chroma chunks 按 doc_id 聚合 + 日志表补入库时间"""
    user_id, username = _require_admin(cred)
    rag = get_rag()
    if rag is None:
        raise HTTPException(status_code=503, detail="服务初始化中，请稍后重试")

    data = rag.vs.chunks_col.get(include=["metadatas"])
    metas = data.get("metadatas") or []
    docs = {}
    for m in metas:
        did = m.get("doc_id", "unknown")
        entry = docs.setdefault(did, {
            "doc_id": did,
            "doc_name": m.get("doc_name", ""),
            "chunk_count": 0,
            "categories": set(),
        })
        entry["chunk_count"] += 1
        if m.get("category"):
            entry["categories"].add(m["category"])

    logs = {l["doc_id"]: l for l in db.list_ingest_logs()}
    result = []
    for did, e in docs.items():
        log = logs.get(did)
        result.append({
            "doc_id": did,
            "doc_name": e["doc_name"],
            "chunk_count": e["chunk_count"],
            "categories": sorted(e["categories"]),
            "created_at": log["created_at"] if log else "—",
            "operator": log["operator"] if log else "",
        })
    result.sort(key=lambda x: x["doc_name"])
    return result


@router.get("/hot-tickets")
def hot_tickets(cred: HTTPAuthorizationCredentials = Depends(_bearer)):
    """当前最热门工单：全量扫描工单库，按累计出现次数 occurrences 降序取 Top 5"""
    user_id, username = _require_admin(cred)
    rag = get_rag()
    if rag is None:
        raise HTTPException(status_code=503, detail="服务初始化中，请稍后重试")

    data = rag.tm.col.get(include=["documents", "metadatas"])
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []
    rows = []
    for text, meta in zip(docs, metas):
        question = _extract_field(text, "问题") or "（未记录问题）"
        rows.append({
            "question": question,
            "category": meta.get("category", "未分类"),
            "occurrences": meta.get("occurrences", 1),
            "last_occurred_at": meta.get("last_occurred_at", ""),
            "resolution": (meta.get("resolution") or "")[:60],
        })
    rows.sort(key=lambda r: r["occurrences"], reverse=True)
    return rows[:5]


@router.delete("/documents/{doc_id}")
def delete_doc(doc_id: str, cred: HTTPAuthorizationCredentials = Depends(_bearer)):
    """删除文档：清 Chroma chunks + 删入库日志（幂等）"""
    user_id, username = _require_admin(cred)
    rag = get_rag()
    if rag is None:
        raise HTTPException(status_code=503, detail="服务初始化中，请稍后重试")

    rag.vs.delete_by_doc(doc_id)
    db.delete_ingest_log(doc_id)
    logger.info(f"[管理员 {username}] 删除文档：{doc_id}")
    return {"deleted": doc_id}
