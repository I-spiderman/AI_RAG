"""
用户表数据访问：连接模式复用 services.memory.long_memory 的写法
（读 .env 的 mysql_password，库 rag_service），模块导入时自动建表。
"""

import logging
import os
import uuid
import pymysql
from dotenv import load_dotenv
from app.security import hash_password

load_dotenv()
logger = logging.getLogger(__name__)

_DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': os.getenv('mysql_password'),
    'database': 'rag_service',
    'charset': 'utf8mb4',
    'connect_timeout': 5,
}

def _get_conn():
    return pymysql.connect(**_DB_CONFIG)

def _ensure_users_table() -> None:
    """建表（幂等）：users 表不存在则创建"""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id            INT AUTO_INCREMENT PRIMARY KEY,
                    username      VARCHAR(50)  NOT NULL UNIQUE,
                    password_hash VARCHAR(256) NOT NULL,
                    user_id       VARCHAR(32)  NOT NULL UNIQUE,
                    created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
                ) DEFAULT CHARSET=utf8mb4
                """
            )
        conn.commit()
    logger.info("users 表就绪（MySQL：rag_service）")

def create_user(username: str, password_hash: str, role: str = "user") -> str:
    """插入新用户，返回业务 user_id（usr_xxxx，后续关联工单/画像/短期记忆）"""
    user_id = "usr_" + uuid.uuid4().hex[:8]
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (username, password_hash, user_id, role) VALUES (%s, %s, %s, %s)",
                (username, password_hash, user_id, role),
            )
        conn.commit()
    return user_id

def get_user_by_username(username: str):
    """按用户名查用户，返回 dict 或 None（含 role，供管理权限判断）"""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, password_hash, user_id, role FROM users WHERE username=%s",
                (username,),
            )
            row = cur.fetchone()
    if not row:
        return None
    return {"id": row[0], "username": row[1], "password_hash": row[2], "user_id": row[3], "role": row[4]}

def _ensure_sessions_table() -> None:
    """建表（幂等）：chat_sessions 会话登记表（session_id ↔ user_id 对应）"""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id         INT AUTO_INCREMENT PRIMARY KEY,
                    user_id    VARCHAR(32)  NOT NULL,
                    session_id VARCHAR(64)  NOT NULL UNIQUE,
                    title      VARCHAR(100) NOT NULL DEFAULT '新对话',
                    created_at DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    KEY idx_user (user_id)
                ) DEFAULT CHARSET=utf8mb4
                """
            )
        conn.commit()
    logger.info("chat_sessions 表就绪（MySQL：rag_service）")

def create_session(user_id: str, session_id: str) -> None:
    """登记一个新会话（session_id ↔ user_id），重复登记幂等"""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT IGNORE INTO chat_sessions (user_id, session_id) VALUES (%s, %s)",
                (user_id, session_id),
            )
        conn.commit()

def list_sessions(user_id: str):
    """该用户的会话列表，最新更新在前"""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT session_id, title, created_at, updated_at FROM chat_sessions "
                "WHERE user_id=%s ORDER BY updated_at DESC",
                (user_id,),
            )
            rows = cur.fetchall()
    return [
        {"session_id": r[0], "title": r[1], "created_at": str(r[2]), "updated_at": str(r[3])}
        for r in rows
    ]

def session_belongs_to(user_id: str, session_id: str) -> bool:
    """校验某会话是否属于该用户（历史接口的归属检查）"""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM chat_sessions WHERE user_id=%s AND session_id=%s",
                (user_id, session_id),
            )
            return cur.fetchone() is not None

def update_title_if_default(session_id: str, title: str) -> bool:
    """会话仍是默认标题时，用第一条用户消息更新标题（取前 20 字）"""
    title = title.strip()[:20]
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_sessions SET title=%s WHERE session_id=%s AND title='新对话'",
                (title, session_id),
            )
            changed = cur.rowcount > 0
        conn.commit()
    return changed

# ==========================================================
# 管理员 / 文档入库日志
# ==========================================================

def _ensure_role_column() -> None:
    """幂等：users 表加 role 列（'admin'/'user'，默认 user）"""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM information_schema.columns "
                "WHERE table_schema = DATABASE() AND table_name='users' AND column_name='role'"
            )
            if cur.fetchone()[0] == 0:
                cur.execute("ALTER TABLE users ADD COLUMN role VARCHAR(10) NOT NULL DEFAULT 'user'")
                conn.commit()
                logger.info("users 表已新增 role 列")
    logger.info("users.role 列就绪")

def _ensure_admin() -> None:
    """幂等：预置管理员账号（username=admin，密码取 .env ADMIN_PASSWORD，默认 admin123）"""
    if get_user_by_username("admin"):
        return
    pwd = os.getenv("ADMIN_PASSWORD", "admin123")
    create_user("admin", hash_password(pwd), role="admin")
    logger.info("已预置管理员账号 admin（密码：ADMIN_PASSWORD，默认 admin123）")

def _ensure_ingest_logs_table() -> None:
    """建表（幂等）：doc_ingest_logs 文档入库日志"""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS doc_ingest_logs (
                    id          INT AUTO_INCREMENT PRIMARY KEY,
                    doc_id      VARCHAR(64)  NOT NULL,
                    doc_name    VARCHAR(255) NOT NULL,
                    chunk_count INT          NOT NULL DEFAULT 0,
                    operator    VARCHAR(50)  NOT NULL DEFAULT '',
                    created_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    KEY idx_doc (doc_id)
                ) DEFAULT CHARSET=utf8mb4
                """
            )
        conn.commit()
    logger.info("doc_ingest_logs 表就绪（MySQL：rag_service）")

def log_ingest(doc_id: str, doc_name: str, chunk_count: int, operator: str = "") -> None:
    """记录一次文档入库"""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO doc_ingest_logs (doc_id, doc_name, chunk_count, operator) VALUES (%s, %s, %s, %s)",
                (doc_id, doc_name, chunk_count, operator),
            )
        conn.commit()

def list_ingest_logs():
    """全部入库日志，最新在前"""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT doc_id, doc_name, chunk_count, operator, created_at FROM doc_ingest_logs "
                "ORDER BY created_at DESC, id DESC"
            )
            rows = cur.fetchall()
    return [
        {"doc_id": r[0], "doc_name": r[1], "chunk_count": r[2], "operator": r[3], "created_at": str(r[4])}
        for r in rows
    ]

def delete_ingest_log(doc_id: str) -> None:
    """删除某文档的入库日志记录"""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM doc_ingest_logs WHERE doc_id=%s", (doc_id,))
        conn.commit()

_ensure_users_table() # 模块导入即建表
_ensure_sessions_table()
_ensure_role_column()          # 兼容旧表：补 role 列
_ensure_admin()                # 预置管理员
_ensure_ingest_logs_table()    # 文档入库日志表
