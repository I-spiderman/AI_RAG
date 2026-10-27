"""
安全模块：密码哈希（PBKDF2-SHA256）+ 轻量签名 Token（HMAC-SHA256）。
零第三方依赖（Python 标准库），token 无状态：服务端不存会话，只校验签名和过期时间。
"""

import base64
import hashlib
import hmac
import json
import os
import time
from dotenv import load_dotenv
load_dotenv()

# token 密钥
TOKEN_SECRET = os.getenv('TOKEN_SECRET', 'dev_secret-change-me')
TOKEN_TTL = 7*24*3600 # token 有效期：7 天
PBKDF2_ITERATIONS = 100_000 # 密码哈希迭代次数

# ---------------- 密码哈希 ----------------
def hash_password(password: str) -> str:
    """生成存储串：'盐hex$哈希hex'（盐随机，同一密码两次结果不同）"""
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac('sha256', password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"{salt.hex()}${dk.hex()}"

def verify_password(password: str, stored: str) -> bool:
    """校验密码与存储串是否匹配（恒定时间比较，防时序攻击）"""
    try:
        salt_hex, dk_hex = stored.split("$")
        dk = hashlib.pbkdf2_hmac('sha256', password.encode("utf-8"),
                                  bytes.fromhex(salt_hex), PBKDF2_ITERATIONS)
        return hmac.compare_digest(dk.hex(), dk_hex)
    except (ValueError, TypeError):
        return False

# ---------------- token 签发/校验 ----------------
def _b64(data: str) -> str:
    # 编码
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def _unb64(s: str) -> bytes:
    # 解码（base64 补位符号必须是 '='，用 '+' 会解码出垃圾字节）
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))

def create_token(user_id: str, username: str) -> str:
    """签发 token：'base64(payload).hmac签名'，payload 含 uid/name/exp"""
    payload = {"uid": user_id, "name": username, "exp": int(time.time()) + TOKEN_TTL}
    body = _b64(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    sig = hmac.new(TOKEN_SECRET.encode(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"

def verify_token(token: str):
    """校验 token：签名正确且未过期则返回 (user_id, username)，否则 None"""
    try:
        body, sig = token.rsplit(".", 1)
        expect = hmac.new(TOKEN_SECRET.encode(), body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expect, sig):
            return None
        payload = json.loads(_unb64(body))
        if payload["exp"] < time.time():
            return None
        return payload["uid"], payload["name"]
    except Exception:
        return None