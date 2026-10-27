"""认证路由：注册 / 登录 / 当前用户"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from app import db, security
from app.schemas import RegisterRequest, LoginRequest, AuthResponse, MeResponse

router = APIRouter(prefix="/api/auth", tags=["auth"])
_bearer = HTTPBearer(auto_error=False)
logger = logging.getLogger(__name__)

@router.post("/register", response_model=AuthResponse)
def register(req: RegisterRequest):
    """注册：查重 → 哈希存库 → 直接签发 token（注册即登录）"""
    if db.get_user_by_username(req.username):
        raise HTTPException(status_code=400, detail="用户名已存在")
    pwd_hash = security.hash_password(req.password)
    user_id = db.create_user(req.username, pwd_hash)
    token = security.create_token(user_id, req.username)
    logger.info(f"新用户注册：{req.username}（{user_id}）")
    return AuthResponse(token=token, user_id=user_id, username=req.username, role="user")

@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest):
    """登录：校验密码 → 签发 token"""
    user = db.get_user_by_username(req.username)
    if not user or not security.verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = security.create_token(user["user_id"], user["username"])
    logger.info(f"用户登录：{req.username}（{user['user_id']}）成功")
    return AuthResponse(token=token, user_id=user["user_id"], username=user["username"], role=user["role"])

@router.get("/me", response_model=MeResponse)
def me(cred: HTTPAuthorizationCredentials = Depends(_bearer)):
    """当前用户（前端刷新页面后恢复登录态用）"""
    info = _resolve_user(cred)
    user = db.get_user_by_username(info[1])
    return MeResponse(user_id=info[0], username=info[1], role=(user or {}).get("role", "user"))

def _resolve_user(cred: HTTPAuthorizationCredentials):
    """解析 Bearer token → (user_id, username)；非法/过期抛 401"""
    if cred is None:
        raise HTTPException(status_code=401, detail="未登录")
    info = security.verify_token(cred.credentials)
    if info is None:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    return info
