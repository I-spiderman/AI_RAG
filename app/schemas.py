"""API 请求/响应模型（Pydantic）"""

from pydantic import BaseModel, Field

class RegisterRequest(BaseModel):
    username: str = Field(min_length=2, max_length=50, description="用户名")
    password: str = Field(min_length=6, max_length=64, description="密码")

class LoginRequest(BaseModel):
    username: str
    password: str

class AuthResponse(BaseModel):
    token: str
    user_id: str
    username: str
    role: str = "user"

class MeResponse(BaseModel):
    user_id: str
    username: str
    role: str = "user"
