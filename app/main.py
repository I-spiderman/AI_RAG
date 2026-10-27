"""FastAPI 应用入口：实例、CORS、路由注册、静态文件挂载"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from app.api import auth, chat, sessions, admin
from app.deps import lifespan

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(title="智能客服 RAG 系统", version="0.1.0", lifespan=lifespan)

# 本地开发允许跨域（前端页面与 API 同源部署，实际可收紧）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由（chat 已接入，后续 ingest / health 在此追加）
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(sessions.router)
app.include_router(admin.router)

# 前端构建产物目录（frontend/dist，npm run build 后生成；开发阶段前端走 Vite 5173）
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"

# 默认首页：有构建产物则返回前端 index.html
@app.get("/", include_in_schema=False)
def index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"msg": "前端请通过 Vite 开发服务器访问（cd frontend && npm run dev）"}

# 前端产物存在才挂载静态文件（避免开发期无构建产物时启动失败）
if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")
    logger.info(f"前端产物已挂载：/static → {FRONTEND_DIR}")
else:
    logger.info("未发现前端构建产物（frontend/dist），跳过静态挂载（开发阶段用 Vite 5173）")

