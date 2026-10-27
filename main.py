# -*- coding: utf-8 -*-
"""
智能客服 RAG 系统：Web 服务统一启动入口

启动：
    python main.py               # 启动 FastAPI（0.0.0.0:8000，reload 开发模式）
                                 # 前端有构建产物（frontend/dist）时由 FastAPI 直接托管，
                                 # 否则浏览器访问 Vite 开发服务器（cd frontend && npm run dev）
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from backend.services.core.rag_service import RAGService

# 日志目录（首次运行时创建）
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(
            filename="logs/app.log", encoding="utf-8", mode="a",
            maxBytes=5 * 1024 * 1024,  # 单文件 5MB 轮转
            backupCount=5,              # 保留 5 个备份
        ),
    ],
)

if __name__ == "__main__":
    import uvicorn
    # app 实例与路由定义在 app/main.py（uvicorn 按字符串导入，日志配置在此已先行完成）
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

