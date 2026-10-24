# -*- coding: utf-8 -*-
"""
智能客服 RAG 系统：命令行交互入口

业务组装在 services/rag_service.py（RAGService），本文件只负责：
- 日志配置（入口职责）
- 命令行问答循环

命令：
    普通输入            → 提问（流式打印）
    "入库:<文件路径>"   → 文档解析分块入库
    "exit"/"退出"       → 退出并归档会话（画像兜底提取）
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from services.core.rag_service import RAGService

# 日志目录（首次运行时创建）
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
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
logger = logging.getLogger("main")

if __name__ == "__main__":
    rag = RAGService()

    print("=" * 50)
    print("智能客服 RAG 系统已就绪")
    print("输入问题直接提问；输入 入库:<文件路径> 入库文档；输入 退出 结束")
    print("=" * 50)

    while True:
        q = input("\n你：").strip()
        if not q:
            continue
        if q in ("exit", "退出", "quit", "q"):
            rag.archive_session()
            print("已归档会话，再见！")
            break
        if q.startswith("入库:"):
            n = rag.ingest_file(q[3:].strip())
            print(f"  入库完成：{n} 块")
            continue

        print("\n助手：", end="", flush=True)
        for piece in rag.ask_stream(q):
            print(piece, end="", flush=True)
        print()
