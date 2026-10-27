"""
LLM 客户端共享模块（单例）

两类模型客户端，按用途区分：
1. ollama 本地模型（get_client / MODEL）：内部后台任务
   摘要合并、画像提取、工单提炼——免费、隐私、省时
2. DeepSeek 云端（get_deepseek_client / DEEPSEEK_MODEL）：面向用户的回答生成
   质量更高、支持流式输出

"""
import os
import logging
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

# ==========================================================
# 1. ollama 本地模型（内部后台任务）
# ==========================================================

MODEL = "qwen2.5:7b-instruct-q4_K_M"
_client = None

def get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
    return _client

# ==========================================================
# 2. DeepSeek 云端模型（面向用户的回答生成）
# ==========================================================

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

_deepseek_client = None

def get_deepseek_client():
    """DeepSeek 客户端（单例）：用户回答生成（流式）"""
    global _deepseek_client
    if _deepseek_client is None:
        if not DEEPSEEK_API_KEY:
            logger.warning(
                "未检测到 DEEPSEEK_API_KEY，请先在 .env 中配置，"
                "否则生成会返回兜底文案"
            )
        _deepseek_client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
    return _deepseek_client
