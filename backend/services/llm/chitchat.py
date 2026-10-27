"""
闲聊模块：意图为 chitchat 时，用 LLM 生成自然对话回复（替代预设话术）。

- 输入：用户消息 + 短期记忆历史（摘要在前、窗口原文在后），保证对话连贯
- 生成：DeepSeek 云端主路（质量高、流式）→ 失败降级 ollama 本地 → 再失败兜底话术
- 画像：闲聊中透露的个人信息由现有"窗口满一批 K 轮 → 后台提取画像"机制覆盖，本模块不重复做
- 工单：闲聊不检索、不生成工单（由 query_knowledge / query_ticket 路径负责）
"""

import logging
from typing import List, Iterator, Dict

from backend.services.llm.llm_client import (
    get_client, MODEL,
    get_deepseek_client, DEEPSEEK_MODEL,
)

logger = logging.getLogger(__name__)

# 闲聊角色设定
_CHITCHAT_SYSTEM = (
    "你是智能客服系统的聊天助手，当前用户想和你闲聊。\n"
    "要求：\n"
    "1. 像朋友一样自然、亲切地回应，语气轻松；\n"
    "2. 每次回复 1~3 句话，简短不啰嗦；\n"
    "3. 不要提知识库、检索、工单等内部技术内容；\n"
    "4. 不要编造用户没说过的事实；\n"
    "5. 如果用户透露了个人信息（职业、爱好、状态、计划等），顺着话题自然接话，"
    "方便后续为其提供更贴心的服务。"
)

# 双路都失败时的兜底话术（不让用户看到空回复）
_CHITCHAT_FALLBACK = "（刚才有点卡顿，咱们换个话题再聊？）"


class ChitchatEngine:
    """闲聊引擎：基于会话历史生成自然对话回复（流式，DeepSeek 主 + ollama 兜底）"""

    def __init__(self, temperature: float = 0.7, max_tokens: int = 400, request_timeout: int = 30):
        """
        :param temperature: 闲聊比知识问答活泼：温度略高（知识问答默认 0.3）
        :param max_tokens: 单次回复上限（闲聊简短，400 足够）
        :param request_timeout: 单次请求超时秒数（DeepSeek 云端快；ollama 本地推理偏慢，放宽到 30s）
        """
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self._deepseek_client = None  # 云端客户端（懒加载）
        self._local_client = None     # ollama 本地客户端（懒加载）

    def chat_stream(self, question: str, history_messages: List[Dict] = None) -> Iterator[str]:
        """流式闲聊：返回回复文本生成器；DeepSeek 失败降级 ollama，再失败兜底话术"""
        messages = [{"role": "system", "content": _CHITCHAT_SYSTEM}]
        if history_messages:
            messages.extend(history_messages)  # 摘要(前) + 窗口原文(后)，对话连贯
        messages.append({"role": "user", "content": question})

        # ---- 主路：DeepSeek 云端（质量高、流式） ----
        try:
            if self._deepseek_client is None:
                self._deepseek_client = get_deepseek_client()
            yield from self._stream(self._deepseek_client, DEEPSEEK_MODEL, messages)
            return
        except Exception as e:
            logger.warning(f"DeepSeek 闲聊调用失败，降级 ollama 本地：{e}")

        # ---- 兜底路：ollama 本地（免费、隐私、不依赖外网） ----
        try:
            if self._local_client is None:
                self._local_client = get_client()
            yield from self._stream(self._local_client, MODEL, messages)
            return
        except Exception as e:
            logger.warning(f"本地闲聊模型调用失败，返回兜底话术：{e}")
            yield _CHITCHAT_FALLBACK

    def _stream(self, client, model: str, messages: List[Dict]) -> Iterator[str]:
        """用指定 client 流式生成一段回复"""
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
            timeout=self.request_timeout,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
