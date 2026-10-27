"""
LLM 生成模块：DeepSeek 云端优先，本地 ollama 降级，兜底文案垫底
DeepSeek 重试 2 次（总3次尝试）→ ollama 尝试 1 次 → 兜底文案

"""

import logging
import time
from typing import List, Dict, Optional, Iterator
from backend.services.llm.llm_client import (get_client, get_deepseek_client, MODEL, DEEPSEEK_MODEL,
                                             DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL)

logger = logging.getLogger(__name__)

# 最终兜底文案
FALLBACK_REPLY = "抱歉，客服系统暂时繁忙，请稍后再试。"
# 输出中中断提示
INTERRUPT_REPLY = "（回答因网络中断，请点击重试）"

class LLMGenerator:
    def __init__(self, model: str = DEEPSEEK_MODEL, local_model: str = MODEL,
                 temperature: float = 0.3, max_tokens: int = 1024,
                 deepseek_retries: int = 2, retry_delay: float = 1.0, request_timeout: int = 30,
                 fallback_reply: str = FALLBACK_REPLY, interrupt_reply: str = INTERRUPT_REPLY):
        """
        :param model: 云端模型名（.env 的 DEEPSEEK_MODEL）
        :param local_model: 本地降级模型（ollama 的 qwen2.5:7b）
        :param temperature: 采样温度；客服场景偏低，回答更稳定
        :param max_tokens: 单次生成最大 token 数
        :param deepseek_retries: DeepSeek 额外重试次数（总尝试 = retries + 1）
        :param retry_delay: 重试间隔秒数（退避）
        :param request_timeout: 单次请求超时秒数
        :param fallback_reply: 两层都失败时的兜底文案
        :param interrupt_reply: 输出中中断时的提示
        """
        self.model = model
        self.local_model = local_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.deepseek_retries = deepseek_retries
        self.retry_delay = retry_delay
        self.request_timeout = request_timeout
        self.fallback_reply = fallback_reply
        self.interrupt_reply = interrupt_reply

    def _deepseek_configured(self) -> bool:
        """检查 DeepSeek 客户端是否已配置"""
        return bool(DEEPSEEK_API_KEY and DEEPSEEK_BASE_URL and self.model)
    def _stream_from(self, client, model, messages, max_tokens, temperature, emitted) -> Iterator[str]:
        """
        向指定客户端发起一次流式请求，逐块 yield 内容。
        - 每吐出一块，把 emitted[0] 置 True（供外层区分"输出前/输出中"失败）
        - 连接或传输异常向上抛出，由外层决定重试/降级/中断
        """
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            timeout=self.request_timeout,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                emitted[0] = True
                yield delta.content

    def generate_stream(self, messages: List[Dict], temperature: Optional[float] = None,
                        max_tokens: Optional[int] = None) -> Iterator[str]:
        """
        流式生成（主入口），含降级链与中断处理：
        - 输出前失败 → DeepSeek 重试 → ollama 尝试 → 兜底文案
        - 输出中失败 → 保留已吐内容 + 中断提示
        """
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        emitted = [False]

        # ---------- 第一层：DeepSeek（失败重试 deepseek_retries 次） ----------
        if not self._deepseek_configured():
            # 配置缺失=确定性失败，重试没意义，直接降级本地模型
            logger.warning(
                "DeepSeek 配置不完整（API_KEY/BASE_URL/MODEL 缺失），直接降级本地模型"
            )
        else:
            for attempt in range(self.deepseek_retries + 1):
                try:
                    yield from self._stream_from(
                        client=get_deepseek_client(),
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        emitted=emitted,
                    )
                    return
                except Exception as e:
                    if emitted[0]:
                        # 输出中失败，不重试不续写，返回已吐内容 + 中断提示
                        logger.warning(f"DeepSeek 输出中断（已吐部分内容）：{e}")
                        yield self.interrupt_reply
                        return
                    if attempt < self.deepseek_retries:
                        logger.warning(
                            f"DeepSeek 请求失败（第{attempt + 1}次），"
                            f"{self.retry_delay}s 后重试：{e}"
                        )
                        time.sleep(self.retry_delay)
                    else:
                        logger.warning(
                            f"DeepSeek 连续 {self.deepseek_retries + 1} 次失败，降级本地模型：{e}"
                        )

        # ---------- 第二层：本地 ollama（失败重试 1 次） ----------
        try:
            yield from self._stream_from(
                client=get_client(),
                model=self.local_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                emitted=emitted,
            )
            return
        except Exception as e:
            if emitted[0]:
                logger.error(f"本地模型输出中断（已吐部分内容）：{e}")
                yield self.interrupt_reply
                return
            logger.error(f"本地模型也失败，返回兜底文案：{e}")

        # ---------- 第三层：兜底文案 ----------
        yield self.fallback_reply

    def generate(self, messages: List[Dict], temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None) -> str:
        """非流式便捷版：内部消费流式接口，拼成完整文本（测试/日志场景用）"""
        return "".join(self.generate_stream(messages, temperature, max_tokens))
