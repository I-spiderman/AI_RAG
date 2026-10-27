"""
短期记忆模块（Redis 存储版）

两层设计：
1. 滑动窗口：保留最近 K 轮完整对话原文（1 轮 = 1 问 1 答），存 Redis List；
2. 滚动摘要：窗口溢出时，把被挤出的对话与旧摘要用 LLM 合并成一份新摘要，
   保留身份/核心问题/结论，剔除寒暄，新信息覆盖旧信息。

存储位置（Redis）：
    memory:{session_id}:window    List    窗口原文（每条 JSON 编码 ["role","content"]）
    memory:{session_id}:summary   String  滚动摘要
    memory:{session_id}:msg_count String  消息计数器（画像提取节拍用，INCR 原子递增）

用户画像提取时机（每满一批 K 轮提取一次）：
    窗口容量 = max_turns*2 条消息（K 轮）。当累计消息数 count 达到 max_turns*2
    的整数倍时，窗口恰好被一整批新的 K 轮对话填满——此时把窗口快照交给
    on_batch_ready 回调（由外部 RAGService 异步提取画像）。节拍对齐：
      第 10 条消息 → 提取 [m1..m10]
      第 20 条消息 → 提取 [m11..m20]
      第 30 条消息 → 提取 [m21..m30]
    会话结束兜底：窗口内最后 count % (max_turns*2) 条未提取的对话（archive_pending）。

设计说明：
- 窗口写入封装成 Lua 原子脚本（满则弹出最老一对 + 写入新消息），天然支持多线程/多 worker 并发；
- 计数器用 Redis INCR（原子），多线程不会漏计；
- 摘要合并是"读-算-写"（GET → LLM 合并 → SET），低频操作，现阶段可接受。
"""

import json
import logging
import os
import redis
from typing import List, Dict, Tuple, Callable, Optional
from backend.services.llm.llm_client import get_client, MODEL

logger = logging.getLogger(__name__)

def local_summarize(old_summary: str, evicted: List[Tuple[str, str]]) -> str:
    """
    本地摘要函数，调用OpenAI API生成摘要
    :param old_summary: 旧摘要
    :param evicted: 被移除的消息列表，每个元素是(role, content)
    :return: 新摘要
    """
    evicted_text = "\n".join(f"{role}: {content}" for role, content in evicted)
    messages = [
        {
            "role": "system",
            "content": (
                "你是对话摘要助手。请把【旧摘要】和【新增对话】合并成一份新摘要。\n"
                "规则：\n"
                "1. 保留关键信息：用户身份、核心问题脉络、关键结论与数字；\n"
                "2. 剔除寒暄客套、与主题无关的过程性细节；\n"
                "3. 若新旧信息冲突，以新信息为准（例如旧摘要写\"笔10元\"，"
                "新对话说\"笔5元\"，则新摘要写\"笔5元\"）；\n"
                "4. 输出控制在500字以内，用简洁陈述句，不要复述对话过程。"
            )
        },
        {
            "role": "user",
            "content": f"【旧摘要】\n{old_summary or '（无）'}\n\n"
                       f"【新增对话】\n{evicted_text}\n\n请输出合并后的新摘要："
        }
    ]
    try:
        resp = get_client().chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=500,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"本地摘要合并失败，降级为直接拼接：{e}")
        return (old_summary + "\n" + evicted_text)[:500]

class MemoryService:
    def __init__(self, max_turns: int = 5, summarize_fn: Optional[Callable[[str, List[Tuple[str, str]]], str]] = None,
                 summary_max_len: int = 500,
                 on_batch_ready: Optional[Callable[[str, List[Tuple[str, str]]], None]] = None):
        """
        :param max_turns: 滑动窗口大小，保留最近几轮完整对话（1轮 = 1问1答）
        :param summarize_fn: 摘要合并函数 (旧摘要, 被挤出对话) -> 新摘要
        :param summary_max_len: 摘要长度上限（字符数）
        :param on_batch_ready: 回调，窗口每填满一批 K 轮新对话时触发
                               (session_id, 窗口快照) -> None；由外部异步提取画像
        """
        self.max_turns = max_turns
        self.summarize_fn = summarize_fn or local_summarize
        self.summary_max_len = summary_max_len
        self.on_batch_ready = on_batch_ready
        self._batch_size = max_turns * 2

        # Redis 连接
        self._redis = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD") or None,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        self._redis.ping() # 启动即检查连接，Redis 不可用直接报错

        # Lua 原子脚本：判断窗口满 → 弹出最老一对 → 写入新消息，返回被弹出的消息列表。
        # Redis 保证脚本整体原子执行，多线程/多 worker 并发 add 同一 session 不会丢消息或重复挤出。
        self._add_script = self._redis.register_script("""
            local len = redis.call("llen", KEYS[1])
            local evicted = {}
            if len >= tonumber(ARGV[1]) then
                local item1 = redis.call('LINDEX', KEYS[1], 0)
                local item2 = redis.call('LINDEX', KEYS[1], 1)
                if item1 then table.insert(evicted, item1) end
                if item2 then table.insert(evicted, item2) end
                redis.call('LTRIM', KEYS[1], 2, -1)
            end
            redis.call('RPUSH', KEYS[1], ARGV[2])
            return evicted
        """)

    # ---------------- 内部工具 ----------------
    @staticmethod
    def _encode(item: Tuple[str, str]) -> str:
        return json.dumps(item, ensure_ascii=False)

    @staticmethod
    def _decode(raw: str) -> Tuple[str, str]:
        return tuple(json.loads(raw))

    def _key(self, session_id: str, part: str) -> str:
        return f"memory:{session_id}:{part}"

    # ---------------- 核心写入 ----------------
    def add(self, session_id: str, role: str, content: str, user_id: str = None) -> None:
        """
        记录一条消息（用户或助手）。
        窗口超限时，自动把最老的对话挤出并合并进摘要。
        每累计满一批 K 轮新对话（计数器到达 batch_size 整数倍），触发画像提取回调。
        :param user_id: 随会话持久化到 Redis，供多 worker 场景画像回调使用
        """
        wkey = self._key(session_id, "window")
        max_messages = self._batch_size

        # 1. user_id 随会话持久化
        if user_id:
            self._redis.set(self._key(session_id, "user_id"), user_id)

        # 2. 原子执行：窗口满 → 弹出最老一对 + 写入新消息
        evicted_raw = self._add_script(
            keys=[wkey],
            args=[str(max_messages), self._encode((role, content))]
        ) or []
        evicted = [self._decode(r) for r in evicted_raw]

        # 3. 有挤出 → 滚动摘要合并
        if evicted:
            old_summary = self._redis.get(self._key(session_id, "summary")) or ""
            merged = self.summarize_fn(old_summary, evicted)
            if len(merged) > self.summary_max_len:
                merged = merged[:self.summary_max_len]
            self._redis.set(self._key(session_id, "summary"), merged)

        # 4. 画像提取时机：消息计数达到 batch_size 整数倍 → 窗口恰好整批更新 → 提取
        if self.on_batch_ready:
            ckey = self._key(session_id, "msg_count")
            count = self._redis.incr(ckey) # 原子计数，天然防并发漏计
            if count % max_messages == 0:
                window_snapshot = self.get_history(session_id) # 整批 K 轮对话
                self.on_batch_ready(session_id, window_snapshot)
    def get_user(self, session_id: str) -> str:
        """取会话所属 user_id（画像回调用），无则返回默认 cli_user"""
        return self._redis.get(self._key(session_id, "user_id")) or "cli_user"

    # ---------------- 读取 ----------------
    def get_summary(self, session_id: str) -> str:
        """取合并内容摘要（空串表示窗口还没溢出过）"""
        return self._redis.get(self._key(session_id, "summary")) or ""

    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        """取窗口内最近 K 轮对话原文"""
        raws = self._redis.lrange(self._key(session_id, "window"), 0, -1)
        return [self._decode(r) for r in raws]

    def get_context(self,session_id: str) -> Dict:
        """
        取完整上下文（给 prompt 模块用的原始结构）：
        {"summary": 合并摘要, "window": 窗口内原文}
        """
        return {
            "summary": self.get_summary(session_id),
            "window": self.get_history(session_id)
        }

    def to_messages(self, session_id: str) -> List[Dict]:
        """
        转成 OpenAI messages 格式（给 LLM 模块直接用）：
        摘要在最前（更早对话的背景），窗口原文按时间顺序在后。
        """
        messages: List[Dict] = []
        summary = self.get_summary(session_id)
        if summary:
            messages.append({"role": "system",
                             "content": f"【旧摘要】\n{summary}"
                             })
        messages.extend(
            {"role": role, "content": content} for role, content in self.get_history(session_id)
        )
        return messages

    # ---------------- 维护 ----------------
    def archive_pending(self, session_id: str) -> List[Tuple[str, str]]:
        """
        会话结束兜底：返回窗口内"还未触发过画像提取"的对话（最后 count % batch_size 条）。
        若最后一整批已提取过，返回空列表。
        """
        count = int(self._redis.get(self._key(session_id, "msg_count")) or 0)
        remain = count % self._batch_size
        if remain == 0:
            return []
        window = self.get_history(session_id)
        return window[-remain:] or []

    def clear(self, session_id: str) -> None:
        """清空某个 session 的全部短期记忆"""
        for part in ("summary", "window", "msg_count", "user_id"):
            self._redis.delete(self._key(session_id, part))

