"""
工单候选线程管理模块（Redis 存储版，多 worker 安全）

工单提取时机：围绕"问题线程"而非"整会话/每条消息"。
一个业务问题（query_knowledge）从提出到收尾是一个线程，线程结束时提炼成一张工单。

线程生命周期：
    open（开线）→ 累积消息（append）→ 超 token 阈值滚动摘要（summarize）
    → 触发归档（archive：用户收尾/话题转移/转人工/空闲超时）→ archived

Redis 结构：
    ticket:thread:{session_id}          Hash  线程元信息
        status: open / archiving / archived
        topic / user_id / opened_at / last_active(unixts)
        tokens / summary / ticket_id
    ticket:thread:{session_id}:msgs     List  线程内消息 JSON ["role","content"]
    ticket:thread:active                Set   所有 open 线程的 session_id（空闲扫描用）

摘要（自建，与 MemoryService 会话滚动摘要完全解耦）：
    - 触发：tokens（字符数近似）> token_threshold
    - 复用 local_summarize（同一 LLM 摘要逻辑），合并后清空 msgs、tokens 归零
    - 归档输入 = summary + 剩余 msgs，不丢内容

空闲检测（全程 Redis）：
    - append 时更新 last_active（unix 时间戳）
    - run_scan_loop 后台线程每 scan_interval 秒扫 active Set，
      超过 idle_seconds 未活跃且仍 open → 归档（reason=idle_timeout）
"""

import json
import logging
import os
import threading
import time
import uuid
from typing import Optional, Tuple

import redis

from backend.services.memory.short_memory_service import local_summarize

logger = logging.getLogger(__name__)

# 收尾词表（A 触发器：用户满意/确认短句）
FINISH_WORDS = ("谢谢", "明白了", "可以了", "解决了", "知道了", "清楚了", "搞定", "没问题了", "感谢", "好的好的")

ACTIVE_SET = "ticket:thread:active"


def is_finish_short(text: str) -> bool:
    """用户收尾信号：<=20 字的短句且命中收尾词表（需结合上一轮业务意图使用）"""
    t = (text or "").strip()
    return 0 < len(t) <= 20 and any(w in t for w in FINISH_WORDS)


class TicketThreadManager:
    """工单候选线程管理：开线/并入/摘要/归档/空闲扫描，全部基于 Redis"""

    def __init__(self, ticket_memory, summarize_fn=None, token_threshold: int = 4000,
                 idle_seconds: int = 3600, scan_interval: int = 300):
        """
        :param ticket_memory: TicketMemory 实例（提炼/入库用）
        :param summarize_fn: 线程滚动摘要函数 (旧摘要, 被挤出消息) -> 新摘要；默认复用 local_summarize
        :param token_threshold: 字符数阈值（近似 token），超过触发一次滚动摘要
        :param idle_seconds: 空闲多少秒未活跃则归档（默认 1 小时）
        :param scan_interval: 空闲扫描间隔秒（默认 5 分钟）
        """
        self.tm = ticket_memory
        self.summarize_fn = summarize_fn or local_summarize
        self.token_threshold = token_threshold
        self.idle_seconds = idle_seconds
        self.scan_interval = scan_interval

        self._redis = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD") or None,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        self._redis.ping()

    # ---------------- keys ----------------
    def _hkey(self, session_id: str) -> str:
        return f"ticket:thread:{session_id}"

    def _mkey(self, session_id: str) -> str:
        return f"ticket:thread:{session_id}:msgs"

    # ---------------- 状态查询 ----------------
    def is_open(self, session_id: str) -> bool:
        return self._redis.hget(self._hkey(session_id), "status") == "open"

    def topic(self, session_id: str) -> str:
        return self._redis.hget(self._hkey(session_id), "topic") or ""

    def tokens(self, session_id: str) -> int:
        return int(self._redis.hget(self._hkey(session_id), "tokens") or 0)

    # ---------------- 开线 ----------------
    def open(self, session_id: str, user_id: str, topic: Optional[str] = None) -> None:
        """开新线程；archived/不存在时重建。

        并行不阻塞（用户选择）：若上一次归档仍在后台进行（status==archiving），
        旧线程的消息已被 archive 时原子 rename 到一次性快照 key，本方法直接重建
        session 的 hkey/mkey 即可，与归档 worker 互不干扰（零等待、零丢消息）。
        旧 worker 提炼完会照常入库，且由 _archive_worker 的 race 防护（非 archiving
        不回写 archived）不会覆盖新线程状态。
        """
        hkey, mkey = self._hkey(session_id), self._mkey(session_id)
        cur = self._redis.hget(hkey, "status")
        if cur == "open":
            return
        # archiving 时也直接重建：旧 msgs 已迁出到快照 key，无冲突，无需等待
        pipe = self._redis.pipeline()
        pipe.delete(hkey, mkey)                      # 重建（archived/归档中旧记录清掉）
        pipe.hset(hkey, mapping={
            "status": "open",
            "user_id": user_id,
            "topic": topic or "",
            "opened_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_active": str(int(time.time())),
            "tokens": 0,
            "summary": "",
            "ticket_id": "",
        })
        pipe.sadd(ACTIVE_SET, session_id)
        pipe.execute()
        logger.info(f"[工单线程] 开线：{session_id}（topic={topic or '未确定'}）")

    # ---------------- 并入消息 ----------------
    def append(self, session_id: str, role: str, content: str) -> bool:
        """
        线程内追加一条消息（仅 open 状态写入）。
        :return: True = 累计 tokens 已超阈值，调用方可触发异步摘要
        """
        hkey, mkey = self._hkey(session_id), self._mkey(session_id)
        if self._redis.hget(hkey, "status") != "open":
            return False
        pipe = self._redis.pipeline()
        pipe.rpush(mkey, json.dumps([role, content], ensure_ascii=False))
        pipe.hincrby(hkey, "tokens", len(content))
        pipe.hset(hkey, "last_active", str(int(time.time())))
        pipe.execute()
        return self.tokens(session_id) > self.token_threshold

    # ---------------- 滚动摘要（token 超阈值） ----------------
    def summarize(self, session_id: str) -> None:
        """把当前 msgs 全部并入线程摘要，清空窗口继续累积（后台线程调用）"""
        hkey, mkey = self._hkey(session_id), self._mkey(session_id)
        if self._redis.hget(hkey, "status") != "open":
            return
        raws = self._redis.lrange(mkey, 0, -1)
        if not raws:
            return
        evicted = [tuple(json.loads(m)) for m in raws]
        old_summary = self._redis.hget(hkey, "summary") or ""
        merged = self.summarize_fn(old_summary, evicted)
        pipe = self._redis.pipeline()
        pipe.hset(hkey, "summary", merged)
        pipe.hset(hkey, "tokens", 0)
        pipe.hset(hkey, "last_active", str(int(time.time())))
        pipe.delete(mkey)
        pipe.execute()
        logger.info(f"[工单线程] 滚动摘要：{session_id}（{len(evicted)} 条并入，摘要 {len(merged)} 字）")

    # ---------------- 归档（统一出口：A/B/C/D 四触发器） ----------------
    def archive(self, session_id: str, user_id: str, reason: str = "manual") -> None:
        """触发归档：先把线程数据原子快照到一次性 key，再起后台 worker 提炼（不阻塞调用方）。

        非 open 状态忽略（防重复归档）。快照后当前线程的 mkey 被 rename 走、summary 被固化到
        快照 key，因此新线程（open）可并行重建 session 的 hkey/mkey，与仍运行的归档 worker 互不
        干扰——这是「并行不阻塞」方案的关键：旧工单数据已脱离共享 key，open 不再需要等待。
        """
        hkey, mkey = self._hkey(session_id), self._mkey(session_id)
        if self._redis.hget(hkey, "status") != "open":
            return
        job_id = uuid.uuid4().hex[:12]
        snap_msgs = f"ticket:archive_job:{session_id}:{job_id}:msgs"
        snap_meta = f"ticket:archive_job:{session_id}:{job_id}:meta"
        # 原子把线程消息列表整体移交给一次性快照 key（旧 mkey 不再存在，open 可安全重建同名）
        # mkey 理论必存在（open 后必 append），但极端情况下（open 后未 append 即归档）可能为空，容错跳过
        try:
            self._redis.rename(mkey, snap_msgs)
        except redis.exceptions.ResponseError:
            logger.warning(f"[工单线程] 归档时 mkey 不存在（无消息），快照为空：{session_id}")
        # 摘要同步固化（归档期间 open 会清空 session 的 summary，必须提前快照）
        self._redis.set(snap_meta, self._redis.hget(hkey, "summary") or "")
        self._redis.hset(hkey, "status", "archiving")   # 占位，防并发重复归档
        t = threading.Thread(
            target=self._archive_worker,
            args=(session_id, user_id, reason, job_id),
            daemon=True,
        )
        t.start()

    def _archive_worker(self, session_id: str, user_id: str, reason: str, job_id: str) -> None:
        """提炼线程（并行版）：只读一次性快照 key，提炼成工单入库，最后清理快照。

        不直接读 session 的 hkey/mkey（它们可能已被 open 并行重建），因此与新一轮
        业务线程零冲突。race 防护：仅当 session 线程仍处于本 job 负责的 archiving 态
        才回写 archived；若已被并行重建（status=open），则跳过，不覆盖新线程状态。
        """
        hkey = self._hkey(session_id)
        snap_msgs = f"ticket:archive_job:{session_id}:{job_id}:msgs"
        snap_meta = f"ticket:archive_job:{session_id}:{job_id}:meta"
        ticket_id = ""
        try:
            summary = self._redis.get(snap_meta) or ""
            raws = self._redis.lrange(snap_msgs, 0, -1)
            parts = []
            if summary:
                parts.append(f"【摘要】\n{summary}")
            if raws:
                msgs = "\n".join(f"{r}: {c}" for r, c in (json.loads(m) for m in raws))
                parts.append(f"【最近对话】\n{msgs}")
            text = "\n".join(parts)
            if not text.strip():
                logger.info(f"[工单线程] 无内容，跳过归档：{session_id}")
                return
            if reason == "human_handoff":
                # 转人工工单已由 create_handoff_ticket 创建（待处理/human_request），
                # 线程归档不再重复建单，只清理线程状态
                logger.info(f"[工单线程] 转人工归档（{reason}）：{session_id}，建单由 create_handoff_ticket 负责")
            else:
                ticket = self.tm.summarize_to_ticket(text)
                if not (ticket["question"] and ticket["resolution"]):
                    logger.warning(f"[工单线程] 提炼缺字段，不生成工单：{session_id}")
                    return
                res = self.tm.add_ticket(
                    user_id, ticket["question"], ticket["process"],
                    ticket["resolution"], ticket["category"],
                    source="auto", status="已解决",
                )
                logger.info(f"[工单线程] 已归档（{reason}）：{session_id} → {res}")
                ticket_id = res.get("ticket_id", "")
        except Exception as e:
            logger.error(f"[工单线程] 归档失败：{session_id}（{reason}）：{e}")
        finally:
            # 清理一次性快照 key（无论成败都要清，避免堆积）
            self._redis.delete(snap_msgs, snap_meta)
            # race 防护：仅当线程仍处于本 job 负责的 archiving 态才落 archived + 记录 ticket_id；
            # 若已被并行重建（status=open），跳过，避免覆盖新线程状态、也不污染新线程 ticket_id
            if self._redis.hget(hkey, "status") == "archiving":
                pipe = self._redis.pipeline()
                if ticket_id:
                    pipe.hset(hkey, "ticket_id", ticket_id)
                pipe.hset(hkey, "status", "archived")
                pipe.srem(ACTIVE_SET, session_id)
                pipe.execute()
                logger.info(f"[工单线程] 归档完成落库：{session_id}（工单 {ticket_id}）")
            else:
                logger.info(f"[工单线程] 归档完成但线程已被并行重建，跳过状态回写：{session_id}")

    def archive_all(self, session_id: str, user_id: str) -> None:
        """会话结束兜底：归档该会话的 open 线程（每个会话至多一个线程）"""
        self.archive(session_id, user_id, reason="session_end")

    # ---------------- 空闲扫描（全程 Redis） ----------------
    def scan_idle(self) -> None:
        """扫描 active Set：last_active 超过 idle_seconds 且仍 open → 归档"""
        now = int(time.time())
        for sid in self._redis.smembers(ACTIVE_SET):
            hkey = self._hkey(sid)
            last = self._redis.hget(hkey, "last_active")
            status = self._redis.hget(hkey, "status")
            if status != "open" or not last:
                continue
            try:
                if now - int(last) > self.idle_seconds:
                    uid = self._redis.hget(hkey, "user_id") or "cli_user"
                    logger.info(f"[工单线程] 空闲超时触发归档：{sid}（{now - int(last)}s 无活跃）")
                    self.archive(sid, uid, reason="idle_timeout")
            except ValueError:
                continue

    def run_scan_loop(self) -> None:
        """后台常驻扫描循环（daemon 线程，随进程退出）"""
        logger.info(f"[工单线程] 空闲扫描启动：每 {self.scan_interval}s 扫一次，空闲 {self.idle_seconds}s 归档")
        while True:
            try:
                self.scan_idle()
            except Exception as e:
                logger.warning(f"[工单线程] 空闲扫描异常：{e}")
            time.sleep(self.scan_interval)
