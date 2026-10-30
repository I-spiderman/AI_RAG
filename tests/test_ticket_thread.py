"""
工单候选线程管理冒烟测试（不调用任何 LLM，Redis 需可用）

验证链路：开线 → 并入消息 → 超阈值滚动摘要 → 四触发器归档 → 工单入库 + Redis 状态流转
运行：.venv/Scripts/python.exe tests/test_ticket_thread.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import redis

from backend.services.memory.ticket_thread import (
    TicketThreadManager, is_finish_short, ACTIVE_SET,
)

TEST_SID = "test_thread_session_001"
TEST_UID = "test_user_001"


class FakeTicketMemory:
    """stub 工单库：不连 Chroma，不调 LLM"""

    def __init__(self):
        self.tickets = []

    def summarize_to_ticket(self, conversation: str) -> dict:
        return {
            "question": "测试问题",
            "process": "处理过程",
            "resolution": "解决方案",
            "category": "测试分类",
        }

    def add_ticket(self, user_id, question, process, resolution, category,
                   source="auto", status="已解决"):
        self.tickets.append({"user_id": user_id, "question": question,
                             "source": source, "status": status})
        return {"ticket_id": f"t{len(self.tickets)}", "action": "added",
                "occurrences": 1, "source": source, "status": status}


def stub_summarize(old_summary, evicted):
    """stub 摘要：直接拼接，不调 LLM"""
    merged = (old_summary or "") + "\n" + "|".join(c for _, c in evicted)
    return merged.strip()[:200]


def _cleanup(r: redis.Redis, sid: str):
    r.delete(f"ticket:thread:{sid}", f"ticket:thread:{sid}:msgs")
    r.srem(ACTIVE_SET, sid)


def main():
    r = redis.Redis(host="localhost", port=6379, db=0,
                    password=os.getenv("REDIS_PASSWORD") or None,
                    decode_responses=True, socket_connect_timeout=3, socket_timeout=3)
    r.ping()
    _cleanup(r, TEST_SID)

    tm = TicketThreadManager(ticket_memory=FakeTicketMemory(),
                             summarize_fn=stub_summarize,
                             token_threshold=100)  # 小阈值便于触发摘要

    # 1. is_finish_short 收尾判定
    assert is_finish_short("谢谢，明白了"), "收尾词未命中"
    assert not is_finish_short("我想了解一下报销流程"), "长句误判收尾"
    assert not is_finish_short(""), "空串误判"

    # 2. 开线
    tm.open(TEST_SID, TEST_UID, topic="报销")
    assert tm.is_open(TEST_SID), "开线后未 open"
    assert r.sismember(ACTIVE_SET, TEST_SID), "active set 未登记"

    # 3. 并入消息（未超阈值）
    over = tm.append(TEST_SID, "user", "报销流程是什么")
    assert not over, "短消息不应触发摘要"
    over = tm.append(TEST_SID, "assistant", "提交报销单即可")
    assert not over, "短消息不应触发摘要"
    assert r.llen(f"ticket:thread:{TEST_SID}:msgs") == 2, "消息未写入 msgs"

    # 4. 超阈值 → 滚动摘要（msgs 清空、summary 有值、tokens 归零）
    tm.append(TEST_SID, "user", "x" * 150)  # 超过 100 阈值
    assert r.llen(f"ticket:thread:{TEST_SID}:msgs") == 3
    tm.summarize(TEST_SID)
    assert r.llen(f"ticket:thread:{TEST_SID}:msgs") == 0, "摘要后 msgs 未清空"
    assert r.hget(f"ticket:thread:{TEST_SID}", "summary"), "摘要为空"
    assert int(r.hget(f"ticket:thread:{TEST_SID}", "tokens")) == 0, "tokens 未归零"

    # 5. 归档（用户收尾触发）→ 后台线程提炼工单
    tm.archive(TEST_SID, TEST_UID, reason="user_finish")
    deadline = time.time() + 10
    while time.time() < deadline and tm.tm.tickets == []:
        time.sleep(0.2)
    assert tm.tm.tickets, "工单未生成"
    assert tm.tm.tickets[0]["user_id"] == TEST_UID, "user_id 未写入工单"
    assert r.hget(f"ticket:thread:{TEST_SID}", "status") == "archived", "状态未置 archived"
    assert not r.sismember(ACTIVE_SET, TEST_SID), "active set 未清理"
    assert r.llen(f"ticket:thread:{TEST_SID}:msgs") == 0, "msgs 未清理"

    # 6. 防重复：再次归档不生效（status=archived）
    before = len(tm.tm.tickets)
    tm.archive(TEST_SID, TEST_UID, reason="user_finish")
    time.sleep(0.5)
    assert len(tm.tm.tickets) == before, "重复归档生成了第二张工单"

    # 7. 重新开线（archived 后重建）
    tm.open(TEST_SID, TEST_UID, topic="故障")
    assert tm.is_open(TEST_SID), "archived 后无法重建线程"
    assert tm.topic(TEST_SID) == "故障", "topic 未更新"
    tm.archive(TEST_SID, TEST_UID, reason="test_done")

    _cleanup(r, TEST_SID)
    print("ALL PASS: 工单线程管理链路正常")


if __name__ == "__main__":
    main()
