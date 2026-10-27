"""
工单查询模块：意图为 query_ticket 时，按用户身份查询历史工单处理记录。

- 数据源：TicketMemory（Chroma 工单库），按 user_id 过滤 + 相似度检索（阈值门控）
- 回答：模板拼装（确定性、零 LLM 调用），命中列出问题/结果/时间/次数
- 边界：进行中工单的进度/催办需要后端工单系统（状态机），属 FastAPI 阶段；
        本模块只查已沉淀的历史工单（同类问题之前怎么处理的）
"""

import logging
from backend.services.memory.long_memory import TicketMemory

logger = logging.getLogger(__name__)

# 未命中时的引导语
NO_TICKET_REPLY = "未找到您相关的历史工单记录。您可以补充工单编号，或选择转接人工客服。"

def _extract_field(text: str, label: str) -> str:
    """从工单文本（【工单】分类/问题/处理过程/结果…）里按标签取值，取不到返回空串"""
    for line in text.splitlines():
        if line.startswith(f"{label}："):
            return line[len(label)+1:].strip()
    return ""

class TicketQueryService:
    """工单查询服务：按用户查历史工单，拼装回答文本"""

    def __init__(self, tm: TicketMemory, top_k: int = 3):
        self.tm = tm
        self.top_k = top_k

    def answer(self, question: str, user_id: str) -> str:
        """查询并组装回答；只查该用户自己的工单（多用户隔离）"""
        hits = self.tm.query_tickets(question, top_k=self.top_k, user_id=user_id)
        if not hits:
            logger.info(f"工单查询未命中（user={user_id}）：{question}")
            return NO_TICKET_REPLY

        lines = [f"为您找到 {len(hits)} 条相关工单记录："]
        for i, h in enumerate(hits, 1):
            meta = h["meta"]
            q_text = _extract_field(h["text"], "问题") or "（未记录问题）"
            res = meta.get("resolution") or _extract_field(h["text"], "结果") or "（未记录结果）"
            occ = meta.get("occurrences", 1)
            lines.append(
                f"{i}. 问题：{q_text}\n"
                f"   处理结果：{res}\n"
                f"   （{meta.get('last_occurred_at', '')}，累计出现 {occ} 次）"
            )
        return "\n\n".join(lines)
