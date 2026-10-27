"""
RAG 问答服务：把检索、短期记忆、画像、工单、prompt、LLM 生成组装成可复用的问答服务。

设计定位：
- 纯业务组装，不含任何交互（命令行/HTTP 都属于调用方的事）；
- ask_stream 返回生成器，命令行逐块打印、FastAPI 可直接包成 SSE 流式响应；
- 组件在 __init__ 中一次性初始化，MySQL 不可用时画像降级为 None，不影响问答。
"""

import logging
import threading
import time
from pathlib import Path
from typing import List, Dict, Iterator

from backend.services.document.file_process import FileParser, DocumentChunker
from backend.services.vector.embedding_service import generate_embedding, embed_texts
from backend.services.vector.vector_store import VectorStore
from backend.services.retrieval.retrieval import Retriever
from backend.services.memory.short_memory_service import MemoryService
from backend.services.memory.long_memory import LongTermMemory, TicketMemory
from backend.services.llm.prompt_builder import PromptBuilder
from backend.services.llm.llm_generator import LLMGenerator
from backend.services.core.intent_recognizer import (
    IntentRecognizer,
    QUERY_KNOWLEDGE, QUERY_TICKET, CHITCHAT, REQUEST_HUMAN,
)
from backend.services.llm.chitchat import ChitchatEngine
from backend.services.memory.ticket_service import TicketQueryService

logger = logging.getLogger(__name__)

# 预定义类别
CATEGORIES = {
    "企业概况": "公司简介、公司全称、成立时间、总部地点、企业性质、使命愿景价值观、"
              "发展历程与里程碑事件、关键运营数据（GMV/用户/商家/SKU）、组织架构、联系方式",
    "人事行政": "公司考勤制度、请假流程、年假病假、报销标准、出差审批、工资福利、入职离职、人事行政规定",
    "技术运维": "服务器部署、系统配置、故障排查、命令行操作、重启维护、环境搭建、数据库管理、网络安全",
    "产品FAQ": "产品功能介绍、客户常见问题、收费标准、使用教程、售后服务、账号管理、功能更新",
    # 新增：财务结算（平台抽成/发票/退款到账）
    "财务结算": "发票开具、对账核对、结算周期、退款到账、平台佣金、账单查询、打款提现、费用标准",
    # 新增：物流配送（发货/物流/配送范围）
    "物流配送": "发货时效、运费标准、物流查询、配送范围、快递时效、签收异常、拒收处理、仓储发货",
    # 新增：营销活动（优惠/会员/积分）
    "营销活动": "优惠券、满减活动、促销规则、会员权益、积分兑换、活动报名、推广资源位、宣传物料",
    # 新增：售后维权（退换货/投诉/赔付）
    "售后维权": "退换货政策、质量问题处理、投诉渠道、售后时效、赔付标准、纠纷处理、假一赔三",
    # 新增：商家服务（B端入驻/店铺/结算）
    "商家服务": "商家入驻、入驻条件、店铺管理、商家后台、结算提现、商家培训、经营报表、产业带赋能",
    # 新增：安全合规（账号/隐私/监管）
    "安全合规": "账号安全、隐私保护、数据安全、信息合规、实名认证、权限管理、未成年人保护、监管要求",
}

# 转人工话术
HANDOFF_REPLY = "已为您转接人工客服，正在安排中，请稍候，人工客服将尽快回复您。"

class RAGService:
    """RAG 问答服务（线程安全要求不高，FastAPI 多 worker 部署时短期记忆需换 Redis）"""

    def __init__(self):
        logger.info("RAGService 初始化...")

        # embed_function 必须注入 embed_texts：类别描述也要向量化
        self.vs = VectorStore(embed_function=embed_texts)
        self.vs.seed_categories(CATEGORIES)

        self.retriever = Retriever(self.vs)
        self.memory = MemoryService(max_turns=5, on_batch_ready=self._on_batch_ready)  # 对话窗口保留 5 轮，满一批触发画像提取
        self.tm = TicketMemory()
        self.ticket_service = TicketQueryService(self.tm)
        self.pb = PromptBuilder()
        self.gen = LLMGenerator()
        self.chitchat = ChitchatEngine()
        self._session_intents = {}
        self._last_intent = {}
        self.intent_recognizer = IntentRecognizer()

        # 画像（MySQL）：连不上不阻塞主流程
        try:
            self.ltm = LongTermMemory()
        except Exception as e:
            logger.warning(f"RAGService 初始化时连接 MySQL 失败：{e}")
            self.ltm = None

        logger.info("RAGService 初始化完成")

    # ==========================================================
    # 文档入库
    # ==========================================================
    def ingest_file(self, file_path: str, doc_id: str = None, doc_name: str = None) -> int:
        """
        单个文档入库：解析 → 清洗/分块 → 向量化 → 自动分类入库。
        同 doc_id 重复入库会覆盖旧 chunk（upsert）。
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"文件不存在：{file_path}")
            return 0

        doc_id = doc_id or path.stem
        doc_name = doc_name or path.name

        logger.info(f"解析文件：{path.name}")
        fragments = FileParser.parse(str(path))
        chunks = DocumentChunker().chunk_document(fragments, doc_id, doc_name)
        logger.info(f"{doc_name} 文件分块分块完成：{len(chunks)} 个 chunk")

        emb_chunks = generate_embedding(chunks)
        self.vs.add_chunks(emb_chunks)
        logger.info(f"入库完成：{path.name}（doc_id={doc_id}）")
        return len(chunks)

    # ==========================================================
    # 问答链路
    # ==========================================================
    @staticmethod
    def _build_tickets_text(ticket_hits: List[Dict]) -> str:
        """把工单检索结果拼成 prompt 文本（带相似度与出现次数），无结果返回空串"""
        if not ticket_hits:
            return ""
        parts = []
        for i, h in enumerate(ticket_hits, 1):
            meta = h["meta"]
            occ = meta.get("occurrences", 1)
            parts.append(f"工单{i}（相似度{h['score']:.2f}，出现{occ}次）：\n{h['text']}")
        return "\n\n".join(parts)

    def _build_messages(self, question: str, session_id: str, user_id: str) -> List[Dict]:
        """检索 → 工单 → 画像 → 记忆 → 拼装 messages（问答链路中不变的部分）"""
        # 混合检索知识
        hits = self.retriever.retrieve(question, recall_k=15, final_top_k=5)
        knowledge = self.retriever.build_context(hits)

        # 历史工单（阈值门控）
        ticket_hits = self.tm.query_tickets(question, top_k=3)
        tickets_text = self._build_tickets_text(ticket_hits)

        # 用户画像
        profile_text = ""
        if self.ltm:
            profile_text = self.ltm.get_profile_text(user_id)

        # 短期记忆上下文
        memory_context = self.memory.get_context(session_id)

        return self.pb.build_messages(
            question=question,
            knowledge=knowledge,
            tickets=tickets_text,
            profile_text=profile_text,
            memory_context=memory_context,
        )

    def ask_stream(self, question: str, session_id: str = "cli_user", user_id: str = "cli_user") -> Iterator[str]:
        """
        流式问答（主入口）：记忆记问 → 拼 prompt → 逐块 yield → 记忆记答。
        调用方负责消费生成器；FastAPI 可包成 StreamingResponse。
        """
        start_all = time.time()

        # 0.意图识别 → 分流（转人工/闲聊/工单查询的模块内容后续细化）
        prev_intent = self._last_intent.get(session_id)
        intent = self.intent_recognizer.recognize(question, prev_intent)
        self._last_intent[session_id] = intent
        logger.info(f"意图识别：{intent}（上一轮：{prev_intent}）")
        self._session_intents.setdefault(session_id, set()).add(intent)

        # 1.短期记忆记录用户问题（所有分支共用）
        self.memory.add(session_id, "user", question, user_id=user_id)

        # ---- 分流：request_human → 转人工（简化版） ----
        if intent == REQUEST_HUMAN:
            yield HANDOFF_REPLY
            self.memory.add(session_id, "assistant", HANDOFF_REPLY, user_id=user_id)
            logger.info(f"[转人工] 用户要求人工客服：{question}")
            return

        # ---- 分流：chitchat → 闲聊（预设话术，不走检索） ----
        if intent == CHITCHAT:
            history = self.memory.to_messages(session_id)
            parts = []
            for piece in self.chitchat.chat_stream(question, history):
                parts.append(piece)
                yield piece
            reply = "".join(parts)
            self.memory.add(session_id, "assistant", reply, user_id=user_id)
            logger.info(f"闲聊完成：耗时 {time.time() - start_all:.2f}s")
            return

        # ---- 分流：query_ticket → 工单查询（简化版） ----
        if intent == QUERY_TICKET:
            reply = self.ticket_service.answer(question, user_id)
            yield reply
            self.memory.add(session_id, "assistant", reply, user_id=user_id)
            return

        # ---- query_knowledge：现有 RAG 流程（检索/工单/画像/记忆） ----
        messages = self._build_messages(question, session_id, user_id)

        # 流式生成（Deepseek → 本地降级 → 兜底）
        parts = []
        for piece in self.gen.generate_stream(messages):
            parts.append(piece)
            yield piece

        # 生成结束后记录回答
        answer = "".join(parts)
        self.memory.add(session_id, "assistant", answer, user_id=user_id)
        logger.info(f"问答完成：耗时 {time.time() - start_all:.2f}s")

    def ask(self, question: str, session_id: str = "cli_user", user_id: str = "cli_user") -> str:
        """非流式便捷版：内部消费生成器，返回完整回答文本"""
        return "".join(self.ask_stream(question, session_id, user_id))

    # ==========================================================
    # 会话归档（退出/会话结束时调用）
    # ==========================================================
    def _on_batch_ready(self, session_id: str, batch: List[dict]) -> None:
        """MemoryService 回调：窗口满一批 K 轮新对话，后台线程异步提取画像，不阻塞回答"""
        if not self.ltm:
            return
        user_id = self.memory.get_user(session_id)
        t = threading.Thread(
            target=self._extract_profile_async,
            args=(user_id, batch),
            daemon=False,  # 非守护线程：程序退出前等待画像提取完成（避免归档丢数据）
        )
        t.start()

    def _extract_profile_async(self, user_id: str, batch: List[dict]) -> None:
        """后台执行的画像提取：窗口快照逐条拼接，交给 LLM 提取并入库"""
        text = "\n".join(f"{role}: {content}" for role, content in batch)
        try:
            self.ltm.extract_and_save(user_id, text)
            logger.info(f"用户 {user_id} 画像提取完成（{len(batch)} 条窗口对话）")
        except Exception as e:
            logger.error(f"用户 {user_id} 画像提取失败：{e}")

    def archive_session(self, session_id: str = "cli_user", user_id: str = "cli_user") -> None:
        """
        会话结束归档（简单版，结束判断后期再细化）：
        1. 画像兜底：窗口内"未满一批未提取过"的对话补一次提取；
        2. 工单兜底：整个会话（摘要+窗口）提炼成一张工单入库。
        """
        # ---------- 1. 画像兜底提取 ----------
        if self.ltm:
            pending = self.memory.archive_pending(session_id)
            if pending:
                text = "\n".join(f"{role}: {content}" for role, content in pending)
                try:
                    self.ltm.extract_and_save(user_id, text)
                    logger.info(f"用户 {user_id} 画像兜底提取完成（{len(pending)} 条）")
                except Exception as e:
                    logger.error(f"用户 {user_id} 画像兜底提取失败：{e}")
            else:
                logger.info("最后一整批画像已提取过，跳过画像归档")

        # ---------- 2. 工单兜底提炼 ----------
        intents = self._session_intents.get(session_id, set())
        if not (intents & {QUERY_KNOWLEDGE, QUERY_TICKET}):
            logger.info("本次会话无实质业务问题（纯闲聊），跳过工单提炼")
            return
        ctx = self.memory.get_context(session_id)
        text = f"【对话摘要】\n{ctx['summary']}\n【最近对话】\n"
        text += "\n".join(f"{role}: {content}" for role, content in ctx["window"])
        if not ctx["summary"] and not ctx["window"]:
            logger.info("本次会话无内容，跳过工单归档")
            return
        try:
            ticket = self.tm.summarize_to_ticket(text)
            if ticket["question"] and ticket["resolution"]:
                self.tm.add_ticket(
                    user_id, ticket["question"], ticket["process"],
                    ticket["resolution"], ticket["category"],
                )
                logger.info(f"工单已归档：{ticket['question'][:30]}")
        except Exception as e:
            logger.error(f"工单提炼失败：{e}")
