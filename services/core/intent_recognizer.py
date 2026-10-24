"""
意图识别模块（语义版）：判断用户消息的意图，决定走哪条处理路径。

识别链路：
    第1层 语义相似度匹配：本地 embedding（Qwen3-Embedding-0.6B）把用户消息
         与各意图"种子句"比对，最高相似度 ≥ 阈值即命中。
         零 LLM 调用、一次 embedding 几十 ms，比关键词正则理解力强得多。
    第2层 LLM 兜底：语义层未命中（相似度低于阈值）才调 ollama 分类。
    默认   query_knowledge：两层都未命中时的安全兜底（宁可多检索，不误转人工）。

query_knowledge 是开放式意图（任何业务问题），无法用种子句覆盖，
所以始终是"默认"；只有 request_human / query_ticket / chitchat 三种特殊意图显式匹配。
"""

import json
import logging
import math
import re
from typing import List, Dict, Optional

from services.vector.embedding_service import embed_texts
from services.llm.llm_client import get_client, MODEL

logger = logging.getLogger(__name__)

# ===================== 4 类意图 =====================
QUERY_KNOWLEDGE = "query_knowledge"
QUERY_TICKET = "query_ticket"
CHITCHAT = "chitchat"
REQUEST_HUMAN = "request_human"
INTENTS = [QUERY_KNOWLEDGE, QUERY_TICKET, CHITCHAT, REQUEST_HUMAN]

# ===================== 第1层：意图种子句（代表"纯意图"的口语化表达） =====================
# 注意：query_knowledge 开放式，不作种子句，始终为默认兜底
INTENT_SEEDS = {
    REQUEST_HUMAN: [
        "我想转人工客服",
        "请帮我接人工",
        "我要找真人客服",
        "麻烦转人工服务",
        "给我转人工处理",
        "人工客服帮我一下",
        "帮我联系人工客服",
        "我要投诉，找人工",
    ],
    QUERY_TICKET: [
        "我的工单处理到哪了",
        "帮我查询工单进度",
        "催办一下我的工单",
        "我的工单现在什么状态",
        "工单什么时候能处理完",
        "我的报销单到哪一步了",
    ],
    CHITCHAT: [
        "你好呀",
        "今天天气不错",
        "随便聊聊",
        "你叫什么名字",
        "谢谢你",
        "在吗",
        "真无聊，陪我聊聊天",
        "哈哈",
    ],
}

# 语义层命中阈值：只拦截"极度明确的意图"（与种子句几乎同义的表达，≥0.75）。
# 实测短句相似度扎堆在 0.4~0.65，不同意图分数差仅 0.01~0.09，单靠阈值切不开边界，
# 所以语义层只做零延迟的"强信号门"；未命中一律交 LLM 分类（主力判断）。
SEMANTIC_THRESHOLD = 0.75

# ===================== LLM 兜底分类的 Prompt =====================
_CLASSIFY_SYSTEM = (
    "你是智能客服的意图分类器。判断用户消息的意图，只输出一个 JSON，不要输出任何其他内容。\n"
    '格式：{"intent": "意图", "reason": "一句话原因"}\n'
    "意图只允许以下四种：\n"
    "query_knowledge：查询知识/制度/流程/业务问题\n"
    "query_ticket：查询自己的工单/催办进度\n"
    "chitchat：打招呼/闲聊/寒暄/情绪表达\n"
    "request_human：明确要求人工客服介入（如'转人工''找真人'），或带有强烈不满情绪的投诉。\n"
    "判断原则：\n"
    "1. 仅询问流程/制度/号码的，即使提到'投诉''人工''电话'，也是 query_knowledge，"
    "例如'投诉流程是什么'；\n"
    "2. 不确定时优先 query_knowledge；有实质业务问题优先 query_knowledge；\n"
    "3. 只有当用户明确要求联系人工/表达强烈不满时才判 request_human。"
)


def _cos_sim(a: List[float], b: List[float]) -> float:
    """余弦相似度（a、b 为 embedding 向量）"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class IntentRecognizer:
    """意图识别：语义相似度第一层 + LLM 兜底，失败默认 query_knowledge"""

    def __init__(self, threshold: float = SEMANTIC_THRESHOLD):
        self.threshold = threshold
        self._seed_vectors = None   # {intent: [向量...]}，懒加载只算一次
        self._client = None         # ollama 客户端，懒加载

    def recognize(self, question: str, prev_intent: Optional[str] = None) -> str:
        """识别意图：返回 4 类之一（永不抛异常）

        :param prev_intent: 上一轮意图（会话状态，用于"闲聊延续"）：
            上一轮是 chitchat 时，本轮除非语义强信号命中其他意图、或 LLM 判定为
            明确的业务问题，否则延续 chitchat——避免闲聊中的承接语（"嗯嗯""然后呢"）
            被误判成 query_knowledge 而误走检索。
        """
        question = (question or "").strip()
        if not question:
            return QUERY_KNOWLEDGE
        # 第一层：语义相似度匹配
        intent = self._semantic_match(question)
        if intent:
            logger.info(f"意图识别（语义强信号）：{intent}")
            return intent
        # 第二层：LLM 兜底分类
        intent = self._llm_classify(question, prev_intent)
        if intent:
            logger.info(f"意图识别（LLM 分类）：{intent}")
            return intent

        logger.info(f"意图识别（默认兜底）：{QUERY_KNOWLEDGE}")
        return QUERY_KNOWLEDGE

    # ----------------------------------------------------------
    def _semantic_match(self, question: str) -> Optional[str]:
        """语义匹配：消息向量与各意图种子句求余弦，最高分 ≥ 阈值即命中"""
        if self._seed_vectors is None:
            self._seed_vectors = self._build_seed_vectors()
        qvec = embed_texts([question])[0]

        best_intent, best_score = None, 0.0
        for intent, seed_vecs in self._seed_vectors.items():
            for svec in seed_vecs:
                score = _cos_sim(qvec, svec)
                if score > best_score:
                    best_intent, best_score = intent, score

        if best_score >= self.threshold:
            logger.info(f"语义强信号命中：{best_intent}（{best_score:.3f} ≥ {self.threshold}）")
            return best_intent
        logger.info(f"语义相似度均低于阈值（最高 {best_score:.3f}），交 LLM 兜底")
        return None

    # ----------------------------------------------------------
    def _build_seed_vectors(self) -> Dict[str, List[List[float]]]:
        """种子句批量向量化（懒加载，进程内只算一次）"""
        vectors = {}
        for intent, seeds in INTENT_SEEDS.items():
            vectors[intent] = embed_texts(seeds)
            logger.info(f"意图种子向量就绪：{intent}（{len(seeds)} 句）")
        return vectors

    # ----------------------------------------------------------
    def _llm_classify(self, question: str, prev_intent: Optional[str] = None) -> Optional[str]:
        """LLM 分类：ollama 本地模型，输出 JSON；失败返回 None（调用方兜底默认意图）

        :param prev_intent: 上一轮意图。闲聊中（prev=chitchat）会给 LLM 注入上下文，
            要求除非明确的业务信号，否则延续闲聊，解决承接语误判问题。
        """
        user_content = f"用户消息：{question}"
        if prev_intent == CHITCHAT:
            user_content = (
                "【上下文】你正在与用户闲聊，上一条消息已判定为 chitchat（闲聊）。\n"
                "除非本轮消息是明确的业务问题（查知识/制度/流程/费用）、工单查询或"
                "要求人工客服，否则应保持 chitchat。\n"
                f"用户消息：{question}"
            )
        try:
            if self._client is None:
                self._client = get_client()
            resp = self._client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": _CLASSIFY_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
                max_tokens=100,
            )
            text = resp.choices[0].message.content.strip()
            m = re.search(r"\{[^{}]*\}", text, re.S)
            if not m:
                logger.warning(f"意图分类输出非 JSON：{text[:100]}")
                return None
            intent = json.loads(m.group(0)).get("intent")
            if intent in INTENTS:
                return intent
            logger.warning(f"意图分类输出非法意图：{intent}")
            return None
        except Exception as e:
            logger.warning(f"意图分类调用失败（降级为默认意图）：{e}")
            return None
