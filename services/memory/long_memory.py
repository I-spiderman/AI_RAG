import json
import logging
import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional
import pymysql
import chromadb
from services.vector.embedding_service import embed_texts, ChromaEmbeddingFunction
from services.llm.llm_client import get_client, MODEL
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

MYSQL_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": os.getenv("mysql_password"),
    "database": "rag_service",
    "charset": "utf8mb4",
    "connect_timeout": 5,
}

def _chat_retry(system_prompt: str, user_content: str,
                validator, max_retries: int = 3) -> str:
    """
    带格式校验的 LLM 调用：输出不过校验自动重试，超过 max_retries 次熔断。

    :param system_prompt: 系统提示词
    :param user_content: 用户内容
    :param validator: 校验函数 text -> (ok: bool, reason: str)
                        ok=True 表示格式合格；reason 是格式不合格的原因（用于反馈给模型）
    :param max_retries: 最大重试次数（含第一次），默认3
    :return: 最后一次模型输出。熔断时不抛异常，已记录错误日志
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    last_text = ""
    for attempt in range(1, max_retries + 1):
        try:
            resp = get_client().chat.completions.create(
                model=MODEL, messages=messages, temperature=0.2, max_tokens=600
            )
            last_text = resp.choices[0].message.content.strip()
        except Exception as e:
            # 网络/服务异常也算一次失败，记录后重试
            logger.error(f"第{attempt}次调用失败：{e}")
            continue

        ok, reason = validator(last_text)
        if ok:
            return last_text

        # 格式不对：把上次输出 + 纠正要求追加进对话，让模型带着反馈重来
        logger.warning(f"模型输出格式不符（第{attempt}次）：{reason}")
        messages.append({"role": "assistant", "content": last_text})
        messages.append({
            "role": "user",
            "content": f"你的输出不符合要求：{reason}。请严格按照要求的格式重新输出，不要解释、不要多余内容。"
        })
    logger.error(f"模型连续{max_retries}次未按格式输出，已熔断，返回最后一次输出")
    return last_text

class LongTermMemory:
    """用户画像：MySQL 持久化，按 user_id 读写"""
    def __init__(self, config: Dict = None):
        self.config = config or MYSQL_CONFIG
        self._ensure_db_and_table()

    def _ensure_db_and_table(self) -> None:
        """数据库/表不存在时自动创建"""
        cfg = dict(self.config)
        db_name = cfg.pop('database')
        # 建库
        conn = None
        try:
            conn = pymysql.connect(**cfg)
            with conn.cursor() as cur:
                cur.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{db_name}` DEFAULT CHARACTER SET utf8mb4"
                )
            conn.commit()
        except pymysql.Error as e:
            logger.error(f"创建数据库{db_name}失败：{e}")
            raise
        finally:
            if conn:
                conn.close()
        # 连库 建表
        conn = None
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(
                """CREATE TABLE IF NOT EXISTS user_profile (
                    user_id      VARCHAR(64) PRIMARY KEY,
                    profile_json JSON,
                    updated_at   DATETIME
                )"""
            )
            conn.commit()
        except Exception as e:
            logger.error(f"创建表user_profile失败：{e}")
            raise
        finally:
            if conn:
                conn.close()
        logger.info(f"用户画像表就绪（MySQL：{db_name}）")

    def _get_conn(self) -> pymysql.connections.Connection:
        return pymysql.connect(**self.config)

    def _get_profile_json(self, user_id: str) -> Dict:
        conn = None
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute("SELECT profile_json FROM user_profile WHERE user_id=%s", (user_id,))
                row = cur.fetchone()
        except Exception as e:
            logger.error(f"读取用户profile失败, user_id={user_id}, err:{e}", exc_info=True)
            raise
        finally:
            if conn:
                conn.close()
        if row and row[0]:
            return json.loads(row[0])
        return {"identity": [], "preference": [], "conclusion": []}

    def _save_profile(self, user_id: str, profile: Dict) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        profile_json = json.dumps(profile, ensure_ascii=False)
        conn = None
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO user_profile (user_id, profile_json, updated_at)"
                    "VALUES (%s, %s, %s)"
                    "ON DUPLICATE KEY UPDATE profile_json=%s, updated_at=%s",
                    (user_id, profile_json, now, profile_json, now)
                )
            conn.commit()
        except Exception as e:
            logger.error(f"保存用户profile失败, user_id={user_id}, err:{e}", exc_info=True)
            raise
        finally:
            if conn:
                conn.close()

    def upsert_fact(self, user_id: str, fact_type: str, content: str, max_per_type: int = 20) -> None:
        """写入一条画像事实（同类型同内容去重，每类最多保留 max_per_type 条）"""
        profile = self._get_profile_json(user_id)
        facts = profile.setdefault(fact_type, [])
        if content not in facts:
            facts.append(content)
            profile[fact_type] = facts[-max_per_type:]
            self._save_profile(user_id, profile)
            logger.info(f"用户 {user_id} 新增 {fact_type} 事实：{content[:30]}")
        else:
            logger.info(f"用户 {user_id} 的 {fact_type} 事实已存在，跳过：{content[:30]}")

    def get_profile_text(self, user_id: str) -> str:
        """把画像拼成文本，新会话开始时注入 prompt"""
        profile = self._get_profile_json(user_id)
        lines = []
        for fact_type in ("identity", "preference", "conclusion"):
            facts = profile.get(fact_type, [])
            if not facts:
                continue
            lines.append(f"{fact_type}:{json.dumps(facts, ensure_ascii=False)}")
        return "\n".join(lines)

    def extract_and_save(self, user_id: str, session_summary: str) -> List[Dict]:
        """用 LLM 从会话摘要中提取/更新用户画像"""
        if not session_summary.strip():
            return []
        old_profile = self.get_profile_text(user_id)
        system_prompt = (
            "你是用户画像提取助手。根据【本次会话摘要】和【已有画像】，"
            "提取该用户的长期特征。\n"
            "画像分为三类，定义如下：\n"
            "- identity（身份信息）：用户的姓名、工号、职位、公司、部门、联系方式等"
            "标识与角色信息，例如\"工号C888\"、\"市场部经理\"；\n"
            "- preference（偏好）：用户喜欢的沟通方式、习惯做法等，"
            "例如\"回复尽量简短\"、\"偏好文字沟通\"；\n"
            "- conclusion（结论与标准）：用户确认过的规则、额度、流程、标准，"
            "例如\"住宿报销上限500元/晚\"、\"年假需提前3天申请\"；\n"
            "规则：\n"
            "1. 只提取上述三类稳定长期信息，宁多勿漏；拿不准归类时，"
            "归入最接近的类型，也不要漏掉；\n"
            "2. 剔除一次性话题、寒暄、临时决定；\n"
            "3. 对照【已有画像】：若新信息与已有条目语义相同，视为已存在，不要重复输出；\n"
            "4. 若新信息与旧画像冲突，以新信息为准；\n"
            "5. 输出格式：每行一条 \"类型: 内容\"，类型只能是 "
            "identity / preference / conclusion，不要输出其他行。"
        )
        user_content = (
            f"【已有画像】\n{old_profile or '（无）'}\n\n"
            f"【本次会话摘要】\n{session_summary}\n\n请提取新画像事实："
        )
        def validator(text: str):
            """至少解析出 1 条合法画像事实才算格式合格"""
            for line in text.splitlines():
                line = line.strip()
                if ":" not in line:
                    continue
                fact_type, _, content = line.partition(":")
                if (fact_type.strip() in ("identity", "preference", "conclusion") and content.strip()):
                    return True, ""
            return False, "没有解析出任何 '类型: 内容' 格式的行（类型只能是 identity/preference/conclusion）"
        result = _chat_retry(system_prompt, user_content, validator)
        new_facts = []
        for line in result.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            fact_type, _, content = line.partition(":")
            fact_type, content = fact_type.strip(), content.strip()
            if fact_type not in ("identity", "preference", "conclusion"):
                continue
            content = content.replace("（已更新）", "").replace("(已更新)", "")
            self.upsert_fact(user_id, fact_type, content)
            new_facts.append({"fact_type": fact_type, "content": content})
        if new_facts:
            logger.info(f"用户 {user_id} 新提取 {len(new_facts)} 条画像事实")
        else:
            logger.info(f"用户 {user_id} 未提取到任何画像事实")
        return new_facts

    def clear(self, user_id: str) -> None:
        """清空用户画像"""
        conn = None
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute("DELETE FROM user_profile WHERE user_id=%s", (user_id,))
                conn.commit()
        except Exception as e:
            logger.error(f"清空用户画像失败, user_id={user_id}, err:{e}", exc_info=True)
            raise
        finally:
            if conn:
                conn.close()

class TicketMemory:
    """历史工单：ChromaDB 向量存储，写入去重、检索阈值门控"""
    def __init__(self, persist_dir: Optional[str] = None, dedup_threshold: float = 0.85, min_score: float = 0.6):
        if persist_dir is None:
            # 工单库目录：项目根 data/ticket_memory（文件在 services/memory/ 下，上三级为项目根）
            persist_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "data", "ticket_memory",
            )
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.col = self.client.get_or_create_collection(
            name="tickets",
            metadata={"hnsw:space": "cosine"},
            embedding_function=ChromaEmbeddingFunction(),
        )
        self.dedup_threshold = dedup_threshold
        self.min_score = min_score
        logger.info(f"工单库准备就绪： tickets={self.col.count()} 条")

    @staticmethod
    def _build_ticket_text(question: str, process: str, resolution: str, category: str = "") -> str:
        return (
            f"【工单】分类：{category or '未分类'}\n"
            f"问题：{question}\n"
            f"处理过程：{process}\n"
            f"结果：{resolution}"
        )

    def add_ticket(self, user_id: str, question: str, process: str,
                   resolution: str, category: str = "未分类") -> Dict:
        """写入工单，相似度 >= dedup_threshold 则合并计数不新增"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        doc = self._build_ticket_text(question, process, resolution, category)
        emb = embed_texts([doc])[0]

        hits = self.col.query(query_embeddings=[emb], n_results=1)
        if hits["ids"] and hits["ids"][0]:
            dist = hits["distances"][0][0]
            score = 1 - dist
            if score >= self.dedup_threshold:
                tid = hits["ids"][0][0]
                old_meta = hits["metadatas"][0][0]
                new_occ = old_meta.get("occurrences", 1) + 1
                self.col.update(
                    ids=[tid],
                    metadatas=[{
                        **old_meta,
                        "occurrences": new_occ,
                        "last_occurred_at": now
                    }]
                )
                logger.info(f"[工单] 重复工单已合并计数: {tid} x{new_occ}")
                return {"ticket_id": tid, "action": "merged", "occurrences": new_occ}

        tid = uuid.uuid4().hex[:12]
        self.col.add(
            ids=[tid],
            documents=[doc],
            embeddings=[emb],
            metadatas=[{
                "ticket_id": tid,
                "user_id": user_id,
                "category": category,
                "resolution": resolution,
                "created_at": now,
                "occurrences": 1,
                "first_occurred_at": now,
                "last_occurred_at": now
            }]
               )
        logger.info(f"[工单] 新增工单: {tid}")
        return {"ticket_id": tid, "action": "added", "occurrences": 1}

    def query_tickets(self, question: str, top_k: int = 5,
                      user_id: Optional[str]=None, category: Optional[str]=None) -> List[dict]:
        """查询相关工单，相似度 < min_score 则不返回"""
        qvec = embed_texts([question])[0]
        where = {}
        if user_id:
            where["user_id"] = user_id
        if category:
            where["category"] = category
        results = self.col.query(
            query_embeddings=[qvec], n_results=top_k, where=where or None
        )
        hits = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                score = 1 - results["distances"][0][i]
                if score < self.min_score:
                    continue
                hits.append({
                    "text": results["documents"][0][i],
                    "score": round(score, 4),
                    "meta": results["metadatas"][0][i]
                })
        return hits

    def summarize_to_ticket(self, conversation: str, category: str = "") -> Dict:
        """用 LLM 把对话提炼成工单结构化摘要 """
        system_prompt = (
            "你是客服工单整理助手。把【对话记录】提炼成一张结构化工单。\n"
            "输出格式（严格四行，必须全部输出）：\n"
            "问题：用户提出的原始问题\n"
            "处理过程：客服排查/解答的关键步骤\n"
            "结果：最终答复或处理结论\n"
            "分类：问题所属类别（如 报销/请假/故障/账号）"
        )
        user_content = f"【对话记录】：\n{conversation}"

        def validator(text: str):
            """四行字段都必须出现且非空"""
            missing = []
            for label, key in (
                    ("问题", "question"),
                    ("处理过程", "process"),
                    ("结果", "resolution"),
                    ("分类", "category"),
            ):
                if not any(
                    line.startswith(f"{label}：") and line[len(label) + 1:].strip()
                    for line in text.splitlines()
                ):
                    missing.append(label)
            if missing:
                return False, f"缺少 {'、'.join(missing)} 这几行，请补全后按四行格式重新输出"
            return True, ""
        text = _chat_retry(system_prompt, user_content, validator)
        result = {"question": "", "process": "", "resolution": "", "category": category}
        for line in text.splitlines():
            for key, label in (
                    ("question", "问题"),
                    ("process", "处理过程"),
                    ("resolution", "结果"),
                    ("category", "分类"),
            ):
                if line.startswith(f"{label}："):
                    result[key] = line[len(label) + 1:].strip()
        return result

    def count(self) -> int:
        return self.col.count()

    def delete_ticket(self, ticket_id: str) -> None:
        self.col.delete(ids=[ticket_id])
        logger.info(f"[工单] 删除工单: {ticket_id}")




