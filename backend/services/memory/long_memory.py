import json
import logging
import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional
import pymysql
import chromadb
from backend.services.vector.embedding_service import embed_texts, ChromaEmbeddingFunction
from backend.services.llm.llm_client import get_client, MODEL
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

MYSQL_CONFIG = {
    "host": os.getenv("mysql_host", "localhost"),
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
        # MySQL 工单表（结构化主存：管理端按状态/来源查询与状态流转用）；
        # 连不上不阻塞主流程，降级为仅 Chroma 向量检索。
        try:
            self._ensure_tickets_table()
        except Exception:
            logger.warning("工单 MySQL 表初始化失败（降级为仅 Chroma）", exc_info=True)
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
                   resolution: str, category: str = "未分类",
                   source: str = "auto", status: str = "已解决") -> Dict:
        """写入工单，相似度 >= dedup_threshold 则合并计数不新增。
        source: auto(会话结束自动提炼) / human_request(用户转人工)
        status: 已解决 / 待处理 / 处理中 / 已关闭
        """
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
                "last_occurred_at": now,
                # 来源与状态：自动提炼 / 用户转人工；已解决 / 待处理 / 处理中 / 已关闭
                "source": source,
                "status": status,
            }]
               )
        logger.info(f"[工单] 新增工单: {tid}（source={source}, status={status}）")
        # 双写 MySQL（结构化主存：管理端按状态/来源查询与状态流转用；Chroma 仅用于向量检索）
        self.upsert_ticket_mysql(tid, user_id, question, process, resolution,
                                category, source, status)
        return {"ticket_id": tid, "action": "added", "occurrences": 1, "source": source, "status": status}

    # ---------------- MySQL 结构化工单主存（管理端查询 / 状态流转） ----------------
    # Chroma 仍保留用于相似工单向量检索；MySQL 负责结构化筛选、认领、结单等运营操作。
    def _ensure_tickets_table(self) -> None:
        """建 tickets 表（连不上不抛异常，降级为仅 Chroma）"""
        conn = None
        try:
            conn = pymysql.connect(**MYSQL_CONFIG)
            with conn.cursor() as cur:
                cur.execute(
                    """CREATE TABLE IF NOT EXISTS tickets (
                        ticket_id  VARCHAR(32) PRIMARY KEY,
                        user_id    VARCHAR(64),
                        question   TEXT,
                        process    TEXT,
                        resolution TEXT,
                        category   VARCHAR(64) DEFAULT '未分类',
                        source     VARCHAR(32) DEFAULT 'auto',
                        status     VARCHAR(32) DEFAULT '已解决',
                        assignee   VARCHAR(64) DEFAULT '',
                        created_at DATETIME,
                        updated_at DATETIME,
                        INDEX idx_tickets_status (status)
                    )"""
                )
            conn.commit()
            logger.info("工单 MySQL 表就绪（tickets）")
        except Exception as e:
            logger.warning(f"工单 MySQL 表初始化失败（降级为仅 Chroma）：{e}")
        finally:
            if conn:
                conn.close()

    def upsert_ticket_mysql(self, ticket_id, user_id, question, process, resolution,
                            category="未分类", source="auto", status="已解决") -> None:
        """写入/更新一条工单到 MySQL（按 ticket_id 幂等）"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = None
        try:
            conn = pymysql.connect(**MYSQL_CONFIG)
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO tickets
                       (ticket_id,user_id,question,process,resolution,category,source,status,created_at,updated_at)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON DUPLICATE KEY UPDATE
                       question=VALUES(question), process=VALUES(process),
                       resolution=VALUES(resolution), category=VALUES(category),
                       source=VALUES(source), status=VALUES(status), updated_at=VALUES(updated_at)""",
                    (ticket_id, user_id, question, process, resolution, category, source, status, now, now),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"[工单] MySQL 写入失败（降级）：{e}")
        finally:
            if conn:
                conn.close()

    def update_ticket_mysql(self, ticket_id, **fields) -> bool:
        """更新 MySQL 工单字段（resolution/status/category/process/assignee）"""
        allowed = ("resolution", "status", "category", "process", "assignee")
        sets = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not sets:
            return False
        sets["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cols = ", ".join(f"{k}=%s" for k in sets)
        conn = None
        try:
            conn = pymysql.connect(**MYSQL_CONFIG)
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE tickets SET {cols} WHERE ticket_id=%s",
                    list(sets.values()) + [ticket_id],
                )
                if cur.rowcount == 0:
                    return False
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"[工单] MySQL 更新失败：{e}")
            return False
        finally:
            if conn:
                conn.close()

    def list_tickets_mysql(self, status=None, source=None, limit=100) -> List[dict]:
        """按状态/来源筛选工单列表（管理端用），最新在前"""
        conn = None
        try:
            conn = pymysql.connect(**MYSQL_CONFIG)
            with conn.cursor(pymysql.cursors.DictCursor) as cur:
                where, params = [], []
                if status:
                    where.append("status=%s")
                    params.append(status)
                if source:
                    where.append("source=%s")
                    params.append(source)
                sql = "SELECT * FROM tickets"
                if where:
                    sql += " WHERE " + " AND ".join(where)
                sql += " ORDER BY created_at DESC LIMIT %s"
                params.append(limit)
                cur.execute(sql, params)
                return cur.fetchall() or []
        except Exception as e:
            logger.error(f"[工单] MySQL 查询失败：{e}")
            return []
        finally:
            if conn:
                conn.close()

    def get_ticket_mysql(self, ticket_id) -> Optional[dict]:
        """按 ticket_id 取单条工单（管理端详情用）"""
        conn = None
        try:
            conn = pymysql.connect(**MYSQL_CONFIG)
            with conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute("SELECT * FROM tickets WHERE ticket_id=%s", (ticket_id,))
                return cur.fetchone()
        except Exception as e:
            logger.error(f"[工单] MySQL 查询失败：{e}")
            return None
        finally:
            if conn:
                conn.close()

    def create_handoff_ticket(self, user_id: str, question: str) -> Dict:
        """用户主动转人工：总是落一张全新的「待处理」工单并同步返回 ticket_id。

        - 不走 add_ticket 的去重合并：每次转人工都是独立事件，需可追踪、
          不能并入已解决的历史工单（否则状态会被旧单覆盖，永远停在「已解决」）；
        - source=human_request 与自动提炼工单（source=auto）区分，便于管理端筛选；
        - 纯「转人工」场景可能没有 open 业务线程，这里保证一定能建单并回传工单号。
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        category = "人工转接"
        process = "用户主动要求转人工客服，等待人工接手（AI 阶段无处理过程记录）。"
        resolution = "待人工客服处理"
        doc = self._build_ticket_text(question, process, resolution, category)
        emb = embed_texts([doc])[0]
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
                "last_occurred_at": now,
                "source": "human_request",
                "status": "待处理",
            }]
        )
        logger.info(f"[转人工] 新增待处理工单: {tid}（user={user_id}）")
        # 双写 MySQL（结构化主存：管理端待处理工单列表/认领/结单用）
        self.upsert_ticket_mysql(tid, user_id, question, process, resolution,
                                category="人工转接", source="human_request", status="待处理")
        return {"ticket_id": tid, "action": "added", "occurrences": 1, "source": "human_request", "status": "待处理"}

    def update_ticket(self, ticket_id: str, **fields) -> bool:
        """人工接手后补录/更新工单字段（resolution / status / category / process 等）。

        仅更新传入的字段，其余保持原值；用于转人工工单被人工处理完后的状态回填。
        返回 True 表示更新成功，False 表示工单不存在。
        """
        try:
            old = self.col.get(ids=[ticket_id], include=["metadatas"])
        except Exception as e:
            logger.error(f"[工单] 查询待更新工单失败: {ticket_id}, {e}")
            return False
        if not old["ids"]:
            logger.warning(f"[工单] 待更新工单不存在: {ticket_id}")
            return False
        old_meta = old["metadatas"][0]
        new_meta = dict(old_meta)
        allowed = ("resolution", "status", "category", "process", "user_id")
        for k in allowed:
            if k in fields and fields[k] is not None:
                new_meta[k] = fields[k]
        new_meta["last_occurred_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.col.update(ids=[ticket_id], metadatas=[new_meta])
        logger.info(f"[工单] 更新工单: {ticket_id}（字段 {list(fields.keys())}）")
        # 同步 MySQL 结构化主存（管理端状态流转用）
        self.update_ticket_mysql(ticket_id, **fields)
        return True

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




