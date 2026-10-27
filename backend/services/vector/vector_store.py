"""
VectorStore：两层 ChromaDB 向量库
- categories 层：类别描述向量，用于入库自动分类与查询自动路由
- chunks 层：分块文本 + 向量 + 元数据，每条带 category 字段
"""

import logging
import os
import re
from typing import List, Dict, Callable, Optional, Tuple

import chromadb

from backend.services.vector.embedding_service import ChromaEmbeddingFunction

logger = logging.getLogger(__name__)

# 类别自动分类的相似度阈值（余弦相似度）。
# top1 相似度 ≥ 此值 → 直接采纳（零延迟、最稳）；
# 否则 → 交本地小模型（ollama）兜底判断，避免"都不高却硬选一个类别"导致检索被错误过滤。
CATEGORY_SIM_THRESHOLD = 0.5


class VectorStore:
    """
    两层向量库（基于 ChromaDB）：

    第一层 categories（类别层）：
        每个类别一条向量，由类别的描述文字 embedding 而来，
        用于入库时自动分类、查询时自动路由。

    第二层 chunks（文本块层）：
        所有分块文本 + 向量 + 元数据，每条都带 category 字段。
    """
    def __init__(self, embed_function: Callable[[List[str]], List[List[float]]] = None):
        # 数据库持久化目录：项目根 data/chroma_db（文件在 services/vector/ 下，上三级为项目根）
        persist_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "chroma_db",
        )
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embed_function = embed_function

        # 第二层：文本块集合，用 cosine 距离（因为embedding是归一化的后的向量）
        self.chunks_col = self.client.get_or_create_collection(
            name="chunks",
            metadata={"hnsw:space": "cosine"},
            embedding_function=ChromaEmbeddingFunction()
        )
        # 第一层：类别集合
        self.categories_col = self.client.get_or_create_collection(
            name="categories",
            metadata={"hnsw:space": "cosine"},
            embedding_function=ChromaEmbeddingFunction()
        )
        logger.info(
            f"向量库初始化完成: chunks={self.chunks_col.count()} 条, "
            f"categories={self.categories_col.count()} 个类别"
        )

    def seed_categories(self, categories: Dict[str, str]) -> None:
        """
        初始化或刷新类别层。把预定义类别的描述文字转成向量存起来。
        只在首次启动、或修改了类别清单时调用一次。

        :param categories: {"人事行政": "考勤、请假、报销等制度", ...}
        """
        if self.embed_function is None:
            raise ValueError("初始化类别需要传入 embed_function（embedding_service.embed_texts）")

        names = list(categories.keys())
        descs = list(categories.values())
        vectors = self.embed_function(descs)

        self.categories_col.upsert(
            ids=names,
            documents=descs,
            metadatas=[{"category": name} for name in names],
            embeddings=vectors
        )
        logger.info(f"已初始化{len(names)}个类别：{names}")

    def classify(self, embedding: List[float], text: str = None,
                 threshold: float = None) -> Dict:
        """
        自动将文档/问题归类（两级判定 + 前二保底）：

        1. 取与类别描述向量最近的类别，算出余弦相似度；
        2. 相似度 ≥ 阈值 → 直接采纳（零延迟、最稳）；
        3. 相似度 < 阈值 且给了文本 → 交云端 DeepSeek 兜底（失败降级本地模型）：
           - 返回有效类别 → 采纳；
           - 返回 '未分类' 或调用失败（两级都无结果）→ 保底取相似度前二类别，扩大检索范围。
        4. 两级都无确定结果时，不再硬选一个类别，而是返回 top2 供上层做「前二类」检索。

        返回 dict：{"category": 单一类别或 None, "top2": [c1, c2], "method": str}
        """
        if self.categories_col.count() == 0:
            return {"category": "未分类", "top2": [], "method": "none"}
        topk = self._classify_topk(embedding, 2)
        top_id, top_sim = topk[0]
        top2 = [c for c, _ in topk]
        # 每次问题归类都先打印「前二相似类别 + 相似度」，便于排查路由结果
        second = (f"；第二名：{topk[1][0]}，相似度：{topk[1][1]:.4f}"
                   if len(topk) > 1 else "")
        logger.info(f"最高相似类别：{top_id}，相似度：{top_sim:.4f}{second}")
        thr = threshold if threshold is not None else CATEGORY_SIM_THRESHOLD
        if top_sim >= thr:
            logger.info(f"自动识别到类别(相似度识别：{top_sim:.4f})：{top_id}")
            return {"category": top_id, "top2": top2, "method": "similarity"}
        # 相似度不够高：交云端 LLM 兜底（失败降级本地）
        if text:
            cats = self._llm_classify_batch([text])
            llm_cat = cats[0] if cats else None
            if llm_cat and llm_cat != "未分类":
                logger.info(f"自动识别到类别(llm识别)：{llm_cat}")
                return {"category": llm_cat, "top2": top2, "method": "llm"}
            # 模型明确"未分类"或调用失败 → 两级均无结果，保底取前二扩大检索
            logger.info(f"两级分类无确定结果（llm={llm_cat}），保底取相似度前二：{top2}")
            return {"category": None, "top2": top2, "method": "top2_fallback"}
        # 没有文本可送 LLM：直接保底前二
        logger.info(f"无文本可分类，保底取相似度前二：{top2}")
        
        return {"category": None, "top2": top2, "method": "top2_fallback"}

    def _classify_topk(self, embedding: List[float], k: int = 2) -> List[Tuple[str, float]]:
        """返回 [(类别名, 余弦相似度), ...] 降序，最多 k 个；库为空返回 []。"""
        if self.categories_col.count() == 0:
            return []
        n = min(k, self.categories_col.count())
        results = self.categories_col.query(query_embeddings=[embedding], n_results=n)
        if not results["ids"] or not results["ids"][0]:
            return []
        return [
            (cid, 1.0 - d)   # chroma 返回余弦距离，转成相似度
            for cid, d in zip(results["ids"][0], results["distances"][0])
        ]

    def _classify_one(self, embedding: List[float]) -> Tuple[str, float]:
        """返回 (最近类别名, 余弦相似度)；库为空时返回 ('未分类', 1.0)。"""
        topk = self._classify_topk(embedding, 1)
        if not topk:
            return "未分类", 1.0
        return topk[0]

    def _load_category_map(self) -> Dict[str, str]:
        """读取类别层：{类别名: 描述文字}"""
        data = self.categories_col.get(include=["documents"])
        cat_map = {}
        for cid, doc in zip(data.get("ids") or [], data.get("documents") or []):
            cat_map[cid] = doc or ""
        return cat_map

    # 单次送给小模型的最大条数：太多会上下文漂移（7B 实测多行会串），分片调用更稳
    LLM_BATCH_SIZE = 15

    def _llm_classify_batch(self, texts: List[str]) -> List[Optional[str]]:
        """
        用本地部署的小模型（ollama）批量做类别兜底判断（自动分片）。
        返回与 texts 等长的列表：类别名 / '未分类' / None（调用失败）。
        兜底原则：宁可判 '未分类'（上游会自动退化为不过滤检索），也不乱塞进错误类别。
        """
        if not texts:
            return []
        results: List[Optional[str]] = []
        for i in range(0, len(texts), self.LLM_BATCH_SIZE):
            results.extend(self._llm_classify_chunk(texts[i:i + self.LLM_BATCH_SIZE]))
        return results

    def _llm_classify_chunk(self, texts: List[str]) -> List[Optional[str]]:
        """对一小批文本调用一次 LLM；输出 '编号|类别'，按编号回填避免行错位。
        主路：云端 DeepSeek；失败降级：本地 ollama。两者都失败返回全 None。"""
        from backend.services.llm.llm_client import (
            get_client, MODEL, get_deepseek_client, DEEPSEEK_MODEL,
        )
        cat_map = self._load_category_map()
        if not cat_map:
            return [None] * len(texts)
        names = list(cat_map.keys())
        items = "\n".join(f"- {n}：{d}" for n, d in cat_map.items())
        numbered = "\n".join(f"{i + 1}. {t[:800]}" for i, t in enumerate(texts))
        system = (
            "你是知识库分类器。根据「类别清单」判断每条「文本」最匹配的类别。\n"
            "输出格式：每条一行，写成 `编号|类别名称`，例如 `1|人事行政`。\n"
            "严格要求：\n"
            "1. 编号必须与输入编号一一对应；\n"
            "2. 类别名称必须「逐字照抄」下方类别清单中的某一个完整名称，"
            "禁止改写、缩写或自造任何名称（例如清单只有「企业概况」，"
            "就绝不能写成「企业发展」「公司简介」之类清单外的词）；\n"
            "3. 若确实都不匹配，则写 `编号|未分类`。\n"
            "不要输出任何解释、多余文字或编号以外的符号。\n\n"
            f"类别清单：\n{items}\n\n待分类文本：\n{numbered}"
        )
        messages = [{"role": "system", "content": system}]
        # 主路：云端 DeepSeek
        raw = self._call_llm(get_deepseek_client, DEEPSEEK_MODEL, messages, len(texts))
        if raw is None:
            # 降级：本地 ollama
            logger.warning("DeepSeek 分类调用失败，降级使用本地模型")
            raw = self._call_llm(get_client, MODEL, messages, len(texts))
        if raw is None:
            return [None] * len(texts)
        return self._parse_category_lines(raw, len(texts), names)

    @staticmethod
    def _call_llm(client_factory, model: str, messages: List[dict], n: int) -> Optional[str]:
        """统一的 LLM 调用（OpenAI 兼容）。失败返回 None，由调用方负责降级/兜底。"""
        try:
            client = client_factory()
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=20 * n + 20,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"LLM 分类调用失败（{model}）：{e}")
            return None

    @staticmethod
    def _parse_category_lines(raw: str, n: int, names: List[str]) -> List[Optional[str]]:
        """
        解析模型输出并按编号回填，兼容三种写法：
        `1|人事行政`、`1. 人事行政`、`人事行政`（无编号时按出现顺序依次回填）。
        编号错乱/漏行也不会串位——这是修复 7B 多行输出漂移的关键。
        """
        results: List[Optional[str]] = [None] * n
        next_slot = 0
        for line in (raw or "").splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(\d+)\s*[|｜:：.、)）]?\s*(.*)$", line)
            if m and 1 <= int(m.group(1)) <= n:
                slot, body = int(m.group(1)) - 1, m.group(2)
            else:
                while next_slot < n and results[next_slot] is not None:
                    next_slot += 1
                if next_slot >= n:
                    break
                slot, body = next_slot, line
            chosen = next((x for x in names if x in body), None)
            if chosen is None and "未分类" in body:
                chosen = "未分类"
            if results[slot] is None:
                results[slot] = chosen
            next_slot = slot + 1
        return results

    def add_chunks(self, chunks: List[Dict], category: str = None,
                   threshold: float = None) -> None:
        """
        把分块向量写入第二层。

        :param chunks: generate_embedding 输出的列表，每项含 text / meta / embedding
        :param category:
            - 传入字符串：强制所有 chunk 归入该类别
            - 不传（None）：每个 chunk 自动分类（相似度 ≥ 阈值直接定，否则小模型兜底）
        :param threshold: 类别相似度阈值（不传用模块默认 CATEGORY_SIM_THRESHOLD）
        """
        if not chunks:
            logger.warning("add_chunks: 输入的 chunks 列表为空")
            return

        thr = threshold if threshold is not None else CATEGORY_SIM_THRESHOLD

        # 1) 先按向量相似度过一遍；低于阈值的收集起来，批量交小模型兜底（一次调用，省时）
        auto_cats: List[Optional[str]] = []
        pending_idx: List[int] = []
        pending_texts: List[str] = []
        for i, chunk in enumerate(chunks):
            if category:
                auto_cats.append(category)
                continue
            top_id, top_sim = self._classify_one(chunk["embedding"])
            if top_sim >= thr:
                auto_cats.append(top_id)
            else:
                auto_cats.append(None)
                pending_idx.append(i)
                pending_texts.append(chunk.get("text", ""))

        if pending_texts:
            llm_cats = self._llm_classify_batch(pending_texts)
            for slot, c in zip(pending_idx, llm_cats):
                # LLM 明确"未分类"或调用失败 → 退化为相似度最高类别，保证每块都有归属
                auto_cats[slot] = (
                    c if (c and c != "未分类")
                    else self._classify_one(chunks[slot]["embedding"])[0]
                )

        # 2) 组装并写入
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        auto_count = {}

        for idx, chunk in enumerate(chunks):
            doc_id = chunk["meta"].get("doc_id", "unknown")
            ids.append(f"{doc_id}_chunk_{idx}")
            documents.append(chunk["text"])
            embeddings.append(chunk["embedding"])
            meta = dict(chunk["meta"])
            cat = auto_cats[idx] or "未分类"
            meta["category"] = cat
            metadatas.append(meta)

            auto_count[cat] = auto_count.get(cat, 0) + 1

        # upsert：相同 id 自动覆盖，不会重复报错
        self.chunks_col.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        logger.info(f"已写入{len(chunks)}条分块，自动分类{auto_count}")

    def query(self, query_embedding: List[float], top_k: int = 5,
              category: str = None, question_text: str = None) -> List[Dict]:
        """
        根据问题向量检索最相关的文本块。

        :param query_embedding: 用户问题的向量
        :param top_k: 返回前几条
        :param category:
            - 指定：在该类别下检索
            - 不指定：先自动判断问题属于哪个类别，再在该类别下检索；
              两级分类都无确定结果时，按相似度前二类别做「前二类」检索，扩大范围
        :param question_text: 用户问题原文，相似度低于阈值时交云端 LLM 兜底分类
        :return: [{"text":..., "meta":..., "distance":...}, ...]
        """
        # 没指定类别就先自动路由
        if category is None:
            cls = self.classify(query_embedding, text=question_text)
            cat = cls["category"]
            top2 = cls["top2"]
            if cat and cat != "未分类":
                logger.info(f"自动判断问题属于类别：{cat}（{cls['method']}）")
            elif top2:
                logger.info(f"分类不确定，按相似度前二扩大检索：{top2}")
        else:
            cat = category
            top2 = None

        # 构造 chroma 的过滤条件：确定单一类别则精确过滤；否则前二类扩大范围
        if cat and cat != "未分类":
            where = {"category": cat}
        elif top2:
            where = {"category": {"$in": top2}}
        else:
            where = None

        results = self.chunks_col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )

        # 整理返回结果
        hits = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                hits.append({
                    "text": results["documents"][0][i],
                    "meta": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if results["distances"] else None,
                })
        return hits

    def list_categories(self) -> List[str]:
        """列出当前所有已注册的类别名"""
        data = self.categories_col.get()
        return [m["category"] for m in data["metadatas"]]

    def delete_by_doc(self, doc_id: str) -> None:
        """按 doc_id 删除某个文档的所有 chunk（重新上传时先调这个）"""
        self.chunks_col.delete(where={"doc_id": doc_id})
        logger.info(f"已删除文档 {doc_id} 的所有 chunk")

