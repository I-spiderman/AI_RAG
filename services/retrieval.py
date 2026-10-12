import logging
import time
from typing import List, Dict
from services.embedding_service import embed_texts
from services.vector_store import VectorStore

logger = logging.getLogger(__name__)

class Retriever:
    """混合检索：语义向量检索 + 关键词检索（BM25），两路通过RRF融合排序"""

    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store

    def retrieve(self, question: str, top_k: int = 5, category: str = None, max_distance: float = 0.8) -> List[Dict]:
        """
        混合检索
        :param question: 用户问题
        :param top_k: 最终返回条数
        :param category: 指定类别，不指定则自动路由
        :param max_distance: 距离过滤阈值
        """
        start_time = time.time()
        logger.info(f"收到问题：{question}")
        query_vec = embed_texts([question])[0]

        # 未指定类别，自动识别类别
        if category is None:
            category = self.vs._classify(query_vec)
            if category and category != "未分类":
                logger.info(f"自动识别到类别：{category}")

        where = {"category": category} if category and category != "未分类" else None

        # 语义向量检索
        vector_hits = self.vs.chunks_col.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where=where
        )

        # 关键词检索
        # chromaDB 自动创建的索引，用于BM25检索
        text_hits = self.vs.chunks_col.query(
            query_texts=[question],
            n_results=top_k * 2,
            where=where
        )

        # 融合排序：RRF 加权排序
        # RRF 公式：score = sum(1 / (K + rank)) ,  K 一般取60
        rrf_k = 60
        scores = {}  # id -> {score, data}

        for rank, (doc, dist, meta) in enumerate(zip(
            vector_hits["documents"][0],
            vector_hits["distances"][0],
            vector_hits["metadatas"][0]
        )):
            chunk_id = vector_hits["ids"][0][rank]
            scores[chunk_id] = {
                "score": 1 / (rrf_k + rank + 1),
                "text": doc, "meta": meta, "dist": dist,
                "vector_rank": rank + 1, "text_rank": None
            }

        for rank, (doc, dist, meta) in enumerate(zip(
                text_hits["documents"][0],
                text_hits["distances"][0],
                text_hits["metadatas"][0]
        )):
            chunk_id = text_hits["ids"][0][rank]
            if chunk_id in scores:
                scores[chunk_id]["score"] += 1 / (rrf_k + rank + 1)
                scores[chunk_id]["text_rank"] = rank + 1
            else:
                scores[chunk_id] = {
                    "score": 1 / (rrf_k + rank + 1),
                    "text": doc, "meta": meta, "dist": dist,
                    "vector_rank": None, "text_rank": rank + 1
                }

        # 按融合分数排序
        merged = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        merged = merged[:top_k]

        # 距离过滤
        hits = []
        for m in merged:
            if m["text_rank"] is None and m["dist"] is not None and m["dist"] > max_distance:
                continue
            hits.append({
                "text": m["text"],
                "meta": m["meta"],
                "distance": m["dist"]
            })
            logger.info(
                f" 命中：向量排名{m['vector_rank']}, 关键词排名{m['text_rank']}，"
                f"分数{m['score']:.4f} | {m['text'][:40]}"
            )
        logger.info(f"混合检索召回 {len(hits)} 条，耗时 {time.time() - start_time:.4f}秒")
        return hits

    def build_context(self, hits: List[Dict]) -> str:
        if not hits:
            return ""
        parts = []
        for i, h in enumerate(hits, 1):
            meta = h["meta"]
            source = meta.get("doc_name", "未知文档")
            chapter = meta.get("chapter_title", "未知章节")
            parts.append(f"[参考资料{i}]（来源：{source} {chapter}）\n{h['text']}")
        return "\n\n".join(parts)


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    from services.embedding_service import embed_texts
    from services.vector_store import VectorStore

    vs = VectorStore(embed_function=embed_texts)
    retriever = Retriever(vs)

    # ===== 第一步：初始化类别 =====
    categories = {
        "人事行政": "公司考勤制度、请假流程、年假病假、报销标准、出差审批、工资福利、入职离职、人事行政规定",
        "技术运维": "服务器部署、系统配置、故障排查、命令行操作、重启维护、环境搭建、数据库管理、网络安全、Docker部署",
        "产品FAQ": "账号注册登录、密码找回、套餐升级、导出数据、功能使用问题、权限管理、企业版功能、支付方式",
    }
    vs.seed_categories(categories)

    # ===== 第二步：写入测试文档 =====
    test_docs = [
        # 人事行政类
        ("hr_001", "考勤制度.txt", "一、考勤规则\n公司实行朝九晚六工作制，迟到15分钟以内扣50元，超过15分钟按旷工半天处理。月度迟到累计3次以上扣当月全勤奖。"),
        ("hr_002", "请假流程.txt", "二、请假流程\n员工请年假需提前3天在OA系统提交申请，病假需提供医院诊断证明，部门负责人审批后交人事部备案。紧急情况可先电话报备，事后补单。"),
        ("hr_003", "出差制度.txt", "三、出差管理\n员工出差需提前在OA提交出差申请单，注明地点、时间和预计费用。住宿标准一线城市不超过500元/晚，其他城市不超过400元/晚。"),
        # 技术运维类
        ("tech_001", "服务器运维.txt", "一、日常运维\n生产服务器出现故障时，先确认故障范围。数据库连接数过高时检查慢查询日志，必要时重启MySQL服务。重启前需通知运维团队。"),
        ("tech_002", "Docker部署.txt", "二、部署规范\n使用Docker部署应用时，先编写Dockerfile构建镜像，再通过docker-compose启动容器集群。Nginx反向代理配置文件位于/etc/nginx/conf.d/目录下。"),
        ("tech_003", "性能排查.txt", "三、性能优化\n前端页面加载缓慢时，先用Chrome开发者工具Network面板查看资源加载耗时，再检查CDN配置。后端接口响应慢时查看慢SQL日志。"),
        # 产品FAQ类
        ("faq_001", "账号问题.txt", "一、账号管理\n忘记登录密码时，在登录页点击「忘记密码」，输入注册手机号获取验证码即可重置。开通企业版需管理员在后台点击「套餐升级」后扫码支付。"),
        ("faq_002", "导出功能.txt", "二、数据导出\n专业版支持一键导出Excel报表，免费版仅支持导出CSV。PDF格式导出需在报表设置中开启高级导出功能，导出文件保留最近30天。"),
    ]

    # 转成 chunks 格式入库
    all_texts = [d[2] for d in test_docs]
    vectors = embed_texts(all_texts)

    chunks = []
    for i, (doc_id, doc_name, text) in enumerate(test_docs):
        chunks.append({
            "text": text,
            "meta": {"doc_id": doc_id, "doc_name": doc_name},
            "embedding": vectors[i],
        })

    vs.add_chunks(chunks)

    # ===== 第三步：测试问题 =====
    test_questions = [
        # 应该命中人事行政
        ("迟到一次扣多少钱？", "人事行政"),
        ("病假需要什么材料？", "人事行政"),
        ("出差住酒店报销标准是多少？", "人事行政"),
        ("公司的日常工作时间是多少？", "人事行政"),
        # 应该命中技术运维
        ("MySQL连不上了怎么排查？", "技术运维"),
        ("怎么用Docker部署项目？", "技术运维"),
        ("网页打开很慢怎么回事？", "技术运维"),
        # 应该命中产品FAQ
        ("密码忘了怎么找回？", "产品FAQ"),
        ("怎么导出Excel报表？", "产品FAQ"),
        ("企业版怎么开通？", "产品FAQ"),
        # 跨类别混合问题
        ("服务器重启需要走什么流程？", "技术运维"),
        ("OA系统提交申请后多久能审批？", "人事行政"),
    ]

    print("\n" + "=" * 60)
    print("开始测试混合检索")
    print("=" * 60)

    correct = 0
    total = len(test_questions)

    for question, expected_category in test_questions:
        hits = retriever.retrieve(question, top_k=3)
        if not hits:
            print(f"\n[FAIL] 问题：{question}")
            print(f"       预期类别：{expected_category}")
            print(f"       结果：未召回任何内容")
            continue

        top_category = hits[0]["meta"].get("category", "未知")
        ok = top_category == expected_category
        if ok:
            correct += 1

        status = "PASS" if ok else "FAIL"
        print(f"\n[{status}] 问题：{question}")
        print(f"       预期：{expected_category} | 实际命中：{top_category}")
        print(f"       Top1：{hits[0]['text'][:50]}...")

    print("\n" + "=" * 60)
    print(f"测试结果：{correct}/{total} 正确")
    print("=" * 60)

