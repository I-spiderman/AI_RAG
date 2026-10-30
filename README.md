# 企业级 RAG 智能客服系统

基于检索增强生成（RAG）的多轮对话智能客服，支持知识问答、闲聊、工单创建与转人工四大意图分流。
后端采用 FastAPI + 本地轻量大模型 + 云端 DeepSeek 的混合架构，前端为 Vue 3 单页应用。

---

## 一、核心特性

- **四路意图分流**：知识问答 / 闲聊 / 工单 / 转人工，由语义强信号 + LLM 兜底识别，自动路由到对应处理链路。
- **混合检索 + 重排**：向量检索（ChromaDB）+ BM25 关键词检索，经 RRF 融合粗排后，再由本地 `Qwen3-Reranker-0.6B` 交叉编码器精排（sigmoid 阈值 0.3）。
- **多轮记忆**：短期记忆基于 Redis 滑动窗口（保留最近 5 轮），窗口溢出由 LLM 滚动摘要；写入通过 Lua 原子脚本保证并发安全。
- **用户画像**：后台线程异步用 LLM 提取用户画像入库 MySQL（不可用时自动降级，不阻塞问答）。
- **工单自动沉淀**：对话中识别到工单意图自动落单；空闲扫描 + 会话结束兜底归档，后台线程提炼成结构化工单。
- **全链路降级**：Embedding / Reranker / 云端 DeepSeek 任意环节失败均有兜底（本地 ollama、原序返回、兜底文案），系统不中断。
- **鉴权与测试**：自研无状态 Token（HMAC-SHA256）+ PBKDF2 密码哈希；附管理员功能冒烟测试与意图识别单测。

---

## 二、技术架构

| 层次 | 组件 | 说明 |
|---|---|---|
| Web 框架 | FastAPI + uvicorn | ASGI 服务，支持 SSE 流式响应 |
| 嵌入模型 | 本地 `Qwen3-Embedding-0.6B` | `SentenceTransformer`，CUDA 优先；批量编码加速入库 |
| 重排模型 | 本地 `Qwen3-Reranker-0.6B` | `CrossEncoder`，懒加载，sigmoid 阈值 0.3 |
| 向量库 | ChromaDB | 本地持久化，全项目共用 `ChromaEmbeddingFunction` |
| 检索 | 向量 + BM25 + RRF 融合 | 粗排后交由 Reranker 精排 |
| 生成大模型 | 云端 DeepSeek（`deepseek-chat`） | 面向用户的回答生成，支持流式 |
| 内部后台模型 | 本地 ollama `qwen2.5:7b` | 摘要合并、画像提取、工单提炼等后台任务 |
| 短期记忆 | Redis | 滑动窗口 + Lua 原子脚本 + 滚动摘要；多 worker 安全 |
| 长期记忆 | MySQL | 用户画像异步入库（`rag_service` 库），降级友好 |
| 鉴权 | 自研 Token | HMAC-SHA256 签名 + 7 天有效期，无状态 |
| 前端 | Vue 3 + Vite + vue-router + axios | 单页应用，开发态接 Vite，生产态由 FastAPI 托管 `dist` |

---

## 三、目录结构

```
项目：智能客服/
├── main.py                 # 启动入口（uvicorn.run，0.0.0.0:8000，reload 开发模式）
├── requirements.txt        # Python 依赖
├── app/                    # FastAPI 层：路由、鉴权、安全
│   ├── main.py             # 创建 app、挂载各 router
│   ├── api/                # 路由：chat / admin / auth / sessions
│   ├── security.py         # 密码哈希 + Token 签发/校验
│   └── schemas.py          # 请求/响应模型
├── backend/                # 核心业务层
│   ├── models/             # 本地模型权重目录（Qwen3-Embedding/Reranker-0.6B，需自行放置，见下文）
│   │   ├── Qwen3-Embedding-0.6B/
│   │   └── Qwen3-Reranker-0.6B/
│   └── services/
│       ├── core/           # RAGService（编排入口）、意图识别
│       ├── retrieval/      # 混合检索、Reranker 重排
│       ├── vector/         # Embedding、向量库
│       ├── llm/            # LLM 客户端、生成、Prompt、闲聊
│       ├── document/       # 文档解析与分块
│       └── memory/         # 短期记忆(Redis)、长期画像(MySQL)、工单线程
├── frontend/               # Vue 3 前端（npm 项目）
│   └── src/ package.json
├── tests/                  # smoke_admin.py（含 403 权限验证）、test_intent.py
└── logs/                   # 运行日志（5MB 轮转，保留 5 份）
```

---

## 四、环境要求

- Python 3.10+
- Node.js 18+（前端开发/构建）
- Redis（短期记忆、工单线程，**必需**）
- MySQL（用户画像；**可选**，不可用时自动降级）
- ollama 本地服务（内部后台模型；不可用时相关后台任务降级）
- 本地模型权重：`Qwen3-Embedding-0.6B`、`Qwen3-Reranker-0.6B`（放置于 `backend/models/`）
- 建议具备 NVIDIA GPU（CUDA），CPU 亦可运行但较慢

> 说明：以上中间件与模型权重的安装、下载、申请由使用者自行完成；本文件仅提供配置与启动说明。

---

## 五、安装与配置

### 1. 后端依赖

```bash
python -m venv .venv
.\.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. 环境变量（`.env`）

项目根目录新建 `.env` 文件，示例：

```ini
# ---- Redis（必需）----
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
# REDIS_PASSWORD=your_redis_password

# ---- MySQL（可选，画像功能；不可用时降级）----
mysql_host=localhost
mysql_password=your_mysql_password
# 数据库名固定为 rag_service，端口 3306，用户 root

# ---- 云端 DeepSeek（用户回答生成；缺省时返回兜底文案）----
DEEPSEEK_API_KEY=sk-xxxx
# DEEPSEEK_BASE_URL=https://api.deepseek.com
# DEEPSEEK_MODEL=deepseek-chat

# ---- 本地 ollama（内部后台任务；缺省时降级）----
# OLLAMA_BASE_URL=http://localhost:11434/v1

# ---- Token 密钥（务必修改默认值）----
TOKEN_SECRET=change-me-to-a-long-random-string
```

### 3. 本地模型权重

将以下模型权重下载并放置到 `backend/models/` 目录（代码通过 `backend/services/vector/embedding_service.py` 上三级定位到 `backend/`，再拼接 `models/...`，与目录一致）：

```
backend/models/Qwen3-Embedding-0.6B/
backend/models/Qwen3-Reranker-0.6B/
```

权重首次加载较慢；Reranker 为懒加载（首次重排时才加载，不拖慢启动）。

### 4. 前端依赖

```bash
cd frontend
npm install
```

---

## 六、启动

### 后端

```bash
python main.py
# 等价于：uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

服务运行于 `http://0.0.0.0:8000`。默认单 worker 开发模式；生产多 worker 部署时，短期记忆/工单线程已使用 Redis 存储，天然支持水平扩展。

### 前端

- **开发模式**（独立 Vite 服务，热更新）：

  ```bash
  cd frontend
  npm run dev
  ```

- **生产模式**：先 `npm run build` 生成 `frontend/dist`，FastAPI 会直接托管该目录；否则访问 Vite 开发服务器。

---

## 七、API 概览

| 方法 | 路径 | 说明 | 鉴权 |
|---|---|---|---|
| POST | `/api/auth/register` | 注册（注册即登录，签发 Token） | 无需 |
| POST | `/api/auth/login` | 登录，返回 Token 与 role | 无需 |
| GET | `/api/auth/me` | 获取当前用户信息 | Token |
| POST | `/api/chat` | SSE 流式问答（四路分流内部完成） | Token |
| GET | `/api/sessions` | 当前用户的会话列表 | Token |
| GET | `/api/sessions/{sid}/messages` | 会话历史（近期原文 + 滚动摘要） | Token |
| POST | `/api/admin/documents` | 上传文档入库 | **admin** |
| GET | `/api/admin/documents` | 已入库文档列表 | **admin** |
| DELETE | `/api/admin/documents/{doc_id}` | 删除文档 | **admin** |
| GET | `/api/admin/hot-tickets` | 热门工单 Top5 | **admin** |

> 所有 `/api/admin/*` 接口要求 `role == admin`，否则返回 `403 Forbidden`。

---

## 八、测试

```bash
# 管理员功能冒烟（登录→上传→列表→热门工单→删除→非管理员 403 验证）
.\.venv\Scripts\python.exe tests\smoke_admin.py

# 意图识别单测
.\.venv\Scripts\python.exe -m pytest tests\test_intent.py
```

`smoke_admin.py` 默认访问 `http://127.0.0.1:8000`，可传入 `base_url` 参数指定其他地址。

---

## 九、降级与容错设计

为适配本地/弱网环境，系统在各关键环节均做了优雅降级：

- **Embedding / Reranker 加载失败**：由上层捕获，检索降级为召回原序，问答不中断。
- **DeepSeek 不可用**：返回友好兜底文案，不抛出致命错误。
- **MySQL 不可用**：用户画像组件置空，问答照常进行，不影响主流程。
- **多 worker / 多线程并发**：短期记忆写入封装为 Redis Lua 原子脚本，工单线程全程 Redis 存储，避免并发丢消息或串台。

---

## 十、许可证

内部项目，仅供学习与演示使用。
