# MewHelp Ch03 向量知识库设计文档

## 意图与动机

ch02 的 `query_faq` 用 `Faq.question LIKE '%关键词%'` 查表，换一种说法就漏召回（README 里
「邮费是多少」正是故意留下的漏召回）。ch03 把它的内核换成 BGE-M3 语义向量 + Milvus Top-K
检索，让「邮费是多少」这类同义改写能命中「运费说明」，同时保持工具入参出参契约不变。

## 目标 / 非目标 / 反目标

**目标**

1. `query_faq(keyword) -> str` 契约不变，内部从关键词查表换成 dense 向量语义检索。
2. 离线文档建库：Markdown 按标题层级结构感知切分；超长内容递归切；块间加重叠且裁到最近句号；
   大表格按行切、每块都复制表头。
3. 对话挖知识 job：分批把历史对话喂 LLM 抽 QA → 进暂存表 → 整体去重 → 入 `knowledge_chunks`。
4. 双写幂等：先写 MySQL 标 `pending`，再写 Milvus 回填 `vector_id`、状态转 `done`；中断重跑只补漏。
5. 每条知识带 `category/questions/answer` 三个进向量字段，另带章节路径、内容类型、是否关键条款、
   前后块指针四类「只存不进向量」的元数据。

**非目标**

- 关键词召回、BM25、混合检索、重排——全部留给 ch04，本章只跑 dense 单路。
- 不改技术选型、不做在线增量写入（建库走离线脚本）、不迁异步数据库。

**反目标**

- 不为了「能跑」牺牲幂等；不让「查不到」被包装成编造；不引入 cron/调度框架（挖知识以离线可重跑
  job 落地，调度器不是本章要解决的问题）。

## 约束

- 技术栈定死：BGE-M3（经硅基流动 OpenAI 兼容接口）、Milvus、MySQL 当原文权威源。
- Python 3.12；沿用现有 FastAPI + SQLAlchemy 2.0 同步栈（pymysql），不迁 asyncmy。
- 工具契约：`query_faq(keyword: str) -> str` 保持不变。

## 备选方案与取舍

1. **Milvus Lite vs Docker Standalone** → 选 **Standalone**。
   - 本机是 Windows，Milvus Lite 的官方包只发 Linux/macOS（pymilvus 的 `milvus-lite` 依赖带
     `sys_platform != 'win32'` 标记），装不上；Standalone 用 Docker 起 etcd + minio + milvus
     三容器，dense 单路完全支持，也是 ch04 本来就要上的形态，等于提前架好。用户已确认改 Standalone。
2. **同步 vs 异步数据库** → 保持**同步**。
   - ch02 的 `pymysql + SessionLocal` 已跑通，迁 asyncmy 是无谓返工；asyncmy 是后续章节的独立迁移。
3. **BGE-M3 远程 API vs 本地推理** → 选**远程**。
   - 走硅基流动 OpenAI 兼容接口（`https://api.siliconflow.cn/v1`，模型 `BAAI/bge-m3`），与参考实现
     一致，无本地 GPU 依赖。用户已确认，Key 由用户补入 `.env`。

## 数据模型与落库

- `knowledge_chunks`：`id`（主键，与 Milvus 主键对齐）、`category`、`questions`、`answer`、
  `section_path`、`content_type`、`is_key_clause`、`prev_chunk_id`、`next_chunk_id`、
  `vector_id`、`vectorize_status`（`pending`/`done`）、时间戳。
- `qa_extraction_staging`：`id`、`batch_no`、`source_ref`、`question`、`answer`、`status`
  （`extracted`/`kept`/`discarded`）、时间戳。
- 建表沿用本仓库 `Base.metadata.create_all`（`scripts/init_db.py`），不新增 `sql/` 目录
  （当前仓库没有 sql/，模型即单一事实源）。

## 模块划分（实现交给 plan 细化签名）

- `app/core/embeddings.py`：`embed_texts` / `embed_query`，AsyncOpenAI 直连硅基流动 BGE-M3。
- `app/core/retrieval.py`：`search_knowledge(query, strategy="vector", top_k, category)`，dense 单路。
- `app/kb/chunking.py`：`MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` +
  句号级重叠 + 表格按行切（复制表头）。
- `app/kb/documents.py`：`Chunk` 数据类 + `build_chunks(md, content_type)` + 关键条款判定。
- `app/kb/dedup.py`：问法归一化去重（staging 内部 + 对已有 knowledge）。
- `app/kb/mining.py`：`extract_qa`（`with_structured_output(method="function_calling")`，
  沿用 ch01）+ `mine`（分批抽取 → staging → 去重 → 入 knowledge_chunks）。
- `app/kb/milvus_client.py`：Milvus Lite 客户端、`ensure_collection`、`upsert_vectors`、
  `dense_search`、`count`、`drop`。
- `app/kb/dualwrite.py`：`write_pending`（写 MySQL + 同文档连 prev/next）+ `vectorize_pending`
  （embed → Milvus upsert → 回填 done）。
- `app/kb/sources.py`：`data/kb/*.md` → content_type 清单。
- `app/db/models.py`：新增两个模型。
- `app/repositories.py`：新增 KB 仓储函数（沿用现有单文件仓储风格）。
- `app/tools/business.py`：`query_faq` 改调 `retrieval.search_knowledge`，保留原返回形态。

## 我的决定（这些不是用户拍板项）

- 文档集：`data/kb/{product-faq,returns-policy,after-sales-manual}.md` 三份；`product-faq.md`
  含「运费怎么算 / 邮费即运费」，满足验收 1。
- 挖出的问答对落库：`category="对话挖知识"`、`content_type="qa_mined"`、`questions=真实问法`。
- `faq` 表与 `search_faq` 保留为 ch02 遗留，`query_faq` 不再使用它们。
- 检索阈值：`dense_min_score=0.40`（cosine），最高分低于此返回「未找到相关问题」；实现期用
  检索标注集校准，若显著偏离再调。
- Milvus 地址：`http://localhost:19530`（Standalone，由 `docker compose` 起），无需本地 `.db`。
- 挖知识以 CLI job + Makefile target 落地（`make kb-mine`），不引入调度框架。

## 实现期需实证的风险点

- Milvus Standalone 需要 Docker Desktop 运行中；COSINE 口径按「`distance` 即相似度（越大越相似）」
  落 `score`，真实一致性由验收阶段 `kb-eval` 实证。
- `MilvusClient.search` 的命中结构（`id`/`distance`/`entity`）与 `count` 用法在 Task 2 用假客户端
  单测锁定参数，真实 Milvus 在验收阶段验证。
- DeepSeek 结构化输出继续走 `function_calling` + `chat_thinking=disabled`（ch01 已实证）。
- 沙箱联网：装新依赖、调嵌入 API 需按审批走。
