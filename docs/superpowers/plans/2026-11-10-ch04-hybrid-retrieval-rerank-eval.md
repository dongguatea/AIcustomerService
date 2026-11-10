# Ch04 混合检索 + 重排 + 评估体系 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 ch03 的 dense 单路语义检索升级为「混合检索(Milvus 原生 BM25 + dense,hybrid_search RRF)+ bge-reranker-v2-m3 精排 + 生成质量控制(引用/拒答/自评落池)」,并建一套四策略分桶评估体系,前端聊天页引用可点、回答带 👍/👎。

**Architecture:** 保持单一 agent 入口 + 单次工具调用不变;`query_faq` 内部升级为完整 RAG 管线(Query 理解 → 品类过滤 → 混合召回 → 重排 → 机械闸/语义闸 → 首尾组装带编号证据);引用元数据经 `ContextVar` 暂存、SSE `citations` 事件下发前端。纯 async 编排,不引入 LangGraph/Langfuse。

**Tech Stack:** Milvus Standalone 2.6.22(pymilvus `Function`/`FunctionType.BM25`/`AnnSearchRequest`/`RRFRanker`/chinese analyzer)、SiliconFlow `/rerank`(`BAAI/bge-reranker-v2-m3`)、SiliconFlow `BAAI/bge-m3` 嵌入(沿用 ch03)、DeepSeek chat(改写/自评/忠实度裁判复用)、FastAPI/SQLAlchemy(沿用)。

**Spec:** `docs/superpowers/specs/2026-11-10-ch04-hybrid-retrieval-rerank-eval-design.md`

## Global Constraints

- **固定选型(定死,不自换)**:Milvus 2.5+ 原生 BM25 + `hybrid_search` RRF;重排 `bge-reranker-v2-m3`;嵌入 `BAAI/bge-m3`。实现中发现矛盾/走不通 → 停,只在选型内换法(见 Task 1 红线)。
- **不引入 LangGraph / Langfuse**,沿用 ch03 纯 async 函数编排。
- **结构化输出用扁平字段**(`useful`/`reason`、`standard`/`expanded`),禁用嵌套对象数组。
- **Milvus `id` = MySQL `knowledge_chunks.id`**(auto_id=False),重跑 upsert 幂等。
- **BM25 `text` 字段 = `category + questions + answer` 拼接**(与 dense 嵌入源同),两路召回同批 chunk。
- **中文分词** `analyzer_params = {"type": "chinese"}`(字段级)。
- **同义词扩展只作用于 BM25 检索文本**,不改 dense query、不改入库。
- **单次工具调用入口不变**;事务型工具(order/product/logistics/ticket)零改动。
- **凭据只在 `.env`**,代码/文档/留痕不写死任何上游地址或密钥。
- 确定性代码走 TDD(红→绿→提交);纯 Prompt/数据任务用标注样例/评估集验证代替 TDD;前端走 Vibe Coding。
- Milvus 单测用假 client(与 ch03 一致),真实验证放冒烟/验收;`milvus_client` 各函数已带 `collection` 参数以隔离测试库。

## Review Focus

以下五类输入/失败模式,spec 没写但最可能咬到使用者,各配一条测试落在所属任务:

1. **BM25 对型号串的分词**:型号含连字符/大小写(如 `MH-LP100`),问 `LP100` 能否命中——若 analyzer 把型号切碎,验收 2 会挂。→ Task 2 `test_bm25_hits_model_number` + Task 11 冒烟实证。
2. **机械闸阈值作用在 rerank_score 而非 Milvus distance**:`rerank_min_score=0.30` 是 bge 的 0-1 相关分,Milvus 的 COSINE distance 是相似度,两者尺度不同。→ Task 9 断言闸用的是 `rerank_score`。
3. **品类过滤清空结果 → 回退不过滤**:category 由模型自由填,常猜错入库值,精确过滤会清空致误拒答。→ Task 9 `test_category_filter_fallback`。
4. **自评上游挂掉 → 按「证据不足」拒答**,绝不放行未过闸内容;改写上游挂掉 → 回落原问题(不崩请求)。→ Task 5/Task 6 各一条回落测试。
5. **前端点到大编号 `[99]`**(模型编造的引用号)无映射时给兜底提示,不白屏。→ Task 14 前端自测项。

---

## 文件布局

**改:** `app/config.py`、`app/kb/milvus_client.py`、`app/kb/sources.py`、`app/core/retrieval.py`、`app/core/prompts.py`、`app/core/llm.py`、`app/tools/business.py`、`app/tools/registry.py`、`app/services/chat.py`、`app/api/chat.py`、`app/db/models.py`、`app/repositories.py`、`app/static/index.html`、`Makefile`、`.env.example`、`scripts/init_db.py`、`tests/conftest.py`、`tests/test_milvus_client.py`、`tests/test_retrieval.py`、`tests/test_chat_service.py`、`tests/test_prompts.py`

**新:** `data/kb/product-specs.md`、`app/core/rerank.py`、`app/core/query_understanding.py`、`app/core/selfcheck.py`、`scripts/smoke_rerank.py`、`scripts/smoke_milvus_bm25.py`、`scripts/eval_ch04.py`、`scripts/validate_eval_ch04.py`、`tests/data/eval_ch04.jsonl`、`tests/data/query_understanding_samples.json`、`tests/data/selfcheck_samples.json`、`tests/kb/test_milvus_hybrid.py`、`tests/core/test_rerank.py`、`tests/core/test_query_understanding.py`、`tests/core/test_selfcheck.py`、`tests/core/test_retrieval_ch04.py`、`tests/db/test_low_confidence.py`、`tests/tools/test_query_faq_rag.py`、`tests/test_chat_citations.py`

---

## Task 1: 配置 + fail-fast 冒烟(红线,GO/NO-GO)

**目的:** 动手改检索主链路前,先用真实服务验证两条新链路(上游 /rerank、Milvus 原生 BM25 + hybrid_search)。不通即停、只在选型内换法、把卡点写进 dev-notes 顶部。非 TDD。

**Files:**
- Modify: `app/config.py`、`.env.example`、`Makefile`
- Create: `scripts/smoke_rerank.py`、`scripts/smoke_milvus_bm25.py`

**Interfaces:**
- Produces: `settings.rerank_base_url`(默认 `https://api.siliconflow.cn/v1`)、`settings.rerank_api_key`(默认 `""`,缺了 hybrid_rerank 跑时才报)、`settings.rerank_model`(默认 `"BAAI/bge-reranker-v2-m3"`)、`settings.recall_top_k=50`、`settings.rerank_top_k=10`、`settings.rerank_min_score=0.30`;`make milvus-up` / `make smoke-rag` / `make eval-rag` / `make eval-check`。

- [ ] **Step 1: 加配置** — `app/config.py` 追加 rerank/召回参数(见 Produces 的默认值);`.env.example` 追加 `RERANK_BASE_URL` / `RERANK_MODEL` / `RERANK_API_KEY` / `RECALL_TOP_K` / `RERANK_TOP_K` / `RERANK_MIN_SCORE`(带注释)。`make eval-rag`/`make eval-check` 目标先占位为 `uv run python scripts/...`(脚本在 Task 13/12 落,占位即可)。

- [ ] **Step 2: 写 rerank 冒烟脚本** — `scripts/smoke_rerank.py`:POST `{rerank_base_url}/rerank`,`payload={"model":"bge-reranker-v2-m3","query":"退货运费谁承担","documents":["满99元包邮,不满收取10元运费。","七天无理由退货,非质量问题退货运费由买家承担。","智能猫砂盆 Pro 支持自动清理。"],"top_n":3}`,Bearer `rerank_api_key`;断言返回 `results` 非空、`max(results, relevance_score)["index"] == 1`;打印 `GO: rerank 链路通`。

- [ ] **Step 3: 写 Milvus BM25 冒烟脚本** — `scripts/smoke_milvus_bm25.py`:临时集合 `smoke_bm25`;schema `id` INT64 主键 + `dense` FLOAT_VECTOR(dim=4) + `text` VARCHAR(enable_analyzer, analyzer chinese) + `sparse` SPARSE_FLOAT_VECTOR;`schema.add_function(Function("text_bm25", ["text"], ["sparse"], FunctionType.BM25))`;索引 dense=AUTOINDEX/COSINE、sparse=SPARSE_INVERTED_INDEX/BM25;插两行(`id=1` 运费文案、`id=2` "智能猫砂盆 Pro 型号支持自动清理和除臭");纯 BM25 查 `"猫砂盆 Pro 型号"` 断言 `id==2`;`AnnSearchRequest` × 2 + `client.hybrid_search(... ranker=RRFRanker(), limit=...)` 断言非空;最后 drop。

- [ ] **Step 4: Makefile 加 milvus 起停 + 冒烟** — `milvus-up` 用 `docker compose up -d etcd minio milvus-standalone` 后轮询 `http://localhost:9091/healthz`;`smoke-rag` 依次跑两个冒烟脚本。`.PHONY` 追加 `milvus-up milvus-down smoke-rag eval-rag eval-check`。

- [ ] **Step 5: 跑两个冒烟(GO/NO-GO)** — `make milvus-up` 后 `uv run python scripts/smoke_milvus_bm25.py` 与 `scripts/smoke_rerank.py`(后者需 `.env` 补真实 `RERANK_API_KEY`)。两个都打印 `GO: ...` 才继续。

- [ ] **Step 6: 提交** — `git add app/config.py .env.example Makefile scripts/smoke_rerank.py scripts/smoke_milvus_bm25.py`;commit `feat(ch04): rerank/召回配置与两条 fail-fast 冒烟`。

**NO-GO 处理(红线):** rerank 报错→先试 `BAAI/bge-reranker-v2-m3` 前缀与 `/v1/rerank` 路径,仍不通→停,把请求/响应/错误写进 dev-notes 顶部并等用户,不自换重排模型。Milvus BM25 报错→在 v2.5.x+ 内换 image tag 重试,仍不通→停+记录,不改回 dense、不弃 BM25。

---

## Task 2: milvus_client 新 schema + 混合/单路检索(TDD)

**Files:**
- Modify: `app/kb/milvus_client.py`
- Modify: `tests/test_milvus_client.py`(ch03 schema 断言随新 schema 更新)
- Create: `tests/kb/test_milvus_hybrid.py`

**Interfaces:**
- Consumes: pymilvus `Function / FunctionType / AnnSearchRequest / RRFRanker / DataType`。
- Produces: `ensure_collection(client, collection=COLLECTION)` 新 schema(dense + text + sparse + 5 标量);`dense_search(client, vector, top_k, category=None, collection=COLLECTION) -> list[dict]`;`bm25_search(client, text, top_k, category=None, collection=COLLECTION) -> list[dict]`;`hybrid_search(client, vector, text, top_k, recall=50, category=None, collection=COLLECTION) -> list[dict]`;hit dict 键 = `{id, score, question, answer, section_path, content_type, category}`;`flush / count / drop / get_client / acall` 保持。

- [ ] **Step 1: 写失败测试** — `tests/kb/test_milvus_hybrid.py`:
  - `test_ensure_collection_builds_bm25_schema`:假 client 建集合后,schema 字段序 = `[("id","INT64"),("dense","FLOAT_VECTOR"),("text","VARCHAR"),("sparse","SPARSE_FLOAT_VECTOR"),("question","VARCHAR"),("answer","VARCHAR"),("section_path","VARCHAR"),("content_type","VARCHAR"),("category","VARCHAR")]`;`text` 字段带 `enable_analyzer=True` 且 `analyzer_params=={"type":"chinese"}`;`schema.functions` 含 `Function(name="text_bm25", input=["text"], output=["sparse"], type=BM25)`;indexes = dense(AUTOINDEX/COSINE) + sparse(SPARSE_INVERTED_INDEX/BM25)。
  - `test_bm25_hits_model_number`:fake 检索回 `id=2` 命中 "猫砂盆 Pro 型号",断言 `bm25_search(...)[0]["id"]==2` 且 `search` 记录 `anns_field=="sparse"`、`search_params=={"metric_type":"BM25"}`。
  - `test_hybrid_returns_rrf_hits`:fake `hybrid_search` 回两个 hit,断言 `hybrid_search(...)` 传 `ranker` 为 `RRFRanker` 实例、`reqs` 两条(dense/sparse 各一)、hit 键齐全。
  - `test_category_filter`:fake 记录断言 `bm25_search(..., category="运费")` 的 filter 为 `category == "运费"`,`hybrid_search(..., category="运费")` 的每条 AnnSearchRequest `expr` 同为该式。

- [ ] **Step 2: 跑测试确认失败** — `uv run pytest tests/kb/test_milvus_hybrid.py -v` → FAIL(`bm25_search`/`hybrid_search` 不存在、schema 无 text/sparse)。

- [ ] **Step 3: 实现** — `ensure_collection` 按新 schema 建集合并 `add_function`;新增 `bm25_search`(search `anns_field="sparse"`, metric BM25)、`hybrid_search`(`AnnSearchRequest(dense, param=COSINE, limit=recall, expr=...)` + `AnnSearchRequest(text, param=BM25, limit=recall, expr=...)`, `client.hybrid_search(..., ranker=RRFRanker(), limit=top_k, output_fields=_OUTPUT)`);`_hit` 映射保持。`test_milvus_client.py` 里 ch03 的 schema 断言与索引断言同步更新为含 text/sparse。

- [ ] **Step 4: 跑测试确认通过** — `uv run pytest tests/kb/test_milvus_hybrid.py tests/test_milvus_client.py -v` → PASS。

- [ ] **Step 5: 提交** — `git add app/kb/milvus_client.py tests/kb/test_milvus_hybrid.py tests/test_milvus_client.py`;commit `feat(ch04): Milvus schema 加 BM25 函数 + 单路/混合检索`。

---

## Task 3: 重排客户端 rerank(TDD)

**Files:**
- Create: `app/core/rerank.py`
- Create: `tests/core/test_rerank.py`

**Interfaces:**
- Consumes: `settings.rerank_base_url / rerank_api_key / rerank_model`。
- Produces: `async rerank(query: str, docs: list[str], top_n: int | None = None) -> list[tuple[int, float]]`(原始索引, 相关分),按分降序、截 top_n。

- [ ] **Step 1: 写失败测试** — `tests/core/test_rerank.py`(monkeypatch `httpx.AsyncClient.post` 为 fake 返回 `{"results":[{"index":0,"relevance_score":0.2},{"index":2,"relevance_score":0.9},{"index":1,"relevance_score":0.5}]}`):
  - `test_rerank_orders_and_truncates`:断言 `await rerank("q", ["a","b","c"], top_n=2) == [(2, 0.9), (1, 0.5)]`。
  - `test_rerank_empty_docs`:断言 `await rerank("q", []) == []` 且不发请求。
  - `test_rerank_url_and_payload`:断言请求 URL 以 `/rerank` 结尾、payload 的 `model==settings.rerank_model`、`top_n` 透传、header 带 Bearer。

- [ ] **Step 2: 跑测试确认失败** — `uv run pytest tests/core/test_rerank.py -v` → FAIL(模块不存在)。

- [ ] **Step 3: 实现** — `_rerank_url()` 把 `rerank_base_url` 拼成 `/rerank`(缺 `/v\d+` 段补 `/v1`);`rerank` 用 `httpx.AsyncClient` 手写 POST(query + documents + model + top_n),按 `relevance_score` 降序排 `(index, score)`,截 top_n。

- [ ] **Step 4: 跑测试确认通过** — `uv run pytest tests/core/test_rerank.py -v` → PASS。

- [ ] **Step 5: 提交** — commit `feat(ch04): bge-reranker-v2-m3 直连 /rerank 客户端`。

---

## Task 4: LLM structured 输出 + RAG 提示词(TDD + 提示词断言)

**Files:**
- Modify: `app/core/llm.py`、`app/core/prompts.py`
- Modify: `tests/test_prompts.py`

**Interfaces:**
- Produces: `get_chat_model(streaming: bool = False, temperature: float = 0.3) -> ChatOpenAI`(新增 temperature 透传);`llm.structured(schema) -> Runnable`(=`get_chat_model().with_structured_output(schema)`);prompts 常量 `QUERY_REWRITE_PROMPT`、`SELF_CHECK_PROMPT`、`RAG_ANSWER_PROMPT`、`FAITHFULNESS_PROMPT`,以及扩展后的 `CUSTOMER_SERVICE_SYSTEM`。

- [ ] **Step 1: 写失败测试** — `tests/test_prompts.py` 追加:
  - `test_rag_prompts_exist`:四个新 prompt 常量存在且 `ChatPromptTemplate`。
  - `test_system_prompt_has_citation_rules`:断言 `CUSTOMER_SERVICE_SYSTEM` 含 `sufficient=false` 拒答语义、`[n]` 引用规则、`以平台售后规则为准`。
  - `test_system_prompt_lists_negative_knowledge`:断言含 `不承诺`、`到账`、`时效`。
  - `tests/core/test_llm_structured.py`:断言 `get_chat_model(temperature=0)` 的 `temperature==0`、默认 `0.3`;`structured` 走 `with_structured_output`。

- [ ] **Step 2: 跑测试确认失败** — `uv run pytest tests/test_prompts.py tests/core/test_llm_structured.py -v` → FAIL。

- [ ] **Step 3: 实现** — `get_chat_model` 加 `temperature` 参数;新增 `structured(schema)`;`prompts.py` 加 `QUERY_REWRITE_SYSTEM/PROMPT`、`SELF_CHECK_SYSTEM/PROMPT`、`RAG_ANSWER_SYSTEM/PROMPT`、`FAITHFULNESS_SYSTEM/PROMPT`(照 spec §4:引用规则、逐字复制型号、条件跟随、拒答规则、负面知识、忠实度裁判口径);`CUSTOMER_SERVICE_SYSTEM` 追加「query_faq 结果处理」段与负面知识段。

- [ ] **Step 4: 跑测试确认通过** — PASS。

- [ ] **Step 5: 提交** — commit `feat(ch04): LLM structured 输出与 RAG 提示词(引用/拒答/自评/忠实度/负面知识)`。

---

## Task 5: Query 理解 query_understanding(标注样例验证)

**Files:**
- Create: `app/core/query_understanding.py`、`tests/core/test_query_understanding.py`、`tests/data/query_understanding_samples.json`

**Interfaces:**
- Consumes: `llm.structured`、`QUERY_REWRITE_PROMPT`。
- Produces: `async understand(query: str) -> dict`(`{"standard": str, "expanded": list[str]}`)。

- [ ] **Step 1: 写样例与失败测试** — `tests/data/query_understanding_samples.json`:6-8 条口语问法 → 期望标准问法与扩展词(如「邮费咋算的呀」→ standard 含「运费」、expanded 含「邮费/运费」)。`tests/core/test_query_understanding.py`:`test_understand_rewrite`(monkeypatch `llm.structured` 回固定 `_Rewrite`,断言 standard/expanded 透传);`test_understand_falls_back_on_error`(fake 抛异常,断言返回 `{"standard": query, "expanded": []}`)。样例验证为**运行脚本**(真实模型)走 `scripts/validate_*` 或标注样例脚本,非单测。

- [ ] **Step 2: 实现** — `_Rewrite(BaseModel)` 扁平字段 `standard`/`expanded: list[str]`;`understand` 用 `QUERY_REWRITE_PROMPT | llm.structured(_Rewrite)` 调模型,异常回落原问题。

- [ ] **Step 3: 跑单测确认通过 + 跑标注样例** — 单测 PASS;真实模型下逐条比对样例(此步需真实 key,放验收轮)。

- [ ] **Step 4: 提交** — commit `feat(ch04): 口语→标准问法改写 + 同义词扩展(检索侧)`。

---

## Task 6: 生成前自评 selfcheck(标注样例验证)

**Files:**
- Create: `app/core/selfcheck.py`、`tests/core/test_selfcheck.py`、`tests/data/selfcheck_samples.json`

**Interfaces:**
- Consumes: `llm.structured`、`SELF_CHECK_PROMPT`。
- Produces: `async check_sufficient(query: str, evidence_texts: list[str]) -> dict`(`{"useful": bool, "reason": str}`)。

- [ ] **Step 1: 写样例与失败测试** — `tests/core/test_selfcheck.py`:`test_check_sufficient_true`(fake `llm.structured` 回 `_Check(useful=True, reason=...)`);`test_check_sufficient_fails_closed`(fake 抛异常 → 断言 `{"useful": False, "reason": "证据自评失败"}`)。样例 JSON:3 条「证据够/不够」各配证据文本。

- [ ] **Step 2: 实现** — `_Check(BaseModel)` 扁平 `useful`/`reason`;`check_sufficient` 拼 `[i]` 证据串,异常按「不足」处理(fail-closed)。

- [ ] **Step 3: 跑单测确认通过** — PASS。

- [ ] **Step 4: 提交** — commit `feat(ch04): 生成前证据充分性自评(fail-closed)`。

---

## Task 7: 检索编排 retrieval.search_knowledge 重写(TDD)

**Files:**
- Modify: `app/core/retrieval.py`
- Create: `tests/core/test_retrieval_ch04.py`

**Interfaces:**
- Consumes: `milvus_client`、`embeddings.embed_query`、`rerank.rerank`、`settings.recall_top_k / rerank_top_k`。
- Produces: `arrange_head_tail(items: list) -> list`(首、尾、居中);`async search_knowledge(query, strategy="hybrid_rerank", top_k=None, category=None, client=None, collection="knowledge", bm25_text=None) -> list[dict]`,strategy ∈ `{vector, bm25, hybrid, hybrid_rerank}`;`hybrid_rerank` 的 hit 多一个 `rerank_score`。

- [ ] **Step 1: 写失败测试** — `tests/core/test_retrieval_ch04.py`(monkeypatch `embeddings.embed_query`、`milvus_client.dense_search/bm25_search/hybrid_search/acall`、`rerank.rerank`):
  - `test_arrange_head_tail`:5 项 → `[0,2,3,4,1]`;2 项 → 原序。
  - `test_strategy_bm25_uses_bm25_text`:传 `bm25_text="标准 邮费 运费"`、`strategy="bm25"`,断言 `bm25_search` 收到该文本。
  - `test_strategy_vector_ignores_bm25_text`:断言走 `dense_search`、用 `embed_query(query)`。
  - `test_strategy_hybrid_no_rerank`:断言 `hybrid_search` 收到 `recall_top_k`、返回截 `top_k`。
  - `test_strategy_hybrid_rerank_adds_rerank_score`:fake `rerank.rerank` 回 `[(1,0.9),(0,0.6)]`,断言输出按该序、每项带 `rerank_score`。
  - `test_unknown_strategy_raises`:原 `test_retrieval.py` 的 `test_search_knowledge_rejects_unknown_strategy` 语义保留(非 vector 时不再一律拒,改为非法值拒)。

- [ ] **Step 2: 跑测试确认失败** — FAIL(`strategy` 分支只有 vector)。

- [ ] **Step 3: 实现** — 按 strategy 分支实现;`hybrid`/`hybrid_rerank` 先 `hybrid_search(recall=settings.recall_top_k)`,hybrid 截 top_k,hybrid_rerank 再 `rerank.rerank(query, docs, top_n=top_k)` 回填 `rerank_score`;`bm25_text or query` 作为 BM25 侧文本。

- [ ] **Step 4: 跑测试确认通过** — PASS(含原 `tests/test_retrieval.py` 更新后)。

- [ ] **Step 5: 提交** — commit `feat(ch04): 检索编排支持四策略 + 首尾组装`。

---

## Task 8: low_confidence_questions 落池(TDD)

**Files:**
- Modify: `app/db/models.py`、`app/repositories.py`
- Create: `tests/db/test_low_confidence.py`

**Interfaces:**
- Produces: `LowConfidenceQuestion`(`__tablename__="low_confidence_questions"`,字段 `id/conversation_id(FK conversations.id, nullable)/raw_question/source/reason/created_at`);`insert_low_confidence(session, conversation_id, raw_question, source, reason) -> int`。

- [ ] **Step 1: 写失败测试** — `tests/db/test_low_confidence.py`(用 `session_factory` 的 sqlite):`test_insert_low_confidence` 写入后断言字段与 `source in {"retrieval_low_conf","self_check","user_feedback"}`;`test_conversation_fk_nullable` 传 `conversation_id=None` 可写入。

- [ ] **Step 2: 跑测试确认失败** — FAIL(表不存在)。

- [ ] **Step 3: 实现** — `models.py` 加模型(用 `Enum("retrieval_low_conf","self_check","user_feedback")`、`server_default=func.now()`);`repositories.py` 加 `insert_low_confidence`。

- [ ] **Step 4: 跑测试确认通过** — PASS。

- [ ] **Step 5: 提交** — commit `feat(ch04): 低置信度问题池模型与写入仓储`。

---

## Task 9: query_faq 升级为 RAG 工具 + registry 按工具超时(TDD)

**Files:**
- Modify: `app/tools/business.py`、`app/tools/registry.py`
- Create: `tests/tools/test_query_faq_rag.py`

**Interfaces:**
- Consumes: `query_understanding.understand`、`retrieval.search_knowledge`、`selfcheck.check_sufficient`、`retrieval.arrange_head_tail`、`insert_low_confidence`(Task 10 用)、`settings.rerank_min_score / rerank_top_k`。
- Produces: `query_faq(keyword: str, category: str | None = None) -> str`(返回证据串或拒答信号);ContextVar `rag_result`(`RagResult(sufficient, source, reason, citations, evidence)`);citation item = `{n, id, section_path, question, answer, content_type}`;`ToolRegistry.register(tool, timeout=None, max_retries=None)` 并让 `execute` 用按工具值。

- [ ] **Step 1: 写失败测试** — `tests/tools/test_query_faq_rag.py`(monkeypatch 上述组件):
  - `test_query_faq_returns_numbered_evidence`:fake 命中 → 返回串含 `[1]`、`[2]`;`rag_result.get()` 的 `sufficient=True`、citations 的 `n` 与 evidence `[n]` 序号一致。
  - `test_query_faq_mechanical_gate`:fake 空命中 → 返回拒答信号;`rag_result` `sufficient=False`、`source="retrieval_low_conf"`。
  - `test_query_faq_selfcheck_gate`:fake `check_sufficient` 回 `useful=False` → `source="self_check"`。
  - `test_query_faq_head_tail_and_citations`:首尾组装后 citations[0] 是最相关、citations[-1] 是次相关。
  - `test_category_filter_fallback`:category 过滤后空 → 回退 `category=None` 重试。
  - `test_registry_per_tool_timeout`:注册带 `timeout=30.0` 的工具,execute 用 30 而非全局。

- [ ] **Step 2: 跑测试确认失败** — FAIL。

- [ ] **Step 3: 实现** — `business.py` 的 `query_faq` 改为 RAG 管线(understand → search hybrid_rerank → 机械闸 → 品类回退 → 语义闸 → arrange_head_tail → citations + evidence 串);`rag_result` ContextVar 置入结构化结果;registry 支持按工具 timeout/max_retries(查询用 `timeout=30.0, max_retries=0`)。

- [ ] **Step 4: 跑测试确认通过** — PASS。

- [ ] **Step 5: 提交** — commit `feat(ch04): query_faq 升级为 RAG 工具(两道生成前闸 + 引用暂存)`。

---

## Task 10: ChatService 拒答落池 + citations SSE(TDD)

**Files:**
- Modify: `app/services/chat.py`、`app/api/chat.py`
- Create: `tests/test_chat_citations.py`

**Interfaces:**
- Consumes: `rag_result` ContextVar、`insert_low_confidence`、`settings.tool_timeout`。
- Produces: `ChatService.stream` 在工具为 `query_faq` 时,读 `rag_result`:`sufficient=False` → 落 `low_confidence_questions(conversation_id, 原话, source, reason)`;`sufficient=True` → 流式前 yield `{"event":"citations","items":[...]}`。`api/chat.py` 把该 frame 原样转 SSE。

- [ ] **Step 1: 写失败测试** — `tests/test_chat_citations.py`:
  - `test_query_faq_sufficient_emits_citations`:fake 模型 tool_call 到 `query_faq`、`rag_result` 置 `sufficient=True` 带 2 条 citations;断言帧序列含 `{"event":"citations","items":...}` 且先于 delta。
  - `test_query_faq_insufficient_writes_low_conf`:置 `sufficient=False, source="self_check"`,断言 `low_confidence_questions` 里新增一行且 `source`/`reason` 正确、无 citations 帧。

- [ ] **Step 2: 跑测试确认失败** — FAIL。

- [ ] **Step 3: 实现** — `chat.py` 工具执行后读取 ContextVar(需在 `current_conversation_id` 同上下文内),按 sufficient 落池或 yield citations 帧;`api/chat.py` 不拦 citations 帧(现为 `data: {json}` 直通)。

- [ ] **Step 4: 跑测试确认通过** — PASS。

- [ ] **Step 5: 提交** — commit `feat(ch04): 拒答落池与引用 SSE 事件下发`。

---

## Task 11: 知识库内容 + 建库/向量化 + BM25 型号命中实证

**Files:**
- Modify: `app/kb/sources.py`(`SOURCE_TYPES` 加 `product-specs.md: "spec"`,`CONTENT_TYPES` 加 `"spec"`)、`scripts/init_db.py`(文案)
- Create: `data/kb/product-specs.md`

- [ ] **Step 1: 写规格手册** — `data/kb/product-specs.md`:易混型号族(猫砂盆 `MH-LP100/LP50/LP200`、饮水机 `MH-W20/W40/W60`、加热垫 `MH-HP12/HP20`、喂食器 `MH-FD10/FD30`、猫爬架 `MH-CT30`、摄像头 `MH-CAM1`、猫窝 `MH-NEST20`),每个型号一节(型号、规格、与同族差异),与 `after-sales-manual.md` 设备品类、`query_product`/种子数据一致。

- [ ] **Step 2: sources.py 挂 spec** — 更新 `SOURCE_TYPES` / `CONTENT_TYPES`。

- [ ] **Step 3: 建库(Standalone)** — `make kb-reset && make kb-build && make kb-vectorize`;断言 Milvus count == MySQL `knowledge_chunks` done 数。

- [ ] **Step 4: BM25 型号命中实证(验收 2 CLI 侧)** — 单条脚本用 `retrieval.search_knowledge("智能猫砂盆Pro 型号", strategy="bm25")` 断言 top hit `section_path` 含该型号。

- [ ] **Step 5: 提交** — commit `feat(ch04): 商品规格手册(spec)入库 + BM25 型号命中实证`。

---

## Task 12: 评估集 + 自检脚本(数据任务)

**Files:**
- Create: `tests/data/eval_ch04.jsonl`、`scripts/validate_eval_ch04.py`

- [ ] **Step 1: 写评估集** — 4 桶 × 20 = 80 题,字段 `{id, bucket, query, expect_section: [str], expect_points: [str], should_refuse: bool, difficulty: easy|medium|hard}`;`A_policy` 政策流程、`B_model` 型号(含易混族)、`C_colloquial` 口语改写、`D_oob` 库外(should_refuse=true,无 expect_section/points)。

- [ ] **Step 2: 写自检脚本** — `scripts/validate_eval_ch04.py`(对 `data/kb/*.md` 源文件校验,不依赖 Milvus):id/query 不重复、四桶各 20;A/B/C 的每个 `expect_section` 至少匹配一个源文件章节标题、每个 `expect_points` 去空白后逐字出现在目标小节正文;D 桶不带 ground-truth。`make eval-check` 接这个脚本。

- [ ] **Step 3: 跑自检确认 0 错** — `make eval-check` → 0 错。

- [ ] **Step 4: 提交** — commit `feat(ch04): 四桶 80 题评估集 + 自检脚本`。

---

## Task 13: 评估脚本 eval_ch04.py + 四策略报告(验收 1)

**Files:**
- Create: `scripts/eval_ch04.py`
- Modify: `Makefile`

- [ ] **Step 1: 写确定性段单测** — `tests/core/test_eval_ch04.py`:直接 import 脚本内纯函数 `_recall_at/_rr/_coverage_mech/_norm`,断言口径(Recall@5 部分分、MRR 倒数取均、证据覆盖度去空白子串匹配)。

- [ ] **Step 2: 写脚本** — 三段:①检索段(确定性)四策略 × A/B/C 桶 Recall@5 + MRR(按 `expect_section` 命中 `section_path`);②证据覆盖度(确定性,`expect_points` 机械匹配);③生成段 hybrid_rerank 的 Faithfulness(A/B/C)+ D 桶拒答率(走 `query_faq` RAG 管线看 sufficient=False)。每次模型调用带超时 + 失败跳过 + 低并发;`--skip-gen` 只跑前两段。产物落 `data/ch04/reports/rag_eval.{txt,json}`。`make eval-rag` 接它。

- [ ] **Step 3: 跑确定性段单测确认通过** — PASS。

- [ ] **Step 4: 跑完整报告(验收 1,真实 key + Milvus)** — `make eval-rag` → 打印分桶表,四策略拉开差距(hybrid_rerank 总体 MRR 最高、BM25 在 B 桶优于 vector)。

- [ ] **Step 5: 提交** — commit `feat(ch04): 四策略分桶评估脚本与报告产物`。

---

## Task 14: 前端可点引用 + 每段回答满意度反馈(Vibe Coding)

**Files:**
- Modify: `app/static/index.html`

- [ ] **Step 1: 读现有 SSE 消费与消息渲染**(`handleSSEBlock`/`renderAssistant`),定位 event 分支与 assistant 气泡结构。
- [ ] **Step 2: 接收 citations 事件** — `handleSSEBlock` 遇 `payload.event === "citations"` 把 `items` 存到当前 assistant 消息 `acc.citations`。
- [ ] **Step 3: 渲染可点角标** — 文本渲染时正则把 `[n]` 换成 `<sup class="cite" data-n="n">[n]</sup>`;点击按 `data-n` 找 citations 项弹浮层(标题=section_path,正文=question + answer);`[99]` 无映射时浮层显示「未找到对应来源」。
- [ ] **Step 4: 满意度反馈** — 有效回答流式结束后在气泡左下角挂 👍/👎;点击点亮所选、另一个淡出、显示「已反馈」并锁定两键(一次性,纯前端)。
- [ ] **Step 5: 浏览器自测截图** — 问「邮费是多少」点引用看来源;同一条回答点 👍 点亮并显示已反馈。截图留 dev-notes。
- [ ] **Step 6: 提交** — commit `feat(ch04): 聊天页可点引用 + 每段回答满意度反馈(Vibe Coding 首版)`。

---

## Task 15: 端到端验收 + 全量回归 + 交付收尾

- [ ] **Step 1: 全量单测** — `uv run pytest -q` → 全绿(ch01-03 旧测试 + ch04 新测试;ch03 的 milvus schema 断言已在 Task 2 更新)。
- [ ] **Step 2: 浏览器验收** — 起服务后:验收 2(型号 BM25 命中)、验收 3(引用可点看原文)、验收 4(库外拒答 + `SELECT ... FROM low_confidence_questions` 有新行)、验收 5(👍/👎 一次性)。
- [ ] **Step 3: 验收 1 复跑留档** — `make eval-rag` 报告贴进 dev-notes。
- [ ] **Step 4: README 补 ch04 使用与演示命令** — `make milvus-up / kb-reset / kb-build / kb-vectorize / eval-rag / eval-check / smoke-rag` 与聊天页演示。
- [ ] **Step 5: dev-notes 补记各阶段 + 提交** — commit `docs(ch04): README 与开发留痕收尾`。

---

## Self-Review

**Spec 覆盖:** §0 目标(混合/重排/过滤/理解/质量控制/评估/前端)→ Task 2/3/7/5/9-10/12-13/14;§2.1 schema → Task 2;§2.2 在线链路 → Task 9/10;§2.3 取舍 → Task 3(手写 rerank)/Task 4(扁平字段);§3 组件 → Task 2/3/5/6/7;§4 质量控制 → Task 4/8/9/10;§5 评估 → Task 11/12/13;§6 前端 → Task 14;§7 数据模型 → Task 8;§8 冒烟/TDD/样例/验收映射 → Task 1/2-10/5-6/11/13/15。无缺口。

**Step scan:** 各测试步给测试名 + 断言,代码步给签名,无 TBD/「补测试」类空步。

**类型一致性:** `search_knowledge(query, strategy, top_k, category, client, collection, bm25_text)` 跨 Task 7/9/13 一致;hit dict(`id/score/question/answer/section_path/content_type/category`,hybrid_rerank 加 `rerank_score`)跨 Task 2/7/9 一致;`insert_low_confidence(session, conversation_id, raw_question, source, reason)` 跨 Task 8/10 一致;citation item(`n/id/section_path/question/answer/content_type`)跨 Task 9/10/14 一致;`settings.recall_top_k/rerank_top_k/rerank_min_score` 跨 Task 1/7/9 一致。

**Review Focus:** 5 条分别落在 Task 2/9/9/5-6/14。

**Proportion:** 计划与 spec 同量级,代码块只出现在「测试断言」与「算法不确定处」。
