# Ch04 设计:混合检索 + 重排 + 评估体系

状态:待用户评审(2026-11-10 brainstorm)。
前置:ch03 已交付 dense 单路语义检索(`query_faq` 内部走 Milvus Standalone dense Top-K)。

## §0 目标与不做

**做**:混合检索(Milvus 原生 BM25 + dense,`hybrid_search` RRF 融合)、重排
(`bge-reranker-v2-m3` 精排 Top-10,首尾组装)、元数据过滤(品类)、Query 理解
(口语→标准问法改写 + 同义词扩展,只在检索侧)、生成质量控制(带引用编号可溯源 /
检索不足或自评不足即拒答并落低置信度问题池 / System Prompt 负面知识)、评估体系
(带难度梯度 + ground-truth 的评估集,检索段 Recall@K + MRR,生成段 Faithfulness,
四策略对比分桶报告)、前端(聊天页可点引用 + 每段回答 👍/👎 反馈)。

**不做**(本章边界):指代消解、多轮改写。

**固定选型(定死,不自换)**:Milvus 2.5+ 原生 BM25 + `hybrid_search` RRF;重排模型
`bge-reranker-v2-m3`;嵌入沿用 ch03 `BAAI/bge-m3`(硅基流动)。

## §1 现状勘察结论(与参考实现的差异)

- 真正的仓库是 `project/`(停在 `ch03` 分支)。同级 `python/` 是全书 10 章参考实现,
  其 ch04 已演进过(ch08 把工具迁到 `builtin/`、图编排迁到 `graph/`),只能当**架构参照**,
  不能照抄;本章按本 spec + Context7 核过的 API 在 `project/` 现有结构上重新落地。
- `project/` 的 ch03 **已经**是 Milvus Standalone(`milvus_uri=http://localhost:19530`,
  `milvusdb/milvus:v2.6.22`),因为 Windows 装不上 Milvus Lite。所以参考设计里
  「Lite → Standalone」这一步**已天然完成**,本章只需在现有集合 schema 上加 BM25 字段。
- 聊天链路是**单一 agent 入口 + 单次工具调用**:`app/services/chat.py` 用 `bind_tools`
  决策 → `ToolRegistry.execute` 跑工具 → 用同一 `CUSTOMER_SERVICE_SYSTEM` 上下文流式生成
  终答。没有独立 agent 类、没有 graph。本章沿用这个结构,`query_faq` 升级为 RAG 管线。
- 聊天模型是 DeepSeek(`.env` 里 `deepseek-flash`,`chat_thinking=disabled`),**不是**参考
  实现用的 glm。改写 / 自评 / 忠实度裁判全部复用这个聊天模型(项目唯一可用模型)。
- 当前知识库只有 3 份文档(faq / policy / manual),**没有型号内容**;验收 2「问具体型号
  BM25 那路能命中」要求新增一份规格手册(易混型号族),见 §5.1。
- `ToolRegistry.execute` 对所有工具共用 `settings.tool_timeout=10s`、重试 2 次;RAG 管线
  要多次上游往返(嵌入 + 混合检索 + 重排 + 自评),需给 `query_faq` 单独更长的超时。

## §2 架构与数据流

### 2.1 Milvus 集合 schema(重建)

集合 `knowledge` 在 ch03 基础上加 BM25 相关字段(`id` 仍 = MySQL `knowledge_chunks.id`,
重跑 upsert 幂等):

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | INT64 主键 | = MySQL chunk id |
| `dense` | FLOAT_VECTOR(1024, COSINE) | bge-m3 稠密向量(不变) |
| `text` | VARCHAR(enable_analyzer=True, analyzer=chinese) | BM25 输入,= `category + questions + answer`(与 dense 同源) |
| `sparse` | SPARSE_FLOAT_VECTOR | BM25 Function 输出,系统自动生成 |
| `question` / `answer` / `section_path` / `content_type` / `category` | VARCHAR | 原文与元数据(不变) |

BM25 Function(`Function(name, input=["text"], output=["sparse"], FunctionType.BM25)`)
在 `create_collection` 前 `schema.add_function`;索引 `dense`→AUTOINDEX/COSINE、
`sparse`→SPARSE_INVERTED_INDEX/BM25。中文分词在**字段级** `analyzer_params={"type":"chinese"}`。
`dualwrite.vectorize_pending` 已产出 `text` 字段,只需 schema 接受它;upsert 后 `flush`
(现有流程已 flush)。

### 2.2 在线主链路(`query_faq` 升级为 RAG 工具)

保持单一入口与单次工具调用。`query_faq` 内部:

```
① Query 理解   口语→标准问法 + 同义词扩展(扩展词只拼进 BM25 检索文本,dense 用标准问法)
② 元数据过滤   按 category 先过滤(过滤后为空/全弱则回退不过滤重试,过滤是优化非硬约束)
③ 混合召回     dense Top-50 + BM25 Top-50 → hybrid_search RRF → 候选
④ 重排         bge-reranker-v2-m3 精排出 Top-10
⑤ 机械闸       无命中或最高 rerank 分 < rerank_min_score → 判「检索证据不足」
⑥ 语义闸       生成前自评:模型判召回证据够不够答(useful?)→ 不够判「自评不足」
⑦ 首尾组装     最相关放首、次相关放尾,其余居中(缓解 lost-in-the-middle)
⑧ 返回         带编号证据串(供模型带引用作答)+ citations 元数据经 ContextVar 暂存
```

`query_faq` @tool 返回**字符串**:证据足时返回「[n] 问:… 答:…」编号证据;证据不足时返回
「未检索到足以回答的知识,请向用户拒答」信号。`sufficient / source / reason / citations`
经 `ContextVar` 暂存,供 `ChatService` 读取(同一 async 上下文,不做跨任务投递)。

`ChatService` 在工具执行后:
- 读 ContextVar:若 `sufficient=False` → 确定性落 `low_confidence_questions`
  (`source=retrieval_low_conf | self_check`),再进入终答(模型按拒答规则作答);
- 若 `sufficient=True` → 在流式 delta 之前多推一个 `citations` SSE 事件,把
  `{n, id, section_path, question, answer, content_type}` 下发前端。

### 2.3 技术取舍

- 不引入 LangGraph / Langfuse,沿用 ch03 纯 async 编排。
- 重排走硅基流动 `/rerank`(Jina/Cohere 形状),手写 httpx 请求,不用 OpenAI 客户端。
- 结构化自评输出用**扁平字段**(`useful`/`reason`),规避嵌套对象数组的解析坑。
- 同义词扩展只作用于 BM25 检索文本,不改 dense query、不改入库(不拆存多份)。

## §3 检索管线组件

| 组件 | 职责 | 测试要点 |
|---|---|---|
| `milvus_client`(扩展) | 新 schema + BM25 Function;`hybrid_search([dense_req,bm25_req], RRFRanker())`;`dense_search` / `bm25_search` 单路(评估用);`category` filter | 假 client:两路各召回、RRF 融合有序、filter 生效、BM25 命中型号词 |
| `rerank`(新) | 直连上游 /rerank 调 bge-reranker-v2-m3,返回 [(原始索引, 相关分)] 降序截 Top-N | mock 上游:重排、截断、空输入安全 |
| `retrieval.search_knowledge`(重写) | `strategy` 分支(`vector/bm25/hybrid/hybrid_rerank`)+ `bm25_text` 分离 + 首尾组装 | mock 子步:管线顺序、strategy 分支、首尾排列正确 |
| `query_understanding`(新,Prompt) | 口语→标准问法 + 同义词扩展(检索侧) | 纯 Prompt → 标注样例验证(非 TDD) |
| `selfcheck`(新,Prompt) | 生成前证据充分性自评(`useful`/`reason`) | 标注样例验证;异常按「不足」处理 |

`strategy` 入参是关键:在线默认 `hybrid_rerank`;评估脚本复用同一 `search_knowledge`
跑四策略,保证「评估的就是线上的」。

## §4 生成质量控制

- **带引用**:`CUSTOMER_SERVICE_SYSTEM` 增加 `query_faq` 结果处理规则——sufficient=true 时
  严格依据编号证据作答、关键结论后标 `[n]`、型号逐字复制、条件跟随;sufficient=false 时
  明确拒答并引导人工。
- **机械闸 + 语义闸**都发生在生成之前(§2.2 ⑤⑥);任一触发即拒答并落池(落池是确定性的,
  不依赖模型)。
- **负面知识**:System Prompt 明列禁止承诺项(不承诺到账时间 / 到货配送时间 / 维修时长 /
  赔偿金额与时效),统一「以平台售后规则为准」。
- `low_confidence_questions` 写入走确定性 repository 方法 → TDD。

## §5 评估体系

### 5.1 知识库内容补充

新增 `data/kb/product-specs.md`(content_type=`spec`):收录易混型号族——智能猫砂盆
`MH-LP100/LP50/LP200`、自动饮水机 `MH-W20/W40/W60`、恒温加热垫 `MH-HP12/HP20`、
自动喂食器 `MH-FD10/FD30`、猫爬架 `MH-CT30`、看护摄像头 `MH-CAM1`、保暖猫窝 `MH-NEST20`
等。使「精确匹配 vs 语义混淆」在四策略上可区分(验收 2 的前提)。内容与 `after-sales-manual.md`
已出现的设备品类、`query_product`/种子数据一致。

### 5.2 评估集与指标

评估集 `tests/data/eval_ch04.jsonl`:**4 桶 × 20 题 = 80 题**,带难度梯度(易/中/难),每题标
`expect_section`(预期命中章节关键词,算 Recall/MRR)与 `expect_points`(标准答案要点):

- `A_policy` 政策/流程通用(20)
- `B_model` 型号/具体名词(20,验收 2;含易混型号族,考验精确匹配)
- `C_colloquial` 口语模糊改写(20,单轮,不做指代/多轮)
- `D_oob` 库里没有→应拒答(20,验收 4;不带 ground-truth)

指标:
- **检索段(确定性,四策略 × A/B/C 桶)**:Recall@5、MRR(按 `expect_section` 命中
  `section_path` 计);另给确定性「证据覆盖度」(Top-K 证据盖住 `expect_points` 的机械比例)。
- **生成段(聊天模型)**:hybrid_rerank 线上管线的 Faithfulness(A/B/C 桶,忠实度裁判);
  D 桶拒答率。
- **四策略对比**:`vector / bm25 / hybrid / hybrid_rerank` 复用 `search_knowledge(strategy=…)`。
- 生成段每次调用带超时 + 失败跳过 + 低并发,单点故障不拖垮整轮;检索段不依赖模型、始终产出。
- 产出:`scripts/eval_ch04.py` → 策略 × 桶 × 指标分桶报告,落 `data/ch04/reports/rag_eval.txt`
  (运行日志)与 `rag_eval.json`(给后续可能的页面读,本章只做 CLI 报告,不做后台评估页)。
  入口 `make eval-rag`;`make eval-check` 自检评估集(id 不重复、四桶题数一致、
  `expect_section` 在库内命中、`expect_points` 逐字出现在目标正文、D 桶无 ground-truth)。

## §6 前端(Vibe Coding,不套 brainstorm/TDD/review)

在现有 `app/static/index.html` 聊天页:
- 回答文本里的 `[n]` 渲染成可点角标,点开浮层显示该引用的 `section_path` + 来源 chunk
  原文(question/answer)。数据来自 SSE 新增 `citations` 事件。
- 每段有效回答左下角挂 👍/👎 满意度反馈:点击点亮所选、另一个淡出、显示「已反馈」并一次性
  锁定;纯前端采集,不落库(`source=user_feedback` 归后续数据飞轮章)。
- 效果由用户描述、迭代式直接改。

## §7 数据模型

新表 `low_confidence_questions`(ORM `LowConfidenceQuestion`,随 `init_db` 自动建表):
`id / conversation_id(FK conversations.id) / raw_question / source
(retrieval_low_conf|self_check|user_feedback) / reason / created_at`。本章写入
`retrieval_low_conf` 与 `self_check` 两个来源;`user_feedback` 本章不接线(留飞轮章)。

## §8 测试策略与验收映射

- TDD(确定性代码,红→绿→提交):milvus 混合检索/filter/RRF、rerank 客户端、检索编排与首尾
  组装、selfcheck 的异常回落、`low_confidence_questions` 写入、ChatService 的拒答落池与
  citations 事件。Milvus 单测用假 client(与 ch03 一致),真实验证放冒烟/验收。
- 标注样例验证(替代 TDD 的纯 Prompt/数据任务):query 改写、自评、faithfulness 裁判、评估集。
- Fail-fast 冒烟(排最前,红线):①硅基流动 `/rerank` 跑通 bge-reranker-v2-m3;②Milvus
  Standalone 原生 BM25 + `hybrid_search` 跑通(chinese 分词命中)。任一不通→停下问用户,
  只在固定选型内换法。

| 验收 | 覆盖 |
|---|---|
| 1 四策略对比报告能跑出数字 | §5 `make eval-rag` 分桶报告 |
| 2 型号具体问题 BM25 那路能命中 | §2.2③ BM25 路 + B 桶 + CLI/浏览器实测 |
| 3 引用编号定位原文、点引用看来源 | §4 引用 + §6 前端 + 浏览器实测 |
| 4 库里没有→明确拒答且进低置信池 | §2.2⑤⑥ + D 桶 + 浏览器实测落库 |
| 5 每段回答可点 👍/👎、点亮并显示已反馈 | §6 前端 + 浏览器实测 |

## §9 决策记录

我的决定(待你确认):
1. **评估报告范围**:只做 CLI 报告(`make eval-rag` → txt/json),**不做**后台 RAG 评估页 /
   编造个案台账 / 裁判校准回归——那些是参考实现的扩展,超出本章提示词范围,留作后续章节。
2. **评估集规模**:4 桶 × 20 = 80 题(参考实现早期也是这个数),不再扩到 300 题跨文档桶。
3. **LLM 复用**:改写 / 自评 / 忠实度裁判全部复用 `.env` 里的 DeepSeek(项目唯一模型),
   不引入第二把模型 key。
4. **新增规格手册** `product-specs.md`(content_type=spec)是验收 2 的必要内容补充,不算换选型。
5. Milvus 镜像沿用 ch03 的 `v2.6.22`(≥2.5 支持原生 BM25),无需改。

## §10 风险闸

- **最大风险**:上游 `/rerank` 与 Milvus Standalone BM25 两条新链路能否跑通 → 排为第一个
  fail-fast 冒烟任务,不通即停。
- **前置依赖**:Docker 起 Milvus 三件套;`.env` 需补 `RERANK_API_KEY`、`EMBED_API_KEY`
  (当前 `.env` 只有 DeepSeek 聊天 key)。
- Milvus 集合 schema 变更需重跑建库 + 向量化(`make kb-reset && make kb-build && make kb-vectorize`)。

## §11 文件布局(预计)

- 改:`app/config.py`、`app/kb/milvus_client.py`、`app/kb/dualwrite.py`、`app/kb/sources.py`、
  `app/core/retrieval.py`、`app/core/prompts.py`、`app/core/llm.py`、`app/tools/business.py`、
  `app/tools/registry.py`(按工具超时)、`app/services/chat.py`、`app/api/chat.py`、
  `app/db/models.py`、`app/repositories.py`、`app/static/index.html`、`docker-compose.yml`、
  `Makefile`、`.env.example`、`scripts/init_db.py`(文案)。
- 新:`data/kb/product-specs.md`、`app/core/rerank.py`、`app/core/query_understanding.py`、
  `app/core/selfcheck.py`、`scripts/eval_ch04.py`、`scripts/validate_eval_ch04.py`、
  `scripts/smoke_rerank.py`、`scripts/smoke_milvus_bm25.py`、`tests/data/eval_ch04.jsonl`、
  相关单测。
