# MewHelp — 电商智能客服(Ch04:混合检索 + 重排 + 评估)

FastAPI + SQLAlchemy + MySQL(Docker)+ LangChain `@tool`。模型经 Function Calling
自己决定调哪个工具,单轮调用后回灌收敛;聊天记录落库,聊天页气泡带工具徽章。

> Windows PowerShell 里 `curl` 是 `Invoke-WebRequest` 的别名,请用 `curl.exe`。
> 先填好 `.env` 再 `make dev`;改完 `.env` 后要重启 `make dev` 才会生效。

## 启动

    cp .env.example .env   # 填入 CHAT_API_KEY
    make db-up             # Docker 起 MySQL(需 Docker Desktop)
    make db-init           # 建四张表 + 灌 FAQ 种子
    make dev               # :8000 应用

    # 浏览器打开聊天页
    http://localhost:8000/

## 验收

浏览器聊天页三条:

1. 问「订单 1001 的物流到哪了」→ 气泡带 `query_logistics` 徽章并按随机物流结果作答。
2. 问「退货政策是什么」→ 气泡带 `query_faq` 徽章并按 FAQ 作答。
3. 问「邮费是多少」→ 答案带 `[n]` 引用角标,点击可看来源 chunk 章节路径与原文。

> PowerShell 里 `curl` 是 `Invoke-WebRequest` 的别名,要用 `curl.exe`。Windows
> PowerShell 5.1 会把 `-d` 双引号剥掉、中文按 GBK 编码,中文/多轮演示直接用
> `scripts\demo_chat.ps1`,不手敲 curl。

    # 两轮流式对话(自动写 UTF-8 请求体,PowerShell 5.1/7 都能用)
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\demo_chat.ps1

    # 5 条标注样例提取验证(需在 project 目录下、.env 已填 key、make dev 运行中)
    make eval

    # 单条流式对话:先写 UTF-8 请求体,再喂给 curl.exe
    [System.IO.File]::WriteAllText("$PWD\body.json", '{"session_id":"s1","message":"hello, 你们卖猫粮吗?"}', (New-Object System.Text.UTF8Encoding($false)))
    curl.exe -N -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" --data-binary "@body.json"

    # 结构化提取
    [System.IO.File]::WriteAllText("$PWD\body.json", '{"text":"订单 MH20260701123 的猫爬架散架了,我要退款"}', (New-Object System.Text.UTF8Encoding($false)))
    curl.exe -X POST http://localhost:8000/api/extract -H "Content-Type: application/json" --data-binary "@body.json"

## 测试

    make test        # 单测(不打真实模型、不打真实 MySQL)
    make eval        # ch01 结构化提取标注样例(需服务运行中、真实 key)
    make eval-tools  # ch02 工具选择标注样例(需真实 key,不打 MySQL)

## 知识库(Ch03)

MySQL 当原文权威源,Milvus Standalone 当向量库,BGE-M3(硅基流动)做嵌入。
`.env` 补上 `EMBED_API_KEY` 后:

    make db-up        # 起 MySQL + Milvus(etcd/minio/milvus,需 Docker Desktop)
    make db-init      # 建表 + FAQ 种子 + 挖知识演示会话
    make kb-build     # 文档切块写入 knowledge_chunks(pending)
    make kb-vectorize # 嵌入后写 Milvus 并回填 done(幂等,中断可重跑)
    make kb-eval      # 5 条检索标注样例(「邮费是多少」应召回运费说明)
    make kb-mine      # 从历史对话挖 QA 去重入库(pending,再 kb-vectorize)

    make kb-reset     # 清空 chunks/暂存并 drop Milvus 集合,重建用
    make kb-mining-eval   # 挖知识标注样例(真实模型)
    make smoke-embed      # 嵌入连通性冒烟(真实 key)

验收两条:

1. 「邮费是多少」这类换说法能召回运费说明并答对(`make kb-eval` + 聊天页)。
2. `make kb-vectorize` 中断后重跑,漏向量化的 pending 块被补齐。

## 混合检索与评估(Ch04)

`query_faq` 内部升级为 RAG 管线:口语改写 + 同义词扩展(仅检索侧)→ 品类过滤 →
Milvus 原生 BM25 + dense 各召回 Top-50 用 `hybrid_search` RRF 融合 →
`bge-reranker-v2-m3` 精排 Top-10 → 机械闸/语义闸两道生成前证据闸 → 首尾组装带编号证据。
证据不足即显式拒答并落 `low_confidence_questions` 池;`.env` 补 `RERANK_API_KEY` 后:

    make milvus-up      # 起 Milvus(etcd/minio/milvus;MySQL 端口 33306,见 .env.example)
    make smoke-rag      # 两条 fail-fast 冒烟:Milvus BM25 + 硅基流动 /rerank
    make kb-reset && make kb-build && make kb-vectorize   # 重建库(4 文档 46 块)
    make eval-rag       # 四策略(vector/bm25/hybrid/hybrid_rerank)分桶评估报告
    make eval-check     # 评估集自检(4 桶 80 题,章节/要点逐字校验)

评估产物落 `data/ch04/reports/rag_eval.{txt,json}`(已 gitignore);`make eval-rag SKIP_GEN=1`
只跑确定性检索段(不碰裁判模型)。聊天页:答案 `[n]` 可点看来源原文,每段回答左下角 👍/👎
一次性反馈(纯前端采集)。

## Ch04 验收

1. `make eval-rag` 四策略对比报告跑出数字:总体 MRR 纯向量 0.943 / 纯 BM25 0.819 /
   混合 0.864 / 混合+重排 0.981;型号类 BM25 优于向量,口语类 BM25 显著偏弱、重排拉回。
2. 问带具体型号(如「MH-LP100 猫砂容量多大」)BM25 那路首条命中。
3. 聊天页问「邮费是多少」,答案 `[n]` 可点、能看到来源章节路径与原文。
4. 问库外内容(如「你们支持货到付款吗」)明确拒答,且 `low_confidence_questions` 新增一行。
