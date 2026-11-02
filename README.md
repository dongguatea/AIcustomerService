# MewHelp — 电商智能客服(Ch02:工具调用)

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
3. 问「邮费是多少」→ `query_faq` 查不到(预期漏召回,留给下一步升级),模型明说不编造。

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
