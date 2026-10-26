# MewHelp — 电商智能客服(Ch01:纯对话)

FastAPI + LangChain,模型上游直连。本章不做工具调用、不做 Agent 循环。

> Windows PowerShell 里 `curl` 是 `Invoke-WebRequest` 的别名,请用 `curl.exe`。
> 先填好 `.env` 再 `make dev`;改完 `.env` 后要重启 `make dev` 才会生效。

## 启动

    cp .env.example .env   # 填入 CHAT_API_KEY
    make dev               # :8000 应用

## 验收

> PowerShell 里 `curl` 是 `Invoke-WebRequest` 的别名,要用 `curl.exe`。而且
> Windows PowerShell 5.1 会把 `-d` 里的双引号剥掉、把中文按 GBK 编码,所以中文/
> 多轮演示直接用 `scripts\demo_chat.ps1`,不手敲 curl。

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

    make test   # 单测(不打真实模型)
    make eval   # 标注样例评估(需服务运行中、.env 已填真实 key)
