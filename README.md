# MewHelp — 电商智能客服(Ch01:纯对话)

FastAPI + LangChain,模型上游直连。本章不做工具调用、不做 Agent 循环。

> Windows PowerShell 里 `curl` 是 `Invoke-WebRequest` 的别名,请用 `curl.exe`。
> 先填好 `.env` 再 `make dev`;改完 `.env` 后要重启 `make dev` 才会生效。

## 启动

    cp .env.example .env   # 填入 CHAT_API_KEY
    make dev               # :8000 应用

## 验收

    # 1. 流式对话(逐 token)
    curl -N http://localhost:8000/api/chat -H 'Content-Type: application/json' \
      -d '{"session_id": "s1", "message": "你们卖猫粮吗?"}'

    # 2. 两轮上下文(第二轮须复述第一轮信息)
    curl -N http://localhost:8000/api/chat -H 'Content-Type: application/json' \
      -d '{"session_id": "s1", "message": "我叫王小明,刚买了智能猫砂盆"}'
    curl -N http://localhost:8000/api/chat -H 'Content-Type: application/json' \
      -d '{"session_id": "s1", "message": "还记得我叫什么、买了什么吗?"}'

    # 3. 结构化提取
    curl http://localhost:8000/api/extract -H 'Content-Type: application/json' \
      -d '{"text": "订单 MH20260701123 的猫爬架散架了,我要退款"}'

## 测试

    make test   # 单测(不打真实模型)
    make eval   # 标注样例评估(需服务运行中、.env 已填真实 key)
