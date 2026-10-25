# MewHelp Ch01 设计:纯对话客服(SSE 流式 + 结构化提取)

## 意图

电商智能客服第一章:跑通纯对话。不做工具调用、不做 Agent 循环。先打通
"模型接入 + 流式 + 多轮上下文 + 结构化输出"这条最薄的竖切,后续章节在
这条骨架上长工具、RAG、Agent。

## 验收标准

1. curl 调对话接口能看到流式回复(逐 token)。
2. 连续问两轮,第二轮能续上第一轮上下文。
3. 发一段售后描述,能拿到结构化 JSON(订单号 / 诉求类型 / 期望方案)。

## 目标 / 非目标 / 反目标

- 目标: FastAPI 单进程 + LangChain 直连上游,两个接口 `/api/chat`(SSE)、
  `/api/extract`(结构化)。
- 非目标: 工具调用、Agent 循环、会话持久化(重启丢历史,接受)、鉴权限流、
  对话中自动触发提取。
- 反目标: 换上游要改代码 / key 进 git / 订单号缺失时模型编造还照单全收。

## 约束

- Python 3.12 + FastAPI + LangChain,依赖管理 uv。
- 应用侧统一 OpenAI 协议直连,地址/模型名/key 进 `.env`,按 `CHAT_*` 分组。
- 默认指向 DeepSeek(`base_url=https://api.deepseek.com`,`model=deepseek-flash`),
  key 由用户手动写入 `.env`,不进 git。`deepseek-chat` 与 `deepseek-v4-flash`
  旧模型名已停用/退役,不作为默认。
- token 预算默认 2000,可配。
- git 仓库建在 `project\` 下,`dev-notes/ch01.md` 放 `project\dev-notes\`,
  目录结构对齐参考 spec。

## 方案(approaches)

- 选定: LangChain `ChatOpenAI` + `astream` 逐 token 流式,服务端按 `session_id`
  存内存历史,`trim_messages` 按 token 预算裁剪,`with_structured_output` 绑
  Pydantic 做提取。与固定技术栈一致,复用 LangChain 的流式与结构化原语。
- 否掉「绕开 LangChain 直接用 openai sdk」: 流式与 `with_structured_output`
  都得手写,违背"技术栈定死 LangChain"。
- 否掉「客户端全量上传 history」: curl 验收不友好,把上下文管理推给调用方。

## 接口契约

### POST /api/chat(SSE 流式对话)

请求:

```json
{"session_id": "abc", "message": "你们家猫粮发什么快递?"}
```

响应 `text/event-stream`,逐 token:

```
data: {"delta": "我"}
data: {"delta": "们"}
...
data: [DONE]
```

出错: `event: error` + 错误信息后关流。

### POST /api/extract(结构化提取)

请求:

```json
{"text": "我上周买的猫爬架,订单号 MH20260701123,到货就散架了,我要退款"}
```

响应(`AfterSalesTicket`):

```json
{"order_id": "MH20260701123", "request_type": "退款", "expected_solution": "商品到货损坏,要求退款"}
```

- `request_type` 枚举: 退款 / 换货 / 维修 / 投诉 / 其他。
- 订单号缺失返回 null,不臆造;上游失败返回 502 JSON。

## Prompt 管理

`core/prompts.py` 集中两个 `ChatPromptTemplate`:

1. 客服对话 prompt: system 写清角色(电商客服)+ 行为约束(不臆造订单/物流
   信息、超范围问题引导回客服话题、语气规范)。
2. 提取 prompt: 面向售后描述的字段提取指令。

## 多轮上下文

内存 `dict[session_id] -> 消息列表`;调用前 `trim_messages` 保 system +
从最近往前保留完整轮次,`TOKEN_BUDGET` 默认 2000。

## 测试与验证

- TDD(pytest,先红后绿,FakeLLM/mock 不打真实模型): memory 裁剪、SSE 组帧、
  schema 校验。
- 标注样例(真模型,用户填完 key 后): 5 条售后描述跑 `/api/extract` 核对字段,
  覆盖订单号缺失、诉求模糊;连发两轮对话验证上下文。

## 关键决策记录

| 决策 | 选择 | 备选与理由 |
|---|---|---|
| 上游接入形态 | 应用直连,配置分 `CHAT_*` 组 | 网关要多一个进程、一份配置,换模型改两处 |
| 会话管理 | 服务端 `session_id` | 客户端全量 history 对 curl 验收不友好 |
| 提取形态 | 独立 `/api/extract` 接口 | 融合进对话需意图判断,超出"纯对话"范围 |
| 依赖管理 | uv | 用户确认 |
| 默认模型 | `deepseek-flash` | 官方当前推荐;旧名 `deepseek-chat`/`deepseek-v4-flash` 已停用 |
