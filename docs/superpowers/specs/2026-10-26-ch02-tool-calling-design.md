# MewHelp Ch02 设计:工具调用客服(单轮 Function Calling + MySQL 持久化)

## 意图

给 ch01 的纯对话客服装上"查数据"能力:模型经 Function Calling 自己决定调
哪个工具,单轮调用后把结果回灌给模型收敛成最终回答。工具链长在客服聊天
入口本身——用户在聊天页问一句,后端走「模型定工具 → 执行 → 回灌收敛」,
最终回答仍逐 token 流式吐出,聊天页在气泡里标出这轮调了哪个工具。

## 验收标准

1. 浏览器打开聊天页,问「订单 1001 的物流到哪了」,能看到模型选中工具
   (气泡带工具徽章),并按返回结果作答。
2. 问「退货政策是什么」,`query_faq` 查得到并作答。
3. 换个说法问「邮费是多少」,确认关键词查表查不出来——这个漏召回是预期
   结果,记下来留给下一步升级。

## 目标 / 非目标 / 反目标

- 目标: FastAPI + SQLAlchemy 分层骨架;Docker 起 MySQL;四张表灌测试数据;
  五个 LangChain `@tool` 业务工具;工具注册/校验/错误处理/超时重试基础设施;
  单轮工具调用接进 SSE 聊天入口;聊天记录(含工具调用与结果)落库;聊天页
  工具轨迹徽章。
- 非目标: 多轮自动循环 Agent Loop、向量检索/RAG、真实电商/物流系统对接
  (三个查询工具内部随机生成数据)。
- 反目标: 模型编造订单/物流数据(没查到的必须明说);工具执行失败把异常
  抛给用户;换上游要改代码;key 或数据库口令进 git。

## 约束

- 技术栈定死: Python 3.12 + FastAPI + SQLAlchemy + MySQL(Docker 起)+
  LangChain `@tool`。走不通停下问,不换方案。
- 应用侧沿用 OpenAI 协议直连上游,`CHAT_*` 配置不变;默认 DeepSeek
  `deepseek-flash`。复用 ch01 结论: DeepSeek 默认思考模式不支持 function
  calling 的 tool_choice,故 `CHAT_THINKING=disabled` 继续生效。
- 只做单轮:模型调一次工具就收敛;工具执行那一段先推一个状态帧。
- 单测不打真实模型(FakeLLM/stub)、不打真实 MySQL(用 SQLite 内存库);
  真实模型与真实 MySQL 只用于验收。

## 方案(approaches)

- 选定: LangChain `ChatOpenAI.bind_tools([...])` 做决策调用,`ainvoke`
  拿 `AIMessage.tool_calls`;命中工具后构造 `ToolMessage` 回灌,再用
  `astream` 流式吐最终回答。持久化用 SQLAlchemy 2.0 同步 ORM +
  PyMySQL,异步工具内部 `asyncio.to_thread` 包同步 DB 访问,避免阻塞
  FastAPI 事件循环。
- 否掉「绕开 LangChain 手写 openai sdk 的 tools 调用」: 违背技术栈定死
  LangChain,且 `bind_tools`/`ToolMessage` 现成原语不用是重复造轮子。
- 否掉「SQLAlchemy async + aiomysql/asyncmy」: 演示项目 DB 访问量小,
  同步 ORM + to_thread 更简单、更好测,少引一个异步驱动。
- 否掉「决策调用也走流式并从 chunk 拼 tool_calls」: 单轮、单工具场景下
  复杂度不值;决策用非流式 `ainvoke` 一次拿全,最终回答再走真流式。

## 分层骨架

```
app/
  config.py          # + database_url / tool_timeout / tool_max_retries
  main.py            # 已服务 index.html(不变)
  db/
    base.py          # Base(DeclarativeBase)
    session.py       # engine / SessionLocal / init_db
    models.py        # Faq / Conversation / Message / Ticket
  repositories.py    # faq / conversation / message / ticket 四个仓储
  tools/
    registry.py      # ToolRegistry:注册/校验/执行(超时+重试)
    business.py      # 五个 @tool 业务工具 + 会话上下文 ContextVar
  services/chat.py   # ChatService:决策→执行→回灌→流式 编排
  api/chat.py        # SSE 路由,调 ChatService,落库
  schemas/chat.py    # ChatRequest(+可选 user)
  static/index.html  # 聊天页(改造:工具轨迹徽章,Vibe Coding)
scripts/init_db.py   # 建四张表 + 灌测试数据
docker-compose.yml   # MySQL 8
```

层间依赖单向: api → service → tools → repositories → db,不得反向。

## 数据模型(四张表)

- `faq`: `id`(int PK)、`question`(text)、`answer`(text)、`category`(str)。
- `conversations`: `id`(int PK)、`session_id`(str unique)、`user`(str 客户
  标识)、`status`(str 处理状态,默认 open)、`created_at`。
- `messages`: `id`(int PK)、`conversation_id`(FK→conversations.id)、
  `role`(user/assistant/tool)、`content`(text)、`tool_call_request`
  (JSON 可空,存 `{name,args}`)、`tool_call_id`(str 可空)、`created_at`。
- `tickets`: `ticket_no`(str PK,形如 `TKT-<epoch>-<4随机>`)、
  `conversation_id`(FK)、`description`、`ticket_type`、`status`(默认
  open)、`created_at`。

种子数据: FAQ 灌「退货政策 / 换货流程 / 发货时效 / 发票」等条目,答案里
明确写清政策;**故意不灌**含「邮费/运费」关键词的条目,以复现验收 3 的
漏召回。

## 业务工具(五个 `@tool`)

均 `async def`,参数用 Pydantic `args_schema`;返回 `str`(序列化成
ToolMessage 的 content)。三个查询工具随机生成数据、不建表、不接真实接口。

| 工具 | 参数 | 行为 |
|---|---|---|
| `query_order` | `order_id: str` | 随机生成订单摘要(下单时间/商品/金额/状态) |
| `query_product` | `keyword: str` | 随机生成商品列表 |
| `query_logistics` | `order_id: str` | 随机生成物流轨迹与当前节点 |
| `query_faq` | `keyword: str` | `faq.question LIKE %kw%` 查表,命中拼成文本,未命中返回"未找到" |
| `create_ticket` | `description: str`,`ticket_type: str` | 写 tickets 表,ticket_no 由服务端生成;conversation_id 从 ContextVar 取 |

`create_ticket` 不把 `conversation_id` 暴露给模型: 路由在进入编排前把当前
conversation 写入 `ContextVar`,工具内部读取,避免模型乱填。

## 工具基础设施

`ToolRegistry`:

- `register(tool)` / `get(name)` / `names()` / `as_list()`(喂给
  `bind_tools`)。
- `execute(name, args, *, timeout, max_retries)`:
  1. 参数 Schema 校验(走工具 args_schema,非法参数抛错转错误结果);
  2. `asyncio.timeout` 包 `tool.ainvoke`,超时或异常按 `max_retries`
     重试(默认 2 次);
  3. 全部失败返回固定错误文案(不回抛),让模型据此向用户解释"查询失败",
     不让异常裸奔到 SSE。

## 单轮工具调用流程(services/chat.py)

```
1. 按 session_id 取/建 conversation,载入最近消息重建历史,trim 到预算。
2. 拼 system prompt + history,先落 user 消息。
3. 决策: decision_model = get_chat_model().bind_tools(registry.as_list())
   ai_msg = await decision_model.ainvoke(messages)
4. 若 ai_msg.tool_calls 非空:
   a. 取第一个 tool_call(单轮),推状态帧
      data: {"status":"tool_call","tool":name,"tool_call_id":id,"args":args}
   b. 落 assistant 消息(role=assistant, tool_call_request={name,args},
      tool_call_id=id, content 存 name 占位)
   c. registry.execute(name,args) → 落 tool 消息(role=tool, content=结果,
      tool_call_id=id)
   d. 回灌: 最终 messages = [..., ai_msg, ToolMessage(...)]
   e. 用 get_chat_model(streaming=True).astream(最终 messages) 逐 token 推
      delta,结束后落 assistant 最终回答消息。
   否则(未调工具): 把 ai_msg.content 切小块逐帧推 delta(单轮、单次调用,
   不二次请求),落 assistant 消息。
5. 推 data: [DONE]。
```

历史重建映射: role=user→HumanMessage;role=assistant 且 tool_call_request
非空→`AIMessage(content, tool_calls=[{name,args,id}])`;role=assistant 纯
文本→`AIMessage(content)`;role=tool→`ToolMessage(content, tool_call_id)`。

## SSE 协议扩展

沿用 `data: {"delta":...}` 与 `data: [DONE]`、`event: error`,新增一个
工具状态帧:

```
data: {"status":"tool_call","tool":"query_logistics","tool_call_id":"call_xxx","args":{"order_id":"1001"}}
```

聊天页收到 `status=tool_call` 时在助手气泡上挂一枚工具徽章
(`⚙ query_logistics`)。

## 聊天页(Vibe Coding,豁免流程)

改造现有 `app/static/index.html`(ch01 已由用户完成,`main.py` 已用
`FileResponse` 服务首页): 在 `handleSSEBlock` 识别新增的
`status=tool_call` 帧,助手气泡内追加一枚工具徽章;`renderAssistant` 重建
innerHTML 时按 `acc.tool` 决定是否带徽章;补一条 `.tool-badge` 样式,并把
页头副标题改为 Ch02。不套 brainstorm/TDD/code review,按效果直接改。

## Prompt 更新

ch02 的 `CUSTOMER_SERVICE_SYSTEM` 从"没有查询权限"改为"可按需调用工具":
订单/物流/商品/FAQ/建工单走工具;工具查不到的(如运费)明确说未收录、引导
人工,禁止编造;语气与职责范围约束沿用 ch01。

## 测试与验证

- TDD(pytest,先红后绿): 四表模型映射、仓储(faq LIKE 命中/漏召回、
  ticket 写入、消息流水)、ToolRegistry(校验失败/超时重试/错误收敛)、
  ChatService(未调工具、单轮工具调用、工具失败仍能回答)、SSE 组帧与
  状态帧、聊天记录落库。
- 纯数据/Prompt 产出(种子数据、新 system prompt): 用标注样例评估替代
  TDD —— 工具选择样例集 `tests/data/tool_eval.json`(含「订单1001物流」→
  query_logistics、「退货政策」→ query_faq、「邮费」→ query_faq 但漏召回),
  真实模型跑一遍核对选中的工具与是否编造。
- 真实验收(需 Docker Desktop + 真实 key): 浏览器问三条验收话术,核对
  工具徽章、回答、落库记录。

## 关键决策记录

| 决策 | 选择 | 备选与理由 |
|---|---|---|
| 决策调用形态 | `bind_tools` + `ainvoke` 非流式 | 流式拼 tool_calls 复杂度不值(单轮单工具) |
| DB 访问 | 同步 ORM + PyMySQL + to_thread | 异步驱动多一个依赖,演示量级用不到 |
| 建表/灌数据 | `scripts/init_db.py` | 不引 Alembic,演示项目从简 |
| create_ticket 会话关联 | ContextVar 注入 | 不把 conversation_id 暴露给模型 |
| 未调工具时流式 | 把已得 content 切小块推帧 | 单次模型调用,不二次请求 |
| 漏召回口径 | FAQ 故意不灌"邮费",工具返回"未找到" | 验收 3 要求复现漏召回 |
