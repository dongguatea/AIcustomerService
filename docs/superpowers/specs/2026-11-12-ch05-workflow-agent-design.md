# Ch05 设计文档:Workflow 确定性编排 + 主力 Agent 混合架构

> 把 ch04 的「单一 agent 入口 + 单次工具调用」升级成 LangGraph 的 Workflow + Agent
> 混合架构:图骨架做确定性编排,主力 Agent 只是骨架里的一个 ReAct 节点。

## 1. 意图与动机

ch04 已经有一套可用的聊天链路(`app/services/chat.py`):一个 system prompt + 绑定工具 +
一次模型调用决定「要不要调工具、调哪个」,query_faq 走专用 RAG 路径,其余工具走
ToolRegistry,最后流式出终答。它的问题不在功能,而在「结构」:所有决策都挤在一条
串行链路里,指代、意图、检索、置信、Agent、日志这些环节没有边界,没法单独演进、
没法单独观测,也没法按意图走不同的确定性流程(比如投诉不该进 Agent、闲聊不该花模型)。

本章把这条链路拆成一张 LangGraph 图:Workflow 确定性编排做骨架,主力 Agent 作为骨架里
的核心节点。骨架负责「谁走哪条路、什么条件下放行」,Agent 只负责「这条路需要模型
自由发挥的那一段」。这是「生产级」的第一步——先有可观测、可扩展的骨架,再往里填肉。

## 2. 目标 / 非目标 / 反目标

目标:

- 用 LangGraph 搭出七节点骨架:指代消解 → 意图识别 → 按意图分流 → [知识检索 →
  置信度闸] → 主力 Agent → 日志记录。
- 七类意图写死分流到四个出口:知识类(强制 RAG→闸→Agent)、业务数据类(直接 Agent)、
  投诉(安抚话术 + 两个可选项)、闲聊(固定话术零模型)。
- 主力 Agent 跑 ReAct 环:简单问题一次工具调用收敛,复杂问题多步,缺信息追问;
  有停止条件和 token 消耗控制;复用 ch02 的 Function Calling 工具,本章不新写业务工具。
- 会话状态用 LangGraph State 一路贯穿,持久化用自带的 checkpointer。
- 转人工、建工单分开、都不后端自动执行:后端只在回复里带建议可选项,前端各渲染成
  独立按钮;点「转人工」前端模拟,点「建工单」才写 tickets 表。
- 指代消解先原样透传,意图识别先做最简 prompt 判七类,置信度闸先做最简阈值版。
- 祛魅热身:先不用框架手写一遍最裸 Agent 循环,看清「它就是带工具的循环」,再上 LangGraph。

非目标(本章不做,留给后面章节):

- 意图识别与指代消解的正式版(多轮改写、coref 模型、意图置信度校准)。
- 上下文管理策略升级(摘要、滑窗、前缀缓存)。
- MCP 接入、统一工具执行引擎(沿用 ch02 的 ToolRegistry)。
- 数据飞轮的入库与审核后台(本章只把弱证据问题写进 `low_confidence_questions` 留坑)。
- Langfuse 可观测、成本账。

反目标(做了反而算失败):

- 把投诉、闲聊也塞进 Agent 或花模型调用(应零模型/确定性)。
- 让 Agent 自动执行「转人工」「建工单」(应前端自选)。
- 照抄 `python/` 里已经演进到 ch08 的 graph(带退款子流程、interrupt、ch09 置信度),
  把本章做成比 spec 更复杂的版本。

## 3. 约束

- 技术选型定死:LangGraph(图编排 + State + checkpointer)、前端原生 HTML/CSS/JS。
- 业务工具复用 ch02,不新写业务工具。
- 检索复用 ch03/ch04 检索器(`app/core/retrieval.py` + `run_query_faq`)。
- 库/API 用法先 Context7 核最新文档再动手(已核:StateGraph / add_conditional_edges /
  checkpointer / stream_mode;实现时对安装到的具体 LangGraph 版本再核一次)。
- 实现中若发现选型矛盾或走不通,停下来问,不自行换方案。
- `python/` 只当架构参照,不照抄。

## 4. 方案取舍(Approaches)

1. **LangGraph StateGraph + 手动 ReAct 环(选定)。** 七节点图 + 一个 `main_agent` 节点
   内部是显式 ReAct 环。优点:骨架即 spec,七节点、四出口、闸位、停止条件全部显式可控,
   每步都能单测和观测;缺点:比「一行 create_react_agent」多写几十行。
2. **`create_react_agent`(LangGraph prebuilt)。** 一行搭 Agent 环。缺点:Agent 是整个应用,
   没法把「强制检索→闸→Agent」「投诉/闲聊确定性出口」这些骨架节点插进去,与本章
   「Workflow 做骨架、Agent 做节点」的目标相反。弃。
3. **不引框架、纯手写编排。** 最可控但重造 State/checkpointer/流式轮子,且与定死的
   LangGraph 选型冲突。弃。

## 5. 现状与改造点

现状(`project/`,ch04 分支):

- 聊天链路 `app/services/chat.py`:`ChatService.stream()` 单 agent + 单次工具调用,
  query_faq 走专用 RAG 路径(含 rerank 阈值 + selfcheck 两道闸),无 graph。
- 工具 `app/tools/business.py`:`query_order / query_logistics / query_product /
  query_faq / create_ticket` 五个;`run_query_faq` 返回结构化 `RagResult.sufficient`。
- 低置信落池 `app/repositories.insert_low_confidence` + `LowConfidenceQuestion` 表已就绪。
- 前端 `app/static/index.html` 的 SSE 只认 `tool_call / delta / citations` 三种帧。
- `pyproject.toml` 没有 `langgraph`,也没有 checkpointer 依赖。

改造点:

- 新增 `app/graph/`(state / nodes / routing / build / runtime)。
- 新增 `app/core/intent.py`(最简意图识别)+ 对应 prompt。
- `/api/chat` 从 `ChatService` 改走 graph 流式;`ChatService` 保留还是移除见 §14。
- 新增 `POST /api/actions/create-ticket`(点「建工单」才写表)。
- 前端 `index.html` 加 `actions` 帧处理 + 「转人工」「建工单」两个独立按钮。
- `pyproject.toml` 加 `langgraph`、`langgraph-checkpoint-sqlite`。

## 6. 图骨架

节点:

| 节点 | 职责 | 触发条件 |
| --- | --- | --- |
| `resolve_reference` | 指代消解(ch05 原样透传) | START |
| `classify_intent` | 意图识别,判七类,输出 JSON | 恒 |
| `retrieve_knowledge` | 知识类强制 RAG 检索 | route=knowledge |
| `confidence_check` | 置信度闸留痕(实体节点) | route=knowledge |
| `main_agent` | 主力 Agent 的 ReAct 推理步 | knowledge(强)/business |
| `agent_tools` | ReAct 行动步,执行工具 | 有 tool_calls 且未超步数 |
| `complaint_reply` | 投诉安抚 + 两可选项 | route=complaint |
| `script_reply` | 闲聊固定话术 | route=chitchat |
| `fallback_reply` | 知识路证据弱兜底 + 落池 | 闸判 weak |
| `log` | 日志 + 落 assistant 消息 | 所有出口 |

边:

```
START → resolve_reference → classify_intent
classify_intent --route_by_intent-->
    knowledge   → retrieve_knowledge → confidence_check
                    --confidence_gate--> strong → main_agent
                                        weak   → fallback_reply
    business    → main_agent
    complaint   → complaint_reply
    chitchat    → script_reply
main_agent --should_continue--> continue → agent_tools → main_agent
                                stop     → log
complaint_reply / script_reply / fallback_reply → log → END
```

## 7. 节点行为详述

### 7.1 分流规则(写死在代码里)

七类意图 → 四出口,单一映射表,不做模型决策:

| 意图 | 出口 route | 行为 |
| --- | --- | --- |
| 商品咨询 | knowledge | 强制 RAG → 闸 → Agent |
| 退款退货 | knowledge | 先检索政策 → 闸 → Agent(Agent 仍可调订单/物流工具) |
| 物流 | business | 不预检索,直接 Agent |
| 订单 | business | 不预检索,直接 Agent |
| 售后 | business | 不预检索,直接 Agent |
| 投诉 | complaint | 不进 Agent,安抚话术 + 两可选项 |
| 闲聊 | chitchat | 固定话术,零模型 |

意图识别失败 / 输出不在七类内:保守归 business(交给 Agent 自己应对)。

### 7.2 resolve_reference(最简)

原样透传:把本轮用户消息写进 `resolved_query`,不改写、不做多轮指代。
```
resolved_query = 用户本轮原文
```

### 7.3 classify_intent(最简)

`app/core/intent.py`:`llm.structured(_Intent)`(Pydantic `Literal[七类]` + confidence,
method=function_calling,复用 ch04 的 DeepSeek 结构化口径),prompt 简单、只判七类。
吃 `resolved_query`(空则回落用户原文);异常 → 归 business 兜底。

### 7.4 retrieve_knowledge

复用 `business.run_query_faq(用户原文)`(它内部已含 Query 理解→混合检索→重排→两道闸)。
把 `RagResult` 落进 State:

- `sufficient=True`:`evidence`(编号证据)、`citations`、`evidence_strong=True`。
- `sufficient=False`:`evidence_strong=False`、记 `fallback_source`
  (`retrieval_low_conf` / `self_check`)与 `fallback_reason`(来自 RagResult.reason)。

### 7.5 confidence_check(最简)

实体节点,只留痕门控决策(`gate_decision = "strong"|"weak"`,供 log 节点打点),实际分流
在条件边 `confidence_gate` 读 `evidence_strong`。强 → main_agent,弱 → fallback_reply。
业务数据类没有检索证据,不走此节点、不过此闸。

### 7.6 main_agent(ReAct 推理步)

每次进入:组装消息(见下)→ `get_chat_model(streaming=True).bind_tools(工具集)` →
`ainvoke` 得完整 AIMessage → 累加 `steps += 1`、`tokens_used += usage.total_tokens` →
返回 `{"messages":[ai], ...}`。

消息组装:

- system:ch04 的 `CUSTOMER_SERVICE_SYSTEM`(基础人设 + 工具使用规则)。
- 知识路且 `evidence_strong`:追加证据提示块(编号证据 + 「据此作答、标引用;
  证据已给,不要再调 query_faq;仍可按需调订单/物流工具」)。
- `messages`(checkpointer 里跨轮累积的历史 + 本轮 HumanMessage)。

工具集(单一、不新写业务工具):

- 查询类:`query_order / query_logistics / query_product / query_faq`。
- 控制类:`suggest_action(action ∈ {transfer_human, create_ticket})`——非业务工具,
  只产出「建议动作」、后端不执行(见 §7.7 / §13)。
- 不含 `create_ticket` 写工具(避免后端自动建单,见 §7.8)。

### 7.7 agent_tools(ReAct 行动步)

读最后一条 AIMessage 的 tool_calls,逐个执行:

- 查询类:走 `ToolRegistry.execute`(沿用 ch02 超时/重试)。
- `suggest_action(action, description=None)`:不执行任何动作,把该建议并入
  `suggested_actions`(`action="create_ticket"` 时 description 必填,作工单 draft,
  ticket_type 缺省「其他」),回一条 ToolMessage「已记录建议,请继续」,让模型接着
  收敛出最终答复。

返回 `{"messages":[ToolMessage...]}`(以及可能追加的 `suggested_actions`)。

### 7.8 complaint_reply / script_reply / fallback_reply

- `complaint_reply`:安抚话术 + `suggested_actions=[transfer_human, create_ticket]`
  (create_ticket 带 `draft`:{description=用户原文, ticket_type="投诉"}),后端不执行。
- `script_reply`:闲聊固定话术,零模型调用。
- `fallback_reply`:知识路弱证据兜底话术 + `insert_low_confidence`(source/reason 取自
  retrieve_knowledge 的 `fallback_source` / `fallback_reason`)+
  `suggested_actions=[transfer_human]`。

### 7.9 log

日志一行(conversation_id / intent / route / resolved_query / gate_decision /
steps / tokens_used),并把最终答复落 MySQL `messages` 一条 assistant 消息
(审计 + 工单外键依赖)。最终答复的取值:确定性出口(complaint/script/fallback)读
`state.answer`;Agent 出口读 `state.messages` 最后一条 AIMessage 的 content。

## 8. State 与 checkpointer

State(`app/graph/state.py`,TypedDict):

```python
class ConversationState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # 跨轮历史 + ReAct 消息
    session_id: str        # 前端 session_id,也做 thread_id
    conversation_id: int   # MySQL 会话 id(工单外键 / 审计)
    user: str
    intent: str            # 七类之一
    route: str             # knowledge|business|complaint|chitchat
    resolved_query: str
    evidence: str          # 知识路编号证据
    citations: list
    evidence_strong: bool
    fallback_source: str
    fallback_reason: str
    answer: str            # 确定性节点答复(agent 答复走流式,不落此字段)
    steps: int
    tokens_used: int
    gate_decision: str     # 知识路闸门决策(strong|weak),供 log 打点
    suggested_actions: list
```

checkpointer:`langgraph-checkpoint-sqlite` 的 `AsyncSqliteSaver`(文件型持久化,
路径走 `settings.checkpointer_db_path`,thread_id = session_id)。测试/演示可注入
`InMemorySaver`。`messages` 用 `add_messages` 归并;其余标量无 reducer,入口每轮显式
清零(直接覆盖),杜绝上一轮 answer / suggested_actions / gate_decision 跨轮残留。

MySQL `messages` 继续落库(审计 + tickets 外键);checkpointer 是图内跨轮历史的来源。
入口映射:每轮由 `get_or_create_conversation(session_id, user)` 拿到 `conversation_id`,
user 消息先落 MySQL,再把 `session_id / conversation_id / user / messages=[HumanMessage]`
传入图。ch05 不做上下文裁剪:checkpointer 里的 history 全量累积,裁剪/摘要升级留给 ch07。

## 9. 流式输出契约与前端按钮

`/api/chat` 保持 SSE,事件字典(前端现有三种 + 新增一种):

| frame | 含义 |
| --- | --- |
| `{"delta": text}` | 答复文本(agent 答复流式;确定性节点整块吐) |
| `{"status":"tool_call","tool":name,"tool_call_id":id,"args":{}}` | 一次工具调用(沿用 ch04 前端形状) |
| `{"event":"citations","items":[..]}` | 知识路引用(沿用 ch04 前端形状) |
| `{"event":"actions","items":[..]}` | 建议动作,末尾吐一次,按 type 去重(新增) |
| `[DONE]` | 结束 |

流式实现:`graph.astream(stream_mode=["messages","updates"])`;`messages` 模式按
`langgraph_node` 过滤出 `main_agent` 的 delta;`updates` 模式取确定性节点 answer /
citations / tool / suggested_actions。token 级细粒度与 `version=` 参数在实现时按安装到的
LangGraph 版本用 Context7 再核 + spike 确认,最坏退化为「整块 delta」(仍满足「流式吐给用户」)。

前端:回复气泡下渲染 `suggested_actions` 为两个独立按钮(不互绑):

- 「转人工」:纯前端,点后展示「已转接人工客服」,再蹦一句
  「您好,我是客服小猫,请问有什么可以帮您的」。
- 「建工单」:点后调 `POST /api/actions/create-ticket`(带 session_id + draft 的
  description/ticket_type),后端调 ch02 的 `create_ticket` 写 `tickets` 表,回 ticket_no,
  前端展示结果。
- 都不点、继续发消息:按普通对话继续,无任何后端动作。

## 10. 祛魅热身:裸 Agent 循环

新增 `scripts/bare_agent.py`,零 LangGraph,直接调 `get_chat_model` + ch02 工具,把
Agent 循环手写出来:

```
messages = [system, *history, human]
for step in range(max_steps):
    ai = model.bind_tools(tools).invoke(messages)
    messages.append(ai)
    if not ai.tool_calls: break          # 没工具调用就收敛
    for tc in ai.tool_calls:
        messages.append(ToolMessage(execute(tc), tc.id))
print(ai.content)
```

配 `make bare-agent`。演示目标:一眼看出「Agent = 带工具的 while 循环」,并证一个
「先查订单、再查物流」的多步样例(验收 5 的图外佐证)。

## 11. 置信度闸(最简版)

卡在知识类检索之后、进 Agent 之前。最简实现直接复用 ch04 `run_query_faq.sufficient`
(rerank 最高分低于 `rerank_min_score` 或 selfcheck `useful=false` 即 insufficient):

- 强 → 证据连同问题交给 main_agent。
- 弱 → fallback_reply 兜底话术,不进 Agent,并把该问题写 `low_confidence_questions`
  (source=retrieval_low_conf / self_check)留数据飞轮。
- 业务数据类无检索证据,不走此闸。

正式置信度闸(证据覆盖度多信号加权)留给可观测那章。

## 12. ReAct 停止条件与 token 控制

停止条件(任一命中即停):

1. 最后一条消息无 `tool_calls` → 收敛出答案。
2. `steps >= settings.max_agent_steps` → 强停(默认 6)。
3. `tokens_used >= settings.agent_token_budget` → 强停(默认 4000)。

token 统计:优先取 `usage_metadata.total_tokens`(DeepSeek thinking=disabled 会回传);
缺失时用 `count_tokens_approximately` 近似累计。强停时以当前已生成内容作为答复,
不掐半句、不假装成功。

## 13. 我的决定(待用户确认)

以下是我在 spec 里替用户拍板的点,评审时请逐条看:

1. checkpointer 用 `AsyncSqliteSaver`(文件型,`data/checkpoints.sqlite`),thread_id=session_id。
2. 祛魅热身做成可跑脚本 `scripts/bare_agent.py` + `make bare-agent`。
3. 给 Agent 加一个**非业务**控制工具 `suggest_action`(转人工/建工单),让「Agent 判断
   合适时」也能产出结构化建议动作;它只记录建议、不执行。若你排除「任何新工具」,
   我退回「只有投诉路线出按钮、Agent 路只用话术文字」。
4. 建工单不进 Agent 自动循环:Agent 工具集不含 `create_ticket`;点「建工单」才由
   `/api/actions/create-ticket` 调 ch02 工具写表。
5. 置信度闸最简版 = 复用 ch04 `run_query_faq.sufficient`,不新加证据覆盖度算法。
6. 意图识别失败/越界保守归 business(Agent 兜底),不新增「其他」类。
7. `ChatService` 保留但不再服务 `/api/chat`(先留作对照/可回退),还是直接删?默认:
   `/api/chat` 改走 graph,`ChatService` 删除,相关测试随之替换。(若你倾向保留,评审时指出。)

## 14. 留给 builder

- 各节点、routing、runtime 的具体代码与日志措辞。
- `intent_samples.json` 标注样例 + `scripts/eval_intent_ch05.py`(意图 prompt 的评估集,
  替代 TDD 对纯 Prompt 的验证)。
- graph 的单测(routing 纯函数 / 停止条件 / 闸决策 / SSE 契约,均用假模型)。
- 前端按钮的确切样式与文案、`actions` 帧的渲染逻辑。
- `.env.example` / README / Makefile 的新增条目。

## 15. 验收标准映射

| 验收 | 由谁满足 |
| --- | --- |
| 1 政策类问题日志见强制检索节点 | route=knowledge + retrieve_knowledge 节点日志 |
| 2 「订单1001物流」Agent 自调工具作答 | business→main_agent→agent_tools(query_logistics) |
| 3 投诉出两按钮;转人工前端模拟;建工单才写表;都不点继续对话 | complaint_reply + 前端按钮 + create-ticket 接口 |
| 4 闲聊固定话术 | script_reply 零模型 |
| 5 复杂问题 ReAct 多步 | 裸循环 + graph 的 steps>1 留痕 |
