# Ch05 Workflow + Agent 混合架构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 LangGraph 把 ch04 的单 agent 链路重构成「Workflow 确定性编排做骨架 + 主力 Agent 做核心节点」的七节点图。

**Architecture:** 一个 StateGraph:指代消解(透传)→ 意图识别(七类 JSON)→ 写死分流(四出口)。知识类走 强制 RAG → 置信度闸 → 主力 Agent;业务数据类直接进 Agent;投诉/闲聊走确定性节点零/少模型。主力 Agent 是图内的 ReAct 环,复用 ch02 工具 + 一个非业务 `suggest_action` 控制工具。State 由 `add_messages` + 自带 checkpointer 贯穿。

**Tech Stack:** LangGraph(StateGraph / add_messages / AsyncSqliteSaver)、LangChain 1.x、FastAPI、原生 HTML/CSS/JS。

**Spec:** `docs/superpowers/specs/2026-11-12-ch05-workflow-agent-design.md`

## Global Constraints

- 技术选型定死:LangGraph(图编排 + State + checkpointer)、前端原生 HTML/CSS/JS。
- 业务工具复用 ch02 五个,不新写业务工具;唯一新增是非业务控制工具 `suggest_action`。
- 检索复用 ch04 `app/core/retrieval.py` + `business.run_query_faq`;置信度闸最简 = `run_query_faq.sufficient`。
- 建工单不进 Agent 自动循环;转人工/建工单都前端自选、后端不自动执行。
- 指代消解原样透传;意图识别七类:物流 / 订单 / 商品咨询 / 退款退货 / 售后 / 投诉 / 闲聊。
- 库/API 用法先 Context7 核最新文档;实现中若选型矛盾,停下来问,不自行换方案。
- ch05 不做上下文裁剪:checkpointer 里的 history 全量累积,裁剪/摘要升级留给 ch07。

## Review Focus

1. 跨轮状态残留:checkpointer 下无 reducer 的标量(answer / suggested_actions / evidence)跨轮不清零,新一轮闲聊/业务会带上一轮的按钮或证据。→ Task 8 的 `test_cross_turn_no_action_leak`。
2. 意图越界/失败:classify 返回 None 或非法值必须保守归 business,不得崩、不得误归闲聊。→ Task 2 `route_by_intent` 未知回退 + Task 3 classify 失败返回 None + Task 4 `classify_intent` 节点兜底。
3. 置信度闸两端:弱证据必须落池且不进 Agent;强证据必须把 evidence 注入 main_agent 并提示不再调 query_faq。→ Task 5 retrieve 节点 + Task 4 fallback 落池 + Task 6 main_agent 证据注入。
4. ReAct 超步数:工具永不收敛时 `max_agent_steps` 强停且仍以最后 AIMessage 作答复。→ Task 2 `should_continue` + Task 6/7 多步测试。
5. 投诉/Agent 建议绝不自动写 tickets:complaint 路径只发 actions、不落 tickets;建工单仅由 `/api/actions/create-ticket` 写。→ Task 7 complaint 端到端断言 tickets 空 + Task 9 接口测试。

---

### Task 1: 配置 + 依赖 + LangGraph 红线冒烟

**Files:**
- Modify: `pyproject.toml`(加 `langgraph`、`langgraph-checkpoint-sqlite`)
- Modify: `app/config.py`(三个新配置)
- Modify: `.env.example`(三个新配置,默认值注释)
- Modify: `tests/test_config.py`(补默认值断言)
- Create: `tests/graph/__init__.py`、`tests/graph/test_smoke_langgraph.py`

**Interfaces:**
- Produces(后续任务依赖的配置值):
  - `settings.checkpointer_db_path: str = "data/checkpoints.sqlite"`
  - `settings.max_agent_steps: int = 6`
  - `settings.agent_token_budget: int = 4000`

- [ ] **Step 1: 写失败测试**

`tests/graph/test_smoke_langgraph.py`:

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

class S(TypedDict, total=False):
    x: int

def bump(state):
    return {"x": state["x"] + 1}

def test_langgraph_compiles_and_runs_with_checkpointer():
    g = StateGraph(S)
    g.add_node("bump", bump)
    g.add_edge(START, "bump")
    g.add_edge("bump", END)
    app = g.compile(checkpointer=InMemorySaver())
    out = app.invoke({"x": 1}, {"configurable": {"thread_id": "t1"}})
    assert out["x"] == 2
```

`tests/test_config.py` 补:

```python
def test_ch05_agent_defaults():
    from app.config import settings
    assert settings.checkpointer_db_path.endswith(".sqlite")
    assert settings.max_agent_steps > 0
    assert settings.agent_token_budget > 0
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/graph/test_smoke_langgraph.py -v`
Expected: FAIL(`ModuleNotFoundError: No module named 'langgraph'`)

- [ ] **Step 3: 加依赖 + 配置**

Run: `uv add langgraph langgraph-checkpoint-sqlite`

`app/config.py` 加三字段(类型/默认值见 Interfaces);`.env.example` 同步加
`CHECKPOINTER_DB_PATH` / `MAX_AGENT_STEPS` / `AGENT_TOKEN_BUDGET` 三行(默认值注释)。

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/graph/test_smoke_langgraph.py tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock app/config.py .env.example tests/graph tests/test_config.py
git commit -m "feat(ch05): 加 langgraph/checkpointer 依赖与 agent 配置 + 红线冒烟"
```

### Task 2: State 与 routing 纯函数

**Files:**
- Create: `app/graph/__init__.py`、`app/graph/state.py`、`app/graph/routing.py`
- Test: `tests/graph/test_routing.py`

**Interfaces:**
- Produces:
  - `ConversationState(TypedDict, total=False)`,字段照 spec §8(含 `fallback_reason` / `gate_decision`)。
  - `INTENT_TO_ROUTE: dict[str, str]`(七类 → knowledge/business/complaint/chitchat)。
  - `def route_by_intent(state: dict) -> str`(未知 → `"business"`)。
  - `def confidence_gate(state: dict) -> str`(`"strong"` / `"weak"`)。
  - `def should_continue(state: dict) -> str`(`"continue"` / `"stop"`;读 `settings.max_agent_steps`)。

- [ ] **Step 1: 写失败测试**

`tests/graph/test_routing.py`:

```python
import pytest
from langchain_core.messages import AIMessage
from app.graph.routing import (
    INTENT_TO_ROUTE, route_by_intent, confidence_gate, should_continue,
)

@pytest.mark.parametrize("intent,route", [
    ("商品咨询", "knowledge"), ("退款退货", "knowledge"),
    ("物流", "business"), ("订单", "business"), ("售后", "business"),
    ("投诉", "complaint"), ("闲聊", "chitchat"),
])
def test_route_by_intent(intent, route):
    assert route_by_intent({"intent": intent}) == route

def test_route_unknown_falls_back_business():
    assert route_by_intent({"intent": "不存在"}) == "business"
    assert route_by_intent({"intent": ""}) == "business"

def test_confidence_gate():
    assert confidence_gate({"evidence_strong": True}) == "strong"
    assert confidence_gate({"evidence_strong": False}) == "weak"

def test_should_continue():
    assert should_continue({"messages": [AIMessage(content="答", tool_calls=[])], "steps": 0}) == "stop"
    assert should_continue({"messages": [AIMessage(content="", tool_calls=[{"name": "query_order", "args": {}, "id": "c1"}])], "steps": 0}) == "continue"

def test_should_continue_stops_at_max_steps(monkeypatch):
    monkeypatch.setattr("app.config.settings.max_agent_steps", 2)
    state = {"messages": [AIMessage(content="", tool_calls=[{"name": "query_order", "args": {}, "id": "c1"}])], "steps": 2}
    assert should_continue(state) == "stop"

def test_conversation_state_fields():
    from app.graph.state import ConversationState
    for k in ("messages", "session_id", "conversation_id", "intent", "route",
              "resolved_query", "evidence", "citations", "evidence_strong",
              "fallback_source", "fallback_reason", "gate_decision", "answer",
              "steps", "tokens_used", "suggested_actions"):
        assert k in ConversationState.__annotations__
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/graph/test_routing.py -v`
Expected: FAIL(ImportError)

- [ ] **Step 3: 实现 `state.py` 与 `routing.py`**

`state.py`:`messages` 用 `Annotated[list[AnyMessage], add_messages]`,其余照 spec §8。
`routing.py`:映射表 + 三个纯函数,签名见 Interfaces。

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/graph/test_routing.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/graph/__init__.py app/graph/state.py app/graph/routing.py tests/graph/test_routing.py
git commit -m "feat(ch05): ConversationState + 分流/闸门/ReAct 停止条件纯函数"
```

### Task 3: 意图识别(最简)+ 标注样例 + 评估脚本

**Files:**
- Modify: `app/core/prompts.py`(加 INTENT 分类 prompt)
- Create: `app/core/intent.py`
- Create: `tests/data/intent_samples.json`、`tests/core/test_intent.py`
- Create: `scripts/eval_intent_ch05.py`(真实模型跑样例,打印分桶准确率)

**Interfaces:**
- Produces:
  - `app.core.intent.INTENTS: tuple[str, ...]`(七类)。
  - `async def classify(query: str) -> dict` → `{"intent": str | None, "confidence": float}`;
    解析失败/越界 → `{"intent": None, "confidence": 0.0}`。

- [ ] **Step 1: 写失败测试**

`tests/core/test_intent.py`:

```python
import asyncio
from pydantic import ValidationError
from app.core import intent

class FakeRunnable:
    def __init__(self, out=None, exc=None):
        self._out = out; self._exc = exc
    async def ainvoke(self, payload):
        if self._exc: raise self._exc
        return self._out

def test_classify_returns_intent(monkeypatch):
    monkeypatch.setattr(intent.llm, "structured", lambda schema:
        FakeRunnable(intent._Intent(intent="物流", confidence=0.9)))
    assert asyncio.run(intent.classify("订单1001到哪了")) == {"intent": "物流", "confidence": 0.9}

def test_classify_failure_returns_none(monkeypatch):
    monkeypatch.setattr(intent.llm, "structured", lambda schema:
        FakeRunnable(exc=ValidationError.from_exception_data("title", [])))
    assert asyncio.run(intent.classify("随便")) == {"intent": None, "confidence": 0.0}
```

`tests/data/intent_samples.json` 至少 14 条(七类各 ≥2,含口语/情绪化问法),形如
`[{"text": "订单1001到哪了", "intent": "物流"}, ...]`。

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/core/test_intent.py -v`
Expected: FAIL(ImportError)

- [ ] **Step 3: 实现 `intent.py`**

`INTENTS` 七类;`_Intent(BaseModel)` 用 `Literal[七类] + confidence`(构造器可按 `_Intent(intent=..., confidence=...)` 直接实例化);
`classify()` 用 `llm.structured(_Intent)`(method=function_calling 已在 `llm.structured` 收口),
异常捕获 → `{"intent": None, "confidence": 0.0}`。`INTENT_CLASSIFY_SYSTEM/PROMPT` 写进
`prompts.py`,只判七类、输出 JSON。

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/core/test_intent.py -v`
Expected: PASS

- [ ] **Step 5: 写评估脚本 + Commit**

`scripts/eval_intent_ch05.py`:加载 `tests/data/intent_samples.json` → 对每条 `await classify(text)` →
比对 label,打印总体准确率与每类准确率(供 Task 12 真跑)。

```bash
git add app/core/intent.py app/core/prompts.py tests/core/test_intent.py tests/data/intent_samples.json scripts/eval_intent_ch05.py
git commit -m "feat(ch05): 七类意图识别 + 标注样例 + 评估脚本"
```

### Task 4: 骨架节点(resolve/classify_intent/confidence/complaint/script/fallback/log)

**Files:**
- Create: `app/graph/nodes.py`(本任务的节点;main_agent/agent_tools/retrieve 后续任务补)
- Test: `tests/graph/test_nodes.py`(本任务的用例)

**Interfaces:**
- Consumes: `app.core.intent.classify`、`app.graph.routing.route_by_intent`、
  `app.repositories.insert_low_confidence / add_message`、`app.db.session.SessionLocal`。
- Produces(节点签名,后续任务复用):
  - `def resolve_reference(state) -> dict`(纯透传,同步)
  - `async def classify_intent(state) -> dict`(写 `intent` + `route`)
  - `async def confidence_check(state) -> dict`(写 `gate_decision`)
  - `async def complaint_reply(state) -> dict` / `script_reply(state) -> dict` / `fallback_reply(state) -> dict`
  - `async def log_node(state) -> dict`
  - 常量 `COMPLAINT_REPLY_TEXT` / `SCRIPT_REPLY_TEXT` / `FALLBACK_REPLY_TEXT`。

- [ ] **Step 1: 写失败测试**

`tests/graph/test_nodes.py` 关键断言(节点除 `resolve_reference` 外均 async,测试统一
`asyncio.run(...)`):

```python
import asyncio
from langchain_core.messages import HumanMessage
from app.graph import nodes

def test_resolve_reference_passthrough():
    state = {"messages": [HumanMessage("订单1001到哪了")]}
    assert nodes.resolve_reference(state)["resolved_query"] == "订单1001到哪了"

def test_classify_intent_node(monkeypatch):
    async def fake(q): return {"intent": "物流", "confidence": 0.9}
    monkeypatch.setattr(nodes.intent, "classify", fake)
    out = asyncio.run(nodes.classify_intent({"resolved_query": "订单1001到哪了"}))
    assert out["route"] == "business"

def test_classify_intent_failure_falls_back_business(monkeypatch):
    async def fake(q): return {"intent": None, "confidence": 0.0}
    monkeypatch.setattr(nodes.intent, "classify", fake)
    out = asyncio.run(nodes.classify_intent({"resolved_query": "x"}))
    assert out["intent"] == "" and out["route"] == "business"

def test_script_reply_returns_fixed_text():
    out = nodes.script_reply({"intent": "闲聊"})
    assert out["answer"] == nodes.SCRIPT_REPLY_TEXT and not out.get("suggested_actions")

def test_complaint_reply_two_actions():
    out = nodes.complaint_reply({"messages": [HumanMessage("我要投诉")]})
    types = [a["type"] for a in out["suggested_actions"]]
    assert types == ["transfer_human", "create_ticket"]
    assert out["suggested_actions"][1]["draft"]["ticket_type"] == "投诉"

def test_fallback_reply_writes_low_conf(monkeypatch, session_factory):
    monkeypatch.setattr(nodes, "SessionLocal", session_factory)
    state = {"conversation_id": 1, "messages": [HumanMessage("你们卖火星车吗")],
             "fallback_source": "self_check", "fallback_reason": "证据不足"}
    out = asyncio.run(nodes.fallback_reply(state))
    assert out["suggested_actions"] == [{"type": "transfer_human"}]
    # 断言 LowConfidenceQuestion 落库 1 行,source/self_check、reason/证据不足、raw_question 原文

def test_log_node_persists_assistant(monkeypatch, session_factory):
    monkeypatch.setattr(nodes, "SessionLocal", session_factory)
    asyncio.run(nodes.log_node({"conversation_id": 1, "answer": "好的"}))
    # 断言 messages 表多一条 assistant 消息
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/graph/test_nodes.py -v`
Expected: FAIL(ImportError)

- [ ] **Step 3: 实现节点**

`classify_intent`:`intent = (await classify(resolved_query or 原文))` → `state.intent = intent or ""`、
`route = route_by_intent({...})`。`fallback_reply`/`log_node` 的 DB 写用
`asyncio.to_thread` + `SessionLocal`。`confidence_check` 只写 `gate_decision`。

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/graph/test_nodes.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/graph/nodes.py tests/graph/test_nodes.py
git commit -m "feat(ch05): 指代透传/意图/闸门/投诉/闲聊/兜底/日志节点"
```

### Task 5: retrieve_knowledge 节点(复用 run_query_faq)

**Files:**
- Modify: `app/graph/nodes.py`(加 `retrieve_knowledge`)
- Test: `tests/graph/test_nodes.py`(补用例)

**Interfaces:**
- Consumes: `app.tools.business.run_query_faq(keyword, category=None) -> RagResult`。
- Produces: `async def retrieve_knowledge(state) -> dict` → 写 `evidence / citations /
  evidence_strong / fallback_source / fallback_reason`。

- [ ] **Step 1: 写失败测试**

```python
import asyncio
from langchain_core.messages import HumanMessage
from app.tools.business import RagResult
from app.graph import nodes

def test_retrieve_knowledge_sufficient(monkeypatch):
    async def fake_rag(keyword, category=None):
        return RagResult(sufficient=True, evidence="[1] 运费:满99包邮",
                         citations=[{"n": 1, "section_path": "运费政策"}])
    monkeypatch.setattr(nodes.business, "run_query_faq", fake_rag)
    out = asyncio.run(nodes.retrieve_knowledge({"messages": [HumanMessage("邮费多少")]}))
    assert out["evidence_strong"] is True and out["evidence"].startswith("[1]")

def test_retrieve_knowledge_insufficient(monkeypatch):
    async def fake_rag(keyword, category=None):
        return RagResult(sufficient=False, source="self_check", reason="证据不足")
    monkeypatch.setattr(nodes.business, "run_query_faq", fake_rag)
    out = asyncio.run(nodes.retrieve_knowledge({"messages": [HumanMessage("火星车")]}))
    assert out["evidence_strong"] is False
    assert out["fallback_source"] == "self_check" and out["fallback_reason"] == "证据不足"
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/graph/test_nodes.py::test_retrieve_knowledge_sufficient tests/graph/test_nodes.py::test_retrieve_knowledge_insufficient -v`
Expected: FAIL(AttributeError)

- [ ] **Step 3: 实现**

`retrieve_knowledge`:`rag = await business.run_query_faq(用户原文)` → 按 `rag.sufficient`
映射 State(见 Interfaces),证据弱时带上 `fallback_source` / `fallback_reason`。

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/graph/test_nodes.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/graph/nodes.py tests/graph/test_nodes.py
git commit -m "feat(ch05): retrieve_knowledge 节点复用 ch04 RAG 管线"
```

### Task 6: suggest_action 控制工具 + ReAct 节点(main_agent / agent_tools)

**Files:**
- Create: `app/tools/control.py`(`suggest_action`)
- Modify: `app/graph/nodes.py`(加 `main_agent` / `agent_tools` / `AGENT_TOOLS` / `configure_registry`)
- Test: `tests/tools/test_suggest_action.py`、`tests/graph/test_nodes.py`(补 ReAct 用例)

**Interfaces:**
- Consumes: `app.tools.business` 四查询工具、`app.tools.registry.ToolRegistry`、
  `app.core.llm.get_chat_model`。
- Produces:
  - `suggest_action`(`BaseTool`,`args_schema=SuggestActionArgs`:`action ∈
    {transfer_human, create_ticket}`,`description: str | None`)。
  - `AGENT_TOOLS: list[BaseTool]` = 四查询工具 + `suggest_action`(不含 create_ticket)。
  - `def configure_registry(registry: ToolRegistry) -> None`(注册四查询工具供 agent_tools 执行)。
  - `async def main_agent(state, config=None) -> dict`(写 messages/steps/tokens_used)。
  - `async def agent_tools(state) -> dict`(写 messages,可能追加 suggested_actions)。

- [ ] **Step 1: 写失败测试**

`tests/tools/test_suggest_action.py`:

```python
def test_suggest_action_tool_shape():
    assert control.suggest_action.name == "suggest_action"
    assert control.SuggestActionArgs.model_validate({"action": "transfer_human"}).action == "transfer_human"
```

`tests/graph/test_nodes.py` 补:

```python
import asyncio
from langchain_core.messages import AIMessage, HumanMessage

class FakeChat:
    def __init__(self, ai): self._ai = ai
    def bind_tools(self, tools): return self
    async def ainvoke(self, messages, config=None): return self._ai

def test_main_agent_converges(monkeypatch):
    monkeypatch.setattr(nodes, "get_chat_model", lambda streaming=True: FakeChat(AIMessage(content="好的", tool_calls=[])))
    out = asyncio.run(nodes.main_agent({"messages": [HumanMessage("在吗")], "steps": 0, "tokens_used": 0}))
    assert out["steps"] == 1 and out["messages"][-1].content == "好的"

def test_main_agent_injects_evidence(monkeypatch):
    seen = {}
    class Cap(FakeChat):
        async def ainvoke(self, messages, config=None):
            seen["msgs"] = messages
            return AIMessage(content="依据[1]", tool_calls=[])
    monkeypatch.setattr(nodes, "get_chat_model", lambda streaming=True: Cap(None))
    asyncio.run(nodes.main_agent({"messages": [HumanMessage("邮费")], "steps": 0, "tokens_used": 0,
                      "route": "knowledge", "evidence_strong": True, "evidence": "[1] 满99包邮"}))
    assert any("满99包邮" in str(m.content) for m in seen["msgs"])

def test_agent_tools_suggest_action_records(monkeypatch):
    last = AIMessage(content="", tool_calls=[{"name": "suggest_action",
                       "args": {"action": "create_ticket", "description": "要退货"}, "id": "c1"}])
    out = asyncio.run(nodes.agent_tools({"messages": [last], "suggested_actions": []}))
    assert out["suggested_actions"] == [{"type": "create_ticket",
        "draft": {"description": "要退货", "ticket_type": "其他"}}]
    assert out["messages"][-1].content == "已记录建议"
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/tools/test_suggest_action.py tests/graph/test_nodes.py -v`
Expected: FAIL(ImportError / AttributeError)

- [ ] **Step 3: 实现**

`control.py`:`@tool(args_schema=SuggestActionArgs)` 的 `suggest_action`,docstring 说明
「只产出建议动作、后端不执行」。`nodes.py`:`main_agent` 按 spec §7.6 组装消息
(知识路强证据注入 EVIDENCE_HINT),`get_chat_model(streaming=True).bind_tools(AGENT_TOOLS)` →
`ainvoke` → 累加 steps/tokens(`usage_metadata.total_tokens`,缺省 0)。`agent_tools` 遍历
tool_calls:`suggest_action` 走建议记录(create_ticket 补 ticket_type="其他"),其余走
`registry.execute`(通过 `configure_registry` 注入的实例)。

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/tools/test_suggest_action.py tests/graph/test_nodes.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/tools/control.py app/graph/nodes.py tests/tools/test_suggest_action.py tests/graph/test_nodes.py
git commit -m "feat(ch05): suggest_action 控制工具 + main_agent/agent_tools ReAct 节点"
```

### Task 7: build.py 组装图 + 图结构/端到端测试

**Files:**
- Create: `app/graph/build.py`
- Test: `tests/graph/test_build.py`

**Interfaces:**
- Produces: `def build_graph(checkpointer=None)` → 编译后的图(compile() 返回值)。
- 图结构固定照 spec §6(10 节点 + 条件边),`build_graph(InMemorySaver())` 供测试/演示。

- [ ] **Step 1: 写失败测试**

`tests/graph/test_build.py` 关键断言。图内 `log_node` 会写 DB,故测试统一
`monkeypatch.setattr(nodes, "SessionLocal", session_factory)`;输入状态固定为
`{"messages":[HumanMessage(...)], "conversation_id":1, "steps":0, "tokens_used":0,
"suggested_actions":[], "answer":"", "resolved_query":""}`。

```python
import asyncio
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select
from app.db.models import Ticket
from app.graph import nodes
from app.graph.build import build_graph

def _input(text):
    return {"messages": [HumanMessage(text)], "conversation_id": 1, "steps": 0,
            "tokens_used": 0, "suggested_actions": [], "answer": "", "resolved_query": ""}

def test_chitchat_end_to_end(monkeypatch, session_factory):
    async def fake(q): return {"intent": "闲聊", "confidence": 0.9}
    monkeypatch.setattr(nodes.intent, "classify", fake)
    monkeypatch.setattr(nodes, "SessionLocal", session_factory)
    out = asyncio.run(build_graph(InMemorySaver()).ainvoke(
        _input("你好"), {"configurable": {"thread_id": "s1"}}))
    assert out["route"] == "chitchat" and out["answer"] == nodes.SCRIPT_REPLY_TEXT

def test_complaint_no_ticket_written(monkeypatch, session_factory):
    async def fake(q): return {"intent": "投诉", "confidence": 0.9}
    monkeypatch.setattr(nodes.intent, "classify", fake)
    monkeypatch.setattr(nodes, "SessionLocal", session_factory)
    out = asyncio.run(build_graph(InMemorySaver()).ainvoke(
        _input("我要投诉"), {"configurable": {"thread_id": "s2"}}))
    types = [a["type"] for a in out["suggested_actions"]]
    assert types == ["transfer_human", "create_ticket"]
    with session_factory() as s:
        assert s.scalars(select(Ticket)).all() == []  # 未自动写单
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/graph/test_build.py -v`
Expected: FAIL(ImportError)

- [ ] **Step 3: 实现 `build.py`**

照 spec §6 接线:START→resolve_reference→classify_intent;条件边
`classify_intent`(route_by_intent)→ knowledge/retrieve_knowledge、business/main_agent、
complaint/complaint_reply、chitchat/script_reply;retrieve_knowledge→confidence_check;
条件边 `confidence_check`(confidence_gate)→ strong/main_agent、weak/fallback_reply;
条件边 `main_agent`(should_continue)→ continue/agent_tools、stop/log;agent_tools→main_agent;
complaint_reply/script_reply/fallback_reply→log→END。

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/graph/test_build.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/graph/build.py tests/graph/test_build.py
git commit -m "feat(ch05): 组装七节点图 + 端到端闲聊/投诉测试"
```

### Task 8: runtime 流式 + /api/chat 改走 graph + 删 ChatService

**Files:**
- Create: `app/graph/runtime.py`
- Modify: `app/api/chat.py`、`app/main.py`(lifespan 调 init_graph/close_graph)
- Delete: `app/services/chat.py`、`tests/test_chat_service.py`、`tests/test_chat_citations.py`
- Modify: `tests/test_chat_api.py`(重写为 graph 版)
- Test: `tests/graph/test_runtime.py`

**Interfaces:**
- Produces:
  - `async def init_graph() -> None` / `async def close_graph() -> None` / `def get_graph()`。
  - `async def stream_turn(session_id: str, user: str | None, message: str) -> AsyncIterator[dict]`
    (帧:`delta` / `{"status":"tool_call",...}` / `{"event":"citations",...}` / `{"event":"actions",...}`)。
  - 内部 `async def _stream_graph(graph, input_state, thread_id) -> AsyncIterator[dict]`(供测试注入 InMemorySaver)。

- [ ] **Step 1: 写失败测试**

`tests/graph/test_runtime.py` 关键断言。约定同 Task 7:`nodes.SessionLocal` 打 `session_factory`,
输入状态用 Task 7 的 `_input(text)` 形状;`collect(ag)` = `[f async for f in ag]`;
monkeypatch `nodes.intent.classify`(定意图)、`nodes.get_chat_model`(FakeChat)、
`nodes.business.run_query_faq`(假 RAG)、`nodes._registry`(假 ToolRegistry)。

```python
def test_stream_chitchat_delta(monkeypatch, session_factory):
    monkeypatch.setattr(nodes, "SessionLocal", session_factory)
    async def fake(q): return {"intent": "闲聊", "confidence": 0.9}
    monkeypatch.setattr(nodes.intent, "classify", fake)
    frames = asyncio.run(collect(_stream_graph(build_graph(InMemorySaver()), _input("你好"), "s1")))
    assert "".join(f["delta"] for f in frames if "delta" in f) == nodes.SCRIPT_REPLY_TEXT

def test_stream_business_tool_and_delta(monkeypatch, session_factory):
    # classify→物流;FakeChat 第一次 ainvoke 返回 AIMessage(tool_calls=[query_logistics]),
    # 第二次返回 AIMessage(content="订单在路上");假 registry.execute 返回 "物流中"
    frames = asyncio.run(collect(_stream_graph(build_graph(InMemorySaver()), _input("订单1001到哪了"), "s2")))
    assert any(f.get("status") == "tool_call" and f["tool"] == "query_logistics" for f in frames)
    assert "".join(f["delta"] for f in frames if "delta" in f) == "订单在路上"

def test_stream_actions_event(monkeypatch, session_factory):
    # classify→投诉
    frames = asyncio.run(collect(_stream_graph(build_graph(InMemorySaver()), _input("我要投诉"), "s3")))
    act = [f for f in frames if f.get("event") == "actions"]
    assert act and {a["type"] for a in act[-1]["items"]} == {"transfer_human", "create_ticket"}

def test_cross_turn_no_action_leak(monkeypatch, session_factory):
    # 同一 thread_id:第 1 轮投诉(有 actions),第 2 轮闲聊(无 actions)
    app = build_graph(InMemorySaver())
    f1 = asyncio.run(collect(_stream_graph(app, _input("我要投诉"), "s4")))
    f2 = asyncio.run(collect(_stream_graph(app, _input("你好"), "s4")))
    assert any(f.get("event") == "actions" for f in f1)
    assert not any(f.get("event") == "actions" for f in f2)
```

`tests/test_chat_api.py` 重写:用 `InMemorySaver` 编译图 + monkeypatch 模型,走 TestClient
POST `/api/chat`,断言 `content-type: text/event-stream`、`[DONE]` 收尾、delta 拼接、tool 帧。

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/graph/test_runtime.py tests/test_chat_api.py -v`
Expected: FAIL(ImportError)

- [ ] **Step 3: 实现 runtime + 改 /api/chat + 删 ChatService**

`runtime.py`:`init_graph` 开 `AsyncSqliteSaver.from_conn_string(settings.checkpointer_db_path)` +
`setup()`,`build_graph(checkpointer=...)`,`configure_registry(四查询工具)`;`stream_turn` 做
`get_or_create_conversation` → 落 user 消息 → `_stream_graph(get_graph(), 输入, thread_id=session_id)`。
`_stream_graph` 用 `graph.astream(input, config, stream_mode=["messages","updates"])`:
`messages` 模式按 `langgraph_node == "main_agent"` 吐 `delta`;`updates` 模式吐确定性节点 answer
(delta)、citations、tool、suggested_actions(末尾去重吐一次 actions)。若 token 级细粒度不达预期,
按 spec §9 spike 结论退化为整块 delta。`/api/chat` 改调 `stream_turn`,保留 `[DONE]` 与 error 帧;
`main.py` 加 lifespan 调 init/close_graph。删除 `app/services/chat.py` 及两个旧测试。

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/graph/test_runtime.py tests/test_chat_api.py -v`
Expected: PASS(再全量 `uv run pytest -q` 确认无残留 import ChatService)

- [ ] **Step 5: Commit**

```bash
git add app/graph/runtime.py app/api/chat.py app/main.py tests/graph/test_runtime.py tests/test_chat_api.py
git rm app/services/chat.py tests/test_chat_service.py tests/test_chat_citations.py
git commit -m "feat(ch05): 图流式 runtime + /api/chat 改走 graph + 移除 ChatService"
```

### Task 9: 建工单接口(/api/actions/create-ticket)

**Files:**
- Create: `app/schemas/actions.py`、`app/api/actions.py`
- Modify: `app/main.py`(include actions router)
- Test: `tests/api/test_actions.py`

**Interfaces:**
- Produces:
  - `class CreateTicketRequest(BaseModel): session_id: str; description: str; ticket_type: str`
  - `class CreateTicketResponse(BaseModel): ticket_no: str; status: str = "open"`
  - `POST /api/actions/create-ticket` → 200,`CreateTicketResponse`(复用 `repositories.create_ticket`)。

- [ ] **Step 1: 写失败测试**

`tests/api/test_actions.py`(注入 session_factory 依赖):

```python
def test_create_ticket_writes_row(session_factory):
    # POST {session_id:"s1", description:"退货运费", ticket_type:"投诉"}
    # 断言 200、resp.ticket_no 前缀 TKT-;tickets 表 1 行,description/ticket_type 正确

def test_create_ticket_requires_fields(session_factory):
    # 缺 description → 422
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/api/test_actions.py -v`
Expected: FAIL(ImportError)

- [ ] **Step 3: 实现**

`api/actions.py` 用 `get_or_create_conversation(session_id)` 取 conversation_id →
`create_ticket(conversation_id, description, ticket_type)` → 返回 ticket_no。依赖注入用
`session_factory`(参照 `api/chat.py` 现有写法,让测试能 override)。

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/api/test_actions.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/schemas/actions.py app/api/actions.py app/main.py tests/api/test_actions.py
git commit -m "feat(ch05): 建工单接口(前端点「建工单」才写 tickets)"
```

### Task 10: 前端两个按钮 + actions 帧(Vibe Coding,无 TDD)

**Files:**
- Modify: `app/static/index.html`

**Interfaces:**
- Consumes:`/api/chat` 的 `{"event":"actions","items":[...]}` 帧、`POST /api/actions/create-ticket`。

- [ ] **Step 1: 在 `handleSSEBlock` 增加 `payload.event === "actions"` 分支**,把
  `acc.actions = payload.items` 存到本轮。
- [ ] **Step 2: `send()` 流结束后**,若 `acc.actions` 非空,在气泡下渲染按钮组:
  每个 action 一个独立按钮;`transfer_human` 文案「转人工」,`create_ticket` 文案「建工单」。
- [ ] **Step 3: 转人工按钮**:纯前端,追加「已转接人工客服」气泡 + 「您好,我是客服小猫,
  请问有什么可以帮您的」气泡,不调任何后端接口。
- [ ] **Step 4: 建工单按钮**:确认后 `fetch("/api/actions/create-ticket",
  {session_id, description, ticket_type})`(draft 来自 action),成功展示工单号,失败展示错误。
  两按钮互不绑定;用户都不点、继续发消息,不产生任何动作。
- [ ] **Step 5: Commit**

```bash
git add app/static/index.html
git commit -m "feat(ch05): 聊天页转人工/建工单独立按钮(Vibe Coding)"
```

### Task 11: 祛魅热身脚本 bare_agent.py

**Files:**
- Create: `scripts/__init__.py`、`scripts/bare_agent.py`
- Test: `tests/graph/test_bare_agent.py`
- Modify: `Makefile`(加 `bare-agent`)

**Interfaces:**
- Produces:
  - `async def run_loop(model, tools, messages, max_steps: int) -> str`(零 LangGraph 裸循环)。
  - `scripts/bare_agent.py` 的 `main()`:读取问题(参数或固定演示),调 DeepSeek + 四查询工具,
    打印每一步 tool_calls 与最终答案。

- [ ] **Step 1: 写失败测试**

`tests/graph/test_bare_agent.py`(FakeModel 先 tool_calls 后 content):

```python
def test_bare_loop_multi_step():
    # FakeModel 第 1 次返回 tool_calls=[query_order],第 2 次返回 content="订单在路上"
    out = asyncio.run(run_loop(fake_model, {查询工具}, [HumanMessage("订单1001到哪了")], 6))
    assert out == "订单在路上" and 调用次数 == 2
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/graph/test_bare_agent.py -v`
Expected: FAIL(ImportError)

- [ ] **Step 3: 实现**

`run_loop` 按 spec §10 伪代码;`main()` 用 `get_chat_model` + ch02 四查询工具
(ToolRegistry 或直接 dict),演示「先查订单、再查物流」样例。Makefile 加
`bare-agent: uv run python scripts/bare_agent.py`。

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/graph/test_bare_agent.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/__init__.py scripts/bare_agent.py tests/graph/test_bare_agent.py Makefile
git commit -m "feat(ch05): 零框架裸 Agent 循环祛魅脚本 + make bare-agent"
```

### Task 12: 意图评估真跑 + 端到端验收 + 文档收尾

**Files:**
- Modify: `README.md`、`Makefile`(加 `eval-intent`)、`.env.example`(核对)

- [ ] **Step 1: 意图评估集真跑**(用户机器,真实 DeepSeek)

Run: `uv run python scripts/eval_intent_ch05.py`
Expected: 打印总体准确率与七类分桶准确率;每类不低于合理基线(记录实际数字)。

- [ ] **Step 2: 祛魅脚本演示**

Run: `make bare-agent`
Expected: 看到「Agent = 带工具的 while 循环」,多步样例走了 ≥2 步 tool_calls。

- [ ] **Step 3: 端到端验收(起服务后逐条过)**

1. 政策类问题 → 日志见 `retrieve_knowledge` 被走到;
2. 「订单 1001 的物流到哪了」→ Agent 自调 `query_logistics`;
3. 「我要投诉」→ 前端出现「转人工」「建工单」两按钮;转人工纯前端;建工单写 tickets;
   都不点继续对话;
4. 闲聊 → 固定话术、无模型调用;
5. 先查订单再查物流的复杂问题 → ReAct 走 ≥2 步。

- [ ] **Step 4: README/Makefile 收尾 + 全量测试**

`Makefile` 加 `eval-intent: uv run python scripts/eval_intent_ch05.py`;README 补 Ch05
使用/演示/验收段。Run: `uv run pytest -q`
Expected: 全量 PASS(记录条数)。

- [ ] **Step 5: Commit**

```bash
git add README.md Makefile .env.example
git commit -m "docs(ch05): README/验收/评估收尾"
```
