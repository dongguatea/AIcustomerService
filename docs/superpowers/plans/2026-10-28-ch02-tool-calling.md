# MewHelp Ch02 工具调用客服 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 ch01 纯对话客服装上单轮 Function Calling:模型决定调哪个工具 → 执行 → 回灌收敛,最终回答流式吐出;聊天记录(含工具调用与结果)落 MySQL,聊天页显示工具徽章。

**Architecture:** FastAPI 单进程(:8000),分层 `api → service → tools → repositories → db`。决策用 `ChatOpenAI.bind_tools([...])` 非流式 `ainvoke` 拿 `tool_calls`,命中工具推一条 `status=tool_call` 状态帧后经 `ToolRegistry` 执行(参数校验 + 超时重试),`ToolMessage` 回灌,再用流式模型逐 token 输出最终回答。SQLAlchemy 2.0 同步 ORM + PyMySQL,异步工具内 `asyncio.to_thread` 包同步 DB 访问。

**Tech Stack:** Python 3.12, uv, FastAPI, SQLAlchemy 2.0 + PyMySQL + cryptography, MySQL 8(Docker), LangChain `@tool` + `bind_tools` + `ToolMessage`, pytest。

**Spec:** `docs/superpowers/specs/2026-10-26-ch02-tool-calling-design.md`

## Global Constraints

- 技术选型定死:FastAPI + SQLAlchemy + MySQL(Docker 起)+ LangChain `@tool`;实现走不通就停下问,不换方案。
- 只做单轮:模型调一次工具即收敛;工具执行段先推一条 `status=tool_call` 状态帧。
- 应用侧沿用 OpenAI 协议直连上游,`CHAT_*` 配置不变;默认 DeepSeek `deepseek-flash`,tool calling 复用 ch01 的 `CHAT_THINKING=disabled`。
- SSE 帧契约:`data: {"delta":"..."}` / `data: {"status":"tool_call","tool":...,"tool_call_id":...,"args":...}` / `data: [DONE]` / `event: error` + `data: {"message":...}`。
- 单测不打真实模型(FakeModel/stub)、不打真实 MySQL(用 SQLite `StaticPool` 内存库);真实模型与真实 MySQL 只用于验收。
- 每个任务完成即 commit。
- `dev-notes/ch02.md` 由主会话按阶段追记,不属于任何任务的文件清单。
- 本机 Windows,`bash` 是 WSL stub;dev/test/eval 走 Makefile(纯 `uv run`)+ PowerShell/浏览器演示。

## Review Focus

1. 工具执行超时或抛异常 → 不回抛,重试后转成错误文案交给模型,让模型向用户解释。(Task 4)
2. 工具参数非法 → 参数 Schema 校验失败立即返回错误文案,不重试、不崩。(Task 4)
3. FAQ 关键词漏召回(如「邮费」)→ `query_faq` 返回"未找到",模型不编造。(Task 5 / Task 8)
4. 未调工具的普通问答 → 仍逐帧流式输出并落库。(Task 6 / Task 7)
5. `create_ticket` 在无会话上下文(ContextVar 为空)时 → 返回明确错误而非崩。(Task 5)

---

### Task 1: 依赖与配置扩展

**Files:**
- Modify: `pyproject.toml`
- Modify: `.env.example`
- Modify: `app/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `app.config.Settings` 新增 `database_url: str`(默认 `mysql+pymysql://mewhelp:mewhelp@127.0.0.1:3306/mewhelp`)、`tool_timeout: float = 10.0`、`tool_max_retries: int = 2`;模块级单例 `settings` 不变。后续任务 `from app.config import settings` 或 `from app.db.session import ...`。

- [ ] **Step 1: 加依赖**

Run: `uv add sqlalchemy pymysql cryptography`
Expected: `pyproject.toml` dependencies 新增 `sqlalchemy>=2.0`、`pymysql`、`cryptography`;`uv.lock` 更新。(PyMySQL 连 MySQL 8 的 caching_sha2_password 需要 cryptography。)

- [ ] **Step 2: 写失败测试**

`tests/test_config.py` 追加:

```python
def test_db_and_tool_settings_have_defaults(monkeypatch):
    for k, v in {"CHAT_MODEL": "m", "CHAT_BASE_URL": "u", "CHAT_API_KEY": "k"}.items():
        monkeypatch.setenv(k, v)
    s = Settings(_env_file=None)
    assert s.database_url.startswith("mysql+pymysql://")
    assert s.tool_timeout == 10.0
    assert s.tool_max_retries == 2


def test_tool_settings_env_override(monkeypatch):
    for k, v in {"CHAT_MODEL": "m", "CHAT_BASE_URL": "u", "CHAT_API_KEY": "k"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("TOOL_TIMEOUT", "3.5")
    monkeypatch.setenv("TOOL_MAX_RETRIES", "5")
    s = Settings(_env_file=None)
    assert s.tool_timeout == 3.5
    assert s.tool_max_retries == 5
```

- [ ] **Step 3: 跑测试确认失败**

Run: `uv run pytest tests/test_config.py -v`
Expected: 两条新增测试 FAIL(`AttributeError: 'Settings' object has no attribute 'database_url'`)。

- [ ] **Step 4: 实现 `app/config.py`**

在 `Settings` 加三个字段,`database_url` 用默认值(对齐 docker-compose),`tool_timeout`/`tool_max_retries` 给默认;不写进 `.env` 也能跑(默认值即本地 MySQL)。

- [ ] **Step 5: 更新 `.env.example`**

追加:

```bash
# 本地 MySQL(默认值已对齐 docker-compose,可留空)
DATABASE_URL=mysql+pymysql://mewhelp:mewhelp@127.0.0.1:3306/mewhelp
TOOL_TIMEOUT=10.0
TOOL_MAX_RETRIES=2
```

- [ ] **Step 6: 跑测试确认通过**

Run: `uv run pytest tests/test_config.py -v`
Expected: 4 passed(原 3 条 + 新增 2 条,注意原文件当前是 3 条)。

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .env.example app/config.py tests/test_config.py
git commit -m "feat(ch02): 数据库与工具配置扩展"
```

---

### Task 2: 数据库骨架与四表模型

**Files:**
- Create: `app/db/__init__.py`, `app/db/base.py`, `app/db/session.py`, `app/db/models.py`
- Test: `tests/conftest.py`, `tests/test_db_models.py`

**Interfaces:**
- Consumes: `app.config.settings.database_url`。
- Produces:
  - `app.db.base.Base`(`DeclarativeBase`)。
  - `app.db.session.engine`、`SessionLocal`(`sessionmaker`)、`init_db()`(调 `Base.metadata.create_all(engine)`)。
  - `app.db.models.Faq / Conversation / Message / Ticket` 四映射类,字段如下:
    - `Faq`: `id:int PK`, `question:str(Text)`, `answer:str(Text)`, `category:str(String(50))`。
    - `Conversation`: `id:int PK`, `session_id:str(String(128), unique)`, `user:str(String(128))`, `status:str(String(32), default "open")`, `created_at:datetime`。
    - `Message`: `id:int PK`, `conversation_id:int FK→conversations.id`, `role:str(String(16))`, `content:str(Text)`, `tool_call_request:dict|None(JSON)`, `tool_call_id:str|None(String(128))`, `created_at:datetime`。
    - `Ticket`: `ticket_no:str(String(64), PK)`, `conversation_id:int FK`, `description:str(Text)`, `ticket_type:str(String(32))`, `status:str(String(32), default "open")`, `created_at:datetime`。

- [ ] **Step 1: 写失败测试**

`tests/conftest.py`(各 DB 测试共用 SQLite 内存库夹具):

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base


@pytest.fixture()
def session_factory():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    Base.metadata.drop_all(engine)
```

`tests/test_db_models.py`:

```python
from datetime import datetime

from app.db.models import Conversation, Faq, Message, Ticket


def test_models_create_tables_and_rows(session_factory):
    with session_factory() as s:
        conv = Conversation(session_id="s1", user="王小明")
        s.add(conv)
        s.flush()
        s.add(Faq(question="退货政策是什么?", answer="7 天无理由", category="售后"))
        s.add(Message(conversation_id=conv.id, role="user", content="你好"))
        s.add(Ticket(ticket_no="TKT-1", conversation_id=conv.id, description="x", ticket_type="投诉"))
        s.commit()
    with session_factory() as s:
        assert s.query(Conversation).count() == 1
        assert s.query(Faq).count() == 1
        assert s.query(Message).count() == 1
        assert s.query(Ticket).count() == 1


def test_message_tool_fields_roundtrip(session_factory):
    with session_factory() as s:
        conv = Conversation(session_id="s2", user="u")
        s.add(conv)
        s.flush()
        s.add(Message(
            conversation_id=conv.id, role="assistant", content="query_logistics",
            tool_call_request={"name": "query_logistics", "args": {"order_id": "1001"}},
            tool_call_id="call_1",
        ))
        s.commit()
    with session_factory() as s:
        m = s.query(Message).one()
        assert m.tool_call_request["name"] == "query_logistics"
        assert m.tool_call_id == "call_1"
        assert isinstance(m.created_at, datetime)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_db_models.py -v`
Expected: FAIL(`ModuleNotFoundError: No module named 'app.db'`)。

- [ ] **Step 3: 实现 `app/db/base.py` / `session.py` / `models.py`**

`base.py` 放 `class Base(DeclarativeBase): pass`;`session.py` 用
`create_engine(settings.database_url, pool_pre_ping=True)` + `sessionmaker`;
`models.py` 用 SQLAlchemy 2.0 `Mapped[...]` + `mapped_column` 声明四表,
`created_at` 用 `default=lambda: datetime.now(timezone.utc)`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_db_models.py -v`
Expected: 2 passed。

- [ ] **Step 5: Commit**

```bash
git add app/db tests/conftest.py tests/test_db_models.py
git commit -m "feat(ch02): 数据库骨架与四表模型"
```

---

### Task 3: 仓储层

**Files:**
- Create: `app/repositories.py`
- Test: `tests/test_repositories.py`

**Interfaces:**
- Consumes: `app.db.models`。
- Produces(均以 `Session` 为第一参数,便于注入 SQLite 会话):
  - `get_or_create_conversation(session, session_id: str, user: str) -> Conversation`
  - `add_message(session, conversation_id: int, role: str, content: str, tool_call_request: dict | None = None, tool_call_id: str | None = None) -> Message`
  - `list_messages(session, conversation_id: int) -> list[Message]`(按 `id` 升序)
  - `search_faq(session, keyword: str) -> list[Faq]`(`Faq.question.like(f"%{keyword}%")`)
  - `create_ticket(session, conversation_id: int, description: str, ticket_type: str) -> Ticket`(ticket_no 形如 `TKT-<epoch毫秒>-<6位大写随机>`)

- [ ] **Step 1: 写失败测试**

`tests/test_repositories.py`:

```python
from app.repositories import (
    add_message, create_ticket, get_or_create_conversation,
    list_messages, search_faq,
)
from app.db.models import Faq


def test_get_or_create_conversation_reuses(session_factory):
    with session_factory() as s:
        a = get_or_create_conversation(s, "s1", "u")
        b = get_or_create_conversation(s, "s1", "u")
        assert a.id == b.id


def test_search_faq_like_hit_and_miss(session_factory):
    with session_factory() as s:
        s.add(Faq(question="退货政策是什么?", answer="7 天无理由", category="售后"))
        s.add(Faq(question="发货时效", answer="48 小时内", category="物流"))
        s.commit()
        assert len(search_faq(s, "退货")) == 1
        assert search_faq(s, "邮费") == []


def test_message_flow_and_ticket(session_factory):
    with session_factory() as s:
        conv = get_or_create_conversation(s, "s1", "u")
        add_message(s, conv.id, "user", "订单 1001 物流")
        add_message(s, conv.id, "assistant", "好的", {"name": "query_logistics", "args": {"order_id": "1001"}}, "call_1")
        add_message(s, conv.id, "tool", "已签收", None, "call_1")
        t = create_ticket(s, conv.id, "快递员态度差", "投诉")
        msgs = list_messages(s, conv.id)
        assert [m.role for m in msgs] == ["user", "assistant", "tool"]
        assert t.ticket_no.startswith("TKT-")
        assert t.status == "open"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_repositories.py -v`
Expected: FAIL(`ModuleNotFoundError: No module named 'app.repositories'`)。

- [ ] **Step 3: 实现 `app/repositories.py`**

按 Interfaces 实现;`get_or_create_conversation` 内 `session.scalar(select(...))`,
未命中则 `session.add(Conversation(...)); session.commit(); session.refresh(...)`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_repositories.py -v`
Expected: 3 passed。

- [ ] **Step 5: Commit**

```bash
git add app/repositories.py tests/test_repositories.py
git commit -m "feat(ch02): 会话/消息/FAQ/工单仓储"
```

---

### Task 4: 工具基础设施 ToolRegistry

**Files:**
- Create: `app/tools/__init__.py`, `app/tools/registry.py`
- Test: `tests/test_tools_registry.py`

**Interfaces:**
- Consumes: `langchain_core.tools.BaseTool`。
- Produces: `class ToolRegistry`:
  - `register(tool: BaseTool) -> None`
  - `get(name: str) -> BaseTool`(未注册抛 `KeyError`)
  - `names() -> list[str]`
  - `as_list() -> list[BaseTool]`
  - `async execute(name: str, args: dict, *, timeout: float, max_retries: int) -> str`:
    1. `name` 不存在返回错误文案;
    2. 用 `tool.args_schema.model_validate(args)` 校验,`ValidationError` 立即返回
       `{"error":"参数校验失败","detail":...}`(不重试);
    3. `asyncio.timeout(timeout)` 包 `tool.ainvoke(args)`,异常/超时按
       `max_retries` 重试,全部失败返回 `{"error":"工具执行失败","detail":...}`;
    4. 成功结果统一转 `str` 返回。

- [ ] **Step 1: 写失败测试**

`tests/test_tools_registry.py`(用 `@tool` 造假工具):

```python
import asyncio

from langchain_core.tools import tool

from app.tools.registry import ToolRegistry


@tool
def ok(x: str) -> str:
    """ok 工具。"""
    return f"ok:{x}"


@tool
def flaky(x: str) -> str:
    """前两次失败第三次成功。"""
    flaky.calls = getattr(flaky, "calls", 0) + 1
    if flaky.calls < 3:
        raise RuntimeError("boom")
    return "recovered"


@tool
def slow(x: str) -> str:
    """慢工具。"""
    asyncio.run(asyncio.sleep(0.2))
    return "late"


def make_registry():
    r = ToolRegistry()
    for t in (ok, flaky, slow):
        r.register(t)
    return r


def test_register_get_names_and_list():
    r = make_registry()
    assert set(r.names()) == {"ok", "flaky", "slow"}
    assert r.get("ok") is ok
    assert len(r.as_list()) == 3


def test_execute_validates_args_and_returns_error_text():
    r = make_registry()
    assert "参数校验失败" in asyncio.run(r.execute("ok", {"x": 1}, timeout=1, max_retries=0))


def test_execute_retries_then_succeeds():
    r = make_registry()
    assert asyncio.run(r.execute("flaky", {"x": "a"}, timeout=1, max_retries=2)) == "recovered"


def test_execute_timeout_returns_error_text():
    r = make_registry()
    assert "工具执行失败" in asyncio.run(r.execute("slow", {"x": "a"}, timeout=0.01, max_retries=0))


def test_execute_unknown_tool_returns_error_text():
    r = make_registry()
    assert "工具不存在" in asyncio.run(r.execute("nope", {}, timeout=1, max_retries=0))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_tools_registry.py -v`
Expected: FAIL(`ModuleNotFoundError: No module named 'app.tools.registry'`)。

- [ ] **Step 3: 实现 `app/tools/registry.py`**

按 Interfaces 实现。`execute` 内统一 `await tool.ainvoke(args)` 即可: LangChain 的
`@tool` 对同步函数生成的 `StructuredTool.ainvoke` 会在线程池执行,不会阻塞事件循环,
registry 无需自己再包 `to_thread`;异步工具的 `ainvoke` 直接 await。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_tools_registry.py -v`
Expected: 5 passed。

- [ ] **Step 5: Commit**

```bash
git add app/tools tests/test_tools_registry.py
git commit -m "feat(ch02): 工具注册/校验/超时重试基础设施"
```

---

### Task 5: 五个业务工具

**Files:**
- Create: `app/tools/business.py`
- Test: `tests/test_tools_business.py`

**Interfaces:**
- Consumes: `app.repositories.search_faq / create_ticket`, `app.db.session.SessionLocal`。
- Produces:
  - `current_conversation_id: ContextVar[int | None]`(默认 None)。
  - 五个 `@tool`(async def,`args_schema` 用 Pydantic):`query_order(order_id: str)`、
    `query_product(keyword: str)`、`query_logistics(order_id: str)`(三者随机生成数据返回
    `str`,不建表、不接真实接口)、`query_faq(keyword: str)`(SQL LIKE 查 faq,命中拼文本,
    未命中返回"未找到相关问题")、`create_ticket(description: str, ticket_type: str)`
    (读 `current_conversation_id`,写入 tickets,返回工单号)。
  - `TOOLS: list[BaseTool]`(五个工具的注册序)。

- [ ] **Step 1: 写失败测试**

`tests/test_tools_business.py`(monkeypatch `app.tools.business.SessionLocal` 为 SQLite 工厂):

```python
import asyncio

from app.db.models import Faq, Ticket
from app.tools import business


def test_query_faq_hit_and_miss(monkeypatch, session_factory):
    monkeypatch.setattr(business, "SessionLocal", session_factory)
    with session_factory() as s:
        s.add(Faq(question="退货政策是什么?", answer="7 天无理由", category="售后"))
        s.commit()
    assert "7 天无理由" in asyncio.run(business.query_faq.ainvoke({"keyword": "退货"}))
    assert "未找到" in asyncio.run(business.query_faq.ainvoke({"keyword": "邮费"}))


def test_mock_tools_echo_input(monkeypatch):
    order = asyncio.run(business.query_order.ainvoke({"order_id": "1001"}))
    assert "1001" in order and "订单" in order
    prod = asyncio.run(business.query_product.ainvoke({"keyword": "猫粮"}))
    assert "猫粮" in prod
    logi = asyncio.run(business.query_logistics.ainvoke({"order_id": "1001"}))
    assert "1001" in logi and "物流" in logi


def test_create_ticket_writes_and_uses_context(monkeypatch, session_factory):
    monkeypatch.setattr(business, "SessionLocal", session_factory)
    with session_factory() as s:
        from app.repositories import get_or_create_conversation
        conv = get_or_create_conversation(s, "s1", "u")
        token = business.current_conversation_id.set(conv.id)
        out = asyncio.run(business.create_ticket.ainvoke({"description": "快递丢了", "ticket_type": "投诉"}))
        assert out.startswith("TKT-")
        t = s.query(Ticket).one()
        assert t.conversation_id == conv.id
        business.current_conversation_id.reset(token)


def test_create_ticket_without_context_returns_error(monkeypatch, session_factory):
    monkeypatch.setattr(business, "SessionLocal", session_factory)
    out = asyncio.run(business.create_ticket.ainvoke({"description": "x", "ticket_type": "投诉"}))
    assert "失败" in out
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_tools_business.py -v`
Expected: FAIL(`ModuleNotFoundError: No module named 'app.tools.business'`)。

- [ ] **Step 3: 实现 `app/tools/business.py`**

三个 mock 工具用 `random` + `await asyncio.sleep(random.uniform(0.02, 0.1))` 模拟
接口延迟;`query_faq`/`create_ticket` 用 `asyncio.to_thread` 包同步仓储访问;
`create_ticket` 在 `current_conversation_id.get()` 为 None 时返回 `{"error":"会话上下文缺失"}`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_tools_business.py -v`
Expected: 4 passed。

- [ ] **Step 5: Commit**

```bash
git add app/tools/business.py tests/test_tools_business.py
git commit -m "feat(ch02): 五个业务工具(order/product/logistics/faq/ticket)"
```

---

### Task 6: ChatService 编排 + Prompt 更新

**Files:**
- Create: `app/services/__init__.py`, `app/services/chat.py`
- Modify: `app/core/prompts.py`
- Test: `tests/test_chat_service.py`, `tests/test_prompts.py`

**Interfaces:**
- Consumes: `app.repositories.*`、`app.tools.registry.ToolRegistry`、`app.core.prompts.CUSTOMER_SERVICE_PROMPT`、`app.core.memory.trim_history`。
- Produces: `class ChatService`:
  - `__init__(self, session_factory, registry: ToolRegistry, model_factory: Callable[[bool], BaseChatModel], token_budget: int)`。
  - `async stream(self, session_id: str, user: str | None, message: str) -> AsyncIterator[dict]`,
    产出 `{"delta": str}` 或 `{"status":"tool_call","tool":str,"tool_call_id":str,"args":dict}`。

    流程: 建/取 conversation → 写 `current_conversation_id`(ContextVar)→ 载历史重建
    `BaseMessage` → `trim_history` → `CUSTOMER_SERVICE_PROMPT.format_messages` → 落 user 消息 →
    `model_factory(False).bind_tools(registry.as_list()).ainvoke(messages)` 得 `ai_msg` →
    (a) 有 `tool_calls`: 取第一个,推状态帧,落 assistant(带 tool_call_request/tool_call_id)、
    执行并落 tool 消息,`ToolMessage` 回灌,`model_factory(True).astream(...)` 逐 chunk 推
    delta,落 assistant 最终回答;
    (b) 无 `tool_calls`: 把 `ai_msg.content` 按小段切块逐帧推 delta,落 assistant 最终回答。

    历史重建映射: `role=user`→`HumanMessage`;`role=assistant` 且 `tool_call_request` 非空→
    `AIMessage(content, tool_calls=[{"name":...,"args":...,"id":tool_call_id}])`;
    `role=assistant` 纯文本→`AIMessage(content)`;`role=tool`→`ToolMessage(content, tool_call_id)`。

- [ ] **Step 1: 写失败测试**

`tests/test_chat_service.py`(FakeModel + SQLite 工厂):

```python
from langchain_core.messages import AIMessage, AIMessageChunk

from app.services.chat import ChatService
from app.tools.registry import ToolRegistry


class FakeModel:
    def __init__(self, tool_calls=None, answer="直接回答"):
        self._tool_calls = tool_calls
        self._answer = answer

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        return AIMessage(content="" if self._tool_calls else self._answer, tool_calls=self._tool_calls or [])

    async def astream(self, messages):
        for ch in self._answer:
            yield AIMessageChunk(content=ch)


async def collect(service, **kw):
    return [frame async for frame in service.stream(**kw)]


def make_service(session_factory, registry, tool_calls=None, answer="直接回答"):
    model = FakeModel(tool_calls=tool_calls, answer=answer)
    return ChatService(session_factory, registry, lambda _: model, token_budget=2000)


def test_no_tool_path_streams_and_persists(session_factory):
    svc = make_service(session_factory, ToolRegistry(), tool_calls=None, answer="您好呀")
    frames = asyncio.run(collect(svc, session_id="s1", user=None, message="在吗"))
    assert "".join(f["delta"] for f in frames) == "您好呀"
    with session_factory() as s:
        from app.repositories import list_messages
        msgs = list_messages(s, 1)
        assert [(m.role, m.content) for m in msgs] == [("user", "在吗"), ("assistant", "您好呀")]


def test_tool_path_emits_status_and_streams(monkeypatch, session_factory):
    from app.tools import business
    monkeypatch.setattr(business, "SessionLocal", session_factory)
    r = ToolRegistry()
    r.register(business.query_faq)
    svc = make_service(
        session_factory, r,
        tool_calls=[{"name": "query_faq", "args": {"keyword": "退货"}, "id": "call_1"}],
        answer="7 天无理由退货。",
    )
    frames = asyncio.run(collect(svc, session_id="s1", user=None, message="退货政策是什么"))
    status = [f for f in frames if f.get("status") == "tool_call"]
    assert len(status) == 1 and status[0]["tool"] == "query_faq"
    assert "".join(f["delta"] for f in frames if "delta" in f) == "7 天无理由退货。"
    with session_factory() as s:
        from app.repositories import list_messages
        roles = [m.role for m in list_messages(s, 1)]
        assert roles == ["user", "assistant", "tool", "assistant"]
```

`tests/test_prompts.py` 的 `test_customer_service_prompt_renders_with_history` 关键字断言
更新为 `("客服", "订单", "禁止编造", "工具")`,并新增:

```python
def test_customer_service_prompt_instructs_tool_use_and_no_fabrication():
    msgs = CUSTOMER_SERVICE_PROMPT.format_messages(history=[])
    content = msgs[0].content
    for kw in ("工具", "调用工具", "禁止编造", "查不到"):
        assert kw in content
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_chat_service.py tests/test_prompts.py -v`
Expected: FAIL(`ModuleNotFoundError: No module named 'app.services.chat'` 与 prompt 关键字缺)。

- [ ] **Step 3: 实现 `app/services/chat.py` + 更新 `app/core/prompts.py`**

`prompts.py` 的 `CUSTOMER_SERVICE_SYSTEM` 从"没有查询权限"改为"可按需调用工具":订单/物流/
商品/FAQ/建工单走工具;工具查不到的(如运费)明确说未收录并引导人工,禁止编造。其余约束沿用。
`services/chat.py` 按 Interfaces 实现;重建映射是本任务唯一需要照抄的算法,见上。无工具时
把 content 按 4 字符一段切块推 delta。服务内所有同步仓储调用(`get_or_create_conversation` /
`add_message` / `list_messages`)统一经 `await asyncio.to_thread(...)` 包装,避免阻塞事件循环。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_chat_service.py tests/test_prompts.py -v`
Expected: chat_service 2 passed + prompts 3 passed。

- [ ] **Step 5: Commit**

```bash
git add app/services app/core/prompts.py tests/test_chat_service.py tests/test_prompts.py
git commit -m "feat(ch02): 单轮工具调用编排与客服 Prompt 更新"
```

---

### Task 7: SSE 路由改造与落库接入

**Files:**
- Modify: `app/schemas/chat.py`, `app/api/chat.py`, `app/core/memory.py`
- Test: `tests/test_chat_api.py`, `tests/test_schemas.py`, `tests/test_memory.py`

**Interfaces:**
- Consumes: `ChatService`、`ToolRegistry`、`app.tools.business.TOOLS`、`app.db.session.SessionLocal`、`app.core.llm.get_chat_model`。
- Produces: `app.api.chat.get_chat_service() -> ChatService`(注册 TOOLS 到 registry,用
  `SessionLocal` + `get_chat_model` 构造),路由 `POST /api/chat` 改调 `service.stream`,
  帧序列化与 `data: [DONE]`、`event: error` 由路由负责。
- 变更: `ChatRequest` 加 `user: str | None = None`;`app.core.memory` 删除 `SessionStore`
  (改由 DB 持久化),保留 `trim_history`。

- [ ] **Step 1: 写失败测试**

`tests/test_schemas.py` 追加:

```python
def test_chat_request_user_defaults_to_none():
    assert ChatRequest(session_id="s1", message="hi").user is None
    assert ChatRequest(session_id="s1", user="王小明", message="hi").user == "王小明"
```

`tests/test_memory.py` 删除 `SessionStore` 相关测试,保留 `trim_history` 测试。

`tests/test_chat_api.py` 重写(用 SQLite 工厂 + FakeModel 覆盖 `get_chat_service`):

```python
import asyncio, json

from fastapi.testclient import TestClient

from app.api import chat as chat_api
from app.main import app
from app.services.chat import ChatService


class FakeModel:
    def __init__(self, tool_calls=None, answer="好的"):
        self._tool_calls = tool_calls
        self._answer = answer
    def bind_tools(self, tools):
        return self
    async def ainvoke(self, messages):
        from langchain_core.messages import AIMessage
        return AIMessage(content="" if self._tool_calls else self._answer, tool_calls=self._tool_calls or [])
    async def astream(self, messages):
        from langchain_core.messages import AIMessageChunk
        for ch in self._answer:
            yield AIMessageChunk(content=ch)


def parse_sse(resp):
    frames, last, error = [], "", None
    for line in resp.iter_lines():
        if line.startswith("event: "):
            error = line[len("event: "):]
        elif line.startswith("data: "):
            payload = line[len("data: "):]
            if payload == "[DONE]":
                last = payload
            else:
                frames.append(json.loads(payload))
    return frames, last, error


def make_client(session_factory, registry, model, teardown):
    svc = ChatService(session_factory, registry, lambda _: model, 2000)
    app.dependency_overrides[chat_api.get_chat_service] = lambda: svc
    client = TestClient(app)
    teardown.append(lambda: app.dependency_overrides.clear())
    return client


def test_chat_api_streams_and_persists(session_factory, registry):
    teardown = []
    client = make_client(session_factory, registry, FakeModel(answer="您好呀"), teardown)
    with client.stream("POST", "/api/chat", json={"session_id": "s1", "message": "在吗"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        frames, last, error = parse_sse(resp)
    assert "".join(f["delta"] for f in frames) == "您好呀"
    assert last == "[DONE]" and error is None
    with session_factory() as s:
        from app.repositories import list_messages
        assert [m.content for m in list_messages(s, 1)] == ["在吗", "您好呀"]


def test_chat_api_emits_tool_status_frame(session_factory, registry):
    teardown = []
    client = make_client(
        session_factory, registry,
        FakeModel(tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}], answer="订单在路上"),
        teardown,
    )
    with client.stream("POST", "/api/chat", json={"session_id": "s2", "message": "订单 1001 到哪了"}) as resp:
        frames, last, error = parse_sse(resp)
    tool = [f for f in frames if f.get("status") == "tool_call"]
    assert tool and tool[0]["tool"] == "query_order"


def test_chat_api_upstream_error_sends_error_frame(session_factory, registry):
    class Boom:
        def bind_tools(self, tools):
            return self
        async def ainvoke(self, messages):
            raise RuntimeError("down")
        async def astream(self, messages):
            raise RuntimeError("down")
            yield
    teardown = []
    client = make_client(session_factory, registry, Boom(), teardown)
    with client.stream("POST", "/api/chat", json={"session_id": "s3", "message": "hi"}) as resp:
        frames, last, error = parse_sse(resp)
    assert error == "error"
    assert last != "[DONE]"
```

注: `registry` fixture 放 `tests/conftest.py`(注册五个 TOOLS):

```python
@pytest.fixture()
def registry():
    from app.tools.business import TOOLS
    from app.tools.registry import ToolRegistry

    r = ToolRegistry()
    for t in TOOLS:
        r.register(t)
    return r
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_chat_api.py tests/test_schemas.py tests/test_memory.py -v`
Expected: FAIL(`get_chat_service` 不存在 / `ChatRequest.user` 不存在 / `SessionStore` 已删)。

- [ ] **Step 3: 实现路由与 schema、清理 memory**

`schemas/chat.py` 加 `user`;`api/chat.py` 用 `get_chat_service` 依赖 + `StreamingResponse`
包 `service.stream`,路由层只做帧序列化、`[DONE]`、`event: error`;`memory.py` 删 `SessionStore`
留 `trim_history`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_chat_api.py tests/test_schemas.py tests/test_memory.py -v`
Expected: chat_api 3 passed + schemas 6 passed + memory 保留条数通过。

- [ ] **Step 5: Commit**

```bash
git add app/schemas/chat.py app/api/chat.py app/core/memory.py tests
git commit -m "feat(ch02): SSE 路由接入工具调用并落库"
```

---

### Task 8: 种子数据 + docker-compose + init_db + Makefile(数据产出)

**Files:**
- Create: `app/db/seed.py`, `scripts/init_db.py`, `docker-compose.yml`
- Modify: `Makefile`, `README.md`
- Test: `tests/test_seed.py`(数据产出用标注断言替代 TDD)

**Interfaces:**
- Produces: `app.db.seed.seed(session) -> None`(灌 FAQ + 一条演示会话/消息);
  `scripts/init_db.py`(main 调 `init_db()` + `seed(SessionLocal())`,print 四表行数);
  `docker-compose.yml`(mysql:8.4,数据库/用户/口令均 `mewhelp`,端口 3306,healthcheck)。

- [ ] **Step 1: 写种子断言**

`tests/test_seed.py`:

```python
from app.db.seed import seed
from app.db.models import Faq
from app.repositories import search_faq


def test_seed_contains_return_policy_and_not_shipping_fee(session_factory):
    with session_factory() as s:
        seed(s)
        assert search_faq(s, "退货")  # 验收 2 可命中
        assert search_faq(s, "邮费") == []  # 验收 3 漏召回
        assert search_faq(s, "运费") == []
        assert s.query(Faq).count() >= 4
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_seed.py -v`
Expected: FAIL(`ModuleNotFoundError: No module named 'app.db.seed'`)。

- [ ] **Step 3: 实现 seed + init_db + docker-compose + Makefile + README**

`seed.py` 的 FAQ 至少含「退货政策 / 换货流程 / 发货时效 / 发票」,**不得**含「邮费/运费」,
`seed(session)` 末尾 `session.commit()`;`scripts/init_db.py` 从 `app.db.session` 与
`app.db.seed` 组装;
`docker-compose.yml` 写 mysql:8.4 服务;`Makefile` 加 `db-up` / `db-down` / `db-init`。
README 增补 ch02 启动与验收命令(`make db-up`、`make db-init`、`make dev`,浏览器打开
http://localhost:8000/ 问三条验收话术)。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_seed.py -v`
Expected: 1 passed。

- [ ] **Step 5: Commit**

```bash
git add app/db/seed.py scripts/init_db.py docker-compose.yml Makefile README.md tests/test_seed.py
git commit -m "feat(ch02): MySQL docker-compose 与种子数据"
```

---

### Task 9: 工具选择标注样例 + eval 脚本(数据产出)

**Files:**
- Create: `tests/data/tool_eval.json`, `scripts/eval_tools.py`
- Modify: `Makefile`(加 `eval-tools` target)

**Interfaces:**
- Produces: `tests/data/tool_eval.json`(样例: `text` / `expect_tool` / 可选 `expect_miss`);
  `scripts/eval_tools.py`(对每条样例用真实模型 `bind_tools(TOOLS).ainvoke` 取第一个
  `tool_calls[0]["name"]`,与 `expect_tool` 比对,打印 PASS/FAIL 与命中工具)。

- [ ] **Step 1: 写样例集**

`tests/data/tool_eval.json` 五条:

```json
[
  {"text": "订单 1001 的物流到哪了", "expect_tool": "query_logistics"},
  {"text": "退货政策是什么", "expect_tool": "query_faq"},
  {"text": "邮费是多少", "expect_tool": "query_faq", "expect_miss": true},
  {"text": "帮我查一下订单 1001 的详情", "expect_tool": "query_order"},
  {"text": "我要投诉快递员态度差", "expect_tool": "create_ticket"}
]
```

- [ ] **Step 2: 实现 eval 脚本**

`scripts/eval_tools.py`: 复用 `app.tools.business.TOOLS` 与 `app.core.llm.get_chat_model`,
`bind_tools` 后逐条 `ainvoke`,打印命中工具与 PASS/FAIL,exit 非 0 若有失败。`expect_miss`
只在打印时标注(实际漏召回由 Task 8 断言 + 浏览器验收覆盖)。

- [ ] **Step 3: 加 Makefile target 并 commit**

`Makefile` 加 `eval-tools: uv run python scripts/eval_tools.py`。

```bash
git add tests/data/tool_eval.json scripts/eval_tools.py Makefile
git commit -m "feat(ch02): 工具选择标注样例与评估脚本"
```

验证: 本任务不打真实模型,真实模型跑 eval 放验收阶段(需真实 key)。

---

### Task 10: 聊天页工具徽章(Vibe Coding,豁免流程)

**Files:**
- Modify: `app/static/index.html`

**Interfaces:**
- Consumes: SSE 新增帧 `{"status":"tool_call","tool":...,"tool_call_id":...,"args":...}`。
- Produces: 助手气泡内工具徽章;页头副标题改「Ch02 · 工具调用」。

- [ ] **Step 1: 改 `handleSSEBlock`**

在 JSON 分支内、`payload.delta` 之前加:

```js
if (payload.status === "tool_call") {
  acc.tool = payload.tool;
  renderAssistant(bubble, acc);
}
```

`send()` 里 `const acc = { text: "" };` 改为 `{ text: "", tool: null };`;
`renderAssistant(bubble, acc)` 在 `formatMarkdown(acc.text)` 前按 `acc.tool` 是否非空,
在气泡内插一个 `<span class="tool-badge">⚙ <工具名></span>`(工具名用 `escapeHtml`)。

- [ ] **Step 2: 加样式**

`.tool-badge`(小号、圆角、强调色浅底),置于气泡文字上方,不挤压布局。

- [ ] **Step 3: 目检并 commit**

本地起服务后在浏览器确认:问「订单 1001 的物流到哪了」气泡带 `⚙ query_logistics` 徽章,
回答逐字打出;「退货政策是什么」带 `⚙ query_faq`。不套 TDD/code review。

```bash
git add app/static/index.html
git commit -m "feat(ch02): 聊天页工具轨迹徽章"
```

---

## 验收(实现完成后,需 Docker Desktop + 真实 key)

1. `make db-up` → `make db-init` → `make dev`,浏览器开 http://localhost:8000/。
2. 问「订单 1001 的物流到哪了」→ 气泡带 `query_logistics` 徽章并按随机物流结果作答。
3. 问「退货政策是什么」→ 气泡带 `query_faq` 徽章并按 FAQ 作答。
4. 问「邮费是多少」→ `query_faq` 返回"未找到",模型明说不编造(预期漏召回,记入
   `dev-notes/ch02.md`)。
5. `make test` 全量绿;`make eval-tools` 5/5 工具选择通过。
