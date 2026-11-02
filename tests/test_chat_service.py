import asyncio

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from app.repositories import list_messages
from app.services.chat import ChatService
from app.tools import business
from app.tools.registry import ToolRegistry


class FakeModel:
    def __init__(self, tool_calls=None, answer="直接回答"):
        self._tool_calls = tool_calls
        self._answer = answer

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        return AIMessage(
            content="" if self._tool_calls else self._answer,
            tool_calls=self._tool_calls or [],
        )

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
        msgs = list_messages(s, 1)
        assert [(m.role, m.content) for m in msgs] == [("user", "在吗"), ("assistant", "您好呀")]


def test_tool_path_emits_status_and_streams(monkeypatch, session_factory):
    monkeypatch.setattr(business, "SessionLocal", session_factory)
    r = ToolRegistry()
    r.register(business.query_faq)
    svc = make_service(
        session_factory,
        r,
        tool_calls=[{"name": "query_faq", "args": {"keyword": "退货"}, "id": "call_1"}],
        answer="7 天无理由退货。",
    )
    frames = asyncio.run(
        collect(svc, session_id="s1", user=None, message="退货政策是什么")
    )
    status = [f for f in frames if f.get("status") == "tool_call"]
    assert len(status) == 1 and status[0]["tool"] == "query_faq"
    assert "".join(f["delta"] for f in frames if "delta" in f) == "7 天无理由退货。"
    with session_factory() as s:
        roles = [m.role for m in list_messages(s, 1)]
        assert roles == ["user", "assistant", "tool", "assistant"]


def test_decision_receives_current_user_message(session_factory):
    captured = {}

    class CapturingModel(FakeModel):
        async def ainvoke(self, messages):
            captured["messages"] = messages
            return AIMessage(content="收到", tool_calls=[])

    svc = ChatService(
        session_factory, ToolRegistry(), lambda _: CapturingModel(), 2000
    )
    asyncio.run(collect(svc, session_id="s1", user=None, message="订单 1001 到哪了"))
    humans = [m for m in captured["messages"] if isinstance(m, HumanMessage)]
    assert humans and humans[-1].content == "订单 1001 到哪了"
