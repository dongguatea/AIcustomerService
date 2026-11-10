import asyncio

from langchain_core.messages import AIMessage, AIMessageChunk
from sqlalchemy import select

from app.db.models import LowConfidenceQuestion
from app.services.chat import ChatService
from app.tools import business
from app.tools.business import RagResult
from app.tools.registry import ToolRegistry


class FakeModel:
    def __init__(self, tool_calls=None, answer="依据证据作答"):
        self._tool_calls = tool_calls
        self._answer = answer

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        return AIMessage(content="", tool_calls=self._tool_calls or [])

    async def astream(self, messages):
        for ch in self._answer:
            yield AIMessageChunk(content=ch)


async def collect(service, **kw):
    return [frame async for frame in service.stream(**kw)]


def make_service(session_factory, tool_calls, answer="依据证据作答"):
    r = ToolRegistry()
    r.register(business.query_faq)
    model = FakeModel(tool_calls=tool_calls, answer=answer)
    return ChatService(session_factory, r, lambda _: model, token_budget=2000)


def test_query_faq_sufficient_emits_citations(monkeypatch, session_factory):
    async def fake_rag(keyword, category=None):
        return RagResult(
            sufficient=True, evidence="[1] 运费政策: 满99包邮",
            citations=[{"n": 1, "id": 1, "section_path": "运费政策",
                        "question": "运费怎么算", "answer": "满99包邮",
                        "content_type": "faq"}],
        )

    monkeypatch.setattr(business, "run_query_faq", fake_rag)
    svc = make_service(
        session_factory,
        tool_calls=[{"name": "query_faq", "args": {"keyword": "邮费"}, "id": "c1"}],
    )
    frames = asyncio.run(collect(svc, session_id="s1", user=None, message="邮费是多少"))
    citations = [f for f in frames if f.get("event") == "citations"]
    deltas = [f for f in frames if "delta" in f]
    assert len(citations) == 1
    assert citations[0]["items"][0]["section_path"] == "运费政策"
    assert frames.index(citations[0]) < frames.index(deltas[0])


def test_query_faq_insufficient_writes_low_conf(monkeypatch, session_factory):
    async def fake_rag(keyword, category=None):
        return RagResult(sufficient=False, source="self_check", reason="证据不足")

    monkeypatch.setattr(business, "run_query_faq", fake_rag)
    svc = make_service(
        session_factory,
        tool_calls=[{"name": "query_faq", "args": {"keyword": "火星车"}, "id": "c1"}],
    )
    frames = asyncio.run(collect(svc, session_id="s1", user=None, message="你们卖火星车吗"))
    assert not [f for f in frames if f.get("event") == "citations"]
    with session_factory() as s:
        row = s.scalars(select(LowConfidenceQuestion)).one()
        assert row.source == "self_check"
        assert row.reason == "证据不足"
        assert row.raw_question == "你们卖火星车吗"
        assert row.conversation_id == 1
