import asyncio

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select

from app.db.models import LowConfidenceQuestion, Ticket
from app.graph import nodes
from app.graph.build import build_graph
from app.tools.business import RagResult


def _input(text):
    return {
        "messages": [HumanMessage(text)],
        "conversation_id": 1,
        "steps": 0,
        "tokens_used": 0,
        "suggested_actions": [],
        "answer": "",
        "resolved_query": "",
    }


def test_chitchat_end_to_end(monkeypatch, session_factory):
    async def fake(q):
        return {"intent": "闲聊", "confidence": 0.9}

    monkeypatch.setattr(nodes.intent, "classify", fake)
    monkeypatch.setattr(nodes, "SessionLocal", session_factory)
    out = asyncio.run(
        build_graph(InMemorySaver()).ainvoke(
            _input("你好"), {"configurable": {"thread_id": "s1"}}
        )
    )
    assert out["route"] == "chitchat"
    assert out["answer"] == nodes.SCRIPT_REPLY_TEXT


def test_complaint_no_ticket_written(monkeypatch, session_factory):
    async def fake(q):
        return {"intent": "投诉", "confidence": 0.9}

    monkeypatch.setattr(nodes.intent, "classify", fake)
    monkeypatch.setattr(nodes, "SessionLocal", session_factory)
    out = asyncio.run(
        build_graph(InMemorySaver()).ainvoke(
            _input("我要投诉"), {"configurable": {"thread_id": "s2"}}
        )
    )
    types = [a["type"] for a in out["suggested_actions"]]
    assert types == ["transfer_human", "create_ticket"]
    with session_factory() as s:
        assert s.scalars(select(Ticket)).all() == []


def test_knowledge_weak_falls_back(monkeypatch, session_factory):
    async def fake_intent(q):
        return {"intent": "商品咨询", "confidence": 0.9}

    async def fake_rag(keyword, category=None):
        return RagResult(sufficient=False, source="self_check", reason="证据不足")

    monkeypatch.setattr(nodes.intent, "classify", fake_intent)
    monkeypatch.setattr(nodes.business, "run_query_faq", fake_rag)
    monkeypatch.setattr(nodes, "SessionLocal", session_factory)
    out = asyncio.run(
        build_graph(InMemorySaver()).ainvoke(
            _input("火星车有吗"), {"configurable": {"thread_id": "s3"}}
        )
    )
    assert out["answer"] == nodes.FALLBACK_REPLY_TEXT
    assert out["gate_decision"] == "weak"
    with session_factory() as s:
        row = s.scalars(select(LowConfidenceQuestion)).one()
        assert row.source == "self_check"
