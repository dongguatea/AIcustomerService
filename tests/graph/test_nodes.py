import asyncio

from langchain_core.messages import HumanMessage
from sqlalchemy import select

from app.db.models import LowConfidenceQuestion, Message
from app.graph import nodes
from app.tools.business import RagResult


def test_resolve_reference_passthrough():
    state = {"messages": [HumanMessage("订单1001到哪了")]}
    assert nodes.resolve_reference(state)["resolved_query"] == "订单1001到哪了"


def test_classify_intent_node(monkeypatch):
    async def fake(q):
        return {"intent": "物流", "confidence": 0.9}

    monkeypatch.setattr(nodes.intent, "classify", fake)
    out = asyncio.run(nodes.classify_intent({"resolved_query": "订单1001到哪了"}))
    assert out["intent"] == "物流"
    assert out["route"] == "business"


def test_classify_intent_failure_falls_back_business(monkeypatch):
    async def fake(q):
        return {"intent": None, "confidence": 0.0}

    monkeypatch.setattr(nodes.intent, "classify", fake)
    out = asyncio.run(nodes.classify_intent({"resolved_query": "x"}))
    assert out["intent"] == ""
    assert out["route"] == "business"


def test_confidence_check():
    assert nodes.confidence_check({"evidence_strong": True})["gate_decision"] == "strong"
    assert nodes.confidence_check({"evidence_strong": False})["gate_decision"] == "weak"


def test_script_reply_returns_fixed_text():
    out = nodes.script_reply({"intent": "闲聊"})
    assert out["answer"] == nodes.SCRIPT_REPLY_TEXT
    assert not out.get("suggested_actions")


def test_complaint_reply_two_actions():
    out = nodes.complaint_reply({"messages": [HumanMessage("我要投诉")]})
    types = [a["type"] for a in out["suggested_actions"]]
    assert types == ["transfer_human", "create_ticket"]
    assert out["suggested_actions"][1]["draft"]["ticket_type"] == "投诉"
    assert out["suggested_actions"][1]["draft"]["description"] == "我要投诉"


def test_fallback_reply_writes_low_conf(monkeypatch, session_factory):
    monkeypatch.setattr(nodes, "SessionLocal", session_factory)
    state = {
        "conversation_id": 1,
        "messages": [HumanMessage("你们卖火星车吗")],
        "fallback_source": "self_check",
        "fallback_reason": "证据不足",
    }
    out = asyncio.run(nodes.fallback_reply(state))
    assert out["suggested_actions"] == [{"type": "transfer_human"}]
    with session_factory() as s:
        row = s.scalars(select(LowConfidenceQuestion)).one()
        assert row.source == "self_check"
        assert row.reason == "证据不足"
        assert row.raw_question == "你们卖火星车吗"
        assert row.conversation_id == 1


def test_log_node_persists_assistant(monkeypatch, session_factory):
    monkeypatch.setattr(nodes, "SessionLocal", session_factory)
    asyncio.run(
        nodes.log_node(
            {
                "conversation_id": 1,
                "answer": "好的",
                "intent": "闲聊",
                "route": "chitchat",
                "resolved_query": "在吗",
            }
        )
    )
    with session_factory() as s:
        rows = s.scalars(select(Message).where(Message.role == "assistant")).all()
        assert [m.content for m in rows] == ["好的"]


def test_retrieve_knowledge_sufficient(monkeypatch):
    async def fake_rag(keyword, category=None):
        return RagResult(
            sufficient=True,
            evidence="[1] 运费:满99包邮",
            citations=[{"n": 1, "section_path": "运费政策"}],
        )

    monkeypatch.setattr(nodes.business, "run_query_faq", fake_rag)
    out = asyncio.run(nodes.retrieve_knowledge({"messages": [HumanMessage("邮费多少")]}))
    assert out["evidence_strong"] is True
    assert out["evidence"].startswith("[1]")
    assert out["citations"] == [{"n": 1, "section_path": "运费政策"}]


def test_retrieve_knowledge_insufficient(monkeypatch):
    async def fake_rag(keyword, category=None):
        return RagResult(sufficient=False, source="self_check", reason="证据不足")

    monkeypatch.setattr(nodes.business, "run_query_faq", fake_rag)
    out = asyncio.run(nodes.retrieve_knowledge({"messages": [HumanMessage("火星车")]}))
    assert out["evidence_strong"] is False
    assert out["fallback_source"] == "self_check"
    assert out["fallback_reason"] == "证据不足"


def test_retrieve_knowledge_exception_fails_closed(monkeypatch):
    async def fake_rag(keyword, category=None):
        raise RuntimeError("milvus down")

    monkeypatch.setattr(nodes.business, "run_query_faq", fake_rag)
    out = asyncio.run(nodes.retrieve_knowledge({"messages": [HumanMessage("邮费多少")]}))
    assert out["evidence_strong"] is False
    assert out["fallback_source"] == "retrieval_low_conf"
    assert "RuntimeError" in out["fallback_reason"]
