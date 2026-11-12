import asyncio

from langchain_core.messages import AIMessage, HumanMessage
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


class FakeChat:
    def __init__(self, ai):
        self._ai = ai

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages, config=None):
        return self._ai


def test_main_agent_converges(monkeypatch):
    monkeypatch.setattr(
        nodes, "get_chat_model",
        lambda streaming=True: FakeChat(AIMessage(content="好的", tool_calls=[])),
    )
    out = asyncio.run(
        nodes.main_agent(
            {"messages": [HumanMessage("在吗")], "steps": 0, "tokens_used": 0}
        )
    )
    assert out["steps"] == 1
    assert out["messages"][-1].content == "好的"


def test_main_agent_counts_usage_when_present(monkeypatch):
    ai = AIMessage(
        content="好的",
        tool_calls=[],
        usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
    )
    monkeypatch.setattr(
        nodes, "get_chat_model", lambda streaming=True: FakeChat(ai)
    )
    out = asyncio.run(
        nodes.main_agent(
            {"messages": [HumanMessage("在吗")], "steps": 0, "tokens_used": 0}
        )
    )
    assert out["tokens_used"] == 110


def test_main_agent_counts_tokens_fallback(monkeypatch):
    # streaming=True 的 DeepSeek 不回传 usage_metadata → 用近似计数兜底,token 预算才不失效
    monkeypatch.setattr(
        nodes, "get_chat_model",
        lambda streaming=True: FakeChat(AIMessage(content="好的", tool_calls=[])),
    )
    out = asyncio.run(
        nodes.main_agent(
            {"messages": [HumanMessage("在吗")], "steps": 0, "tokens_used": 0}
        )
    )
    assert out["tokens_used"] > 0


def test_main_agent_injects_evidence(monkeypatch):
    seen = {}

    class Cap(FakeChat):
        async def ainvoke(self, messages, config=None):
            seen["msgs"] = messages
            return AIMessage(content="依据[1]", tool_calls=[])

    monkeypatch.setattr(nodes, "get_chat_model", lambda streaming=True: Cap(None))
    asyncio.run(
        nodes.main_agent(
            {
                "messages": [HumanMessage("邮费")],
                "steps": 0,
                "tokens_used": 0,
                "route": "knowledge",
                "evidence_strong": True,
                "evidence": "[1] 满99包邮",
            }
        )
    )
    assert any("满99包邮" in str(m.content) for m in seen["msgs"])


def test_agent_tools_suggest_action_records(monkeypatch):
    last = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "suggest_action",
                "args": {"action": "create_ticket", "description": "要退货"},
                "id": "c1",
            }
        ],
    )
    out = asyncio.run(
        nodes.agent_tools({"messages": [last], "suggested_actions": []})
    )
    assert out["suggested_actions"] == [
        {"type": "create_ticket", "draft": {"description": "要退货", "ticket_type": "其他"}}
    ]
    assert out["messages"][-1].content == "已记录建议"


def test_agent_tools_executes_query_tool(monkeypatch):
    class FakeRegistry:
        async def execute(self, name, args, *, timeout, max_retries):
            return '{"订单号":"1001","状态":"已发货"}'

    monkeypatch.setattr(nodes, "_registry", FakeRegistry())
    last = AIMessage(
        content="",
        tool_calls=[
            {"name": "query_order", "args": {"order_id": "1001"}, "id": "c2"}
        ],
    )
    out = asyncio.run(
        nodes.agent_tools({"messages": [last], "suggested_actions": []})
    )
    assert out["messages"][-1].content == '{"订单号":"1001","状态":"已发货"}'
