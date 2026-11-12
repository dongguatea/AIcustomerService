import asyncio

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import nodes
from app.graph.build import build_graph
from app.graph.runtime import _stream_graph


def _input(text):
    return {
        "messages": [HumanMessage(text)],
        "conversation_id": 1,
        "steps": 0,
        "tokens_used": 0,
        "suggested_actions": [],
        "answer": "",
        "resolved_query": "",
        "evidence": "",
        "evidence_strong": False,
        "citations": [],
        "fallback_source": "",
        "fallback_reason": "",
        "gate_decision": "",
        "intent": "",
        "route": "",
    }


async def collect(ag):
    return [f async for f in ag]


def _classify(intent_name):
    async def fake(q):
        return {"intent": intent_name, "confidence": 0.9}

    return fake


class SeqChat:
    def __init__(self, ais):
        self._ais = list(ais)
        self._i = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages, config=None):
        ai = self._ais[self._i]
        self._i += 1
        return ai


class FakeRegistry:
    async def execute(self, name, args, *, timeout, max_retries):
        return '{"订单号":"1001","物流":"运输中"}'


def test_stream_chitchat_delta(monkeypatch, session_factory):
    monkeypatch.setattr(nodes, "SessionLocal", session_factory)
    monkeypatch.setattr(nodes.intent, "classify", _classify("闲聊"))
    frames = asyncio.run(
        collect(_stream_graph(build_graph(InMemorySaver()), _input("你好"), "s1"))
    )
    assert "".join(f["delta"] for f in frames if "delta" in f) == nodes.SCRIPT_REPLY_TEXT


def test_stream_business_tool_and_delta(monkeypatch, session_factory):
    monkeypatch.setattr(nodes, "SessionLocal", session_factory)
    monkeypatch.setattr(nodes.intent, "classify", _classify("物流"))
    chat = SeqChat(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}
                ],
            ),
            AIMessage(content="订单在路上", tool_calls=[]),
        ]
    )
    monkeypatch.setattr(nodes, "get_chat_model", lambda streaming=True: chat)
    monkeypatch.setattr(nodes, "_registry", FakeRegistry())
    frames = asyncio.run(
        collect(
            _stream_graph(
                build_graph(InMemorySaver()), _input("订单1001到哪了"), "s2"
            )
        )
    )
    assert any(
        f.get("status") == "tool_call" and f["tool"] == "query_logistics"
        for f in frames
    )
    assert "".join(f["delta"] for f in frames if "delta" in f) == "订单在路上"


def test_stream_actions_event(monkeypatch, session_factory):
    monkeypatch.setattr(nodes, "SessionLocal", session_factory)
    monkeypatch.setattr(nodes.intent, "classify", _classify("投诉"))
    frames = asyncio.run(
        collect(_stream_graph(build_graph(InMemorySaver()), _input("我要投诉"), "s3"))
    )
    act = [f for f in frames if f.get("event") == "actions"]
    assert act and {a["type"] for a in act[-1]["items"]} == {
        "transfer_human",
        "create_ticket",
    }


def test_cross_turn_no_action_leak(monkeypatch, session_factory):
    monkeypatch.setattr(nodes, "SessionLocal", session_factory)
    app = build_graph(InMemorySaver())
    monkeypatch.setattr(nodes.intent, "classify", _classify("投诉"))
    f1 = asyncio.run(collect(_stream_graph(app, _input("我要投诉"), "s4")))
    monkeypatch.setattr(nodes.intent, "classify", _classify("闲聊"))
    f2 = asyncio.run(collect(_stream_graph(app, _input("你好"), "s4")))
    assert any(f.get("event") == "actions" for f in f1)
    assert not any(f.get("event") == "actions" for f in f2)
