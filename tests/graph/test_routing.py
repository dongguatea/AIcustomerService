import pytest
from langchain_core.messages import AIMessage

from app.graph.routing import (
    INTENT_TO_ROUTE,
    confidence_gate,
    route_by_intent,
    should_continue,
)


@pytest.mark.parametrize(
    "intent,route",
    [
        ("商品咨询", "knowledge"),
        ("退款退货", "knowledge"),
        ("物流", "business"),
        ("订单", "business"),
        ("售后", "business"),
        ("投诉", "complaint"),
        ("闲聊", "chitchat"),
    ],
)
def test_route_by_intent(intent, route):
    assert route_by_intent({"intent": intent}) == route


def test_route_unknown_falls_back_business():
    assert route_by_intent({"intent": "不存在"}) == "business"
    assert route_by_intent({"intent": ""}) == "business"


def test_confidence_gate():
    assert confidence_gate({"evidence_strong": True}) == "strong"
    assert confidence_gate({"evidence_strong": False}) == "weak"


def test_should_continue():
    assert (
        should_continue(
            {"messages": [AIMessage(content="答", tool_calls=[])], "steps": 0}
        )
        == "stop"
    )
    assert (
        should_continue(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "query_order", "args": {}, "id": "c1"}],
                    )
                ],
                "steps": 0,
            }
        )
        == "continue"
    )


def test_should_continue_stops_at_max_steps(monkeypatch):
    monkeypatch.setattr("app.graph.routing.settings.max_agent_steps", 2)
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "query_order", "args": {}, "id": "c1"}],
            )
        ],
        "steps": 2,
    }
    assert should_continue(state) == "stop"


def test_conversation_state_fields():
    from app.graph.state import ConversationState

    for k in (
        "messages",
        "session_id",
        "conversation_id",
        "intent",
        "route",
        "resolved_query",
        "evidence",
        "citations",
        "evidence_strong",
        "fallback_source",
        "fallback_reason",
        "gate_decision",
        "answer",
        "steps",
        "tokens_used",
        "suggested_actions",
    ):
        assert k in ConversationState.__annotations__
