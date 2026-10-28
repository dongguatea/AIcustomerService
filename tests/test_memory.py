from langchain_core.messages import AIMessage, HumanMessage

from app.core.memory import trim_history


def test_trim_keeps_recent_and_starts_on_human():
    msgs = []
    for i in range(20):
        msgs.append(HumanMessage(f"问题{i}:" + "喵" * 50))
        msgs.append(AIMessage(f"回答{i}:" + "喵" * 50))
    trimmed = trim_history(msgs, max_tokens=200)
    assert 0 < len(trimmed) < len(msgs)
    assert trimmed[0].type == "human"
    assert trimmed[-1] == msgs[-1]


def test_trim_noop_when_under_budget():
    msgs = [HumanMessage("hi"), AIMessage("hello")]
    assert trim_history(msgs, max_tokens=2000) == msgs
