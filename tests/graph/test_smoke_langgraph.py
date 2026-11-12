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
