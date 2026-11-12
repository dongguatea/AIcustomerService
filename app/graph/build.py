from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.routing import confidence_gate, route_by_intent, should_continue
from app.graph.state import ConversationState


def build_graph(checkpointer=None):
    """组装七节点图:指代→意图→分流→[知识检索→闸]→主力 Agent(ReAct)→日志。"""
    b = StateGraph(ConversationState)
    b.add_node("resolve_reference", nodes.resolve_reference)
    b.add_node("classify_intent", nodes.classify_intent)
    b.add_node("retrieve_knowledge", nodes.retrieve_knowledge)
    b.add_node("confidence_check", nodes.confidence_check)
    b.add_node("main_agent", nodes.main_agent)
    b.add_node("agent_tools", nodes.agent_tools)
    b.add_node("complaint_reply", nodes.complaint_reply)
    b.add_node("script_reply", nodes.script_reply)
    b.add_node("fallback_reply", nodes.fallback_reply)
    b.add_node("log", nodes.log_node)

    b.add_edge(START, "resolve_reference")
    b.add_edge("resolve_reference", "classify_intent")
    b.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "knowledge": "retrieve_knowledge",
            "business": "main_agent",
            "complaint": "complaint_reply",
            "chitchat": "script_reply",
        },
    )
    b.add_edge("retrieve_knowledge", "confidence_check")
    b.add_conditional_edges(
        "confidence_check",
        confidence_gate,
        {"strong": "main_agent", "weak": "fallback_reply"},
    )
    b.add_conditional_edges(
        "main_agent",
        should_continue,
        {"continue": "agent_tools", "stop": "log"},
    )
    b.add_edge("agent_tools", "main_agent")
    b.add_edge("complaint_reply", "log")
    b.add_edge("script_reply", "log")
    b.add_edge("fallback_reply", "log")
    b.add_edge("log", END)
    return b.compile(checkpointer=checkpointer)
