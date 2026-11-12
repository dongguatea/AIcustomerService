from langchain_core.messages import AIMessage

from app.config import settings

# 七类意图 → 四出口,写死的分流规则(单一来源,不做模型决策)
INTENT_TO_ROUTE: dict[str, str] = {
    "商品咨询": "knowledge",
    "退款退货": "knowledge",
    "物流": "business",
    "订单": "business",
    "售后": "business",
    "投诉": "complaint",
    "闲聊": "chitchat",
}


def route_by_intent(state: dict) -> str:
    """按意图分流;未知/空意图保守归 business(交给 Agent 自己应对)。"""
    return INTENT_TO_ROUTE.get(state.get("intent", ""), "business")


def confidence_gate(state: dict) -> str:
    """知识路生成前证据闸:强放行、弱兜底。"""
    return "strong" if state.get("evidence_strong") else "weak"


def should_continue(state: dict) -> str:
    """ReAct 停止条件:无 tool_calls 收敛,步数封顶强停,否则继续。"""
    last = state["messages"][-1]
    has_tool_calls = isinstance(last, AIMessage) and bool(last.tool_calls)
    if not has_tool_calls:
        return "stop"
    if state.get("steps", 0) >= settings.max_agent_steps:
        return "stop"
    return "continue"
