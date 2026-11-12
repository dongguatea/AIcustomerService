from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ConversationState(TypedDict, total=False):
    """ch05 图状态:一条状态贯穿七节点,checkpointer 按 thread_id 持久化。"""

    messages: Annotated[list[AnyMessage], add_messages]  # 跨轮历史 + ReAct 消息
    session_id: str  # 前端 session_id,也做 thread_id
    conversation_id: int  # MySQL 会话 id(工单外键 / 审计)
    user: str
    intent: str  # 七类之一
    route: str  # knowledge | business | complaint | chitchat
    resolved_query: str
    evidence: str  # 知识路编号证据
    citations: list
    evidence_strong: bool
    fallback_source: str
    fallback_reason: str
    answer: str  # 确定性节点答复(agent 答复走流式,不落此字段)
    steps: int
    tokens_used: int
    gate_decision: str  # 知识路闸门决策(strong|weak),供 log 打点
    suggested_actions: list

