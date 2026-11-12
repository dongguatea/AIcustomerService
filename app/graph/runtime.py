"""图运行时:持久 checkpointer + 流式输出(SSE 帧)。"""
import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import settings
from app.db.session import SessionLocal
from app.graph import nodes
from app.graph.build import build_graph
from app.repositories import add_message, get_or_create_conversation
from app.tools.business import query_faq, query_logistics, query_order, query_product
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

ANSWER_NODES = {"main_agent"}  # 答复 token 走 messages 流式
DETERMINISTIC_ANSWER_NODES = {"script_reply", "complaint_reply", "fallback_reply"}
CITATION_NODES = {"retrieve_knowledge"}

_graph = None
_cm = None  # AsyncSqliteSaver context manager,持有以免被 GC


async def init_graph() -> None:
    """lifespan 起:开持久 checkpointer + setup + 注入工具执行器 + 编译图。"""
    global _graph, _cm
    Path(settings.checkpointer_db_path).parent.mkdir(parents=True, exist_ok=True)
    _cm = AsyncSqliteSaver.from_conn_string(settings.checkpointer_db_path)
    checkpointer = await _cm.__aenter__()
    await checkpointer.setup()
    registry = ToolRegistry()
    for tool in (query_order, query_logistics, query_product, query_faq):
        registry.register(tool)
    nodes.configure_registry(registry)
    _graph = build_graph(checkpointer=checkpointer)
    logger.info("图已编译,checkpointer=%s", settings.checkpointer_db_path)


async def close_graph() -> None:
    global _graph, _cm
    if _cm is not None:
        await _cm.__aexit__(None, None, None)
        _cm = None
    _graph = None


def get_graph():
    if _graph is None:
        raise RuntimeError("图未初始化,应在 FastAPI lifespan 调 init_graph()")
    return _graph


def _graph_input(
    session_id: str, user: str, message: str, conversation_id: int
) -> dict:
    """每轮入口:messages 走 add_messages 累加;无 reducer 标量显式清零防跨轮残留。"""
    return {
        "messages": [HumanMessage(message)],
        "session_id": session_id,
        "conversation_id": conversation_id,
        "user": user,
        "steps": 0,
        "tokens_used": 0,
        "answer": "",
        "suggested_actions": [],
        "evidence": "",
        "evidence_strong": False,
        "citations": [],
        "fallback_source": "",
        "fallback_reason": "",
        "gate_decision": "",
        "resolved_query": "",
        "intent": "",
        "route": "",
    }


def _dedup_actions(actions: list) -> list:
    seen: set = set()
    out: list = []
    for a in actions:
        t = a.get("type")
        if t in seen:
            continue
        seen.add(t)
        out.append(a)
    return out


async def _stream_graph(graph, input_state: dict, thread_id: str) -> AsyncIterator[dict]:
    """astream 多模式 → SSE 帧。供 stream_turn 调用,测试可直接注入 InMemorySaver 图。"""
    config = {"configurable": {"thread_id": thread_id}}
    actions: list = []
    async for mode, chunk in graph.astream(
        input_state, config, stream_mode=["messages", "updates"]
    ):
        if mode == "messages":
            msg, meta = chunk
            if meta.get("langgraph_node") in ANSWER_NODES:
                text = msg.content if isinstance(msg.content, str) else ""
                if text:
                    yield {"delta": text}
        elif mode == "updates":
            for node, upd in chunk.items():
                if not isinstance(upd, dict):
                    continue
                if node == "main_agent":
                    for m in upd.get("messages", []):
                        for tc in m.tool_calls or []:
                            yield {
                                "status": "tool_call",
                                "tool": tc.get("name"),
                                "tool_call_id": tc.get("id"),
                                "args": tc.get("args") or {},
                            }
                if node in DETERMINISTIC_ANSWER_NODES and upd.get("answer"):
                    yield {"delta": upd["answer"]}
                if node in CITATION_NODES and upd.get("citations"):
                    yield {"event": "citations", "items": upd["citations"]}
                if upd.get("suggested_actions"):
                    actions.extend(upd["suggested_actions"])
    if actions:
        yield {"event": "actions", "items": _dedup_actions(actions)}


async def stream_turn(
    session_id: str, user: str | None, message: str
) -> AsyncIterator[dict]:
    """流式一轮:落 user 消息 → 跑图 → 事件 dict(供 /api/chat 转 SSE)。"""

    def _conv():
        with SessionLocal() as s:
            return get_or_create_conversation(s, session_id, user or session_id)

    conv = await asyncio.to_thread(_conv)

    def _add_user():
        with SessionLocal() as s:
            add_message(s, conv.id, "user", message)

    await asyncio.to_thread(_add_user)
    input_state = _graph_input(session_id, user or session_id, message, conv.id)
    async for frame in _stream_graph(get_graph(), input_state, session_id):
        yield frame
