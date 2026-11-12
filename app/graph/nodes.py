"""图节点:骨架里的确定性环节与 ReAct 主力 Agent(分任务逐步补全)。"""
import asyncio
import logging

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from app.config import settings
from app.core import intent
from app.core.llm import get_chat_model
from app.core.prompts import CUSTOMER_SERVICE_SYSTEM
from app.db.session import SessionLocal
from app.graph.routing import route_by_intent
from app.repositories import add_message, insert_low_confidence
from app.tools import business, control

logger = logging.getLogger(__name__)

COMPLAINT_REPLY_TEXT = "非常抱歉给您带来不好的体验,您的心情我理解。您可以选择「转人工」让真人客服跟进,或「建工单」把问题记下来,我们会尽快处理。"
SCRIPT_REPLY_TEXT = "你好呀,我是喵喵优选客服「小喵」,有什么购物上的问题都可以问我哦~"
FALLBACK_REPLY_TEXT = "暂时没有查到足够的信息回答您。建议点下方「转人工」联系人工客服进一步核实。"
EVIDENCE_HINT = (
    "\n\n## 已检索到的知识证据(请据此作答,每个关键结论后标注来源编号如[1];"
    "证据已给,不要再调用 query_faq;仍可按需调用订单/物流等工具)\n"
)

# 主力 Agent 的工具集:ch02 四查询工具 + 非业务控制工具(不含 create_ticket 写工具)
AGENT_TOOLS = [
    business.query_order,
    business.query_logistics,
    business.query_product,
    business.query_faq,
    control.suggest_action,
]

_registry = None


def configure_registry(registry) -> None:
    """注入查询工具执行器(agent_tools 用它跑 query_* 工具)。"""
    global _registry
    _registry = registry


def _user_text(state: dict) -> str:
    """取本轮用户原文(消息列表里最后一条 human 消息)。"""
    for m in reversed(state.get("messages", [])):
        if getattr(m, "type", "") == "human" and isinstance(m.content, str):
            return m.content
    return ""


def _resolve_answer(state: dict) -> str:
    """确定性出口读 state['answer'];Agent 出口读最后一条 AIMessage 的 content。"""
    if state.get("answer"):
        return state["answer"]
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content:
            return m.content
    return ""


def resolve_reference(state: dict) -> dict:
    """指代消解(ch05 最简:原样透传)。"""
    return {"resolved_query": _user_text(state)}


async def classify_intent(state: dict) -> dict:
    """意图识别:判七类;失败/越界保守归 business(Agent 兜底)。"""
    query = state.get("resolved_query") or _user_text(state)
    r = await intent.classify(query)
    intent_name = r["intent"] or ""
    route = route_by_intent({"intent": intent_name})
    return {"intent": intent_name, "route": route}


def confidence_check(state: dict) -> dict:
    """置信度闸留痕:实际分流在条件边 confidence_gate(读 evidence_strong)。"""
    return {"gate_decision": "strong" if state.get("evidence_strong") else "weak"}


async def retrieve_knowledge(state: dict) -> dict:
    """知识类强制 RAG 检索(复用 ch04 管线),产出编号证据 + 证据强弱信号。"""
    try:
        rag = await business.run_query_faq(_user_text(state))
    except Exception as exc:  # noqa: BLE001 - 检索管线异常按证据不足处理(fail-closed)
        logger.warning("知识检索失败,按证据不足处理", exc_info=True)
        return {
            "evidence_strong": False,
            "fallback_source": "retrieval_low_conf",
            "fallback_reason": f"检索管线异常:{type(exc).__name__}",
        }
    if rag.sufficient:
        return {
            "evidence": rag.evidence,
            "citations": rag.citations,
            "evidence_strong": True,
        }
    return {
        "evidence_strong": False,
        "fallback_source": rag.source or "retrieval_low_conf",
        "fallback_reason": rag.reason or "",
    }


def _agent_messages(state: dict) -> list:
    """主力 Agent 的输入:system + 证据提示(知识路强证据时)+ 历史消息。"""
    system = CUSTOMER_SERVICE_SYSTEM
    if state.get("route") == "knowledge" and state.get("evidence_strong"):
        system = system + EVIDENCE_HINT + state.get("evidence", "")
    return [SystemMessage(content=system), *state.get("messages", [])]


async def main_agent(state: dict, config: RunnableConfig | None = None) -> dict:
    """ReAct 推理步:调模型(带工具),累加 steps 与 token 消耗。"""
    model = get_chat_model(streaming=True).bind_tools(AGENT_TOOLS)
    ai: AIMessage = await model.ainvoke(_agent_messages(state), config)
    usage = getattr(ai, "usage_metadata", None) or {}
    used = usage.get("total_tokens", 0)
    return {
        "messages": [ai],
        "steps": state.get("steps", 0) + 1,
        "tokens_used": state.get("tokens_used", 0) + used,
    }


async def agent_tools(state: dict) -> dict:
    """ReAct 行动步:查询工具走 registry,suggest_action 只记录建议。"""
    last = state["messages"][-1]
    tool_msgs: list[ToolMessage] = []
    actions = list(state.get("suggested_actions", []))
    for tc in last.tool_calls:
        name = tc["name"]
        args = tc.get("args") or {}
        if name == "suggest_action":
            if args.get("action") == "create_ticket":
                actions.append(
                    {
                        "type": "create_ticket",
                        "draft": {
                            "description": args.get("description") or "",
                            "ticket_type": "其他",
                        },
                    }
                )
            else:
                actions.append({"type": "transfer_human"})
            tool_msgs.append(ToolMessage(content="已记录建议", tool_call_id=tc["id"]))
        else:
            result = await _registry.execute(
                name,
                args,
                timeout=settings.tool_timeout,
                max_retries=settings.tool_max_retries,
            )
            tool_msgs.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    return {"messages": tool_msgs, "suggested_actions": actions}


def script_reply(state: dict) -> dict:
    """闲聊:固定话术,零模型调用。"""
    return {"answer": SCRIPT_REPLY_TEXT}


def complaint_reply(state: dict) -> dict:
    """投诉:安抚话术 + 转人工/建工单两个可选项(后端不执行,前端自选)。"""
    return {
        "answer": COMPLAINT_REPLY_TEXT,
        "suggested_actions": [
            {"type": "transfer_human"},
            {
                "type": "create_ticket",
                "draft": {"description": _user_text(state), "ticket_type": "投诉"},
            },
        ],
    }


async def fallback_reply(state: dict) -> dict:
    """知识路证据弱兜底:回兜底话术 + 落低置信池(数据飞轮留坑),不进 Agent。"""
    source = state.get("fallback_source") or "retrieval_low_conf"
    reason = state.get("fallback_reason") or ""
    cid = state.get("conversation_id")
    raw = _user_text(state)

    def _write():
        with SessionLocal() as s:
            insert_low_confidence(s, cid, raw, source, reason)

    await asyncio.to_thread(_write)
    return {
        "answer": FALLBACK_REPLY_TEXT,
        "suggested_actions": [{"type": "transfer_human"}],
    }


async def log_node(state: dict) -> dict:
    """日志 + 落 MySQL 一条 assistant 消息(审计 + 工单外键依赖)。"""
    logger.info(
        "turn conv=%s intent=%s route=%s resolved=%r gate=%s steps=%s tokens=%s",
        state.get("conversation_id"),
        state.get("intent"),
        state.get("route"),
        state.get("resolved_query"),
        state.get("gate_decision"),
        state.get("steps"),
        state.get("tokens_used"),
    )
    answer = _resolve_answer(state)
    cid = state.get("conversation_id")
    if cid and answer:

        def _write():
            with SessionLocal() as s:
                add_message(s, cid, "assistant", answer)

        await asyncio.to_thread(_write)
    return {}
