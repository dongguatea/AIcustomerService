from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from app.core.prompts import (
    CUSTOMER_SERVICE_PROMPT,
    CUSTOMER_SERVICE_SYSTEM,
    EXTRACT_PROMPT,
    FAITHFULNESS_PROMPT,
    MINING_PROMPT,
    QUERY_REWRITE_PROMPT,
    RAG_ANSWER_PROMPT,
    SELF_CHECK_PROMPT,
)


def test_customer_service_prompt_renders_with_history():
    msgs = CUSTOMER_SERVICE_PROMPT.format_messages(
        history=[HumanMessage("你们卖猫粮吗?")]
    )
    assert msgs[0].type == "system"
    assert msgs[-1].content == "你们卖猫粮吗?"
    for keyword in ("客服", "订单", "禁止编造", "工具"):
        assert keyword in msgs[0].content


def test_customer_service_prompt_instructs_tool_use_and_no_fabrication():
    msgs = CUSTOMER_SERVICE_PROMPT.format_messages(history=[])
    content = msgs[0].content
    for kw in ("工具", "调用工具", "禁止编造", "查不到"):
        assert kw in content


def test_extract_prompt_renders_text():
    msgs = EXTRACT_PROMPT.format_messages(text="订单 MH1 坏了要退款")
    assert msgs[0].type == "system"
    assert "订单 MH1 坏了要退款" in msgs[-1].content
    assert "禁止编造" in msgs[0].content


def test_mining_prompt_renders():
    msgs = MINING_PROMPT.format_messages(
        conversations="user: 满多少包邮\nassistant: 满99包邮"
    )
    content = msgs[0].content
    for kw in ("复用", "通用", "以平台售后规则为准"):
        assert kw in content


def test_rag_prompts_exist():
    for p in (QUERY_REWRITE_PROMPT, SELF_CHECK_PROMPT, RAG_ANSWER_PROMPT, FAITHFULNESS_PROMPT):
        assert isinstance(p, ChatPromptTemplate)


def test_system_prompt_has_citation_and_refusal_rules():
    for kw in ("sufficient", "编号", "拒答", "以平台售后规则为准"):
        assert kw in CUSTOMER_SERVICE_SYSTEM


def test_system_prompt_lists_negative_knowledge():
    for kw in ("不承诺", "到账", "时效"):
        assert kw in CUSTOMER_SERVICE_SYSTEM
