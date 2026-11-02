from langchain_core.messages import HumanMessage

from app.core.prompts import CUSTOMER_SERVICE_PROMPT, EXTRACT_PROMPT


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
