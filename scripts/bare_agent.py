"""祛魅热身:零 LangGraph 手写最裸 Agent 循环。

一个 while 循环:调 LLM → 有工具调用就执行并把结果喂回去 → 没工具调用就收敛出答案。
复用 ch02 查询工具 + ch04 的 DeepSeek 模型,不引任何编排框架。
"""
import asyncio
import json

from langchain_core.messages import HumanMessage, ToolMessage

from app.core.llm import get_chat_model
from app.tools.business import query_faq, query_logistics, query_order, query_product


async def run_loop(model, tools, messages, max_steps: int) -> str:
    """裸 Agent 循环:tools 为 {name: tool};收敛返回最终答复文本。"""
    bound = model.bind_tools(list(tools.values()))
    for _ in range(max_steps):
        ai = await bound.ainvoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            return ai.content if isinstance(ai.content, str) else str(ai.content)
        for tc in ai.tool_calls:
            tool = tools.get(tc["name"])
            if tool is None:
                result = json.dumps(
                    {"error": "工具不存在", "name": tc["name"]}, ensure_ascii=False
                )
            else:
                result = await tool.ainvoke(tc.get("args") or {})
                result = result if isinstance(result, str) else str(result)
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    last = messages[-1]
    return last.content if isinstance(last.content, str) else ""


async def main() -> None:
    tools = {
        t.name: t for t in (query_order, query_logistics, query_product, query_faq)
    }
    model = get_chat_model(streaming=False)
    demo = "订单 1001 到哪了?顺便帮我看看这个订单的物流。"
    print("用户:", demo)
    answer = await run_loop(model, tools, [HumanMessage(demo)], max_steps=6)
    print("\n最终答复:", answer)


if __name__ == "__main__":
    asyncio.run(main())
