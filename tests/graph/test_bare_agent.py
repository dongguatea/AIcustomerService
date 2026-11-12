import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from scripts.bare_agent import run_loop


class FakeModel:
    def __init__(self, ais):
        self._ais = list(ais)
        self._i = 0
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        ai = self._ais[self._i]
        self._i += 1
        return ai


class FakeTool:
    async def ainvoke(self, args):
        return '{"订单号":"1001","物流":"运输中"}'


def test_bare_loop_multi_step():
    model = FakeModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}
                ],
            ),
            AIMessage(content="订单在路上", tool_calls=[]),
        ]
    )
    tools = {"query_order": FakeTool()}
    out = asyncio.run(
        run_loop(model, tools, [HumanMessage("订单1001到哪了")], max_steps=6)
    )
    assert out == "订单在路上"
    assert model.calls == 2
