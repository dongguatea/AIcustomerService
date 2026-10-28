import asyncio
import json

from langchain_core.tools import BaseTool
from pydantic import ValidationError


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

    def as_list(self) -> list[BaseTool]:
        return list(self._tools.values())

    async def execute(
        self, name: str, args: dict, *, timeout: float, max_retries: int
    ) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return json.dumps({"error": "工具不存在", "name": name}, ensure_ascii=False)

        if tool.args_schema is not None:
            try:
                tool.args_schema.model_validate(args)
            except ValidationError as exc:
                return json.dumps(
                    {"error": "参数校验失败", "detail": str(exc)}, ensure_ascii=False
                )

        last_err: Exception | None = None
        for _ in range(max_retries + 1):
            try:
                async with asyncio.timeout(timeout):
                    result = await tool.ainvoke(args)
                return result if isinstance(result, str) else str(result)
            except TimeoutError as exc:
                last_err = exc
            except Exception as exc:
                last_err = exc

        return json.dumps(
            {"error": "工具执行失败", "detail": str(last_err)}, ensure_ascii=False
        )
