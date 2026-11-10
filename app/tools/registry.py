import asyncio
import json

from langchain_core.tools import BaseTool
from pydantic import ValidationError


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._timeouts: dict[str, float] = {}
        self._retries: dict[str, int] = {}

    def register(
        self, tool: BaseTool, timeout: float | None = None, max_retries: int | None = None
    ) -> None:
        self._tools[tool.name] = tool
        if timeout is not None:
            self._timeouts[tool.name] = timeout
        if max_retries is not None:
            self._retries[tool.name] = max_retries

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
        timeout = self._timeouts.get(name, timeout)
        max_retries = self._retries.get(name, max_retries)

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
