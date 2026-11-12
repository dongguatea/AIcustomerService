"""非业务控制工具:suggest_action 只产出「建议动作」,后端不执行。"""
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class SuggestActionArgs(BaseModel):
    action: Literal["transfer_human", "create_ticket"] = Field(
        description="建议动作类型"
    )
    description: str | None = Field(default=None, description="建工单时的问题描述")


@tool(args_schema=SuggestActionArgs)
async def suggest_action(action: str, description: str | None = None) -> str:
    """当判断用户需要转人工或建工单时调用,产出「建议动作」(只记录、后端不执行)。
    这些建议会作为可选项交给用户,由用户在前端点按钮自行决定。"""
    return f"已记录建议:{action}"
