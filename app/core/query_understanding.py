"""口语→标准问法改写 + 同义词扩展(检索侧)。"""
import logging

from pydantic import BaseModel, Field

from app.core import llm
from app.core.prompts import QUERY_REWRITE_PROMPT

logger = logging.getLogger(__name__)


class _Rewrite(BaseModel):
    standard: str = Field(description="标准问法")
    expanded: list[str] = Field(default_factory=list, description="同义/近义扩展词")


async def understand(query: str) -> dict:
    """返回 {"standard": str, "expanded": list[str]};模型异常回落原问题、扩展词留空。"""
    model = llm.structured(_Rewrite)
    try:
        r: _Rewrite = await (QUERY_REWRITE_PROMPT | model).ainvoke({"query": query})
    except Exception:
        logger.warning("query 改写失败,回落原问题 query=%r", query, exc_info=True)
        return {"standard": query, "expanded": []}
    return {"standard": r.standard or query, "expanded": list(r.expanded or [])}
