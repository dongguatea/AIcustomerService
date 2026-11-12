"""最简意图识别:七类 JSON 输出(不做多轮改写、不做置信度校准)。"""
import logging
from typing import Literal

from pydantic import BaseModel, Field

from app.core import llm
from app.core.prompts import INTENT_CLASSIFY_PROMPT

logger = logging.getLogger(__name__)

INTENTS = ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊")


class _Intent(BaseModel):
    intent: Literal["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊"] = Field(
        description="七类意图之一"
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="判断把握 0-1")


async def classify(query: str) -> dict:
    """返回 {"intent": str | None, "confidence": float};模型异常/越界 → intent=None。"""
    model = llm.structured(_Intent)
    try:
        r: _Intent = await (INTENT_CLASSIFY_PROMPT | model).ainvoke({"query": query})
    except Exception:
        logger.warning("意图识别失败,回落未知 query=%r", query, exc_info=True)
        return {"intent": None, "confidence": 0.0}
    return {"intent": r.intent, "confidence": float(r.confidence)}
