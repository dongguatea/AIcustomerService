import asyncio
import json
import random
from contextvars import ContextVar

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from app.db.session import SessionLocal
from app.repositories import create_ticket as _create_ticket
from app.repositories import search_faq as _search_faq

current_conversation_id: ContextVar[int | None] = ContextVar(
    "current_conversation_id", default=None
)


class OrderArgs(BaseModel):
    order_id: str = Field(description="订单号")


class KeywordArgs(BaseModel):
    keyword: str = Field(description="搜索关键词")


class TicketArgs(BaseModel):
    description: str = Field(description="问题描述")
    ticket_type: str = Field(description="工单类型,如 退款/换货/维修/投诉/其他")


@tool(args_schema=OrderArgs)
async def query_order(order_id: str) -> str:
    """根据订单号查询订单摘要(演示数据,不接真实接口)。"""
    await asyncio.sleep(random.uniform(0.02, 0.1))
    return json.dumps(
        {
            "订单号": order_id,
            "下单时间": "2026-10-20 14:30",
            "商品": "智能猫砂盆 x1",
            "金额": "699.00 元",
            "状态": random.choice(["待付款", "已付款", "已发货", "已完成"]),
        },
        ensure_ascii=False,
    )


@tool(args_schema=KeywordArgs)
async def query_product(keyword: str) -> str:
    """按关键词查询商品(演示数据,不接真实接口)。"""
    await asyncio.sleep(random.uniform(0.02, 0.1))
    return json.dumps(
        {
            "关键词": keyword,
            "商品列表": [
                {"名称": f"{keyword}经典款", "价格": "199.00 元", "库存": random.randint(1, 99)},
                {"名称": f"{keyword}升级款", "价格": "299.00 元", "库存": random.randint(0, 50)},
            ],
        },
        ensure_ascii=False,
    )


@tool(args_schema=OrderArgs)
async def query_logistics(order_id: str) -> str:
    """根据订单号查询物流轨迹(演示数据,不接真实接口)。"""
    await asyncio.sleep(random.uniform(0.02, 0.1))
    return json.dumps(
        {
            "订单号": order_id,
            "物流单号": f"SF{random.randint(1000000000, 9999999999)}",
            "当前节点": random.choice(["运输中", "已到达分拨中心", "派送中", "已签收"]),
            "轨迹": ["已揽收", "运输中", "到达目的城市"],
        },
        ensure_ascii=False,
    )


@tool(args_schema=KeywordArgs)
async def query_faq(keyword: str) -> str:
    """按关键词在常见问题表里检索(问题字段 LIKE 匹配)。"""

    def _run() -> str:
        with SessionLocal() as session:
            rows = _search_faq(session, keyword)
        if not rows:
            return "未找到相关问题"
        return "\n".join(f"问: {r.question}\n答: {r.answer}" for r in rows)

    return await asyncio.to_thread(_run)


@tool(args_schema=TicketArgs)
async def create_ticket(description: str, ticket_type: str) -> str:
    """创建人工工单,写入工单表。"""
    conv_id = current_conversation_id.get()
    if conv_id is None:
        return json.dumps({"error": "会话上下文缺失,创建工单失败"}, ensure_ascii=False)

    def _run() -> str:
        with SessionLocal() as session:
            return _create_ticket(session, conv_id, description, ticket_type).ticket_no

    return await asyncio.to_thread(_run)


TOOLS: list[BaseTool] = [
    query_order,
    query_product,
    query_logistics,
    query_faq,
    create_ticket,
]
