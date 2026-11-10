import asyncio
import json
import random
from contextvars import ContextVar
from dataclasses import dataclass, field

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from app.config import settings
from app.core import query_understanding, retrieval, selfcheck
from app.db.session import SessionLocal
from app.repositories import create_ticket as _create_ticket

current_conversation_id: ContextVar[int | None] = ContextVar(
    "current_conversation_id", default=None
)


class OrderArgs(BaseModel):
    order_id: str = Field(description="订单号")


class KeywordArgs(BaseModel):
    keyword: str = Field(description="搜索关键词")


class FaqArgs(BaseModel):
    keyword: str = Field(description="用户咨询的政策/规则/操作类问题")
    category: str | None = Field(default=None, description="可选:按品类过滤,如「运费」「商品手册」")


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


@dataclass
class RagResult:
    sufficient: bool
    source: str | None = None
    reason: str | None = None
    evidence: str = ""
    citations: list[dict] = field(default_factory=list)


REFUSAL_MSG = "未检索到足以回答的知识,请向用户拒答"


async def run_query_faq(keyword: str, category: str | None = None) -> RagResult:
    """RAG 检索管线:Query 理解 → 品类过滤 → 混合检索+重排 → 两道生成前闸 → 首尾组装。
    返回结构化 RagResult;ChatService 直接调用本函数以拿到 citations/source 等元数据。"""
    u = await query_understanding.understand(keyword)
    query = u["standard"]
    # 同义词扩展只拼进 BM25 检索文本(检索侧),不改 dense query、不改入库
    bm25_text = query + (" " + " ".join(u["expanded"]) if u["expanded"] else "")

    hits = await retrieval.search_knowledge(
        query, strategy="hybrid_rerank", category=category, bm25_text=bm25_text)

    # 品类过滤稳健性:过滤后为空/全弱则回退不过滤重试(过滤是优化、非硬约束)
    if category and (not hits or hits[0].get("rerank_score", 0.0) < settings.rerank_min_score):
        hits = await retrieval.search_knowledge(
            query, strategy="hybrid_rerank", category=None, bm25_text=bm25_text)

    # 机械闸:无召回或最高 rerank 分低于阈值
    top = hits[0].get("rerank_score", 0.0) if hits else 0.0
    if not hits or top < settings.rerank_min_score:
        return RagResult(sufficient=False, source="retrieval_low_conf",
                         reason=f"检索证据不足(top={top:.3f})")

    # 语义闸:生成前自评证据够不够
    ev_texts = [f"{h['question']} {h['answer']}" for h in hits]
    chk = await selfcheck.check_sufficient(query, ev_texts)
    if not chk["useful"]:
        return RagResult(sufficient=False, source="self_check", reason=chk["reason"])

    # 首尾组装 + 编号
    arranged = retrieval.arrange_head_tail(hits)
    citations = [
        {"n": i + 1, "id": h["id"], "section_path": h["section_path"],
         "question": h["question"], "answer": h["answer"],
         "content_type": h["content_type"]}
        for i, h in enumerate(arranged)
    ]
    evidence = "\n".join(f"[{c['n']}] {c['question']}: {c['answer']}" for c in citations)
    return RagResult(sufficient=True, evidence=evidence, citations=citations)


@tool(args_schema=FaqArgs)
async def query_faq(keyword: str, category: str | None = None) -> str:
    """查询常见问题/政策知识库(混合检索+重排)。用于政策、规则、时效、费用、商品手册等通用问题。
    返回带编号证据供作答引用;证据不足时返回拒答信号,请据此向用户拒答。"""
    r = await run_query_faq(keyword, category)
    return r.evidence if r.sufficient else REFUSAL_MSG


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
