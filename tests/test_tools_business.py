import asyncio

from app.db.models import Faq, Ticket
from app.repositories import get_or_create_conversation
from app.tools import business


def test_query_faq_hit_and_miss(monkeypatch, session_factory):
    monkeypatch.setattr(business, "SessionLocal", session_factory)
    with session_factory() as s:
        s.add(Faq(question="退货政策是什么?", answer="7 天无理由", category="售后"))
        s.commit()
    assert "7 天无理由" in asyncio.run(business.query_faq.ainvoke({"keyword": "退货"}))
    assert "未找到" in asyncio.run(business.query_faq.ainvoke({"keyword": "邮费"}))


def test_mock_tools_echo_input():
    order = asyncio.run(business.query_order.ainvoke({"order_id": "1001"}))
    assert "1001" in order and "订单" in order
    prod = asyncio.run(business.query_product.ainvoke({"keyword": "猫粮"}))
    assert "猫粮" in prod
    logi = asyncio.run(business.query_logistics.ainvoke({"order_id": "1001"}))
    assert "1001" in logi and "物流" in logi


def test_create_ticket_writes_and_uses_context(monkeypatch, session_factory):
    monkeypatch.setattr(business, "SessionLocal", session_factory)
    with session_factory() as s:
        conv_id = get_or_create_conversation(s, "s1", "u").id
    token = business.current_conversation_id.set(conv_id)
    try:
        out = asyncio.run(
            business.create_ticket.ainvoke({"description": "快递丢了", "ticket_type": "投诉"})
        )
    finally:
        business.current_conversation_id.reset(token)
    assert out.startswith("TKT-")
    with session_factory() as s:
        t = s.query(Ticket).one()
        assert t.conversation_id == conv_id


def test_create_ticket_without_context_returns_error(monkeypatch, session_factory):
    monkeypatch.setattr(business, "SessionLocal", session_factory)
    out = asyncio.run(
        business.create_ticket.ainvoke({"description": "x", "ticket_type": "投诉"})
    )
    assert "失败" in out
