from app.db.models import Faq
from app.repositories import (
    add_message,
    create_ticket,
    get_or_create_conversation,
    list_messages,
    search_faq,
)


def test_get_or_create_conversation_reuses(session_factory):
    with session_factory() as s:
        a = get_or_create_conversation(s, "s1", "u")
        b = get_or_create_conversation(s, "s1", "u")
        assert a.id == b.id


def test_search_faq_like_hit_and_miss(session_factory):
    with session_factory() as s:
        s.add(Faq(question="退货政策是什么?", answer="7 天无理由", category="售后"))
        s.add(Faq(question="发货时效", answer="48 小时内", category="物流"))
        s.commit()
        assert len(search_faq(s, "退货")) == 1
        assert search_faq(s, "邮费") == []


def test_message_flow_and_ticket(session_factory):
    with session_factory() as s:
        conv = get_or_create_conversation(s, "s1", "u")
        add_message(s, conv.id, "user", "订单 1001 物流")
        add_message(
            s,
            conv.id,
            "assistant",
            "好的",
            {"name": "query_logistics", "args": {"order_id": "1001"}},
            "call_1",
        )
        add_message(s, conv.id, "tool", "已签收", None, "call_1")
        t = create_ticket(s, conv.id, "快递员态度差", "投诉")
        msgs = list_messages(s, conv.id)
        assert [m.role for m in msgs] == ["user", "assistant", "tool"]
        assert t.ticket_no.startswith("TKT-")
        assert t.status == "open"
