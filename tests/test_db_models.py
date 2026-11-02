from datetime import datetime

from app.db.models import Conversation, Faq, Message, Ticket


def test_models_create_tables_and_rows(session_factory):
    with session_factory() as s:
        conv = Conversation(session_id="s1", user="王小明")
        s.add(conv)
        s.flush()
        s.add(Faq(question="退货政策是什么?", answer="7 天无理由", category="售后"))
        s.add(Message(conversation_id=conv.id, role="user", content="你好"))
        s.add(Ticket(ticket_no="TKT-1", conversation_id=conv.id, description="x", ticket_type="投诉"))
        s.commit()
    with session_factory() as s:
        assert s.query(Conversation).count() == 1
        assert s.query(Faq).count() == 1
        assert s.query(Message).count() == 1
        assert s.query(Ticket).count() == 1


def test_message_tool_fields_roundtrip(session_factory):
    with session_factory() as s:
        conv = Conversation(session_id="s2", user="u")
        s.add(conv)
        s.flush()
        s.add(Message(
            conversation_id=conv.id, role="assistant", content="query_logistics",
            tool_call_request={"name": "query_logistics", "args": {"order_id": "1001"}},
            tool_call_id="call_1",
        ))
        s.commit()
    with session_factory() as s:
        m = s.query(Message).one()
        assert m.tool_call_request["name"] == "query_logistics"
        assert m.tool_call_id == "call_1"
        assert isinstance(m.created_at, datetime)
