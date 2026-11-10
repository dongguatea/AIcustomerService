from app.db.models import LowConfidenceQuestion
from app.repositories import insert_low_confidence


def test_insert_low_confidence(session_factory):
    with session_factory() as s:
        rid = insert_low_confidence(s, 1, "你们卖火星车吗", "retrieval_low_conf", "无召回")
        row = s.get(LowConfidenceQuestion, rid)
        assert row.raw_question == "你们卖火星车吗"
        assert row.source == "retrieval_low_conf"
        assert row.reason == "无召回"
        assert row.conversation_id == 1


def test_insert_low_confidence_nullable_conversation(session_factory):
    with session_factory() as s:
        rid = insert_low_confidence(s, None, "你们卖火星车吗", "self_check", "证据不足")
        row = s.get(LowConfidenceQuestion, rid)
        assert row.conversation_id is None
        assert row.source == "self_check"
