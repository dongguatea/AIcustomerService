import asyncio

from app.db.models import KnowledgeChunk, QaExtractionStaging
from app.kb import mining
from app.repositories import (
    add_message,
    get_or_create_conversation,
    insert_knowledge_chunk,
)


def test_mine_dedups_existing(monkeypatch, session_factory):
    monkeypatch.setattr(mining, "SessionLocal", session_factory)

    async def fake_extract(conversation_texts, model=None):
        return [mining.QaPair(question="满多少包邮", answer="满99包邮")]
    monkeypatch.setattr(mining, "extract_qa", fake_extract)

    with session_factory() as s:
        c = get_or_create_conversation(s, "s1", "u")
        add_message(s, c.id, "user", "满多少包邮")
        add_message(s, c.id, "assistant", "满99包邮")
        insert_knowledge_chunk(s, "运费", "满多少包邮", "满99包邮")

    stats = asyncio.run(mining.mine(batch_size=20))
    assert stats["kept"] == 0 and stats["discarded"] == 1
    with session_factory() as s:
        assert s.query(QaExtractionStaging).count() == 1
        assert s.query(KnowledgeChunk).count() == 1


def test_mine_writes_new_qa(monkeypatch, session_factory):
    monkeypatch.setattr(mining, "SessionLocal", session_factory)

    async def fake_extract(conversation_texts, model=None):
        return [mining.QaPair(question="多久发货", answer="48小时内")]
    monkeypatch.setattr(mining, "extract_qa", fake_extract)

    with session_factory() as s:
        c = get_or_create_conversation(s, "s2", "u")
        add_message(s, c.id, "user", "多久发货")
        add_message(s, c.id, "assistant", "48小时内")

    stats = asyncio.run(mining.mine(batch_size=20))
    assert stats["kept"] == 1
    with session_factory() as s:
        row = s.query(KnowledgeChunk).one()
        assert row.category == "对话挖知识" and row.content_type == "qa_mined"
        assert row.questions == "多久发货" and row.vectorize_status == "pending"
