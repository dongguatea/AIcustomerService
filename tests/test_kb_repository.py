from app.db.models import Conversation, KnowledgeChunk, Message
from app.repositories import (
    count_chunks_by_content_types,
    insert_knowledge_chunk,
    insert_staging,
    list_all_questions,
    list_conversations_with_messages,
    list_pending_chunks,
    list_staging_by_status,
    mark_chunk_vectorized,
    set_chunk_neighbors,
    set_staging_status,
)


def test_chunk_insert_neighbors_pending_and_done(session_factory):
    with session_factory() as s:
        a = insert_knowledge_chunk(s, "运费", "运费怎么算", "满99包邮",
                                   section_path="开票与配送 / 运费", content_type="faq")
        b = insert_knowledge_chunk(s, "运费", "偏远地区", "另计", content_type="faq")
        set_chunk_neighbors(s, a, None, b)
        set_chunk_neighbors(s, b, a, None)
        assert [c.id for c in list_pending_chunks(s)] == [a, b]
        mark_chunk_vectorized(s, a, str(a))
        assert [c.id for c in list_pending_chunks(s)] == [b]
        assert s.get(KnowledgeChunk, a).vector_id == str(a)


def test_staging_and_questions_and_count(session_factory):
    with session_factory() as s:
        insert_knowledge_chunk(s, "运费", "运费怎么算\n偏远地区怎么算", "满99包邮",
                               content_type="faq")
        i = insert_staging(s, "b1", "conv:1", "运费怎么算", "满99包邮")
        assert [r.status for r in list_staging_by_status(s, "extracted")] == ["extracted"]
        set_staging_status(s, [i], "kept")
        assert list_staging_by_status(s, "extracted") == []
        assert list_all_questions(s) == ["运费怎么算", "偏远地区怎么算"]
        assert count_chunks_by_content_types(s, {"faq"}) == 1
        assert count_chunks_by_content_types(s, {"policy"}) == 0


def test_list_conversations_with_messages(session_factory):
    with session_factory() as s:
        c = Conversation(session_id="s1", user="u")
        s.add(c)
        s.flush()
        s.add_all([Message(conversation_id=c.id, role="user", content="满多少包邮"),
                   Message(conversation_id=c.id, role="assistant", content="满99包邮")])
        s.commit()
        rows = list_conversations_with_messages(s)
        assert len(rows) == 1 and rows[0][0] == c.id
        assert [m.content for m in rows[0][1]] == ["满多少包邮", "满99包邮"]
