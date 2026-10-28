from app.db.models import KnowledgeChunk, QaExtractionStaging


def test_knowledge_chunk_defaults_and_neighbors(session_factory):
    with session_factory() as s:
        a = KnowledgeChunk(category="运费", questions="运费怎么算", answer="满99包邮")
        s.add(a)
        s.flush()
        b = KnowledgeChunk(category="运费", questions="偏远地区", answer="另计",
                           prev_chunk_id=a.id)
        s.add(b)
        s.flush()
        a.next_chunk_id = b.id
        s.commit()
        assert a.vectorize_status == "pending"
        assert a.is_key_clause == 0
        assert b.prev_chunk_id == a.id and a.next_chunk_id == b.id


def test_staging_default_status(session_factory):
    with session_factory() as s:
        s.add(QaExtractionStaging(batch_no="b1", source_ref="conv:1",
                                  question="运费怎么算", answer="满99包邮"))
        s.commit()
        row = s.query(QaExtractionStaging).one()
        assert row.status == "extracted"


def test_chunk_neighbor_fk_sets_null_on_delete():
    prev_fk = list(KnowledgeChunk.__table__.c.prev_chunk_id.foreign_keys)
    next_fk = list(KnowledgeChunk.__table__.c.next_chunk_id.foreign_keys)
    assert prev_fk and prev_fk[0].ondelete == "SET NULL"
    assert next_fk and next_fk[0].ondelete == "SET NULL"
