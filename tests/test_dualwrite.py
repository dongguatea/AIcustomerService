import asyncio

from app.db.models import KnowledgeChunk
from app.kb import dualwrite
from app.kb.documents import Chunk
from app.repositories import mark_chunk_vectorized


def _chunks(n=3):
    return [Chunk(category="运费", questions=f"q{i}", answer=f"a{i}",
                  section_path="开票与配送 / 运费", content_type="faq") for i in range(n)]


def test_write_pending_inserts_and_links(monkeypatch, session_factory):
    monkeypatch.setattr(dualwrite, "SessionLocal", session_factory)
    ids = asyncio.run(dualwrite.write_pending(_chunks()))
    with session_factory() as s:
        rows = s.query(KnowledgeChunk).order_by(KnowledgeChunk.id).all()
        assert [r.id for r in rows] == ids
        assert rows[1].prev_chunk_id == ids[0] and rows[1].next_chunk_id == ids[2]
        assert all(r.vectorize_status == "pending" for r in rows)


def test_vectorize_pending_resumes_and_is_idempotent(monkeypatch, session_factory):
    monkeypatch.setattr(dualwrite, "SessionLocal", session_factory)

    async def fake_embed(texts):
        return [[0.0] for _ in texts]
    monkeypatch.setattr(dualwrite.embeddings, "embed_texts", fake_embed)

    upserts, flushes = [], []
    monkeypatch.setattr(dualwrite.milvus_client, "get_client", lambda *a, **k: object())
    monkeypatch.setattr(dualwrite.milvus_client, "ensure_collection", lambda *a, **k: None)
    monkeypatch.setattr(dualwrite.milvus_client, "upsert_vectors",
                        lambda c, rows, collection="knowledge": upserts.append([r["id"] for r in rows]))
    monkeypatch.setattr(dualwrite.milvus_client, "flush", lambda *a, **k: flushes.append(1))

    async def inline(fn, *a, **k):
        return fn(*a, **k)
    monkeypatch.setattr(dualwrite.milvus_client, "acall", inline)

    ids = asyncio.run(dualwrite.write_pending(_chunks(3)))
    with session_factory() as s:
        mark_chunk_vectorized(s, ids[0], str(ids[0]))

    assert asyncio.run(dualwrite.vectorize_pending(batch_size=1)) == 2
    assert upserts == [[ids[1]], [ids[2]]]
    assert asyncio.run(dualwrite.vectorize_pending(batch_size=1)) == 0
    with session_factory() as s:
        assert all(r.vectorize_status == "done" for r in s.query(KnowledgeChunk))
