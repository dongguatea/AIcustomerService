import asyncio

import pytest

from app.core import retrieval


def _hit(id, **kw):
    h = {"id": id, "score": 0.8, "question": f"q{id}", "answer": f"a{id}",
         "section_path": f"s{id}", "content_type": "faq", "category": "c"}
    h.update(kw)
    return h


async def _fake_acall(fn, *a, **k):
    return fn(*a, **k)


def _install_common(monkeypatch):
    async def fake_embed(query):
        return [1.0, 2.0]

    monkeypatch.setattr(retrieval.embeddings, "embed_query", fake_embed)
    monkeypatch.setattr(retrieval.milvus_client, "get_client", lambda *a, **k: object())
    monkeypatch.setattr(retrieval.milvus_client, "ensure_collection", lambda *a, **k: None)
    monkeypatch.setattr(retrieval.milvus_client, "acall", _fake_acall)


def test_arrange_head_tail():
    assert retrieval.arrange_head_tail([0, 1, 2, 3, 4]) == [0, 2, 3, 4, 1]
    assert retrieval.arrange_head_tail([0, 1]) == [0, 1]


def test_strategy_bm25_uses_bm25_text(monkeypatch):
    _install_common(monkeypatch)
    captured = {}

    def fake_bm25(client, text, top_k, category=None, collection="knowledge"):
        captured.update(text=text, top_k=top_k)
        return [_hit(1)]

    monkeypatch.setattr(retrieval.milvus_client, "bm25_search", fake_bm25)
    hits = asyncio.run(retrieval.search_knowledge(
        "邮费", strategy="bm25", top_k=5, bm25_text="标准 邮费 运费"))
    assert hits[0]["id"] == 1
    assert captured == {"text": "标准 邮费 运费", "top_k": 5}


def test_strategy_vector_uses_clean_query(monkeypatch):
    _install_common(monkeypatch)
    captured = {}

    def fake_dense(client, vector, top_k, category=None, collection="knowledge"):
        captured.update(vector=vector, top_k=top_k)
        return [_hit(2)]

    monkeypatch.setattr(retrieval.milvus_client, "dense_search", fake_dense)
    asyncio.run(retrieval.search_knowledge(
        "邮费", strategy="vector", top_k=3, bm25_text="扩展词"))
    assert captured == {"vector": [1.0, 2.0], "top_k": 3}


def test_strategy_hybrid_no_rerank(monkeypatch):
    _install_common(monkeypatch)
    captured = {}

    def fake_hybrid(client, vector, text, top_k, recall=50, category=None, collection="knowledge"):
        captured.update(top_k=top_k, recall=recall, text=text)
        return [_hit(i) for i in range(12)]

    monkeypatch.setattr(retrieval.milvus_client, "hybrid_search", fake_hybrid)
    hits = asyncio.run(retrieval.search_knowledge("邮费", strategy="hybrid", top_k=10))
    assert len(hits) == 10
    assert captured["top_k"] == 50
    assert captured["recall"] == 50


def test_strategy_hybrid_rerank_adds_rerank_score(monkeypatch):
    _install_common(monkeypatch)
    monkeypatch.setattr(
        retrieval.milvus_client, "hybrid_search",
        lambda client, vector, text, top_k, recall=50, category=None, collection="knowledge":
            [_hit(0), _hit(1), _hit(2)],
    )

    async def fake_rerank(query, docs, top_n=None):
        return [(1, 0.9), (0, 0.6)]

    monkeypatch.setattr(retrieval.rerank, "rerank", fake_rerank)
    hits = asyncio.run(retrieval.search_knowledge("邮费", strategy="hybrid_rerank", top_k=2))
    assert [h["id"] for h in hits] == [1, 0]
    assert [h["rerank_score"] for h in hits] == [0.9, 0.6]


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        asyncio.run(retrieval.search_knowledge("x", strategy="foo"))
