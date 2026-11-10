import asyncio

import pytest

from app.core import retrieval


def test_search_knowledge_vector_path(monkeypatch):
    async def fake_embed(query):
        return [1.0, 2.0]
    monkeypatch.setattr(retrieval.embeddings, "embed_query", fake_embed)
    captured = {}

    def fake_dense(client, vector, top_k, category=None, collection="knowledge"):
        captured.update(top_k=top_k, category=category)
        return [{"id": 1, "score": 0.9, "question": "运费怎么算", "answer": "满99包邮",
                 "section_path": "", "content_type": "faq", "category": "运费"}]

    async def fake_acall(fn, *a, **k):
        return fn(*a, **k)

    monkeypatch.setattr(retrieval.milvus_client, "get_client", lambda *a, **k: object())
    monkeypatch.setattr(retrieval.milvus_client, "ensure_collection", lambda *a, **k: None)
    monkeypatch.setattr(retrieval.milvus_client, "dense_search", fake_dense)
    monkeypatch.setattr(retrieval.milvus_client, "acall", fake_acall)
    hits = asyncio.run(retrieval.search_knowledge(
        "邮费是多少", strategy="vector", top_k=3, category="运费"))
    assert hits[0]["answer"] == "满99包邮"
    assert captured == {"top_k": 3, "category": "运费"}


def test_search_knowledge_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        asyncio.run(retrieval.search_knowledge("x", strategy="foo"))
