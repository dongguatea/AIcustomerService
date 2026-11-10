import asyncio

from app.tools import business


def _hit(id, **kw):
    h = {"id": id, "score": 0.9, "rerank_score": 0.9, "question": f"q{id}",
         "answer": f"a{id}", "section_path": f"s{id}", "content_type": "faq",
         "category": "c"}
    h.update(kw)
    return h


def _install(monkeypatch, hits, useful=True, check_reason=""):
    async def fake_understand(query):
        return {"standard": "运费怎么算", "expanded": ["邮费"]}

    async def fake_search(query, strategy="hybrid_rerank", top_k=None, category=None,
                          client=None, collection="knowledge", bm25_text=None):
        fake_search.calls.append({"query": query, "category": category, "bm25_text": bm25_text})
        return hits

    async def fake_check(query, evidence_texts):
        return {"useful": useful, "reason": check_reason}

    fake_search.calls = []
    monkeypatch.setattr(business.query_understanding, "understand", fake_understand)
    monkeypatch.setattr(business.retrieval, "search_knowledge", fake_search)
    monkeypatch.setattr(business.selfcheck, "check_sufficient", fake_check)
    return fake_search


def test_query_faq_returns_numbered_evidence(monkeypatch):
    hits = [_hit(1), _hit(2)]
    _install(monkeypatch, hits)
    result = asyncio.run(business.run_query_faq("邮费怎么算"))
    assert "[1]" in result.evidence and "[2]" in result.evidence
    assert result.sufficient is True
    assert [c["n"] for c in result.citations] == [1, 2]
    assert result.citations[0]["id"] == 1


def test_query_faq_mechanical_gate(monkeypatch):
    _install(monkeypatch, [])
    result = asyncio.run(business.run_query_faq("你们卖火星车吗"))
    assert result.sufficient is False
    assert result.source == "retrieval_low_conf"


def test_query_faq_selfcheck_gate(monkeypatch):
    _install(monkeypatch, [_hit(1)], useful=False, check_reason="证据不够")
    result = asyncio.run(business.run_query_faq("你们卖火星车吗"))
    assert result.sufficient is False
    assert result.source == "self_check"
    assert result.reason == "证据不够"


def test_query_faq_head_tail_and_citations(monkeypatch):
    hits = [_hit(1), _hit(2), _hit(3), _hit(4)]
    _install(monkeypatch, hits)
    result = asyncio.run(business.run_query_faq("邮费怎么算"))
    # arrange_head_tail([1,2,3,4]) = [1,3,4,2]
    assert [c["id"] for c in result.citations] == [1, 3, 4, 2]
    assert result.evidence.startswith("[1] q1: a1")


def test_category_filter_fallback(monkeypatch):
    _install(monkeypatch, [])

    async def fake_search_with_hits(query, strategy="hybrid_rerank", top_k=None, category=None,
                                    client=None, collection="knowledge", bm25_text=None):
        fake_search_with_hits.calls.append(category)
        return [] if category else [_hit(1)]

    fake_search_with_hits.calls = []
    monkeypatch.setattr(business.retrieval, "search_knowledge", fake_search_with_hits)
    result = asyncio.run(business.run_query_faq("邮费怎么算", category="运费"))
    assert "[1]" in result.evidence
    assert fake_search_with_hits.calls == ["运费", None]
    assert result.sufficient is True
