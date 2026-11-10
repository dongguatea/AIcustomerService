import asyncio

from app.config import settings
from app.core import rerank


RERANK_RESULTS = {
    "results": [
        {"index": 0, "relevance_score": 0.2},
        {"index": 2, "relevance_score": 0.9},
        {"index": 1, "relevance_score": 0.5},
    ]
}


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeAsyncClient:
    last_call = None

    def __init__(self):
        self._resp = FakeResponse(RERANK_RESULTS)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, json=None, headers=None, timeout=None):
        FakeAsyncClient.last_call = dict(url=url, json=json, headers=headers, timeout=timeout)
        return self._resp


def _install(monkeypatch):
    FakeAsyncClient.last_call = None
    monkeypatch.setattr(rerank.httpx, "AsyncClient", FakeAsyncClient)


def test_rerank_orders_and_truncates(monkeypatch):
    _install(monkeypatch)
    out = asyncio.run(rerank.rerank("q", ["a", "b", "c"], top_n=2))
    assert out == [(2, 0.9), (1, 0.5)]


def test_rerank_empty_docs_makes_no_request(monkeypatch):
    _install(monkeypatch)
    out = asyncio.run(rerank.rerank("q", []))
    assert out == []
    assert FakeAsyncClient.last_call is None


def test_rerank_url_and_payload(monkeypatch):
    _install(monkeypatch)
    asyncio.run(rerank.rerank("q", ["a", "b"], top_n=3))
    call = FakeAsyncClient.last_call
    assert call["url"].endswith("/rerank")
    assert call["json"]["model"] == settings.rerank_model
    assert call["json"]["documents"] == ["a", "b"]
    assert call["json"]["top_n"] == 3
    assert call["headers"]["Authorization"] == f"Bearer {settings.rerank_api_key}"
