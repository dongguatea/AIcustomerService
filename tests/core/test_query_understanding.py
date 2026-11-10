import asyncio

from pydantic import BaseModel

from app.core import query_understanding as qu


class _FakeRewrite(BaseModel):
    standard: str = ""
    expanded: list[str] = []


def test_understand_rewrite(monkeypatch):
    async def fake_model(messages):
        return _FakeRewrite(standard="运费怎么算", expanded=["邮费", "运费"])

    monkeypatch.setattr(
        qu.llm, "structured",
        lambda schema: fake_model,
    )
    out = asyncio.run(qu.understand("邮费咋算的呀"))
    assert out == {"standard": "运费怎么算", "expanded": ["邮费", "运费"]}


def test_understand_falls_back_on_error(monkeypatch):
    async def fake_model(messages):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        qu.llm, "structured",
        lambda schema: fake_model,
    )
    out = asyncio.run(qu.understand("邮费咋算的呀"))
    assert out == {"standard": "邮费咋算的呀", "expanded": []}
