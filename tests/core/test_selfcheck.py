import asyncio

from pydantic import BaseModel

from app.core import selfcheck as sc


class _FakeCheck(BaseModel):
    useful: bool = True
    reason: str = ""


def test_check_sufficient_true(monkeypatch):
    async def fake_model(messages):
        return _FakeCheck(useful=True, reason="证据够")

    monkeypatch.setattr(
        sc.llm, "structured",
        lambda schema: fake_model,
    )
    out = asyncio.run(sc.check_sufficient("运费怎么算", ["满99元包邮,不满收取10元运费"]))
    assert out == {"useful": True, "reason": "证据够"}


def test_check_sufficient_fails_closed(monkeypatch):
    async def fake_model(messages):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        sc.llm, "structured",
        lambda schema: fake_model,
    )
    out = asyncio.run(sc.check_sufficient("运费怎么算", ["满99元包邮"]))
    assert out == {"useful": False, "reason": "证据自评失败"}
