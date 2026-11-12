import asyncio

from app.core import intent


def test_classify_returns_intent(monkeypatch):
    async def fake_model(messages):
        return intent._Intent(intent="物流", confidence=0.9)

    monkeypatch.setattr(intent.llm, "structured", lambda schema: fake_model)
    assert asyncio.run(intent.classify("订单1001到哪了")) == {
        "intent": "物流",
        "confidence": 0.9,
    }


def test_classify_failure_returns_none(monkeypatch):
    async def fake_model(messages):
        raise RuntimeError("down")

    monkeypatch.setattr(intent.llm, "structured", lambda schema: fake_model)
    assert asyncio.run(intent.classify("随便")) == {"intent": None, "confidence": 0.0}

