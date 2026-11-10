from pydantic import BaseModel

from app.core import llm


class _Out(BaseModel):
    x: int


def test_get_chat_model_temperature_default():
    assert llm.get_chat_model().temperature == 0.3


def test_get_chat_model_temperature_override():
    assert llm.get_chat_model(temperature=0).temperature == 0


def test_structured_calls_with_structured_output(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.called = None

        def with_structured_output(self, schema):
            self.called = schema
            return self

    fake = FakeModel()
    monkeypatch.setattr(llm, "get_chat_model", lambda streaming=False: fake)
    llm.structured(_Out)
    assert fake.called is _Out
