import json

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk

from app.api import chat as chat_api
from app.main import app
from app.repositories import list_messages
from app.services.chat import ChatService


class FakeModel:
    def __init__(self, tool_calls=None, answer="好的"):
        self._tool_calls = tool_calls
        self._answer = answer

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        return AIMessage(
            content="" if self._tool_calls else self._answer,
            tool_calls=self._tool_calls or [],
        )

    async def astream(self, messages):
        for ch in self._answer:
            yield AIMessageChunk(content=ch)


def parse_sse(resp):
    frames, last, error = [], "", None
    for line in resp.iter_lines():
        if line.startswith("event: "):
            error = line[len("event: "):]
        elif line.startswith("data: "):
            payload = line[len("data: "):]
            if payload == "[DONE]":
                last = payload
            else:
                frames.append(json.loads(payload))
    return frames, last, error


def make_client(session_factory, registry, model):
    svc = ChatService(session_factory, registry, lambda _: model, 2000)
    app.dependency_overrides[chat_api.get_chat_service] = lambda: svc
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def test_chat_api_streams_and_persists(session_factory, registry):
    client = make_client(session_factory, registry, FakeModel(answer="您好呀"))
    with client.stream("POST", "/api/chat", json={"session_id": "s1", "message": "在吗"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        frames, last, error = parse_sse(resp)
    assert "".join(f["delta"] for f in frames) == "您好呀"
    assert last == "[DONE]" and error is None
    with session_factory() as s:
        assert [m.content for m in list_messages(s, 1)] == ["在吗", "您好呀"]


def test_chat_api_emits_tool_status_frame(session_factory, registry):
    client = make_client(
        session_factory,
        registry,
        FakeModel(
            tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}],
            answer="订单在路上",
        ),
    )
    with client.stream("POST", "/api/chat", json={"session_id": "s2", "message": "订单 1001 到哪了"}) as resp:
        frames, last, error = parse_sse(resp)
    tool = [f for f in frames if f.get("status") == "tool_call"]
    assert tool and tool[0]["tool"] == "query_order"
    assert "".join(f["delta"] for f in frames if "delta" in f) == "订单在路上"


def test_chat_api_upstream_error_sends_error_frame(session_factory, registry):
    class Boom:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            raise RuntimeError("down")

        async def astream(self, messages):
            raise RuntimeError("down")
            yield

    client = make_client(session_factory, registry, Boom())
    with client.stream("POST", "/api/chat", json={"session_id": "s3", "message": "hi"}) as resp:
        frames, last, error = parse_sse(resp)
    assert error == "error"
    assert last != "[DONE]"
