import json

from fastapi.testclient import TestClient

from app.api import chat as chat_api
from app.main import app


def parse_sse(resp):
    frames, last, error = [], "", None
    for line in resp.iter_lines():
        if line.startswith("event: "):
            error = line[len("event: ") :]
        elif line.startswith("data: "):
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                last = payload
            else:
                frames.append(json.loads(payload))
    return frames, last, error


def test_chat_api_streams_delta_and_done(monkeypatch):
    async def fake_stream(session_id, user, message):
        yield {"delta": "您好呀"}

    monkeypatch.setattr(chat_api, "stream_turn", fake_stream)
    client = TestClient(app)
    with client.stream(
        "POST", "/api/chat", json={"session_id": "s1", "message": "在吗"}
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        frames, last, error = parse_sse(resp)
    assert "".join(f["delta"] for f in frames) == "您好呀"
    assert last == "[DONE]" and error is None


def test_chat_api_passes_through_tool_frame(monkeypatch):
    async def fake_stream(session_id, user, message):
        yield {
            "status": "tool_call",
            "tool": "query_order",
            "tool_call_id": "c1",
            "args": {"order_id": "1001"},
        }
        yield {"delta": "订单在路上"}

    monkeypatch.setattr(chat_api, "stream_turn", fake_stream)
    client = TestClient(app)
    with client.stream(
        "POST", "/api/chat", json={"session_id": "s2", "message": "订单到哪了"}
    ) as resp:
        frames, last, error = parse_sse(resp)
    tool = [f for f in frames if f.get("status") == "tool_call"]
    assert tool and tool[0]["tool"] == "query_order"
    assert "".join(f["delta"] for f in frames if "delta" in f) == "订单在路上"


def test_chat_api_upstream_error_sends_error_frame(monkeypatch):
    async def fake_stream(session_id, user, message):
        raise RuntimeError("down")
        yield

    monkeypatch.setattr(chat_api, "stream_turn", fake_stream)
    client = TestClient(app)
    with client.stream(
        "POST", "/api/chat", json={"session_id": "s3", "message": "hi"}
    ) as resp:
        frames, last, error = parse_sse(resp)
    assert error == "error"
    assert last != "[DONE]"
