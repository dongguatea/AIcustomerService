from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api import actions as actions_api
from app.db.models import Ticket
from app.main import app


def test_create_ticket_writes_row(monkeypatch, session_factory):
    monkeypatch.setattr(actions_api, "SessionLocal", session_factory)
    client = TestClient(app)
    resp = client.post(
        "/api/actions/create-ticket",
        json={"session_id": "s1", "description": "退货运费", "ticket_type": "投诉"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticket_no"].startswith("TKT-")
    with session_factory() as s:
        row = s.scalars(select(Ticket)).one()
        assert row.description == "退货运费"
        assert row.ticket_type == "投诉"
        assert row.ticket_no == body["ticket_no"]


def test_create_ticket_requires_fields(monkeypatch, session_factory):
    monkeypatch.setattr(actions_api, "SessionLocal", session_factory)
    client = TestClient(app)
    resp = client.post(
        "/api/actions/create-ticket",
        json={"session_id": "s1", "description": "x"},
    )
    assert resp.status_code == 422
