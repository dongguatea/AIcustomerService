import asyncio

from fastapi import APIRouter

from app.db.session import SessionLocal
from app.repositories import create_ticket as _create_ticket
from app.repositories import get_or_create_conversation
from app.schemas.actions import CreateTicketRequest, CreateTicketResponse

router = APIRouter()


@router.post("/api/actions/create-ticket", response_model=CreateTicketResponse)
async def create_ticket(req: CreateTicketRequest):
    """点「建工单」才写 tickets 表(前端自选触发,后端不自动建单)。"""

    def _run() -> str:
        with SessionLocal() as s:
            conv = get_or_create_conversation(s, req.session_id, req.session_id)
            return _create_ticket(s, conv.id, req.description, req.ticket_type).ticket_no

    ticket_no = await asyncio.to_thread(_run)
    return CreateTicketResponse(ticket_no=ticket_no, status="open")
