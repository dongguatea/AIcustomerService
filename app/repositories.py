import time
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Conversation, Faq, Message, Ticket


def get_or_create_conversation(
    session: Session, session_id: str, user: str
) -> Conversation:
    conv = session.scalar(
        select(Conversation).where(Conversation.session_id == session_id)
    )
    if conv is None:
        conv = Conversation(session_id=session_id, user=user)
        session.add(conv)
        session.commit()
        session.refresh(conv)
    return conv


def add_message(
    session: Session,
    conversation_id: int,
    role: str,
    content: str,
    tool_call_request: dict | None = None,
    tool_call_id: str | None = None,
) -> Message:
    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        tool_call_request=tool_call_request,
        tool_call_id=tool_call_id,
    )
    session.add(msg)
    session.commit()
    session.refresh(msg)
    return msg


def list_messages(session: Session, conversation_id: int) -> list[Message]:
    return list(
        session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
        )
    )


def search_faq(session: Session, keyword: str) -> list[Faq]:
    return list(
        session.scalars(select(Faq).where(Faq.question.like(f"%{keyword}%")))
    )


def create_ticket(
    session: Session, conversation_id: int, description: str, ticket_type: str
) -> Ticket:
    ticket = Ticket(
        ticket_no=f"TKT-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6].upper()}",
        conversation_id=conversation_id,
        description=description,
        ticket_type=ticket_type,
    )
    session.add(ticket)
    session.commit()
    session.refresh(ticket)
    return ticket
