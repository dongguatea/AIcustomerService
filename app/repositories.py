import time
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import (
    Conversation,
    Faq,
    KnowledgeChunk,
    LowConfidenceQuestion,
    Message,
    QaExtractionStaging,
    Ticket,
)


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


def insert_knowledge_chunk(
    session: Session,
    category: str,
    questions: str,
    answer: str,
    *,
    section_path: str | None = None,
    content_type: str | None = None,
    is_key_clause: int = 0,
) -> int:
    chunk = KnowledgeChunk(
        category=category,
        questions=questions,
        answer=answer,
        section_path=section_path,
        content_type=content_type,
        is_key_clause=is_key_clause,
    )
    session.add(chunk)
    session.commit()
    session.refresh(chunk)
    return chunk.id


def set_chunk_neighbors(
    session: Session, chunk_id: int, prev_id: int | None, next_id: int | None
) -> None:
    chunk = session.get(KnowledgeChunk, chunk_id)
    chunk.prev_chunk_id = prev_id
    chunk.next_chunk_id = next_id
    session.commit()


def list_pending_chunks(session: Session) -> list[KnowledgeChunk]:
    return list(
        session.scalars(
            select(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == "pending")
            .order_by(KnowledgeChunk.id)
        )
    )


def mark_chunk_vectorized(session: Session, chunk_id: int, vector_id: str) -> None:
    chunk = session.get(KnowledgeChunk, chunk_id)
    chunk.vectorize_status = "done"
    chunk.vector_id = vector_id
    session.commit()


def insert_staging(
    session: Session, batch_no: str, source_ref: str | None, question: str, answer: str
) -> int:
    row = QaExtractionStaging(
        batch_no=batch_no, source_ref=source_ref, question=question, answer=answer
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row.id


def list_staging_by_status(session: Session, status: str) -> list[QaExtractionStaging]:
    return list(
        session.scalars(
            select(QaExtractionStaging)
            .where(QaExtractionStaging.status == status)
            .order_by(QaExtractionStaging.id)
        )
    )


def set_staging_status(session: Session, ids: list[int], status: str) -> None:
    for row_id in ids:
        row = session.get(QaExtractionStaging, row_id)
        if row is not None:
            row.status = status
    session.commit()


def list_all_questions(session: Session) -> list[str]:
    out: list[str] = []
    for questions in session.scalars(select(KnowledgeChunk.questions)):
        out.extend(questions.split("\n"))
    return out


def list_conversations_with_messages(
    session: Session,
) -> list[tuple[int, list[Message]]]:
    conversations = list(
        session.scalars(select(Conversation).order_by(Conversation.id))
    )
    out: list[tuple[int, list[Message]]] = []
    for conv in conversations:
        messages = list(
            session.scalars(
                select(Message)
                .where(Message.conversation_id == conv.id)
                .order_by(Message.id)
            )
        )
        out.append((conv.id, messages))
    return out


def count_chunks_by_content_types(
    session: Session, content_types: set[str]
) -> int:
    return len(
        list(
            session.scalars(
                select(KnowledgeChunk.id).where(
                    KnowledgeChunk.content_type.in_(content_types)
                )
            )
        )
    )


def insert_low_confidence(
    session: Session,
    conversation_id: int | None,
    raw_question: str,
    source: str,
    reason: str | None,
) -> int:
    row = LowConfidenceQuestion(
        conversation_id=conversation_id,
        raw_question=raw_question,
        source=source,
        reason=reason,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row.id
