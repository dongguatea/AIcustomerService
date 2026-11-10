from datetime import datetime, timezone

from sqlalchemy import JSON, BigInteger, DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# BIGINT 主键在 SQLite 不自增(SQLite 只对 INTEGER PRIMARY KEY 自增),用 variant 在
# SQLite 落到 Integer、MySQL 保持 BIGINT。
BIGINT = BigInteger().with_variant(Integer, "sqlite")


class Faq(Base):
    __tablename__ = "faq"

    id: Mapped[int] = mapped_column(primary_key=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(50))


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(String(128), unique=True)
    user: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), default="open")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"))
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    tool_call_request: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class Ticket(Base):
    __tablename__ = "tickets"

    ticket_no: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"))
    description: Mapped[str] = mapped_column(Text)
    ticket_type: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), default="open")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String(255))
    questions: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    section_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    is_key_clause: Mapped[int] = mapped_column(Integer, default=0)
    prev_chunk_id: Mapped[int | None] = mapped_column(
        BIGINT, ForeignKey("knowledge_chunks.id", ondelete="SET NULL"), nullable=True
    )
    next_chunk_id: Mapped[int | None] = mapped_column(
        BIGINT, ForeignKey("knowledge_chunks.id", ondelete="SET NULL"), nullable=True
    )
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    vectorize_status: Mapped[str] = mapped_column(String(16), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )


class QaExtractionStaging(Base):
    __tablename__ = "qa_extraction_staging"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    batch_no: Mapped[str] = mapped_column(String(64))
    source_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="extracted")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class LowConfidenceQuestion(Base):
    __tablename__ = "low_confidence_questions"

    id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversations.id"), nullable=True
    )
    raw_question: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(
        Enum("retrieval_low_conf", "self_check", "user_feedback")
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
