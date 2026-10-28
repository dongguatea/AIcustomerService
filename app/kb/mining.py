import asyncio
import logging
from datetime import datetime

from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.prompts import MINING_PROMPT
from app.db.session import SessionLocal
from app.kb import dedup
from app.repositories import (
    insert_knowledge_chunk,
    insert_staging,
    list_all_questions,
    list_conversations_with_messages,
    list_staging_by_status,
    set_staging_status,
)

logger = logging.getLogger(__name__)


class QaPair(BaseModel):
    question: str
    answer: str


class QaExtraction(BaseModel):
    """LLM 结构化输出 schema:并列标量数组一一对应。"""
    questions: list[str] = Field(
        default_factory=list, description="问法列表,与 answers 一一对应;无可复用问答则为空"
    )
    answers: list[str] = Field(
        default_factory=list, description="答案列表,与 questions 一一对应"
    )


async def extract_qa(conversation_texts: list[str], model=None) -> list[QaPair]:
    model = model or get_chat_model()
    chain = MINING_PROMPT | model.with_structured_output(
        QaExtraction, method="function_calling"
    )
    result: QaExtraction = await chain.ainvoke(
        {"conversations": "\n---\n".join(conversation_texts)}
    )
    qs, ans = result.questions, result.answers
    if len(qs) != len(ans):
        logger.warning("挖知识并列数组长度不一致 questions=%d answers=%d", len(qs), len(ans))
    n = min(len(qs), len(ans))
    return [QaPair(question=qs[i], answer=ans[i]) for i in range(n)]


async def _load_conversation_texts() -> list[tuple[str, str]]:
    def _run():
        with SessionLocal() as s:
            out: list[tuple[str, str]] = []
            for conv_id, msgs in list_conversations_with_messages(s):
                lines = [f"{m.role}: {m.content}" for m in msgs if m.content]
                if lines:
                    out.append((f"conv:{conv_id}", "\n".join(lines)))
            return out
    return await asyncio.to_thread(_run)


async def mine(batch_size: int = 20, model=None) -> dict:
    """可重跑 job:对话 → 分批 LLM 抽取写 staging → 整体去重 → kept 入 knowledge_chunks。"""
    sources = await _load_conversation_texts()
    batch_no = datetime.now().strftime("mine-%Y%m%d%H%M%S")

    for start in range(0, len(sources), batch_size):
        batch = sources[start:start + batch_size]
        pairs = await extract_qa([t for _, t in batch], model=model)
        if not pairs:
            continue
        source_ref = batch[0][0]

        def _insert(pairs=pairs):
            with SessionLocal() as s:
                for p in pairs:
                    insert_staging(s, batch_no, source_ref, p.question, p.answer)
        await asyncio.to_thread(_insert)

    def _dedup():
        with SessionLocal() as s:
            staged = [x for x in list_staging_by_status(s, "extracted")
                      if x.batch_no == batch_no]
            existing = list_all_questions(s)
            return dedup.dedupe(staged, existing)
    kept, discarded = await asyncio.to_thread(_dedup)

    def _finalize():
        with SessionLocal() as s:
            set_staging_status(s, [x.id for x in kept], "kept")
            set_staging_status(s, [x.id for x in discarded], "discarded")
            for p in kept:
                insert_knowledge_chunk(
                    s, "对话挖知识", p.question, p.answer,
                    section_path="对话挖知识", content_type="qa_mined",
                )
    await asyncio.to_thread(_finalize)

    return {"sources": len(sources), "extracted": len(kept) + len(discarded),
            "kept": len(kept), "discarded": len(discarded)}
