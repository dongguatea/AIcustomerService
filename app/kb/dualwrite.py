import asyncio

from app.core import embeddings
from app.db.session import SessionLocal
from app.kb import milvus_client
from app.kb.documents import Chunk
from app.repositories import (
    insert_knowledge_chunk,
    list_pending_chunks,
    mark_chunk_vectorized,
    set_chunk_neighbors,
)


async def write_pending(chunks: list[Chunk]) -> list[int]:
    """按顺序把一份文档的 chunks 写入 MySQL(pending),并在同文档内连 prev/next 指针。"""
    def _run() -> list[int]:
        with SessionLocal() as s:
            ids = [insert_knowledge_chunk(
                s, c.category, c.questions, c.answer,
                section_path=c.section_path, content_type=c.content_type,
                is_key_clause=c.is_key_clause,
            ) for c in chunks]
            for i, cid in enumerate(ids):
                prev_id = ids[i - 1] if i > 0 else None
                next_id = ids[i + 1] if i < len(ids) - 1 else None
                set_chunk_neighbors(s, cid, prev_id, next_id)
            return ids
    return await asyncio.to_thread(_run)


def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


async def vectorize_pending(
    client=None, batch_size: int = 64, collection: str = milvus_client.COLLECTION
) -> int:
    """幂等可重跑:取 pending → 拼三字段文本 → 嵌入 → Milvus upsert(PK=id) → 回填 done。"""
    def _list_pending():
        with SessionLocal() as s:
            return list_pending_chunks(s)
    pending = await asyncio.to_thread(_list_pending)
    done = 0
    for batch in _batches(pending, batch_size):
        texts = [f"{r.category}\n{r.questions}\n{r.answer}" for r in batch]
        vectors = await embeddings.embed_texts(texts)
        rows = [
            {"id": r.id, "dense": v, "text": t,
             "question": r.questions, "answer": r.answer,
             "section_path": r.section_path or "", "content_type": r.content_type or "",
             "category": r.category or ""}
            for r, v, t in zip(batch, vectors, texts)
        ]

        def work_upsert(rows=rows):
            c = client or milvus_client.get_client()
            milvus_client.ensure_collection(c, collection=collection)
            milvus_client.upsert_vectors(c, rows, collection=collection)
        await milvus_client.acall(work_upsert)

        def _mark_done():
            with SessionLocal() as s:
                for r in batch:
                    mark_chunk_vectorized(s, r.id, str(r.id))
        await asyncio.to_thread(_mark_done)
        done += len(batch)

    if done:
        def work_flush():
            c = client or milvus_client.get_client()
            milvus_client.flush(c, collection=collection)
        await milvus_client.acall(work_flush)
    return done
