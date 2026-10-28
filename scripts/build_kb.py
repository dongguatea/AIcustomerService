import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.session import SessionLocal
from app.kb import documents, dualwrite
from app.kb.sources import KB_DIR, SOURCE_TYPES as DOCS
from app.repositories import count_chunks_by_content_types


async def main() -> None:
    def _count() -> int:
        with SessionLocal() as s:
            return count_chunks_by_content_types(s, set(DOCS.values()))

    existing = await asyncio.to_thread(_count)
    if existing:
        print(f"已存在 {existing} 条文档块,跳过以防重复插入。重建请先: make kb-reset")
        return
    total = 0
    for fname, ctype in DOCS.items():
        md = (KB_DIR / fname).read_text(encoding="utf-8")
        chunks = documents.build_chunks(md, content_type=ctype)
        ids = await dualwrite.write_pending(chunks)
        total += len(ids)
        print(f"  {fname}: {len(ids)} 块")
    print(f"建库(pending):共 {total} 块。下一步: make kb-vectorize")


if __name__ == "__main__":
    asyncio.run(main())
