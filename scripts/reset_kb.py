import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import delete

from app.db.models import KnowledgeChunk, QaExtractionStaging
from app.db.session import SessionLocal
from app.kb import milvus_client


def main() -> None:
    client = milvus_client.get_client()
    milvus_client.drop(client, milvus_client.COLLECTION)
    with SessionLocal() as s:
        s.execute(delete(KnowledgeChunk))
        s.execute(delete(QaExtractionStaging))
        s.commit()
    print("已清空 knowledge_chunks / qa_extraction_staging 并 drop Milvus 集合 knowledge")


if __name__ == "__main__":
    main()
