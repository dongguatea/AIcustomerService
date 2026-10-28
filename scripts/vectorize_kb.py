import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.kb import dualwrite, milvus_client


async def main() -> None:
    client = milvus_client.get_client()
    milvus_client.ensure_collection(client)
    n = await dualwrite.vectorize_pending(client)
    print(f"本次向量化 {n} 块;Milvus 现有 {milvus_client.count(client)} 条")


if __name__ == "__main__":
    asyncio.run(main())
