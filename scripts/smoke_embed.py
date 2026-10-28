import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core import embeddings


async def main() -> None:
    vec = await embeddings.embed_query("你好")
    print(f"嵌入维度 {len(vec)};前 5 位 {vec[:5]}")


if __name__ == "__main__":
    asyncio.run(main())
