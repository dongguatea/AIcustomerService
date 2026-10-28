from app.config import settings
from app.core import embeddings
from app.kb import milvus_client


async def search_knowledge(
    query: str,
    strategy: str = "vector",
    top_k: int | None = None,
    category: str | None = None,
    client=None,
    collection: str = "knowledge",
) -> list[dict]:
    """ch03 仅 dense 单路:问题向量化后到 Milvus 按相似度取 Top-K。"""
    if strategy != "vector":
        raise ValueError(f"ch03 仅支持 strategy='vector', 收到 {strategy!r}")
    top_k = top_k or settings.dense_top_k
    vec = await embeddings.embed_query(query)

    def work():
        c = client or milvus_client.get_client()
        milvus_client.ensure_collection(c, collection=collection)
        return milvus_client.dense_search(c, vec, top_k, category, collection)

    return await milvus_client.acall(work)
