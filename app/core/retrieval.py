from app.config import settings
from app.core import embeddings, rerank
from app.kb import milvus_client

_STRATEGIES = ("vector", "bm25", "hybrid", "hybrid_rerank")


def arrange_head_tail(items: list) -> list:
    """最相关放首、次相关放尾,其余按序居中(缓解 lost-in-the-middle)。"""
    if len(items) <= 2:
        return items
    return [items[0], *items[2:], items[1]]


async def search_knowledge(
    query: str,
    strategy: str = "hybrid_rerank",
    top_k: int | None = None,
    category: str | None = None,
    client=None,
    collection: str = "knowledge",
    bm25_text: str | None = None,
) -> list[dict]:
    """四策略检索。dense 用干净 query,BM25 用 bm25_text(缺省=query)。
    hybrid_rerank 的 hit 多带 rerank_score;首尾组装由调用侧按需处理。"""
    if strategy not in _STRATEGIES:
        raise ValueError(f"未知 strategy={strategy!r}, 应为 {'/'.join(_STRATEGIES)}")
    top_k = top_k or settings.rerank_top_k
    bt = bm25_text or query

    if strategy == "bm25":
        def work():
            c = client or milvus_client.get_client()
            milvus_client.ensure_collection(c, collection=collection)
            return milvus_client.bm25_search(c, bt, top_k, category, collection)
        return await milvus_client.acall(work)

    if strategy == "vector":
        vec = await embeddings.embed_query(query)

        def work():
            c = client or milvus_client.get_client()
            milvus_client.ensure_collection(c, collection=collection)
            return milvus_client.dense_search(c, vec, top_k, category, collection)
        return await milvus_client.acall(work)

    # hybrid / hybrid_rerank:先两路各召回 recall_top_k 融合,再截断/精排
    vec = await embeddings.embed_query(query)

    def work():
        c = client or milvus_client.get_client()
        milvus_client.ensure_collection(c, collection=collection)
        return milvus_client.hybrid_search(
            c, vec, bt, top_k=settings.recall_top_k, recall=settings.recall_top_k,
            category=category, collection=collection)
    hits = await milvus_client.acall(work)

    if strategy == "hybrid":
        return hits[:top_k]

    docs = [f"{h['question']} {h['answer']}" for h in hits]
    ranked = await rerank.rerank(query, docs, top_n=top_k)
    out: list[dict] = []
    for idx, score in ranked:
        h = dict(hits[idx])
        h["rerank_score"] = score
        out.append(h)
    return out
