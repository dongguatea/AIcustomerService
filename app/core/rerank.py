"""直连上游 /rerank 调 bge-reranker-v2-m3。非 OpenAI 协议,手写请求。"""
import re

import httpx

from app.config import settings

_VERSION_SEG = re.compile(r"/v\d+$")


def _rerank_url() -> str:
    """把 rerank_base_url 拼成 /rerank 端点;地址缺版本段(如只填域名)时补 /v1。"""
    base = settings.rerank_base_url.rstrip("/")
    if not _VERSION_SEG.search(base):
        base += "/v1"
    return base + "/rerank"


async def rerank(query: str, docs: list[str], top_n: int | None = None) -> list[tuple[int, float]]:
    """返回 [(原始索引, 相关分)],按相关分降序,截 top_n;空输入直接返回空。"""
    if not docs:
        return []
    payload = {
        "model": settings.rerank_model,
        "query": query,
        "documents": docs,
        "top_n": top_n or len(docs),
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            _rerank_url(),
            json=payload,
            headers={"Authorization": f"Bearer {settings.rerank_api_key}"},
            timeout=60,
        )
    resp.raise_for_status()
    results = resp.json()["results"]
    ranked = sorted(
        ((r["index"], float(r["relevance_score"])) for r in results),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:top_n] if top_n else ranked
