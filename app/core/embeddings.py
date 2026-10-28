from openai import AsyncOpenAI

from app.config import settings

_CLIENT: AsyncOpenAI | None = None


def _client() -> AsyncOpenAI:
    """直连嵌入上游(硅基流动的 bge-m3,OpenAI 兼容),进程内单例。"""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    if not settings.embed_api_key:
        raise RuntimeError("EMBED_API_KEY 未配置,请先写入 .env")
    _CLIENT = AsyncOpenAI(base_url=settings.embed_base_url, api_key=settings.embed_api_key)
    return _CLIENT


async def embed_texts(texts: list[str]) -> list[list[float]]:
    resp = await _client().embeddings.create(model=settings.embed_model, input=texts)
    return [d.embedding for d in resp.data]


async def embed_query(text: str) -> list[float]:
    return (await embed_texts([text]))[0]
