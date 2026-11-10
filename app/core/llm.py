from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable

from app.config import settings


def get_chat_model(streaming: bool = False, temperature: float = 0.3) -> ChatOpenAI:
    """直连聊天上游。地址、模型名、密钥全从 .env 来,换上游不动这里。"""
    extra_body = {}
    if settings.chat_thinking:
        extra_body["thinking"] = {"type": settings.chat_thinking}
    return ChatOpenAI(
        model=settings.chat_model,
        base_url=settings.chat_base_url,
        api_key=settings.chat_api_key,
        streaming=streaming,
        temperature=temperature,
        extra_body=extra_body,
    )


def structured(schema) -> Runnable:
    """返回带结构化输出的 runnable(基于 Pydantic schema)。评估裁判传 temperature=0。"""
    return get_chat_model(streaming=False).with_structured_output(schema)
