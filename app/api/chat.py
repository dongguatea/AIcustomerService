import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.config import settings
from app.core.llm import get_chat_model
from app.db.session import SessionLocal
from app.schemas.chat import ChatRequest
from app.services.chat import ChatService
from app.tools.business import TOOLS
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)
router = APIRouter()

_registry = ToolRegistry()
for _tool in TOOLS:
    _registry.register(_tool)


def get_chat_service() -> ChatService:
    return ChatService(SessionLocal, _registry, get_chat_model, settings.token_budget)


@router.post("/api/chat")
async def chat(req: ChatRequest, service: ChatService = Depends(get_chat_service)):
    async def event_stream() -> AsyncIterator[str]:
        try:
            async for frame in service.stream(req.session_id, req.user, req.message):
                yield f"data: {json.dumps(frame, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception:
            logger.exception("聊天处理失败 session_id=%s", req.session_id)
            yield "event: error\n"
            yield (
                "data: "
                + json.dumps({"message": "上游模型暂时不可用,请稍后重试"}, ensure_ascii=False)
                + "\n\n"
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")
