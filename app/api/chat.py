import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.graph.runtime import stream_turn
from app.schemas.chat import ChatRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/chat")
async def chat(req: ChatRequest):
    async def event_stream() -> AsyncIterator[str]:
        try:
            async for frame in stream_turn(req.session_id, req.user, req.message):
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
