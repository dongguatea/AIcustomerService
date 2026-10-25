import logging

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.runnables import Runnable

from app.core.llm import get_chat_model
from app.core.prompts import EXTRACT_PROMPT
from app.schemas.extract import AfterSalesTicket, ExtractRequest

logger = logging.getLogger(__name__)
router = APIRouter()


def get_extractor() -> Runnable:
    model = get_chat_model()
    # DeepSeek 不支持 json_schema response_format,用 function_calling(tool_choice)路径;
    # 配合 llm.py 里 chat_thinking=disabled,关闭思考模式后 tool_choice 才可用。
    return EXTRACT_PROMPT | model.with_structured_output(
        AfterSalesTicket, method="function_calling"
    )


@router.post("/api/extract", response_model=AfterSalesTicket)
async def extract(
    req: ExtractRequest, extractor: Runnable = Depends(get_extractor)
) -> AfterSalesTicket:
    try:
        return await extractor.ainvoke({"text": req.text})
    except Exception as exc:
        logger.exception("结构化提取失败")
        raise HTTPException(status_code=502, detail="上游模型暂时不可用,请稍后重试") from exc
