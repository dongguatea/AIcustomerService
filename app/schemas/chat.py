from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, description="会话 ID,同一会话多轮复用")
    user: str | None = Field(default=None, description="客户标识,缺省用 session_id")
    message: str = Field(min_length=1, description="用户本轮消息")
