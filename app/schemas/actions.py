from pydantic import BaseModel, Field


class CreateTicketRequest(BaseModel):
    session_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    ticket_type: str = Field(min_length=1)


class CreateTicketResponse(BaseModel):
    ticket_no: str
    status: str = "open"

