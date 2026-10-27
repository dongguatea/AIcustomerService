import asyncio
from collections.abc import AsyncIterator, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.config import settings
from app.core.memory import trim_history
from app.core.prompts import CUSTOMER_SERVICE_PROMPT
from app.db.models import Message
from app.repositories import add_message, get_or_create_conversation, list_messages
from app.tools.business import current_conversation_id
from app.tools.registry import ToolRegistry

ModelFactory = Callable[[bool], BaseChatModel]


def _rebuild_history(rows: list[Message]) -> list[BaseMessage]:
    history: list[BaseMessage] = []
    for row in rows:
        if row.role == "user":
            history.append(HumanMessage(row.content))
        elif row.role == "assistant":
            if row.tool_call_request:
                history.append(
                    AIMessage(
                        content=row.content,
                        tool_calls=[
                            {
                                "name": row.tool_call_request.get("name"),
                                "args": row.tool_call_request.get("args", {}),
                                "id": row.tool_call_id,
                            }
                        ],
                    )
                )
            else:
                history.append(AIMessage(row.content))
        elif row.role == "tool":
            history.append(
                ToolMessage(content=row.content, tool_call_id=row.tool_call_id or "")
            )
    return history


class ChatService:
    def __init__(
        self,
        session_factory,
        registry: ToolRegistry,
        model_factory: ModelFactory,
        token_budget: int,
    ) -> None:
        self._session_factory = session_factory
        self._registry = registry
        self._model_factory = model_factory
        self._token_budget = token_budget

    async def stream(
        self, session_id: str, user: str | None, message: str
    ) -> AsyncIterator[dict]:
        def _conv():
            with self._session_factory() as session:
                return get_or_create_conversation(
                    session, session_id, user or session_id
                )

        conv = await asyncio.to_thread(_conv)
        token = current_conversation_id.set(conv.id)
        try:
            def _history():
                with self._session_factory() as session:
                    return list_messages(session, conv.id)

            rows = await asyncio.to_thread(_history)
            history = _rebuild_history(rows)
            history.append(HumanMessage(message))
            history = trim_history(history, max_tokens=self._token_budget)
            messages = CUSTOMER_SERVICE_PROMPT.format_messages(history=history)

            def _add_user():
                with self._session_factory() as session:
                    add_message(session, conv.id, "user", message)

            await asyncio.to_thread(_add_user)

            decision_model = self._model_factory(False).bind_tools(
                self._registry.as_list()
            )
            ai_msg = await decision_model.ainvoke(messages)

            if ai_msg.tool_calls:
                tc = ai_msg.tool_calls[0]
                name = tc.get("name") or ""
                args = tc.get("args") or {}
                tc_id = tc.get("id") or ""
                yield {
                    "status": "tool_call",
                    "tool": name,
                    "tool_call_id": tc_id,
                    "args": args,
                }

                def _add_assistant_tool():
                    with self._session_factory() as session:
                        add_message(
                            session,
                            conv.id,
                            "assistant",
                            "",
                            {"name": name, "args": args},
                            tc_id,
                        )

                await asyncio.to_thread(_add_assistant_tool)

                result = await self._registry.execute(
                    name,
                    args,
                    timeout=settings.tool_timeout,
                    max_retries=settings.tool_max_retries,
                )

                def _add_tool():
                    with self._session_factory() as session:
                        add_message(session, conv.id, "tool", result, None, tc_id)

                await asyncio.to_thread(_add_tool)

                final_messages = [
                    *messages,
                    ai_msg,
                    ToolMessage(content=result, tool_call_id=tc_id),
                ]
                chunks: list[str] = []
                async for chunk in self._model_factory(True).astream(final_messages):
                    text = chunk.content if isinstance(chunk.content, str) else ""
                    if text:
                        chunks.append(text)
                        yield {"delta": text}

                def _add_assistant_final():
                    with self._session_factory() as session:
                        add_message(session, conv.id, "assistant", "".join(chunks))

                await asyncio.to_thread(_add_assistant_final)
            else:
                content = ai_msg.content if isinstance(ai_msg.content, str) else ""
                for i in range(0, len(content), 4):
                    yield {"delta": content[i : i + 4]}

                def _add_assistant_final():
                    with self._session_factory() as session:
                        add_message(session, conv.id, "assistant", content)

                await asyncio.to_thread(_add_assistant_final)
        finally:
            current_conversation_id.reset(token)
