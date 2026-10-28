from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages


def trim_history(messages: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
    return trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=max_tokens,
        start_on="human",
        allow_partial=False,
    )
