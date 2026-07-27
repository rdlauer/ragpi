from typing import Literal
from pydantic import BaseModel, Field

from src.config import get_settings

settings = get_settings()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]


class CreateChatRequest(BaseModel):
    sources: list[str] | None = None
    model: str = settings.DEFAULT_CHAT_MODEL
    # Optional per-request override; only used when the Responses API path is enabled,
    # otherwise ignored (does not alter the Chat Completions request).
    reasoning_effort: ReasoningEffort | None = None
    messages: list[ChatMessage] = Field(min_length=1)


class ChatResponse(BaseModel):
    message: str | None
