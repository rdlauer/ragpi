from __future__ import annotations
from typing import TYPE_CHECKING

from src.llm_providers.constants import ChatProvider, EmbeddingProvider

if TYPE_CHECKING:
    from src.config import Settings


def validate_provider_settings(settings: Settings):
    # OpenAI
    if settings.CHAT_PROVIDER == ChatProvider.OPENAI and not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY must be set when CHAT_PROVIDER is 'openai'")

    if (
        settings.EMBEDDING_PROVIDER == EmbeddingProvider.OPENAI
        and not settings.OPENAI_API_KEY
    ):
        raise ValueError(
            "OPENAI_API_KEY must be set when EMBEDDING_PROVIDER is 'openai'"
        )

    # OLLAMA
    if settings.CHAT_PROVIDER == ChatProvider.OLLAMA and not settings.OLLAMA_BASE_URL:
        raise ValueError("OLLAMA_BASE_URL must be set when CHAT_PROVIDER is 'ollama'")

    if (
        settings.EMBEDDING_PROVIDER == EmbeddingProvider.OLLAMA
        and not settings.OLLAMA_BASE_URL
    ):
        raise ValueError(
            "OLLAMA_BASE_URL must be set when EMBEDDING_PROVIDER is 'ollama'"
        )

    # DeepSeek
    if (
        settings.CHAT_PROVIDER == ChatProvider.DEEPSEEK
        and not settings.DEEPSEEK_API_KEY
    ):
        raise ValueError(
            "DEEPSEEK_API_KEY must be set when CHAT_PROVIDER is 'deepseek'"
        )

    # OpenAI Compatible
    if (
        settings.CHAT_PROVIDER == ChatProvider.OPENAI_COMPATIBLE
        and not settings.CHAT_OPENAI_COMPATIBLE_BASE_URL
    ):
        raise ValueError(
            "CHAT_OPENAI_COMPATIBLE_BASE_URL must be set when CHAT_PROVIDER is 'openai-compatible'"
        )

    if (
        settings.CHAT_PROVIDER == ChatProvider.OPENAI_COMPATIBLE
        and not settings.CHAT_OPENAI_COMPATIBLE_API_KEY
    ):
        raise ValueError(
            "CHAT_OPENAI_COMPATIBLE_API_KEY must be set when CHAT_PROVIDER is 'openai-compatible'"
        )

    if (
        settings.EMBEDDING_PROVIDER == EmbeddingProvider.OPENAI_COMPATIBLE
        and not settings.EMBEDDING_OPENAI_COMPATIBLE_BASE_URL
    ):
        raise ValueError(
            "EMBEDDING_OPENAI_COMPATIBLE_BASE_URL must be set when EMBEDDING_PROVIDER is 'openai-compatible'"
        )

    if (
        settings.EMBEDDING_PROVIDER == EmbeddingProvider.OPENAI_COMPATIBLE
        and not settings.EMBEDDING_OPENAI_COMPATIBLE_API_KEY
    ):
        raise ValueError(
            "EMBEDDING_OPENAI_COMPATIBLE_API_KEY must be set when EMBEDDING_PROVIDER is 'openai-compatible'"
        )

    # Responses API interlock: only OpenAI implements /v1/responses, so this path can
    # never be enabled for deepseek/ollama/openai-compatible providers.
    if settings.CHAT_USE_RESPONSES_API and settings.CHAT_PROVIDER != ChatProvider.OPENAI:
        raise ValueError(
            "CHAT_USE_RESPONSES_API is only supported when CHAT_PROVIDER is 'openai'. "
            "Deepseek, Ollama, and OpenAI-compatible providers do not implement the Responses API."
        )

    # Only reject the ZDR/manual-replay case when the Responses path is actually on,
    # so an irrelevant setting never blocks a legacy Chat Completions deployment.
    if settings.CHAT_USE_RESPONSES_API and not settings.OPENAI_RESPONSES_STORE:
        raise ValueError(
            "OPENAI_RESPONSES_STORE=false (Zero-Data-Retention / manual reasoning replay) "
            "is not yet supported; leave it true."
        )

    return settings
