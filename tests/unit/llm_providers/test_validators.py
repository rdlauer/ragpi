import pytest
from pydantic import ValidationError

from src.config import Settings


def _settings(**overrides) -> Settings:
    base: dict = dict(OPENAI_API_KEY="test-key")
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


def test_responses_api_allowed_with_openai() -> None:
    settings = _settings(CHAT_PROVIDER="openai", CHAT_USE_RESPONSES_API=True)
    assert settings.CHAT_USE_RESPONSES_API is True


def test_responses_api_rejected_for_non_openai_provider() -> None:
    with pytest.raises(ValidationError, match="only supported when CHAT_PROVIDER is 'openai'"):
        _settings(
            CHAT_PROVIDER="deepseek",
            DEEPSEEK_API_KEY="dk",
            CHAT_USE_RESPONSES_API=True,
        )


def test_store_false_rejected_only_when_responses_enabled() -> None:
    with pytest.raises(ValidationError, match="not yet supported"):
        _settings(
            CHAT_PROVIDER="openai",
            CHAT_USE_RESPONSES_API=True,
            OPENAI_RESPONSES_STORE=False,
        )


def test_store_false_ignored_when_responses_disabled() -> None:
    # An irrelevant OPENAI_RESPONSES_STORE=false must not block a legacy deployment.
    settings = _settings(
        CHAT_PROVIDER="deepseek",
        DEEPSEEK_API_KEY="dk",
        OPENAI_RESPONSES_STORE=False,
    )
    assert settings.CHAT_USE_RESPONSES_API is False
