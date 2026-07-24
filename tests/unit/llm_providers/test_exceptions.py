from unittest.mock import Mock

import pytest
from openai import APIError

from src.common.exceptions import KnownException, ResourceNotFoundException, ResourceType
from src.llm_providers.exceptions import handle_openai_client_error


def _api_error(message: str, **body) -> APIError:
    return APIError(request=Mock(), message=message, body=body or None)


def test_directional_error_advises_enabling_responses() -> None:
    err = _api_error(
        "Function tools with reasoning_effort are not supported for gpt-5.6-sol in "
        "/v1/chat/completions. To use function tools, use /v1/responses or set "
        "reasoning_effort to 'none'.",
        param="reasoning_effort",
        type="invalid_request_error",
    )
    with pytest.raises(KnownException, match="Responses API"):
        handle_openai_client_error(err, "gpt-5.6-sol")


def test_responses_only_error_does_not_advise_enabling_responses() -> None:
    # Responses is already enabled and the model isn't supported there — the handler
    # must NOT tell the operator to enable Responses (it would be contradictory).
    err = _api_error(
        "The model is not supported in /v1/responses", type="invalid_request_error"
    )
    with pytest.raises(KnownException) as exc:
        handle_openai_client_error(err, "some-model")
    assert "CHAT_USE_RESPONSES_API" not in str(exc.value)


def test_reasoning_effort_on_responses_error_does_not_advise_enabling() -> None:
    # reasoning_effort rejected on /v1/responses (Responses already on), telling us to
    # use chat/completions — must NOT advise enabling Responses despite the keywords.
    err = _api_error(
        "reasoning_effort is not supported in /v1/responses; use chat/completions",
        param="reasoning_effort",
        type="invalid_request_error",
    )
    with pytest.raises(KnownException) as exc:
        handle_openai_client_error(err, "gpt-5.6-sol")
    assert "CHAT_USE_RESPONSES_API" not in str(exc.value)


def test_model_not_found_still_maps_to_resource_not_found() -> None:
    err = _api_error("Model not found", code="model_not_found", param="model", type="not_found")
    with pytest.raises(ResourceNotFoundException) as exc:
        handle_openai_client_error(err, "missing-model")
    assert exc.value.resource_type == ResourceType.MODEL
