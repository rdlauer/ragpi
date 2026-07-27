import logging
from openai import APIError

from src.common.exceptions import (
    KnownException,
    ResourceNotFoundException,
    ResourceType,
)

logger = logging.getLogger(__name__)


def handle_openai_client_error(e: APIError, model: str) -> None:
    # OpenAI Model Not Found
    if e.code == "model_not_found":
        raise ResourceNotFoundException(
            ResourceType.MODEL,
            model,
            f"Model '{model}' not found, or you do not have access to it.",
        )

    # OpenAI model not supporting 'system' prompt.
    # TODO: Test if system prompt update is required for o3
    if "does not support 'system' with this model" in e.message:
        raise KnownException(f"Model '{model}' is not supported.")

    # OpenAI model not supporting 'tools'
    if "'tools is not supported in this model" in e.message:
        raise KnownException(f"Model '{model}' is not supported.")

    # Deepseek Model Not Found
    if "Model Not Exists" in e.message:
        raise ResourceNotFoundException(
            ResourceType.MODEL,
            model,
            f"Model '{model}' not found, or you do not have access to it.",
        )

    # Deepseek model not supporting 'tools'
    if "does not support Function Calling" in e.message:
        raise KnownException(f"Model '{model}' is not supported.")

    # Reasoning model that needs the Responses API for function tools + active
    # reasoning. Key on param/type + a narrow message check (OpenAI's `code` is
    # often null here) so this only fires on a genuine endpoint-capability error.
    message = (e.message or "").lower()
    # Only fire on an explicit *directional* phrase that the request should move TO the
    # Responses API. Anything looser (a bare "v1/responses", or reasoning_effort + a
    # "chat/completions" mention) misfires on errors like "reasoning_effort is not
    # supported in /v1/responses; use chat/completions" when Responses is already on.
    if (
        "use /v1/responses" in message
        or "use the responses api" in message
        # e.g. "This model is only supported in v1/responses and not in v1/chat/completions."
        or "only supported in v1/responses" in message
        or "only supported in /v1/responses" in message
    ):
        raise KnownException(
            f"Model '{model}' requires the OpenAI Responses API for this request. Enable it "
            "with CHAT_USE_RESPONSES_API=true (and CHAT_PROVIDER=openai) to use reasoning models."
        )

    logging.error(e)

    raise KnownException(
        "Error calling model. Verify the model exists, you have access, and it supports function/tool calling."
    )
