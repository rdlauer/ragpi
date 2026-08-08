from datetime import datetime
from types import SimpleNamespace
import pytest
from pytest_mock import MockerFixture
from openai import APIError, OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.responses import ResponseFunctionToolCall

from src.chat.service import ChatService
from src.chat.schemas import ChatMessage, ChatResponse, CreateChatRequest
from src.chat.tools.definitions import TOOL_DEFINITIONS
from src.common.exceptions import KnownException, ResourceNotFoundException, ResourceType
from src.document_store.schemas import Document
from src.sources.service import SourceService


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(
            id="1",
            url="test.com",
            title="Test title 1",
            content="Test content 1",
            created_at=datetime(2024, 1, 1, 0, 0, 0),
        ),
        Document(
            id="2",
            url="test.com",
            title="Test title 2",
            content="Test content 2",
            created_at=datetime(2024, 1, 2, 0, 0, 0),
        ),
    ]


@pytest.fixture
def mock_source_service(mocker: MockerFixture) -> SourceService:
    return mocker.Mock(spec=SourceService)


@pytest.fixture
def mock_openai_client(mocker: MockerFixture) -> OpenAI:
    client = mocker.Mock(spec=OpenAI)
    client.chat = mocker.Mock()
    client.chat.completions = mocker.Mock()
    client.responses = mocker.Mock()
    return client


@pytest.fixture
def chat_service(
    mock_source_service: SourceService,
    mock_openai_client: OpenAI,
) -> ChatService:
    return ChatService(
        source_service=mock_source_service,
        openai_client=mock_openai_client,
        project_name="Test Project",
        project_description="This is a test project.",
        base_system_prompt="You are a helpful assistant.",
        tool_definitions=[],
        chat_history_limit=10,
        max_iterations=3,
        retrieval_top_k=5,
    )


@pytest.fixture
def sample_chat_input() -> CreateChatRequest:
    return CreateChatRequest(
        messages=[
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!"),
            ChatMessage(role="user", content="What is the weather?"),
        ],
        model="test-model",
        sources=["source1", "source2"],
    )


def test_generate_response_direct_answer(
    chat_service: ChatService,
    mock_openai_client: OpenAI,
    sample_chat_input: CreateChatRequest,
    mocker: MockerFixture,
) -> None:
    mock_completion = ChatCompletion(
        id="test-id",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content="This is a direct answer",
                    role="assistant",
                ),
            )
        ],
        created=1234567890,
        model="gpt-4",
        object="chat.completion",
    )
    mock_create_completion = mocker.patch.object(
        mock_openai_client.chat.completions,
        "create",
        return_value=mock_completion,
    )

    response = chat_service.generate_response(sample_chat_input)

    assert isinstance(response, ChatResponse)
    assert response.message == "This is a direct answer"
    mock_create_completion.assert_called_once()


def test_generate_response_with_tool_calls(
    chat_service: ChatService,
    mock_openai_client: OpenAI,
    mock_source_service: SourceService,
    sample_chat_input: CreateChatRequest,
    sample_documents: list[Document],
    mocker: MockerFixture,
) -> None:
    # Mock first response with tool call
    tool_call_completion = ChatCompletion(
        id="test-id-1",
        choices=[
            Choice(
                finish_reason="tool_calls",
                index=0,
                message=ChatCompletionMessage(
                    content=None,
                    role="assistant",
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call-1",
                            type="function",
                            function=Function(
                                name="retrieve_documents",
                                arguments='{"source_name": "source1", "semantic_query": "test semantic query", "full_text_query": "test full text query"}',
                            ),
                        )
                    ],
                ),
            )
        ],
        created=1234567890,
        model="gpt-4",
        object="chat.completion",
    )

    # Mock final response with answer
    final_completion = ChatCompletion(
        id="test-id-2",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content="Here is the answer based on the search",
                    role="assistant",
                ),
            )
        ],
        created=1234567890,
        model="gpt-4",
        object="chat.completion",
    )

    mock_create_completion = mocker.patch.object(
        mock_openai_client.chat.completions,
        "create",
        side_effect=[tool_call_completion, final_completion],
    )

    # Mock source service search results
    mock_search_source = mocker.patch.object(
        mock_source_service,
        "search_source",
        return_value=sample_documents,
    )

    response = chat_service.generate_response(sample_chat_input)

    assert isinstance(response, ChatResponse)
    assert response.message == "Here is the answer based on the search"
    assert mock_create_completion.call_count == 2
    mock_search_source.assert_called_once_with(
        source_name="source1",
        semantic_query="test semantic query",
        full_text_query="test full text query",
        top_k=5,
    )


def test_generate_response_max_iterations_exceeded(
    chat_service: ChatService,
    mock_openai_client: OpenAI,
    mock_source_service: SourceService,
    sample_chat_input: CreateChatRequest,
    sample_documents: list[Document],
    mocker: MockerFixture,
) -> None:
    # Mock tool call response that keeps searching
    tool_call_completion = ChatCompletion(
        id="test-id",
        choices=[
            Choice(
                finish_reason="tool_calls",
                index=0,
                message=ChatCompletionMessage(
                    content=None,
                    role="assistant",
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call-1",
                            type="function",
                            function=Function(
                                name="retrieve_documents",
                                arguments='{"source_name": "source1", "semantic_query": "test semantic query", "full_text_query": "test full text query"}',
                            ),
                        )
                    ],
                ),
            )
        ],
        created=1234567890,
        model="gpt-4",
        object="chat.completion",
    )

    mock_create_completion = mocker.patch.object(
        mock_openai_client.chat.completions,
        "create",
        return_value=tool_call_completion,
    )

    # Mock source service search results
    mocker.patch.object(
        mock_source_service,
        "search_source",
        return_value=sample_documents,
    )

    response = chat_service.generate_response(sample_chat_input)

    assert isinstance(response, ChatResponse)
    assert (
        response.message
        == "I'm sorry, but I don't have the information you're looking for."
    )
    assert mock_create_completion.call_count == 3


def test_generate_response_model_not_found(
    chat_service: ChatService,
    mock_openai_client: OpenAI,
    sample_chat_input: CreateChatRequest,
    mocker: MockerFixture,
) -> None:
    mocker.patch.object(
        mock_openai_client.chat.completions,
        "create",
        side_effect=APIError(
            request=mocker.Mock(),
            message="Model not found",
            body={"code": "model_not_found", "param": "model", "type": "not_found"},
        ),
    )

    with pytest.raises(ResourceNotFoundException) as exc_info:
        chat_service.generate_response(sample_chat_input)

    assert exc_info.value.resource_type == ResourceType.MODEL
    assert exc_info.value.identifier == sample_chat_input.model


# --------------------------------------------------------------------------- #
# Responses API path (opt-in, dual-path)                                      #
# --------------------------------------------------------------------------- #
@pytest.fixture
def responses_chat_service(
    mock_source_service: SourceService,
    mock_openai_client: OpenAI,
) -> ChatService:
    # Real tool definitions so the tool path (and flat tool schema) is exercised.
    return ChatService(
        source_service=mock_source_service,
        openai_client=mock_openai_client,
        project_name="Test Project",
        project_description="This is a test project.",
        base_system_prompt="You are a helpful assistant.",
        tool_definitions=TOOL_DEFINITIONS,
        chat_history_limit=10,
        max_iterations=3,
        retrieval_top_k=5,
        use_responses_api=True,
        reasoning_effort=None,
        responses_store=True,
    )


def _fake_response(*, output: list, output_text: str, response_id: str) -> SimpleNamespace:
    return SimpleNamespace(output=output, output_text=output_text, id=response_id)


def _function_call(call_id: str, source: str) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        type="function_call",
        call_id=call_id,
        name="retrieve_documents",
        arguments=(
            f'{{"source_name": "{source}", "semantic_query": "sq", "full_text_query": "ftq"}}'
        ),
    )


def test_responses_direct_answer(
    responses_chat_service: ChatService,
    mock_openai_client: OpenAI,
    sample_chat_input: CreateChatRequest,
    mocker: MockerFixture,
) -> None:
    create = mocker.patch.object(
        mock_openai_client.responses,
        "create",
        return_value=_fake_response(output=[], output_text="Direct answer", response_id="r1"),
    )

    response = responses_chat_service.generate_response(sample_chat_input)

    assert response.message == "Direct answer"
    create.assert_called_once()
    kwargs = create.call_args.kwargs
    assert kwargs["store"] is True
    assert kwargs["instructions"]  # system prompt passed via instructions=
    assert "tools" in kwargs and kwargs["tools"][0]["type"] == "function"
    assert "reasoning" not in kwargs  # no effort configured


def test_responses_with_tool_calls(
    responses_chat_service: ChatService,
    mock_openai_client: OpenAI,
    mock_source_service: SourceService,
    sample_chat_input: CreateChatRequest,
    sample_documents: list[Document],
    mocker: MockerFixture,
) -> None:
    create = mocker.patch.object(
        mock_openai_client.responses,
        "create",
        side_effect=[
            _fake_response(
                output=[_function_call("fc-1", "source1")], output_text="", response_id="r1"
            ),
            _fake_response(output=[], output_text="Answer from search", response_id="r2"),
        ],
    )
    search = mocker.patch.object(
        mock_source_service, "search_source", return_value=sample_documents
    )

    response = responses_chat_service.generate_response(sample_chat_input)

    assert response.message == "Answer from search"
    assert create.call_count == 2
    search.assert_called_once_with(
        source_name="source1", semantic_query="sq", full_text_query="ftq", top_k=5
    )
    # Second call continues the same server-side response and sends the tool output.
    second = create.call_args_list[1].kwargs
    assert second["previous_response_id"] == "r1"
    assert second["input"][0]["type"] == "function_call_output"
    assert second["input"][0]["call_id"] == "fc-1"


def test_responses_max_iterations_exceeded(
    responses_chat_service: ChatService,
    mock_openai_client: OpenAI,
    mock_source_service: SourceService,
    sample_chat_input: CreateChatRequest,
    sample_documents: list[Document],
    mocker: MockerFixture,
) -> None:
    create = mocker.patch.object(
        mock_openai_client.responses,
        "create",
        return_value=_fake_response(
            output=[_function_call("fc-x", "source1")], output_text="", response_id="r1"
        ),
    )
    mocker.patch.object(
        mock_source_service, "search_source", return_value=sample_documents
    )

    response = responses_chat_service.generate_response(sample_chat_input)

    assert (
        response.message
        == "I'm sorry, but I don't have the information you're looking for."
    )
    assert create.call_count == 3


def test_responses_reasoning_effort_forwarded(
    responses_chat_service: ChatService,
    mock_openai_client: OpenAI,
    mocker: MockerFixture,
) -> None:
    create = mocker.patch.object(
        mock_openai_client.responses,
        "create",
        return_value=_fake_response(output=[], output_text="ok", response_id="r1"),
    )
    request = CreateChatRequest(
        messages=[ChatMessage(role="user", content="hi")],
        model="gpt-5.6-sol",
        reasoning_effort="max",
        sources=["source1"],
    )

    responses_chat_service.generate_response(request)

    assert create.call_args.kwargs["reasoning"] == {"effort": "max"}


def test_flat_tool_schema_does_not_mutate_chat_tools(
    responses_chat_service: ChatService,
) -> None:
    responses_tool = responses_chat_service.responses_tools[0]
    assert responses_tool["type"] == "function"
    assert responses_tool["name"] == "retrieve_documents"
    assert "function" not in responses_tool  # flat, not nested
    # The chat-completions tool keeps its nested shape (deep copy protected it).
    assert "function" in dict(responses_chat_service.tools[0])


def test_reasoning_model_error_on_chat_path_maps_to_actionable_message(
    chat_service: ChatService,
    mock_openai_client: OpenAI,
    sample_chat_input: CreateChatRequest,
    mocker: MockerFixture,
) -> None:
    # The production case: the DEFAULT Chat Completions path (Responses OFF) rejects
    # function tools + active reasoning, and we guide the operator to enable Responses.
    mocker.patch.object(
        mock_openai_client.chat.completions,
        "create",
        side_effect=APIError(
            request=mocker.Mock(),
            message=(
                "Function tools with reasoning_effort are not supported for gpt-5.6-sol in "
                "/v1/chat/completions. To use function tools, use /v1/responses or set "
                "reasoning_effort to 'none'."
            ),
            body={"code": None, "param": "reasoning_effort", "type": "invalid_request_error"},
        ),
    )

    with pytest.raises(KnownException, match="Responses API"):
        chat_service.generate_response(sample_chat_input)


def test_reasoning_effort_ignored_when_responses_disabled(
    chat_service: ChatService,
    mock_openai_client: OpenAI,
    mocker: MockerFixture,
) -> None:
    # chat_service has use_responses_api=False (the default / legacy path).
    completion = ChatCompletion(
        id="c1",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(content="answer", role="assistant"),
            )
        ],
        created=1,
        model="gpt-4o",
        object="chat.completion",
    )
    create = mocker.patch.object(
        mock_openai_client.chat.completions, "create", return_value=completion
    )
    responses_create = mocker.patch.object(mock_openai_client.responses, "create")

    request = CreateChatRequest(
        messages=[ChatMessage(role="user", content="hi")],
        model="gpt-4o",
        reasoning_effort="high",  # supplied, but path is off
        sources=["source1"],
    )
    response = chat_service.generate_response(request)

    assert response.message == "answer"
    create.assert_called_once()
    responses_create.assert_not_called()  # legacy path never touches client.responses
    # reasoning_effort must not leak into the Chat Completions request.
    assert "reasoning" not in create.call_args.kwargs
    assert "reasoning_effort" not in create.call_args.kwargs


def test_responses_refusal_is_returned_as_message(
    responses_chat_service: ChatService,
    mock_openai_client: OpenAI,
    sample_chat_input: CreateChatRequest,
    mocker: MockerFixture,
) -> None:
    refusal_block = SimpleNamespace(type="refusal", refusal="I can't help with that.")
    message_item = SimpleNamespace(type="message", content=[refusal_block])
    mocker.patch.object(
        mock_openai_client.responses,
        "create",
        return_value=SimpleNamespace(output=[message_item], output_text="", id="r1"),
    )

    response = responses_chat_service.generate_response(sample_chat_input)

    assert response.message == "I can't help with that."


def test_responses_incomplete_maps_to_known_exception(
    responses_chat_service: ChatService,
    mock_openai_client: OpenAI,
    sample_chat_input: CreateChatRequest,
    mocker: MockerFixture,
) -> None:
    mocker.patch.object(
        mock_openai_client.responses,
        "create",
        return_value=SimpleNamespace(
            output=[],
            output_text="",
            id="r1",
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        ),
    )

    with pytest.raises(KnownException, match="incomplete: max_output_tokens"):
        responses_chat_service.generate_response(sample_chat_input)


def test_responses_incomplete_with_function_call_maps_to_known_exception(
    responses_chat_service: ChatService,
    mock_openai_client: OpenAI,
    mock_source_service: SourceService,
    sample_chat_input: CreateChatRequest,
    mocker: MockerFixture,
) -> None:
    # A max_output_tokens-truncated response can carry a function call whose JSON
    # arguments are cut off. Completeness must be checked BEFORE tool processing so
    # this maps to a clean 400 (KnownException), not a JSONDecodeError -> 500.
    truncated_call = ResponseFunctionToolCall(
        type="function_call",
        call_id="fc-trunc",
        name="retrieve_documents",
        arguments='{"source_name": "sou',  # truncated mid-JSON
    )
    mocker.patch.object(
        mock_openai_client.responses,
        "create",
        return_value=SimpleNamespace(
            output=[truncated_call],
            output_text="",
            id="r1",
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        ),
    )
    search = mocker.patch.object(mock_source_service, "search_source")

    with pytest.raises(KnownException, match="incomplete: max_output_tokens"):
        responses_chat_service.generate_response(sample_chat_input)
    search.assert_not_called()  # the truncated tool call never reached the executor


def test_unknown_tool_call_maps_to_known_exception(
    responses_chat_service: ChatService,
    mock_openai_client: OpenAI,
    sample_chat_input: CreateChatRequest,
    mocker: MockerFixture,
) -> None:
    hallucinated = ResponseFunctionToolCall(
        type="function_call", call_id="fc-1", name="delete_all_sources", arguments="{}"
    )
    mocker.patch.object(
        mock_openai_client.responses,
        "create",
        return_value=SimpleNamespace(
            output=[hallucinated], output_text="", id="r1", status="completed"
        ),
    )

    with pytest.raises(KnownException, match="unknown tool"):
        responses_chat_service.generate_response(sample_chat_input)


def test_malformed_tool_arguments_map_to_known_exception(
    responses_chat_service: ChatService,
    mock_openai_client: OpenAI,
    sample_chat_input: CreateChatRequest,
    mocker: MockerFixture,
) -> None:
    bad_args = ResponseFunctionToolCall(
        type="function_call",
        call_id="fc-1",
        name="retrieve_documents",
        arguments='{"wrong_field": true}',  # valid JSON, fails schema validation
    )
    mocker.patch.object(
        mock_openai_client.responses,
        "create",
        return_value=SimpleNamespace(
            output=[bad_args], output_text="", id="r1", status="completed"
        ),
    )

    with pytest.raises(KnownException, match="invalid tool call"):
        responses_chat_service.generate_response(sample_chat_input)


def test_nonexistent_source_in_tool_call_feeds_error_back_to_model(
    chat_service: ChatService,
    mock_openai_client: OpenAI,
    mock_source_service: SourceService,
    sample_chat_input: CreateChatRequest,
    mocker: MockerFixture,
) -> None:
    # The model may hallucinate a source name (e.g. one mentioned in prompt prose but
    # not registered). The error must be returned AS the tool result so the model can
    # self-correct — not propagate as a 404 for the whole chat request.
    bad_source_completion = ChatCompletion(
        id="c1",
        choices=[
            Choice(
                finish_reason="tool_calls",
                index=0,
                message=ChatCompletionMessage(
                    content=None,
                    role="assistant",
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call-1",
                            type="function",
                            function=Function(
                                name="retrieve_documents",
                                arguments='{"source_name": "no-such-source", '
                                '"semantic_query": "q", "full_text_query": "q"}',
                            ),
                        )
                    ],
                ),
            )
        ],
        created=1,
        model="gpt-4o",
        object="chat.completion",
    )
    final_completion = ChatCompletion(
        id="c2",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(content="answer", role="assistant"),
            )
        ],
        created=1,
        model="gpt-4o",
        object="chat.completion",
    )
    create = mocker.patch.object(
        mock_openai_client.chat.completions,
        "create",
        side_effect=[bad_source_completion, final_completion],
    )
    mocker.patch.object(
        mock_source_service,
        "search_source",
        side_effect=ResourceNotFoundException(ResourceType.SOURCE, "no-such-source"),
    )
    mocker.patch.object(
        mock_source_service,
        "list_sources",
        return_value=[SimpleNamespace(name="real-source", description="d")],
    )

    response = chat_service.generate_response(sample_chat_input)

    assert response.message == "answer"
    # The second call's messages include the error tool-result with the valid options.
    second_messages = create.call_args_list[1].kwargs["messages"]
    tool_msgs = [m for m in second_messages if isinstance(m, dict) and m.get("role") == "tool"]
    assert tool_msgs and "does not exist" in tool_msgs[-1]["content"]
    assert "real-source" in tool_msgs[-1]["content"]


def test_empty_messages_list_is_rejected_by_schema() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CreateChatRequest(messages=[], model="gpt-4o", sources=["s1"])


def test_responses_empty_output_maps_to_known_exception(
    responses_chat_service: ChatService,
    mock_openai_client: OpenAI,
    sample_chat_input: CreateChatRequest,
    mocker: MockerFixture,
) -> None:
    mocker.patch.object(
        mock_openai_client.responses,
        "create",
        return_value=SimpleNamespace(
            output=[], output_text="", id="r1", status="completed"
        ),
    )

    with pytest.raises(KnownException, match="no answer content"):
        responses_chat_service.generate_response(sample_chat_input)
