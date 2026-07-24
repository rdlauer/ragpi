import copy
import json
from typing import Any
from openai import APIError, OpenAI, pydantic_function_tool
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionMessageToolCall,
)
from openai.types.responses import ResponseFunctionToolCall

from src.chat.exceptions import ChatException
from src.chat.prompts import get_system_prompt
from src.chat.schemas import ChatResponse, CreateChatRequest
from src.chat.tools.definitions import ToolDefinition
from src.chat.tools.schamas import RetrieveDocuments
from src.common.exceptions import KnownException
from src.llm_providers.exceptions import handle_openai_client_error
from src.sources.metadata.schemas import SourceMetadata
from src.sources.service import SourceService


class ChatService:
    def __init__(
        self,
        *,
        source_service: SourceService,
        openai_client: OpenAI,
        project_name: str,
        project_description: str,
        base_system_prompt: str,
        tool_definitions: list[ToolDefinition],
        chat_history_limit: int,
        max_iterations: int,
        retrieval_top_k: int,
        use_responses_api: bool = False,
        reasoning_effort: str | None = None,
        responses_store: bool = True,
    ):
        self.chat_client = openai_client
        self.source_service = source_service
        self.project_name = project_name
        self.project_description = project_description
        self.base_system_prompt = base_system_prompt
        self.chat_history_limit = chat_history_limit
        self.max_iterations = max_iterations
        self.retrieval_top_k = retrieval_top_k
        self.use_responses_api = use_responses_api
        self.reasoning_effort = reasoning_effort
        self.responses_store = responses_store
        self.tools = [
            pydantic_function_tool(
                model=tool.model,
                name=tool.name,
                description=tool.description,
            )
            for tool in tool_definitions
        ]
        # Flat Responses-API tool schema, deep-copied from the chat-completions tools
        # so flattening can never mutate the (nested) chat schema.
        self.responses_tools: list[dict[str, Any]] = [
            self._to_responses_tool(tool) for tool in self.tools
        ]

    @staticmethod
    def _to_responses_tool(chat_tool: Any) -> dict[str, Any]:
        fn = copy.deepcopy(dict(chat_tool)["function"])
        return {
            "type": "function",
            "name": fn["name"],
            "description": fn.get("description"),
            "parameters": fn.get("parameters"),
            "strict": fn.get("strict", True),
        }

    def _get_sources(
        self, source_names: list[str] | None = None
    ) -> list[SourceMetadata]:
        """Retrieve and validate sources."""
        if not source_names:
            sources = self.source_service.list_sources()
        else:
            sources = [self.source_service.get_source(name) for name in source_names]

        if not sources:
            raise ChatException("No sources found.")

        return sources

    def _create_chat_messages(
        self,
        system_prompt: str,
        chat_history: list[ChatCompletionMessageParam],
        user_message: str,
    ) -> list[ChatCompletionMessageParam]:
        """Create the complete list of chat messages."""
        return [
            ChatCompletionSystemMessageParam(role="system", content=system_prompt),
            *chat_history,
            ChatCompletionUserMessageParam(role="user", content=user_message),
        ]

    def _retrieve_documents(self, name: str, arguments: str) -> str:
        """Run the retrieve_documents tool and return its JSON result. Shared by the
        Chat Completions and Responses tool-call loops (only the surrounding message
        shapes differ)."""
        if name != "retrieve_documents":
            raise ValueError(f"Unknown tool call: {name}")

        args = json.loads(arguments)
        source_input = RetrieveDocuments(**args)
        documents = self.source_service.search_source(
            source_name=source_input.source_name,
            semantic_query=source_input.semantic_query,
            full_text_query=source_input.full_text_query,
            top_k=self.retrieval_top_k,
        )
        return json.dumps(
            [{"url": doc.url, "content": doc.content} for doc in documents]
        )

    def _handle_tool_call(
        self,
        tool_call: ChatCompletionMessageToolCall,
    ) -> ChatCompletionToolMessageParam:
        """Handle a Chat Completions tool call."""
        content = self._retrieve_documents(
            tool_call.function.name, tool_call.function.arguments
        )
        return ChatCompletionToolMessageParam(
            tool_call_id=tool_call.id,
            content=content,
            role="tool",
        )

    def generate_response(self, chat_input: CreateChatRequest) -> ChatResponse:
        """Generate a response based on chat input, dispatching to the opt-in
        Responses API path (OpenAI reasoning models) or the default Chat Completions
        path used by every other provider/model."""
        try:
            sources = self._get_sources(chat_input.sources)
            system_prompt = get_system_prompt(
                project_name=self.project_name,
                project_description=self.project_description,
                base_prompt=self.base_system_prompt,
                sources=sources,
                max_attempts=self.max_iterations,
            )

            if self.use_responses_api:
                return self._generate_via_responses(chat_input, system_prompt)
            return self._generate_via_chat_completions(chat_input, system_prompt)

        except ChatException as e:
            raise KnownException(str(e))
        except APIError as e:
            handle_openai_client_error(e, chat_input.model)
            raise e

    def _generate_via_chat_completions(
        self, chat_input: CreateChatRequest, system_prompt: str
    ) -> ChatResponse:
        # Prepare chat history
        chat_history: list[ChatCompletionMessageParam] = [
            (
                ChatCompletionUserMessageParam(role="user", content=msg.content)
                if msg.role == "user"
                else ChatCompletionAssistantMessageParam(
                    role="assistant", content=msg.content
                )
            )
            for msg in chat_input.messages[-self.chat_history_limit : -1]
        ]

        messages = self._create_chat_messages(
            system_prompt, chat_history, chat_input.messages[-1].content
        )

        # Generate response
        for _ in range(self.max_iterations):
            response = self.chat_client.chat.completions.create(
                model=chat_input.model,
                messages=messages,
                tools=self.tools,
            )

            message = response.choices[0].message
            messages.append(message)  # type: ignore

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_response = self._handle_tool_call(tool_call)  # type: ignore
                    messages.append(tool_response)
            elif message.content:
                return ChatResponse(message=message.content)
            else:
                # ChatException (not a raw error) so this maps to a clean 400, not a 500.
                raise ChatException(
                    "The model returned no content or tool call."
                )

        return ChatResponse(
            message="I'm sorry, but I don't have the information you're looking for."
        )

    def _generate_via_responses(
        self, chat_input: CreateChatRequest, system_prompt: str
    ) -> ChatResponse:
        """Reasoning + tools via the OpenAI Responses API. Reasoning continuity is
        preserved across the internal tool-call loop (previous_response_id) within
        this single request; conversation history is passed as input items."""
        effort = chat_input.reasoning_effort or self.reasoning_effort

        input_items: list[dict[str, Any]] = [
            {"role": msg.role, "content": msg.content}
            for msg in chat_input.messages[-self.chat_history_limit : -1]
        ]
        input_items.append(
            {"role": "user", "content": chat_input.messages[-1].content}
        )

        kwargs: dict[str, Any] = {}
        if self.responses_tools:
            kwargs["tools"] = self.responses_tools
        if effort:
            kwargs["reasoning"] = {"effort": effort}

        previous_response_id: str | None = None
        for _ in range(self.max_iterations):
            response = self.chat_client.responses.create(
                model=chat_input.model,
                instructions=system_prompt,
                input=input_items,  # type: ignore[arg-type]
                store=self.responses_store,
                previous_response_id=previous_response_id,
                **kwargs,
            )

            function_calls = [
                item
                for item in response.output
                if isinstance(item, ResponseFunctionToolCall)
            ]

            if function_calls:
                previous_response_id = response.id
                input_items = []
                for call in function_calls:
                    output = self._retrieve_documents(call.name, call.arguments)
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": output,
                        }
                    )
                continue

            # No tool calls: a normal answer, a safeguard refusal, an incomplete
            # response (e.g. token exhaustion), or genuinely empty output. Map the
            # non-answer cases to ChatException (clean 400) rather than a raw 500.
            if response.output_text:
                return ChatResponse(message=response.output_text)
            refusal = self._extract_refusal(response)
            if refusal:
                return ChatResponse(message=refusal)
            if getattr(response, "status", None) == "incomplete":
                details = getattr(response, "incomplete_details", None)
                reason = getattr(details, "reason", None) or "unknown"
                raise ChatException(
                    f"The model could not complete the response (incomplete: {reason})."
                )
            raise ChatException("The model returned no answer content.")

        return ChatResponse(
            message="I'm sorry, but I don't have the information you're looking for."
        )

    @staticmethod
    def _extract_refusal(response: Any) -> str | None:
        """Return the refusal text if the model declined the request, else None."""
        for item in response.output:
            if getattr(item, "type", None) == "message":
                for block in getattr(item, "content", None) or []:
                    if getattr(block, "type", None) == "refusal":
                        return getattr(block, "refusal", None)
        return None
