"""Anthropic Messages API client with multimodal and structured-output support."""

import base64
import json
import logging
from datetime import datetime
from typing import List, Tuple, Any, Optional

from anthropic import Anthropic

from .base_client import BaseAIClient
from .response import DiscardedAttempts, LLMResponse, Usage
from .pricing import apply_costs, calculate_cost_components
from .reasoning import uncounted_reasoning
from .utils import (
    attach_discarded,
    billed_cost,
    error_payload,
    extract_json_from_text,
    rejects_parameter,
    usage_counts,
    usage_of,
)

logger = logging.getLogger(__name__)

# Known model prefixes that reject temperature. _send handles other models at runtime.
TEMPERATURE_FREE_MODEL_PREFIXES = (
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
)

# Known model prefixes that reject forced tool use; start with an optional tool.
# Other models use the same fallback after a tool_choice rejection.
FORCED_TOOL_FREE_MODEL_PREFIXES = (
    "claude-fable-5-1",
    "claude-mythos-5-1",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
)

OPTIONAL_TOOL_DESCRIPTION = (
    "Record your final answer. Call this tool exactly once, after reading the whole input, "
    "with every item you extracted. There is no tool result to wait for."
)


def _unknown_keys(json_schema: dict, payload: Any) -> List[str]:
    """
    Detect undeclared top-level keys that Pydantic could silently ignore.

    Resolve a root #/$defs/ reference. Skip roots without a properties mapping,
    roots permitting additional properties or declaring nonempty patternProperties, and
    non-object payloads. Otherwise return sorted keys absent from properties.
    This heuristic does not validate values or match property patterns.
    """
    root = json_schema
    ref = root.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        root = json_schema.get("$defs", {}).get(ref[len("#/$defs/") :])
    if not isinstance(root, dict) or not isinstance(payload, dict):
        return []

    known = root.get("properties")
    if not isinstance(known, dict):
        return []
    if root.get("additionalProperties") not in (None, False) or root.get("patternProperties"):
        return []
    return sorted(key for key in payload if key not in known)


class ClaudeClient(BaseAIClient):
    """
    Anthropic client for text, images, caching, and structured output.

    Structured requests use a forced tool when supported. Models that reject it
    receive an optional tool with the same schema. Plain text is the final fallback.
    """

    PROVIDER_ID = "anthropic"
    SUPPORTS_MULTIMODAL = True
    SUPPORTS_TOOLS = True

    def _init_client(self):
        """Initialize the Anthropic client with the provided API key."""
        self.api_client = Anthropic(api_key=self.api_key, timeout=300.0)

    def _prepare_content_with_images(
        self,
        prompt: str,
        images: List[str],
        cache_images: bool = False,
        file_content: str = "",
        content_order=None,
    ) -> List[dict]:
        """
        Prepare Anthropic content with text, files, and images.

        Args:
            prompt: The text prompt
            images: List of image paths/URLs
            cache_images: Mark each image with cache_control for prompt caching
            file_content: Text content from files (empty string if none)
            content_order: Content ordering policy override

        Returns:
            List of content blocks for Anthropic API
        """
        prompt_parts = [{"type": "text", "text": prompt}]
        files_parts = [{"type": "text", "text": file_content}] if file_content else []
        image_parts = []

        for resource in images:
            try:
                if self.is_url(resource):
                    import requests

                    response = requests.get(resource)
                    if response.status_code == 200:
                        base64_image = base64.b64encode(response.content).decode("utf-8")
                    else:
                        logger.error(
                            f"Failed to fetch image from URL {resource}: {response.status_code}"
                        )
                        continue
                else:
                    with open(resource, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

                from .utils import detect_image_mime_type

                media_type = detect_image_mime_type(resource)

                image_block = {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_image,
                    },
                }

                if cache_images:
                    image_block["cache_control"] = {"type": "ephemeral"}
                    logger.info(f"Marking image for prompt caching: {resource}")

                image_parts.append(image_block)
            except Exception as e:
                logger.error(f"Error processing image {resource}: {e}")

        return self._order_content_parts(
            {"prompt": prompt_parts, "images": image_parts, "files": files_parts},
            content_order,
        )

    def _do_prompt(
        self,
        model: str,
        prompt: str,
        messages: Optional[List[dict]] = None,
        images: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        response_format: Optional[Any] = None,
        cache: bool = False,
        file_content: str = "",
        **kwargs,
    ) -> LLMResponse:
        """
        Send a prompt to the Claude model and get the response.

        Claude supports prompt caching via cache_control blocks.

        Args:
            model: The Claude model identifier (e.g., "claude-3-opus-20240229")
            prompt: The text prompt to send
            messages: Optional conversation history (multi-turn)
            images: List of image paths/URLs
            system_prompt: System prompt to use
            response_format: Optional Pydantic model for structured output
            cache: Enable cache_control blocks for files and images
            file_content: File content to mark for caching (when cache=True)
            **kwargs: Additional Claude-specific parameters

        Returns:
            LLMResponse object with the provider's response
        """
        images = images or []
        content_order = kwargs.pop("_content_order", None)

        # Build system blocks with optional caching
        system_blocks = []

        # When caching is active, files go to the system block (Claude-specific behaviour).
        # Otherwise they are passed as a "files" user-content slot below.
        file_content_for_user = ""
        if file_content and cache:
            system_blocks.append(
                {
                    "type": "text",
                    "text": f"Reference documents:\n\n{file_content}",
                    "cache_control": {"type": "ephemeral"},
                }
            )
        else:
            file_content_for_user = file_content

        # Add system prompt (not cached by default)
        if system_prompt:
            system_blocks.append({"type": "text", "text": system_prompt})

        # Build API messages
        if messages and len(messages) > 1:
            # Multi-turn conversation
            api_messages = []
            for i, msg in enumerate(messages):
                if (
                    msg["role"] == "user"
                    and i == len(messages) - 1
                    and (images or file_content_for_user)
                ):
                    image_parts = []
                    for resource in images:
                        try:
                            if self.is_url(resource):
                                import requests

                                response = requests.get(resource)
                                if response.status_code == 200:
                                    base64_image = base64.b64encode(response.content).decode(
                                        "utf-8"
                                    )
                                else:
                                    logger.error(
                                        f"Failed to fetch image from URL {resource}: {response.status_code}"
                                    )
                                    continue
                            else:
                                with open(resource, "rb") as image_file:
                                    base64_image = base64.b64encode(image_file.read()).decode(
                                        "utf-8"
                                    )

                            from .utils import detect_image_mime_type

                            media_type = detect_image_mime_type(resource)

                            image_block = {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_image,
                                },
                            }

                            if cache:
                                image_block["cache_control"] = {"type": "ephemeral"}
                                logger.info(
                                    f"Marking image for prompt caching in conversation: {resource}"
                                )

                            image_parts.append(image_block)
                        except Exception as e:
                            logger.error(f"Error processing image {resource}: {e}")

                    files_parts = (
                        [{"type": "text", "text": file_content_for_user}]
                        if file_content_for_user
                        else []
                    )
                    content_blocks = self._order_content_parts(
                        {
                            "prompt": [{"type": "text", "text": msg["content"]}],
                            "images": image_parts,
                            "files": files_parts,
                        },
                        content_order,
                    )
                    api_messages.append({"role": msg["role"], "content": content_blocks})
                else:
                    api_messages.append(msg)
        else:
            # Single-turn request
            content = self._prepare_content_with_images(
                prompt,
                images,
                cache_images=cache,
                file_content=file_content_for_user,
                content_order=content_order,
            )
            api_messages = [{"role": "user", "content": content}]

        # Determine max_tokens based on model
        if "opus" in model.lower():
            default_max_tokens = 4096
        elif "sonnet" in model.lower():
            default_max_tokens = 8192
        else:
            default_max_tokens = 4096

        # Extract Claude-specific parameters
        params = {
            "model": model,
            "messages": api_messages,
            "max_tokens": kwargs.get(
                "max_tokens", self.settings.get("max_tokens", default_max_tokens)
            ),
            "timeout": 300.0,
        }

        # Add system blocks or simple system prompt
        if system_blocks:
            params["system"] = system_blocks
        elif system_prompt:
            params["system"] = system_prompt

        # Add optional parameters if specified
        optional_params = ["temperature", "top_p", "top_k"]
        for param in optional_params:
            value = kwargs.get(param, self.settings.get(param))
            if value is not None:
                params[param] = value

        if self._rejects_temperature(model):
            params.pop("temperature", None)

        # Handle tool calling
        tool_definitions = kwargs.pop("_tool_definitions", None)
        if tool_definitions:
            # Convert to Claude tools format
            params["tools"] = [
                {
                    "name": tool.get("function", tool.get("name", "unknown")),
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("parameters", {}),
                }
                for tool in tool_definitions
            ]
            # Omitting tool_choice lets the model choose whether to call a tool.

        # Keep discarded usage local for thread safety. Outer retries re-enter this
        # method, so the accumulator covers only this invocation's fallback attempts.
        discarded = DiscardedAttempts()

        # Handle structured output using tools
        if response_format and hasattr(response_format, "model_json_schema"):
            json_schema = response_format.model_json_schema()

            tools = [
                {
                    "name": "extract_structured_data",
                    "description": "Extract structured data according to the provided schema",
                    "input_schema": json_schema,
                }
            ]

            forced_tool_refused = self._rejects_forced_tool(model)

            if not forced_tool_refused:
                response, forced_tool_refused = self._attempt_tool(
                    params,
                    tools,
                    {"type": "tool", "name": "extract_structured_data"},
                    self._create_response_from_tool,
                    (model, response_format),
                    model,
                    discarded,
                    "tools",
                )
                if response is not None:
                    return response

            if forced_tool_refused:
                # Preserve the original schema and identify the tool call as the final answer.
                optional_tools = [{**tools[0], "description": OPTIONAL_TOOL_DESCRIPTION}]
                response, _ = self._attempt_tool(
                    params,
                    optional_tools,
                    {"type": "auto"},
                    self._create_response_from_optional_tool,
                    (model, response_format, json_schema),
                    model,
                    discarded,
                    "optional tool",
                )
                if response is not None:
                    return response

        # Send the request to Anthropic
        try:
            raw_response = self._send(params, model)
        except Exception as e:
            raise attach_discarded(e, discarded)

        return self._build_response(
            self._create_response_from_raw, raw_response, model, discarded=discarded
        )

    @staticmethod
    def _rejects_temperature(model: str) -> bool:
        """Check whether the model matches a known temperature-rejecting prefix."""
        return model.split("/")[-1].lower().startswith(TEMPERATURE_FREE_MODEL_PREFIXES)

    @staticmethod
    def _rejects_forced_tool(model: str) -> bool:
        """Check whether the model matches a known forced-tool-rejecting prefix."""
        return model.split("/")[-1].lower().startswith(FORCED_TOOL_FREE_MODEL_PREFIXES)

    def _attempt_tool(
        self,
        params: dict,
        tools: List[dict],
        tool_choice: dict,
        builder,
        builder_args: tuple,
        model: str,
        discarded: DiscardedAttempts,
        label: str,
    ) -> Tuple[Optional[LLMResponse], bool]:
        """
        Attempt structured output through a tool and record reported usage on failure.

        Args:
            params: Request parameters; tools and tool_choice are removed on failure
            tools: Tool definitions for this attempt
            tool_choice: Tool choice for this attempt
            builder: Response builder passed to _build_response
            builder_args: Builder arguments after the raw response
            model: Model identifier
            discarded: Accumulator for this call
            label: Attempt name for logging

        Returns:
            The response or None, and whether tool_choice was rejected before a response
        """
        params["tools"] = tools
        params["tool_choice"] = tool_choice
        raw_response = None
        try:
            raw_response = self._send(params, model)
            return (
                self._build_response(builder, raw_response, *builder_args, discarded=discarded),
                False,
            )
        except Exception as e:
            wasted = self._record_discarded_attempt(
                discarded, e, raw_response, self.PROVIDER_ID, model
            )
            del params["tools"]
            del params["tool_choice"]
            # Parameter rejections before generation do not incur token usage.
            refused = raw_response is None and rejects_parameter(e, "tool_choice")
            if refused:
                logger.warning(
                    f"Model {model} refuses forced tool use; retrying with an optional tool. "
                    f"Add its prefix to FORCED_TOOL_FREE_MODEL_PREFIXES to skip this round trip."
                )
            else:
                logger.warning(
                    f"Structured output via {label} failed: {e}. Falling back to text mode."
                    f"{wasted} A second request will be billed."
                )
            return None, refused

    def _send(self, params: dict, model: str):
        """
        Send a request and retry once without temperature if the model rejects it.

        Args:
            params: Request parameters, corrected in place when the retry fires
            model: Model identifier, for the log line

        Returns:
            The provider's response
        """
        try:
            return self.api_client.messages.create(**params)
        except Exception as error:
            if not rejects_parameter(error, "temperature") or "temperature" not in params:
                raise
            del params["temperature"]
            logger.warning(
                f"Model {model} has retired the temperature parameter; retrying without "
                f"it. Add its prefix to TEMPERATURE_FREE_MODEL_PREFIXES to skip this "
                f"round trip."
            )
            return self.api_client.messages.create(**params)

    @staticmethod
    def _record_discarded_attempt(
        discarded: DiscardedAttempts,
        error: Exception,
        raw_response: Any,
        provider: str,
        model: str,
    ) -> str:
        """
        Record reported usage and cost for a failed structured-output attempt.

        Prefer response usage because parsing can fail after generation succeeds.

        Args:
            discarded: Accumulator for this call
            error: Exception raised by the failed attempt
            raw_response: Response from the attempt, if one arrived
            provider: Provider ID, for the pricing lookup
            model: Model identifier as requested

        Returns:
            A usage summary for logging, or an empty string if usage is unavailable
        """
        payload = raw_response if usage_counts(usage_of(raw_response)) else error_payload(error)
        counts = usage_counts(usage_of(payload))
        if counts is None:
            return ""

        input_tokens, output_tokens, total = counts
        reasoning = uncounted_reasoning(
            input_tokens, output_tokens, total, payload, provider=provider, model=model
        )
        # Prefer the provider's billed cost over a list-price estimate.
        cost = billed_cost(usage_of(payload))
        if cost is None:
            costs = calculate_cost_components(
                provider, model, input_tokens, output_tokens, reasoning
            )
            cost = costs.total_cost_usd if costs is not None else None
        discarded.add(input_tokens, output_tokens, cost, reasoning)

        priced = f" (${cost:.6f})" if cost is not None else ""
        return f" Discarded attempt billed {input_tokens} input + {output_tokens} output{priced}."

    def _create_response_from_optional_tool(
        self,
        raw_response: Any,
        model: str,
        response_format: Any,
        json_schema: dict,
        discarded: Optional[DiscardedAttempts] = None,
    ) -> LLMResponse:
        """
        Build a response after checking optional-tool input for unknown keys.

        Reject undeclared wrapper keys that Pydantic could ignore when filling
        defaults. Schemas permitting extra keys or declaring property patterns bypass the key
        check. Text responses are accepted without this check.

        Args:
            raw_response: Raw Anthropic response object
            model: Model identifier
            response_format: Pydantic model for validation
            json_schema: JSON schema the tool was offered with
            discarded: Usage from earlier failed attempts

        Returns:
            LLMResponse object

        Raises:
            ValueError: If the key check identifies an undeclared top-level key
        """
        for block in raw_response.content or []:
            if block.type == "tool_use" and block.name == "extract_structured_data":
                unknown = _unknown_keys(json_schema, block.input)
                if unknown:
                    raise ValueError(f"tool input has keys the schema does not define: {unknown}")
                break
        return self._create_response_from_tool(raw_response, model, response_format, discarded)

    def _create_response_from_tool(
        self,
        raw_response: Any,
        model: str,
        response_format: Any,
        discarded: Optional[DiscardedAttempts] = None,
    ) -> LLMResponse:
        """
        Create LLMResponse from Claude tool-based response (structured output).

        Args:
            raw_response: Raw Anthropic response object
            model: Model identifier
            response_format: Pydantic model for validation

        Returns:
            LLMResponse object
        """
        # Extract tool use from response
        text = ""
        parsed_data = None
        for block in raw_response.content:
            if block.type == "tool_use" and block.name == "extract_structured_data":
                try:
                    # Validate with Pydantic and convert to JSON
                    structured = response_format(**block.input)
                    text = structured.model_dump_json()
                    parsed_data = block.input  # Store the parsed dict
                except Exception as e:
                    logger.warning(f"Pydantic validation failed: {e}")
                    # Use raw tool input without validation
                    text = json.dumps(block.input)
                    parsed_data = block.input  # Still store it as parsed
                break
            elif block.type == "text":
                text = block.text
                # Try to extract JSON from text (works with or without response_format)
                if not parsed_data:  # Only if not already set from tool use
                    parsed_data = extract_json_from_text(text)

        usage = Usage()
        if hasattr(raw_response, "usage") and raw_response.usage:
            # Extract Claude cache tokens
            cache_creation_tokens = getattr(raw_response.usage, "cache_creation_input_tokens", 0)
            cache_read_tokens = getattr(raw_response.usage, "cache_read_input_tokens", 0)

            usage = Usage(
                input_tokens=raw_response.usage.input_tokens,
                output_tokens=raw_response.usage.output_tokens,
                total_tokens=raw_response.usage.input_tokens + raw_response.usage.output_tokens,
                cache_creation_tokens=cache_creation_tokens,
                cache_read_tokens=cache_read_tokens,
            )

            # Log cache usage (safely handle potential Mock objects in tests)
            if isinstance(cache_creation_tokens, (int, float)) and cache_creation_tokens > 0:
                logger.info(f"Claude cache created: {cache_creation_tokens} tokens")
            if isinstance(cache_read_tokens, (int, float)) and cache_read_tokens > 0:
                logger.info(f"Claude cache read: {cache_read_tokens} tokens")

            usage.reasoning_tokens = uncounted_reasoning(
                usage.input_tokens,
                usage.output_tokens,
                usage.total_tokens,
                raw_response,
                provider=self.PROVIDER_ID,
                model=model,
            )
            apply_costs(usage, self.PROVIDER_ID, model)

        if discarded is not None:
            discarded.apply_to(usage)

        return LLMResponse(
            text=text,
            model=model,
            provider=self.PROVIDER_ID,
            finish_reason=raw_response.stop_reason or "unknown",
            usage=usage,
            raw_response=raw_response,
            parsed=parsed_data,
        )

    def _create_response_from_raw(
        self,
        raw_response: Any,
        model: str,
        discarded: Optional[DiscardedAttempts] = None,
    ) -> LLMResponse:
        """
        Create LLMResponse from Claude raw response.

        Args:
            raw_response: Raw Anthropic response object
            model: Model identifier

        Returns:
            LLMResponse object
        """
        # Extract text and tool calls from content blocks
        text = ""
        tool_calls = []
        if raw_response.content:
            for block in raw_response.content:
                if block.type == "text":
                    text = block.text
                elif block.type == "tool_use":
                    # Claude uses tool_use blocks for tool calls
                    tool_calls.append(
                        {
                            "id": block.id,
                            "name": block.name,
                            "arguments": block.input,
                        }
                    )

        # Try to extract JSON from text
        parsed_data = extract_json_from_text(text)

        usage = Usage()
        if hasattr(raw_response, "usage") and raw_response.usage:
            # Extract Claude cache tokens
            cache_creation_tokens = getattr(raw_response.usage, "cache_creation_input_tokens", 0)
            cache_read_tokens = getattr(raw_response.usage, "cache_read_input_tokens", 0)

            usage = Usage(
                input_tokens=raw_response.usage.input_tokens,
                output_tokens=raw_response.usage.output_tokens,
                total_tokens=raw_response.usage.input_tokens + raw_response.usage.output_tokens,
                cache_creation_tokens=cache_creation_tokens,
                cache_read_tokens=cache_read_tokens,
            )

            # Log cache usage (safely handle potential Mock objects in tests)
            if isinstance(cache_creation_tokens, (int, float)) and cache_creation_tokens > 0:
                logger.info(f"Claude cache created: {cache_creation_tokens} tokens")
            if isinstance(cache_read_tokens, (int, float)) and cache_read_tokens > 0:
                logger.info(f"Claude cache read: {cache_read_tokens} tokens")

            usage.reasoning_tokens = uncounted_reasoning(
                usage.input_tokens,
                usage.output_tokens,
                usage.total_tokens,
                raw_response,
                provider=self.PROVIDER_ID,
                model=model,
            )
            apply_costs(usage, self.PROVIDER_ID, model)

        if discarded is not None:
            discarded.apply_to(usage)

        return LLMResponse(
            text=text,
            model=model,
            provider=self.PROVIDER_ID,
            finish_reason=raw_response.stop_reason or "unknown",
            usage=usage,
            raw_response=raw_response,
            parsed=parsed_data,
            tool_calls=tool_calls if tool_calls else None,
        )

    def get_model_list(self) -> List[Tuple[str, Optional[str]]]:
        """
        Get a list of available models from Claude.

        Returns:
            List of tuples (model_id, created_date)
        """
        if self.api_client is None:
            raise ValueError("Claude client is not initialized.")

        model_list = []
        raw_list = self.api_client.models.list()

        for model in raw_list:
            try:
                readable_date = datetime.fromisoformat(str(model.created_at)).strftime("%Y-%m-%d")
            except (ValueError, TypeError, AttributeError):
                readable_date = None
            model_list.append((model.id, readable_date))

        return model_list

    def _build_tool_messages(
        self,
        original_prompt: str,
        tool_calls: List[dict],
        tool_results: List[dict],
    ) -> List[dict]:
        """
        Build Claude message format with tool calls and results.

        Args:
            original_prompt: The original user prompt
            tool_calls: List of tool calls made by LLM
            tool_results: List of tool execution results

        Returns:
            List of message dicts in Claude format
        """
        return [
            {"role": "user", "content": original_prompt},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["arguments"],
                    }
                    for tc in tool_calls
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": result["tool_call_id"],
                        "content": result["content"],
                    }
                    for result in tool_results
                ],
            },
        ]
