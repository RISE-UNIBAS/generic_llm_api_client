"""
OpenAI-specific implementation of the BaseAIClient.

This module provides the OpenAIClient class, which implements the BaseAIClient
interface specifically for OpenAI's API, supporting both text-only and
multimodal content (text + images).

Also used for OpenAI-compatible APIs like OpenRouter and sciCORE.
"""

import base64
import json
import logging
from datetime import datetime, timezone
from typing import List, Tuple, Any, Optional

from openai import OpenAI

from .base_client import BaseAIClient
from .response import DiscardedAttempts, LLMResponse, Usage
from .pricing import apply_costs, calculate_cost_components
from .reasoning import uncounted_reasoning
from .utils import (
    attach_discarded,
    attach_payload,
    billed_cost,
    error_payload,
    extract_json_from_text,
    rejects_parameter,
    usage_counts,
    usage_of,
)

logger = logging.getLogger(__name__)

# The two rules below have different boundaries and must not be merged: every gpt-5.x
# needs the renamed token cap, but gpt-5.1 and newer accept a custom temperature while
# gpt-5, gpt-5-mini, gpt-5-nano and the o-series do not. Both verified against the API.

# Models whose chat endpoint takes max_completion_tokens instead of max_tokens. A loose
# prefix, since the whole series renamed the parameter.
MAX_COMPLETION_TOKENS_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")

# Models that accept only their default temperature. Matched as a family -- the exact
# name, or a "-" suffix of it -- so "gpt-5-nano" is covered while "gpt-5.1" is not.
# Discarding a temperature a model would have honoured silently changes sampling, so
# this list stays narrow and _call_with_parameter_retry recovers for anything it misses.
FIXED_TEMPERATURE_FAMILIES = ("gpt-5", "o1", "o3", "o4")


class OpenAIClient(BaseAIClient):
    """
    OpenAI-specific implementation of the BaseAIClient.

    This class implements the BaseAIClient interface for OpenAI's API,
    handling both text-only and multimodal (text + images) requests through
    the OpenAI Chat Completions API.

    Key features:
    - Full support for multimodal content via GPT-4 Vision models
    - Image encoding and formatting specific to OpenAI's requirements
    - Support for OpenAI-specific parameters like frequency_penalty, top_p, etc.
    - Support for structured output via response_format
    - Custom base_url support for OpenRouter, sciCORE, and other OpenAI-compatible APIs
    """

    PROVIDER_ID = "openai"
    SUPPORTS_MULTIMODAL = True
    SUPPORTS_TOOLS = True

    def _init_client(self):
        """Initialize the OpenAI client with the provided API key and optional base URL."""
        kwargs = {"api_key": self.api_key}

        # Support custom base URLs (for OpenRouter, sciCORE, etc.)
        if self.base_url:
            kwargs["base_url"] = self.base_url

        # Add custom headers if provided
        if "default_headers" in self.settings:
            kwargs["default_headers"] = self.settings["default_headers"]

        self.api_client = OpenAI(**kwargs)

    def _prepare_message_with_images(
        self,
        prompt: str,
        images: List[str],
        system_prompt: str,
        file_content: str = "",
        content_order=None,
    ) -> List[dict]:
        """
        Prepare an OpenAI message object with text, files, and images.

        Args:
            prompt: The text prompt
            images: List of image paths/URLs
            system_prompt: System prompt
            file_content: Text content from files (empty string if none)
            content_order: Content ordering policy override

        Returns:
            List of OpenAI message objects with content
        """
        prompt_parts = [{"type": "text", "text": prompt}]
        files_parts = [{"type": "text", "text": file_content}] if file_content else []
        image_parts = []

        for resource in images:
            if self.is_url(resource):
                image_parts.append({"type": "image_url", "image_url": {"url": resource}})
            else:
                try:
                    with open(resource, "rb") as image_file:
                        image_data = image_file.read()
                        base64_image = base64.b64encode(image_data).decode("utf-8")

                    from .utils import detect_image_mime_type

                    mime_type = detect_image_mime_type(resource)
                    image_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                        }
                    )
                except Exception as e:
                    logger.error(f"Error reading image file {resource}: {e}")

        user_content = self._order_content_parts(
            {"prompt": prompt_parts, "images": image_parts, "files": files_parts},
            content_order,
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

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
        Send a prompt to the OpenAI model and get the response.

        OpenAI has automatic prompt caching for 1024+ tokens.

        Args:
            model: The OpenAI model identifier (e.g., "gpt-4o", "gpt-3.5-turbo")
            prompt: The text prompt to send
            messages: Optional conversation history (multi-turn)
            images: List of image paths/URLs to include
            system_prompt: System prompt to use
            response_format: Optional Pydantic model for structured output
            cache: Informational only (caching is always automatic)
            file_content: Not used (files already appended to prompt)
            **kwargs: Additional OpenAI-specific parameters
                api_style: Which OpenAI endpoint to use. One of:
                    "chat"        (default) v1/chat/completions  — GPT-4o, GPT-4.1, o-series
                    "responses"             v1/responses          — Codex, o3, o4-mini
                    "completions"           v1/completions        — legacy text-completion models
                    Can also be set once at client level via settings["api_style"].
                prompt_cache_key: Routing hint for cache optimization
                prompt_cache_retention: "in_memory" (5-10min) or "24h"

        Returns:
            LLMResponse object with the provider's response
        """
        # Handle conversation messages
        images = images or []
        content_order = kwargs.pop("_content_order", None)
        api_style = kwargs.pop("api_style", self.settings.get("api_style", "chat"))

        if messages and len(messages) > 1:
            # Multi-turn conversation: use provided messages
            api_messages = []

            # Add system prompt first
            if system_prompt:
                api_messages.append({"role": "system", "content": system_prompt})

            # Add conversation messages, attaching images and files to the last user message
            for i, msg in enumerate(messages):
                if msg["role"] == "user" and i == len(messages) - 1 and (images or file_content):
                    image_parts = []
                    for resource in images:
                        if self.is_url(resource):
                            image_parts.append(
                                {"type": "image_url", "image_url": {"url": resource}}
                            )
                        else:
                            try:
                                with open(resource, "rb") as image_file:
                                    image_data = image_file.read()
                                    base64_image = base64.b64encode(image_data).decode("utf-8")
                                from .utils import detect_image_mime_type

                                mime_type = detect_image_mime_type(resource)
                                image_parts.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime_type};base64,{base64_image}"
                                        },
                                    }
                                )
                            except Exception as e:
                                logger.error(f"Error reading image file {resource}: {e}")

                    files_parts = [{"type": "text", "text": file_content}] if file_content else []
                    content = self._order_content_parts(
                        {
                            "prompt": [{"type": "text", "text": msg["content"]}],
                            "images": image_parts,
                            "files": files_parts,
                        },
                        content_order,
                    )
                    api_messages.append({"role": msg["role"], "content": content})
                else:
                    api_messages.append(msg)
        else:
            # Single-turn request
            api_messages = self._prepare_message_with_images(
                prompt, images, system_prompt, file_content, content_order
            )

        # Extract OpenAI-specific parameters from settings and kwargs
        params = {
            "model": model,
            "messages": api_messages,
        }

        # Add OpenAI caching parameters (from kwargs)
        if "prompt_cache_key" in kwargs:
            params["prompt_cache_key"] = kwargs["prompt_cache_key"]

        if "prompt_cache_retention" in kwargs and kwargs["prompt_cache_retention"] in [
            "in_memory",
            "24h",
        ]:
            params["prompt_cache_retention"] = kwargs["prompt_cache_retention"]

        # Add optional parameters if specified
        optional_params = [
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "n",
            "stop",
        ]
        for param in optional_params:
            value = kwargs.get(param, self.settings.get(param))
            if value is not None:
                params[param] = value

        # Route to the appropriate endpoint before any chat-specific handling
        if api_style == "completions":
            return self._do_completions_api(params, model, response_format)
        if api_style == "responses":
            return self._do_responses_api(
                params, model, prompt, images, file_content, system_prompt, response_format, kwargs
            )

        # gpt-5 and newer reject max_tokens on the chat endpoint. Renaming it up front for
        # the families we know about avoids a rejected round trip; _call_with_token_cap_retry
        # covers the ones released after this version.
        if self._uses_max_completion_tokens(model):
            self._rename_token_cap(params)
        if self._rejects_custom_temperature(model):
            params.pop("temperature", None)

        # Handle tool calling
        tool_definitions = kwargs.pop("_tool_definitions", None)
        if tool_definitions:
            # Convert to OpenAI tools format
            params["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("function", tool.get("name", "unknown")),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                    },
                }
                for tool in tool_definitions
            ]
            params["tool_choice"] = "auto"

        # A failed attempt is still billed, so its tokens are accumulated here and reported
        # beside the successful response. Local to this call, never on self: one client
        # object serves many worker threads, and shared state would attribute one thread's
        # waste to another's response. Note _do_prompt_with_retry re-enters this method on
        # failure, so the totals cover the final attempt only.
        discarded = DiscardedAttempts()

        # Handle structured output
        if response_format:
            # Check if it's a Pydantic model (v1 or v2)
            is_pydantic_v2 = hasattr(response_format, "model_json_schema")
            is_pydantic_v1 = hasattr(response_format, "schema")

            if is_pydantic_v2:
                # Use beta.chat.completions.parse for Pydantic v2 structured output
                try:
                    params["response_format"] = response_format
                    raw_response = self._parse_structured(params, model)
                    return self._build_response(
                        self._create_response_from_parsed,
                        raw_response,
                        model,
                        discarded=discarded,
                    )
                except Exception as e:
                    wasted = self._record_discarded_attempt(discarded, e, self.PROVIDER_ID, model)
                    if self._is_non_chat_model_error(e):
                        logger.warning(
                            f"Model {model} does not support chat completions, "
                            f"falling back to v1/completions: {e}"
                        )
                        params.pop("response_format", None)
                        return self._do_completions_api(
                            params, model, response_format, discarded
                        )  # auto-fallback
                    logger.warning(
                        f"Structured output failed, falling back to JSON mode: {e}."
                        f"{wasted} A second request will be billed."
                    )
                    # Fall through to JSON object mode

            # Fallback to JSON object mode (for Pydantic v1 or when v2 parse fails)
            params["response_format"] = {"type": "json_object"}

            # Get JSON schema (support both Pydantic v1 and v2)
            if is_pydantic_v2:
                schema_dict = response_format.model_json_schema()
            elif is_pydantic_v1:
                schema_dict = response_format.schema()
            else:
                schema_dict = None

            if schema_dict:
                schema_prompt = f"\n\nYou MUST respond with valid JSON matching this exact schema: {json.dumps(schema_dict)}"

                # Find the last user message and append the schema prompt
                last_user_idx = None
                for i in range(len(params["messages"]) - 1, -1, -1):
                    if params["messages"][i].get("role") == "user":
                        last_user_idx = i
                        break

                if last_user_idx is not None:
                    msg = params["messages"][last_user_idx]
                    # Handle both string and array content
                    if isinstance(msg["content"], str):
                        msg["content"] += schema_prompt
                    elif isinstance(msg["content"], list):
                        # Content is an array (multimodal) - append text block
                        msg["content"].append({"type": "text", "text": schema_prompt})
                else:
                    # No user message found, add a new one
                    params["messages"].append({"role": "user", "content": schema_prompt})

        # Send the request to OpenAI
        try:
            raw_response = self._call_with_parameter_retry(
                self.api_client.chat.completions.create, params, model
            )
        except Exception as e:
            if self._is_non_chat_model_error(e):
                logger.warning(
                    f"Model {model} does not support chat completions, "
                    f"falling back to v1/completions: {e}"
                )
                params.pop("response_format", None)
                return self._do_completions_api(params, model, response_format, discarded)
            # Only earlier attempts travel here. This failure's own usage is recorded
            # by _create_error_response; adding it would count the same call twice.
            raise attach_discarded(e, discarded)

        return self._build_response(
            self._create_response_from_raw,
            raw_response,
            model,
            response_format,
            discarded=discarded,
        )

    def _do_responses_api(
        self,
        params: dict,
        model: str,
        prompt: str,
        images: List[str],
        file_content: str,
        system_prompt: Optional[str],
        response_format: Optional[Any],
        extra_kwargs: dict,
    ) -> LLMResponse:
        """
        Call v1/responses (Responses API) for models like Codex, o3, o4-mini.

        Key differences from chat completions:
        - ``input`` instead of ``messages`` (string or array of input items)
        - ``instructions`` instead of a system role message
        - ``max_output_tokens`` instead of ``max_tokens``
        - Content type literals: ``input_text`` / ``input_image`` (not ``text`` / ``image_url``)
        - Image URL is a flat string field, not nested under ``image_url.url``
        """
        # Build the input: plain string for text-only with no files,
        # message content array when images or file content are present.
        if images or file_content:
            content = [{"type": "input_text", "text": prompt}]
            if file_content:
                content.append({"type": "input_text", "text": file_content})
            for resource in images:
                if self.is_url(resource):
                    content.append({"type": "input_image", "image_url": resource})
                else:
                    try:
                        with open(resource, "rb") as f:
                            image_data = f.read()
                        base64_image = base64.b64encode(image_data).decode("utf-8")
                        from .utils import detect_image_mime_type

                        mime_type = detect_image_mime_type(resource)
                        content.append(
                            {
                                "type": "input_image",
                                "image_url": f"data:{mime_type};base64,{base64_image}",
                            }
                        )
                    except Exception as e:
                        logger.error(f"Error reading image file {resource}: {e}")
            input_value = [{"type": "message", "role": "user", "content": content}]
        else:
            input_value = prompt

        responses_params: dict = {"model": model, "input": input_value}

        if system_prompt:
            responses_params["instructions"] = system_prompt

        # Map common settings; note the renamed max_tokens → max_output_tokens
        param_map = {
            "temperature": "temperature",
            "max_tokens": "max_output_tokens",
            "top_p": "top_p",
            "stop": "stop",
        }
        for src, dst in param_map.items():
            value = extra_kwargs.get(src, params.get(src))
            if value is not None:
                responses_params[dst] = value

        # Structured output via text.format
        if response_format and hasattr(response_format, "model_json_schema"):
            schema = response_format.model_json_schema()
            responses_params["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": schema.get("title", "response"),
                    "schema": schema,
                    "strict": True,
                }
            }

        raw_response = self.api_client.responses.create(**responses_params)
        return self._build_response(
            self._create_response_from_responses, raw_response, model, response_format
        )

    def _create_response_from_responses(
        self,
        raw_response: Any,
        model: str,
        response_format: Optional[Any],
        discarded: Optional[DiscardedAttempts] = None,
    ) -> LLMResponse:
        """Create LLMResponse from a v1/responses API response."""
        text = getattr(raw_response, "output_text", "") or ""
        parsed_data = extract_json_from_text(text)

        is_pydantic = response_format and (
            hasattr(response_format, "model_json_schema") or hasattr(response_format, "schema")
        )
        if is_pydantic and parsed_data:
            try:
                validated = response_format(**parsed_data)
                text = (
                    validated.model_dump_json()
                    if hasattr(validated, "model_dump_json")
                    else validated.json()
                )
            except Exception as e:
                logger.warning(f"Failed to validate responses API output with Pydantic: {e}")

        usage = Usage()
        if hasattr(raw_response, "usage") and raw_response.usage:
            usage = Usage(
                input_tokens=raw_response.usage.input_tokens,
                output_tokens=raw_response.usage.output_tokens,
                total_tokens=raw_response.usage.total_tokens,
            )
            self._finalize_usage(usage, raw_response, model)

        if discarded is not None:
            discarded.apply_to(usage)

        return LLMResponse(
            text=text,
            model=raw_response.model,
            provider=self.PROVIDER_ID,
            finish_reason=getattr(raw_response, "status", None) or "unknown",
            usage=usage,
            raw_response=raw_response,
            parsed=parsed_data,
        )

    @staticmethod
    def _bare_name(model: str) -> str:
        """Strip any provider prefix and routing suffix from a model id."""
        return model.split("/")[-1].split(":")[0].lower()

    @classmethod
    def _uses_max_completion_tokens(cls, model: str) -> bool:
        """
        Return True if the model's chat endpoint requires max_completion_tokens.

        Matched as a prefix on the bare id, so a routed name like "openai/gpt-5" or a
        pinned "gpt-5:groq" resolves, while an unrelated id that merely contains "o1"
        does not.
        """
        return cls._bare_name(model).startswith(MAX_COMPLETION_TOKENS_PREFIXES)

    @classmethod
    def _rejects_custom_temperature(cls, model: str) -> bool:
        """
        Return True if the model accepts only its default temperature.

        Deliberately narrower than the token cap rule: gpt-5.1 and newer honour a
        temperature, and silently discarding one would change a caller's sampling.
        """
        bare = cls._bare_name(model)
        return any(
            bare == family or bare.startswith(f"{family}-") for family in FIXED_TEMPERATURE_FAMILIES
        )

    @staticmethod
    def _rename_token_cap(params: dict) -> bool:
        """Rename max_tokens to max_completion_tokens, reporting whether it was present."""
        if "max_tokens" not in params:
            return False
        params["max_completion_tokens"] = params.pop("max_tokens")
        return True

    @staticmethod
    def _is_token_cap_error(error: Exception) -> bool:
        """Return True if the provider rejected max_tokens in favour of the newer name."""
        message = str(error).lower()
        return "max_tokens" in message and "max_completion_tokens" in message

    def _finalize_usage(self, usage: Usage, raw_response: Any, model: str) -> None:
        """
        Fill a response's reasoning count and cost fields.

        Priced on the model as requested rather than the id the response echoed back:
        routers rewrite names, and a rewritten name misses the pricing table. A cost the
        provider reports having billed is kept as it stands, on every path rather than
        only the plain chat one.

        Args:
            usage: Usage built from the response, filled in place
            raw_response: The provider's response
            model: Model identifier as requested
        """
        usage.reasoning_tokens = uncounted_reasoning(
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens,
            raw_response,
            provider=self.PROVIDER_ID,
            model=model,
        )
        billed = billed_cost(getattr(raw_response, "usage", None))
        if billed is not None:
            usage.estimated_cost_usd = billed
        apply_costs(usage, self.PROVIDER_ID, model)

    def _parse_structured(self, params: dict, model: str):
        """
        Call the structured-output endpoint, keeping the body if parsing fails.

        The SDK validates after the request has returned, and two of its three failure
        modes raise exceptions carrying nothing: a content filter, and output that does
        not match the schema. Those tokens are billed all the same, so the body is taken
        first and attached to the exception for the usage recovery to find.

        Args:
            params: Request parameters
            model: Model identifier as requested

        Returns:
            The parsed completion
        """
        endpoint = self.api_client.beta.chat.completions
        with_raw_response = getattr(endpoint, "with_raw_response", None)
        if with_raw_response is None:  # an SDK that cannot show the body first
            return self._call_with_parameter_retry(endpoint.parse, params, model)

        response = self._call_with_parameter_retry(with_raw_response.parse, params, model)
        try:
            return response.parse()
        except Exception as error:
            raise attach_payload(error, self._response_body(response))

    @staticmethod
    def _response_body(response):
        """Return a raw response body as a mapping, or None if it cannot be read."""
        try:
            body = json.loads(response.text)
        except Exception:
            return None
        return body if isinstance(body, dict) else None

    def _call_with_parameter_retry(self, send, params: dict, model: str):
        """
        Call a chat endpoint, correcting parameters the model refuses.

        Newer models rename the token cap and accept only their default temperature.
        Known families are corrected before the call; this recovers for the rest, so a
        model released after this version still works. Each rejection is a 400 raised
        before anything is generated, so nothing is billed and no attempt is recorded.

        Args:
            send: Bound SDK method taking the request parameters
            params: Request parameters, corrected in place when a retry fires
            model: Model identifier, for the log line

        Returns:
            The provider's response
        """
        # One attempt, plus one for each correction this method knows how to make.
        for _ in range(3):
            try:
                return send(**params)
            except Exception as error:
                if self._is_token_cap_error(error) and self._rename_token_cap(params):
                    logger.warning(
                        f"Model {model} rejected max_tokens; retrying with "
                        f"max_completion_tokens. Add its prefix to "
                        f"MAX_COMPLETION_TOKENS_PREFIXES to skip this round trip."
                    )
                    continue
                if rejects_parameter(error, "temperature") and "temperature" in params:
                    del params["temperature"]
                    logger.warning(
                        f"Model {model} rejected a custom temperature; retrying without "
                        f"one. Add its family to FIXED_TEMPERATURE_FAMILIES to skip this "
                        f"round trip."
                    )
                    continue
                raise
        return send(**params)

    @staticmethod
    def _record_discarded_attempt(
        discarded: DiscardedAttempts, error: Exception, provider: str, model: str
    ) -> str:
        """
        Record the tokens a failed attempt was billed for, if it reported any.

        Args:
            discarded: Accumulator for this call
            error: Exception raised by the failed attempt
            provider: Provider ID, for the pricing lookup
            model: Model identifier as requested

        Returns:
            A phrase naming what was wasted, or an empty string if nothing was billed
        """
        payload = error_payload(error)
        counts = usage_counts(usage_of(payload))
        if counts is None:
            return ""

        input_tokens, output_tokens, total = counts
        reasoning = uncounted_reasoning(
            input_tokens, output_tokens, total, payload, provider=provider, model=model
        )
        # A cost the provider billed is the accurate one; list prices are the fallback.
        cost = billed_cost(usage_of(payload))
        if cost is None:
            costs = calculate_cost_components(
                provider, model, input_tokens, output_tokens, reasoning
            )
            cost = costs.total_cost_usd if costs is not None else None
        discarded.add(input_tokens, output_tokens, cost, reasoning)

        priced = f" (${cost:.6f})" if cost is not None else ""
        return f" Discarded attempt billed {input_tokens} input + {output_tokens} output{priced}."

    @staticmethod
    def _is_non_chat_model_error(error: Exception) -> bool:
        """Return True if the error indicates the model only supports the v1/completions endpoint."""
        msg = str(error).lower()
        return "not a chat model" in msg or "v1/completions" in msg

    def _do_completions_api(
        self,
        params: dict,
        model: str,
        response_format: Optional[Any],
        discarded: Optional[DiscardedAttempts] = None,
    ) -> LLMResponse:
        """
        Call v1/completions (text completions) instead of v1/chat/completions.

        Converts the chat-style messages list into a single prompt string and
        calls ``api_client.completions.create()``.  Used when ``completions_api=True``
        is set, or as an automatic fallback when the model rejects chat completions.
        """
        # Build a single prompt string from chat messages
        parts = []
        for msg in params.get("messages", []):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Flatten multimodal content - only keep text parts
                content = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")

        legacy_prompt = "\n".join(parts)

        legacy_params = {
            "model": model,
            "prompt": legacy_prompt,
        }
        for p in (
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "n",
            "stop",
        ):
            if p in params:
                legacy_params[p] = params[p]

        # The legacy endpoint still takes max_tokens, even if the chat path renamed it.
        if "max_tokens" not in legacy_params and "max_completion_tokens" in params:
            legacy_params["max_tokens"] = params["max_completion_tokens"]

        raw_response = self.api_client.completions.create(**legacy_params)
        return self._build_response(
            self._create_response_from_completions,
            raw_response,
            model,
            response_format,
            discarded=discarded,
        )

    def _create_response_from_completions(
        self,
        raw_response: Any,
        model: str,
        response_format: Optional[Any],
        discarded: Optional[DiscardedAttempts] = None,
    ) -> LLMResponse:
        """Create LLMResponse from a v1/completions response."""
        choice = raw_response.choices[0]
        text = choice.text or ""
        parsed_data = extract_json_from_text(text)

        is_pydantic = response_format and (
            hasattr(response_format, "model_json_schema") or hasattr(response_format, "schema")
        )
        if is_pydantic and parsed_data:
            try:
                validated = response_format(**parsed_data)
                text = (
                    validated.model_dump_json()
                    if hasattr(validated, "model_dump_json")
                    else validated.json()
                )
            except Exception as e:
                logger.warning(f"Failed to validate completions response with Pydantic: {e}")

        usage = Usage()
        if hasattr(raw_response, "usage") and raw_response.usage:
            usage = Usage(
                input_tokens=raw_response.usage.prompt_tokens,
                output_tokens=raw_response.usage.completion_tokens,
                total_tokens=raw_response.usage.total_tokens,
            )
            self._finalize_usage(usage, raw_response, model)

        if discarded is not None:
            discarded.apply_to(usage)

        return LLMResponse(
            text=text,
            model=raw_response.model,
            provider=self.PROVIDER_ID,
            finish_reason=getattr(choice, "finish_reason", None) or "unknown",
            usage=usage,
            raw_response=raw_response,
            parsed=parsed_data,
        )

    def _create_response_from_parsed(
        self,
        raw_response: Any,
        model: str,
        discarded: Optional[DiscardedAttempts] = None,
    ) -> LLMResponse:
        """
        Create LLMResponse from OpenAI parsed response (structured output).

        Args:
            raw_response: Raw OpenAI response object
            model: Model identifier

        Returns:
            LLMResponse object
        """
        choice = raw_response.choices[0]

        # For structured output, the parsed object is in message.parsed
        parsed_data = None
        if hasattr(choice.message, "parsed") and choice.message.parsed:
            text = choice.message.parsed.model_dump_json()
            # Parse the JSON to dict/list for convenience
            try:
                parsed_data = json.loads(text)
            except Exception:
                pass
        else:
            text = choice.message.content or ""
            # Try to extract JSON even if structured output wasn't used
            parsed_data = extract_json_from_text(text)

        usage = Usage()
        if hasattr(raw_response, "usage") and raw_response.usage:
            # Extract cached tokens from prompt_tokens_details
            cached_tokens = 0
            if hasattr(raw_response.usage, "prompt_tokens_details"):
                details = raw_response.usage.prompt_tokens_details
                if hasattr(details, "cached_tokens"):
                    cached_tokens = details.cached_tokens
                elif isinstance(details, dict):
                    cached_tokens = details.get("cached_tokens", 0)

            usage = Usage(
                input_tokens=raw_response.usage.prompt_tokens,
                output_tokens=raw_response.usage.completion_tokens,
                total_tokens=raw_response.usage.total_tokens,
                cached_tokens=(
                    cached_tokens
                    if isinstance(cached_tokens, (int, float)) and cached_tokens > 0
                    else None
                ),
            )
            self._finalize_usage(usage, raw_response, model)

        if discarded is not None:
            discarded.apply_to(usage)

        return LLMResponse(
            text=text,
            model=raw_response.model,
            provider=self.PROVIDER_ID,
            finish_reason=choice.finish_reason or "unknown",
            usage=usage,
            raw_response=raw_response,
            parsed=parsed_data,
        )

    def _create_response_from_raw(
        self,
        raw_response: Any,
        model: str,
        response_format: Optional[Any],
        discarded: Optional[DiscardedAttempts] = None,
    ) -> LLMResponse:
        """
        Create LLMResponse from OpenAI raw response.

        Args:
            raw_response: Raw OpenAI response object
            model: Model identifier
            response_format: Pydantic model if structured output was requested

        Returns:
            LLMResponse object
        """
        choice = raw_response.choices[0]
        # Some reasoning models (e.g. Alibaba qwen-thinking, DeepSeek reasoner) return the
        # final answer in `reasoning_content` with an empty `content` field.
        text = choice.message.content or getattr(choice.message, "reasoning_content", None) or ""
        parsed_data = None

        # Extract tool calls if present
        tool_calls = None
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                }
                for tc in choice.message.tool_calls
            ]

        # Try to extract JSON from the response (works with or without response_format)
        extracted_json = extract_json_from_text(text)

        # If response_format was provided, validate with Pydantic (v1 or v2)
        is_pydantic = response_format and (
            hasattr(response_format, "model_json_schema") or hasattr(response_format, "schema")
        )

        if is_pydantic and extracted_json:
            try:
                # Validate with Pydantic (works for both v1 and v2)
                validated = response_format(**extracted_json)
                # Use appropriate dump method based on Pydantic version
                if hasattr(validated, "model_dump_json"):
                    # Pydantic v2
                    text = validated.model_dump_json()
                else:
                    # Pydantic v1
                    text = validated.json()
                parsed_data = extracted_json
            except Exception as e:
                logger.warning(f"Failed to validate structured response with Pydantic: {e}")
                # Keep original text but still set parsed_data if JSON was valid
                parsed_data = extracted_json
        elif extracted_json:
            # No response_format, but we found valid JSON - populate parsed attribute
            parsed_data = extracted_json

        usage = Usage()
        if hasattr(raw_response, "usage") and raw_response.usage:
            # Extract cached tokens from prompt_tokens_details (correct location)
            cached_tokens = 0
            if hasattr(raw_response.usage, "prompt_tokens_details"):
                details = raw_response.usage.prompt_tokens_details
                if hasattr(details, "cached_tokens"):
                    cached_tokens = details.cached_tokens
                elif isinstance(details, dict):
                    cached_tokens = details.get("cached_tokens", 0)

            # Safely handle cached_tokens (might be Mock in tests)
            cached_tokens_value = None
            if isinstance(cached_tokens, (int, float)) and cached_tokens > 0:
                cached_tokens_value = cached_tokens

            usage = Usage(
                input_tokens=raw_response.usage.prompt_tokens,
                output_tokens=raw_response.usage.completion_tokens,
                total_tokens=raw_response.usage.total_tokens,
                cached_tokens=cached_tokens_value,
            )
            self._finalize_usage(usage, raw_response, model)

        if discarded is not None:
            discarded.apply_to(usage)

        response = LLMResponse(
            text=text,
            model=raw_response.model,
            provider=self.PROVIDER_ID,
            finish_reason=choice.finish_reason or "unknown",
            usage=usage,
            raw_response=raw_response,
            parsed=parsed_data,
            tool_calls=tool_calls,
        )

        return response

    def get_model_list(self) -> List[Tuple[str, Optional[str]]]:
        """
        Get a list of available models from OpenAI.

        Returns:
            List of tuples (model_id, created_date), where created_date is None if the
            endpoint does not report one. DeepSeek and dedicated HuggingFace endpoints
            both omit it, and reading it blindly fails the whole listing.
        """
        if self.api_client is None:
            raise ValueError("OpenAI client is not initialized.")

        model_list = []

        for model in self.api_client.models.list():
            created = getattr(model, "created", None)
            readable_date = None
            if isinstance(created, (int, float)):
                readable_date = datetime.fromtimestamp(created, tz=timezone.utc).strftime(
                    "%Y-%m-%d"
                )
            model_list.append((model.id, readable_date))

        return model_list

    def _build_tool_messages(
        self,
        original_prompt: str,
        tool_calls: List[dict[str, Any]],
        tool_results: List[dict[str, Any]],
    ) -> List[dict[str, Any]]:
        """
        Build OpenAI message format with tool calls and results.

        Args:
            original_prompt: The original user prompt
            tool_calls: List of tool calls made by LLM
            tool_results: List of tool execution results

        Returns:
            List of message dicts in OpenAI format
        """
        return [
            {"role": "user", "content": original_prompt},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]),
                        },
                    }
                    for tc in tool_calls
                ],
            },
            *[
                {
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": result["content"],
                }
                for result in tool_results
            ],
        ]
