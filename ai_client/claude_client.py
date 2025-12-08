"""
Anthropic Claude-specific implementation of the BaseAIClient.

This module provides the ClaudeClient class, which implements the BaseAIClient
interface specifically for Anthropic's Claude API, supporting both text and multimodal
interactions.
"""

import base64
import json
import logging
from datetime import datetime
from typing import List, Tuple, Any, Optional

from anthropic import Anthropic

from .base_client import BaseAIClient
from .response import LLMResponse, Usage
from .pricing import calculate_cost

logger = logging.getLogger(__name__)


class ClaudeClient(BaseAIClient):
    """
    Anthropic Claude-specific implementation of the BaseAIClient.

    This class implements the BaseAIClient interface for Anthropic's Claude API,
    supporting both text-only and multimodal requests with tool-based structured output.

    Key features:
    - Integration with Anthropic's Messages API
    - Support for multimodal content (text + images)
    - Support for Claude-specific parameters like top_p, top_k
    - Structured output via tools API
    """

    PROVIDER_ID = "anthropic"
    SUPPORTS_MULTIMODAL = True  # Claude supports images

    def _init_client(self):
        """Initialize the Anthropic client with the provided API key."""
        self.api_client = Anthropic(api_key=self.api_key, timeout=300.0)  # 5 minutes timeout

    def _prepare_content_with_images(self, prompt: str, images: List[str]) -> List[dict]:
        """
        Prepare Anthropic content with text and images.

        Args:
            prompt: The text prompt
            images: List of image paths/URLs

        Returns:
            List of content blocks for Anthropic API
        """
        content = [{"type": "text", "text": prompt}]

        # Add images if any
        for resource in images:
            try:
                if self.is_url(resource):
                    # For URLs, we need to fetch and encode
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
                    # For local files, read and encode
                    with open(resource, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

                # Detect media type from file extension
                from .utils import detect_image_mime_type

                media_type = detect_image_mime_type(resource)

                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_image,
                        },
                    }
                )
            except Exception as e:
                logger.error(f"Error processing image {resource}: {e}")

        return content

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
            cache: Enable cache_control blocks for files
            file_content: File content to mark for caching (when cache=True)
            **kwargs: Additional Claude-specific parameters

        Returns:
            LLMResponse object with the provider's response
        """
        images = images or []

        # Build system blocks with optional caching
        system_blocks = []

        # Add cached content FIRST (Claude caches trailing system blocks)
        if file_content and cache:
            system_blocks.append(
                {
                    "type": "text",
                    "text": f"Reference documents:\n\n{file_content}",
                    "cache_control": {"type": "ephemeral"},  # Mark for caching
                }
            )

        # Add system prompt (not cached by default)
        if system_prompt:
            system_blocks.append({"type": "text", "text": system_prompt})

        # Build API messages
        if messages and len(messages) > 1:
            # Multi-turn conversation
            api_messages = []
            for i, msg in enumerate(messages):
                content_blocks = []

                # Add images to last user message only
                if msg["role"] == "user" and i == len(messages) - 1 and images:
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

                            content_blocks.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": base64_image,
                                    },
                                }
                            )
                        except Exception as e:
                            logger.error(f"Error processing image {resource}: {e}")

                # Add text content
                content_blocks.append({"type": "text", "text": msg["content"]})

                api_messages.append({"role": msg["role"], "content": content_blocks})
        else:
            # Single-turn request
            content = self._prepare_content_with_images(prompt, images)
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

            params["tools"] = tools
            params["tool_choice"] = {"type": "tool", "name": "extract_structured_data"}

            try:
                raw_response = self.api_client.messages.create(**params)
                return self._create_response_from_tool(raw_response, model, response_format)
            except Exception as e:
                logger.warning(
                    f"Structured output via tools failed: {e}. Falling back to text mode."
                )
                # Remove tools and try again
                del params["tools"]
                del params["tool_choice"]

        # Send the request to Anthropic
        raw_response = self.api_client.messages.create(**params)

        return self._create_response_from_raw(raw_response, model)

    def _create_response_from_tool(
        self, raw_response: Any, model: str, response_format: Any
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

            # Calculate cost if pricing data is available
            costs = calculate_cost(
                self.PROVIDER_ID,
                model,
                usage.input_tokens,
                usage.output_tokens,
            )
            if costs is not None:
                usage.input_cost_usd, usage.output_cost_usd, usage.estimated_cost_usd = costs

        return LLMResponse(
            text=text,
            model=model,
            provider=self.PROVIDER_ID,
            finish_reason=raw_response.stop_reason or "unknown",
            usage=usage,
            raw_response=raw_response,
            parsed=parsed_data,
        )

    def _create_response_from_raw(self, raw_response: Any, model: str) -> LLMResponse:
        """
        Create LLMResponse from Claude raw response.

        Args:
            raw_response: Raw Anthropic response object
            model: Model identifier

        Returns:
            LLMResponse object
        """
        # Extract text from content blocks
        text = ""
        if raw_response.content:
            for block in raw_response.content:
                if block.type == "text":
                    text = block.text
                    break

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

            # Calculate cost if pricing data is available
            costs = calculate_cost(
                self.PROVIDER_ID,
                model,
                usage.input_tokens,
                usage.output_tokens,
            )
            if costs is not None:
                usage.input_cost_usd, usage.output_cost_usd, usage.estimated_cost_usd = costs

        return LLMResponse(
            text=text,
            model=model,
            provider=self.PROVIDER_ID,
            finish_reason=raw_response.stop_reason or "unknown",
            usage=usage,
            raw_response=raw_response,
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
