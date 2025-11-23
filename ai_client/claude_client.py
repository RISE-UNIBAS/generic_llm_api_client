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
        images: List[str],
        system_prompt: str,
        response_format: Optional[Any],
        **kwargs,
    ) -> LLMResponse:
        """
        Send a prompt to the Claude model and get the response.

        Args:
            model: The Claude model identifier (e.g., "claude-3-opus-20240229")
            prompt: The text prompt to send
            images: List of image paths/URLs
            system_prompt: System prompt to use
            response_format: Optional Pydantic model for structured output
            **kwargs: Additional Claude-specific parameters

        Returns:
            LLMResponse object with the provider's response
        """
        # Prepare content with images
        content = self._prepare_content_with_images(prompt, images)

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
            "messages": [{"role": "user", "content": content}],
            "system": system_prompt,
            "max_tokens": kwargs.get(
                "max_tokens", self.settings.get("max_tokens", default_max_tokens)
            ),
            "timeout": 300.0,
        }

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
            usage = Usage(
                input_tokens=raw_response.usage.input_tokens,
                output_tokens=raw_response.usage.output_tokens,
                total_tokens=raw_response.usage.input_tokens + raw_response.usage.output_tokens,
            )
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
            usage = Usage(
                input_tokens=raw_response.usage.input_tokens,
                output_tokens=raw_response.usage.output_tokens,
                total_tokens=raw_response.usage.input_tokens + raw_response.usage.output_tokens,
            )
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
            except:
                readable_date = None
            model_list.append((model.id, readable_date))

        return model_list
