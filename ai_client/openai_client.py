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
from .response import LLMResponse, Usage

logger = logging.getLogger(__name__)


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
        self, prompt: str, images: List[str], system_prompt: str
    ) -> List[dict]:
        """
        Prepare an OpenAI message object with text and images.

        Args:
            prompt: The text prompt
            images: List of image paths/URLs
            system_prompt: System prompt

        Returns:
            List of OpenAI message objects with content
        """
        # Prepare user content with text
        user_content = [{"type": "text", "text": prompt}]

        # Add images if any
        for resource in images:
            if self.is_url(resource):
                # For URLs, directly use the URL in the prompt
                user_content.append({"type": "image_url", "image_url": {"url": resource}})
            else:
                # For local files, read and encode as base64
                try:
                    with open(resource, "rb") as image_file:
                        image_data = image_file.read()
                        base64_image = base64.b64encode(image_data).decode("utf-8")

                    # Detect MIME type from file extension
                    from .utils import detect_image_mime_type

                    mime_type = detect_image_mime_type(resource)

                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                        }
                    )
                except Exception as e:
                    logger.error(f"Error reading image file {resource}: {e}")

        # Return the complete message structure
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

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
        Send a prompt to the OpenAI model and get the response.

        Args:
            model: The OpenAI model identifier (e.g., "gpt-4o", "gpt-3.5-turbo")
            prompt: The text prompt to send
            images: List of image paths/URLs to include
            system_prompt: System prompt to use
            response_format: Optional Pydantic model for structured output
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            LLMResponse object with the provider's response
        """
        # Prepare messages with any images
        messages = self._prepare_message_with_images(prompt, images, system_prompt)

        # Extract OpenAI-specific parameters from settings and kwargs
        params = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.settings.get("temperature", 0.5)),
        }

        # Add optional parameters if specified
        optional_params = [
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

        # Handle structured output
        if response_format:
            # Check if it's a Pydantic model
            if hasattr(response_format, "model_json_schema"):
                # Use beta.chat.completions.parse for structured output
                try:
                    params["response_format"] = response_format
                    raw_response = self.api_client.beta.chat.completions.parse(**params)
                    return self._create_response_from_parsed(raw_response, model)
                except Exception as e:
                    logger.warning(f"Structured output failed, falling back to JSON mode: {e}")
                    # Fall through to JSON object mode

            # Fallback to JSON object mode
            params["response_format"] = {"type": "json_object"}
            if hasattr(response_format, "model_json_schema"):
                schema_prompt = f"\n\nYou MUST respond with valid JSON matching this exact schema: {json.dumps(response_format.model_json_schema())}"
                params["messages"][0]["content"] += schema_prompt

        # Send the request to OpenAI
        raw_response = self.api_client.chat.completions.create(**params)

        return self._create_response_from_raw(raw_response, model, response_format)

    def _create_response_from_parsed(self, raw_response: Any, model: str) -> LLMResponse:
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
        if hasattr(choice.message, "parsed") and choice.message.parsed:
            text = choice.message.parsed.model_dump_json()
        else:
            text = choice.message.content or ""

        usage = Usage()
        if hasattr(raw_response, "usage") and raw_response.usage:
            usage = Usage(
                input_tokens=raw_response.usage.prompt_tokens,
                output_tokens=raw_response.usage.completion_tokens,
                total_tokens=raw_response.usage.total_tokens,
            )

        return LLMResponse(
            text=text,
            model=raw_response.model,
            provider=self.PROVIDER_ID,
            finish_reason=choice.finish_reason or "unknown",
            usage=usage,
            raw_response=raw_response,
        )

    def _create_response_from_raw(
        self, raw_response: Any, model: str, response_format: Optional[Any]
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
        text = choice.message.content or ""

        # If response_format was provided and response is JSON, try to parse it
        if response_format and hasattr(response_format, "model_json_schema"):
            try:
                # Try to extract JSON if wrapped in markdown
                content = text
                if "```json" in content:
                    import re

                    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", content)
                    if json_match:
                        content = json_match.group(1)
                elif "```" in content:
                    import re

                    json_match = re.search(r"```\s*([\s\S]*?)\s*```", content)
                    if json_match:
                        content = json_match.group(1)

                # Parse and validate with Pydantic
                json_data = json.loads(content)
                validated = response_format(**json_data)
                text = validated.model_dump_json()
            except Exception as e:
                logger.warning(f"Failed to parse/validate structured response: {e}")
                # Keep original text

        usage = Usage()
        if hasattr(raw_response, "usage") and raw_response.usage:
            usage = Usage(
                input_tokens=raw_response.usage.prompt_tokens,
                output_tokens=raw_response.usage.completion_tokens,
                total_tokens=raw_response.usage.total_tokens,
            )
            # OpenRouter may include cost information
            if hasattr(raw_response.usage, "cost"):
                usage.estimated_cost_usd = raw_response.usage.cost
            if hasattr(raw_response.usage, "cached_tokens"):
                usage.cached_tokens = raw_response.usage.cached_tokens

        return LLMResponse(
            text=text,
            model=raw_response.model,
            provider=self.PROVIDER_ID,
            finish_reason=choice.finish_reason or "unknown",
            usage=usage,
            raw_response=raw_response,
        )

    def get_model_list(self) -> List[Tuple[str, Optional[str]]]:
        """
        Get a list of available models from OpenAI.

        Returns:
            List of tuples (model_id, created_date)
        """
        if self.api_client is None:
            raise ValueError("OpenAI client is not initialized.")

        raw_list = self.api_client.models.list()
        model_list = []

        for model in raw_list:
            readable_date = datetime.fromtimestamp(model.created, tz=timezone.utc).strftime(
                "%Y-%m-%d"
            )
            model_list.append((model.id, readable_date))

        return model_list
