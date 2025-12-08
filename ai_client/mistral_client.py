"""
Mistral-specific implementation of the BaseAIClient.

This module provides the MistralClient class, which implements the BaseAIClient
interface specifically for Mistral's API, supporting both text-only and multimodal
interactions.
"""

import base64
import json
import logging
from typing import List, Tuple, Any, Optional

from mistralai import Mistral

from .base_client import BaseAIClient
from .response import LLMResponse, Usage
from .pricing import calculate_cost

logger = logging.getLogger(__name__)


class MistralClient(BaseAIClient):
    """
    Mistral-specific implementation of the BaseAIClient.

    This class implements the BaseAIClient interface for Mistral's API,
    supporting text-only and multimodal requests via the chat completion API.

    Key features:
    - Integration with Mistral's chat completion API
    - Support for multimodal content (text + images)
    - Support for Mistral-specific parameters
    - Structured output via JSON mode
    """

    PROVIDER_ID = "mistral"
    SUPPORTS_MULTIMODAL = True  # Mistral supports images

    def _init_client(self):
        """Initialize the Mistral client with the provided API key."""
        self.api_client = Mistral(api_key=self.api_key)

    def _prepare_content_with_images(self, prompt: str, images: List[str]) -> List[dict]:
        """
        Prepare Mistral content with text and images.

        Args:
            prompt: The text prompt
            images: List of image paths/URLs

        Returns:
            List of content blocks for Mistral API
        """
        content = [{"type": "text", "text": prompt}]

        # Add images if any
        for resource in images:
            try:
                if self.is_url(resource):
                    # For URLs, use directly
                    data_uri = resource
                else:
                    # For local files, encode as base64
                    with open(resource, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

                        # Detect MIME type from file extension
                        from .utils import detect_image_mime_type

                        mime_type = detect_image_mime_type(resource)

                        data_uri = f"data:{mime_type};base64,{base64_image}"

                content.append({"type": "image_url", "image_url": {"url": data_uri}})
            except Exception as e:
                logger.error(f"Error processing image {resource}: {e}")

        return content

    def _do_prompt(
        self,
        model,
        prompt,
        messages=None,
        images=None,
        system_prompt=None,
        response_format=None,
        cache=False,
        file_content="",
        **kwargs,
    ) -> LLMResponse:
        """
        Send a prompt to the Mistral model and get the response.

        Args:
            model: The Mistral model identifier
            prompt: The text prompt to send
            messages: Optional conversation history (multi-turn)
            images: List of image paths/URLs
            system_prompt: System prompt to use
            response_format: Optional Pydantic model for structured output
            cache: Not used (Mistral doesn't support caching)
            file_content: Not used (files already appended to prompt)
            **kwargs: Additional Mistral-specific parameters

        Returns:
            LLMResponse object with the provider's response
        """
        # Prepare content with images
        content = self._prepare_content_with_images(prompt, images)

        # Build messages
        messages = [{"role": "user", "content": content}]

        # Add system message if provided
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        params = {"messages": messages, "model": model}

        # Handle structured output
        if response_format and hasattr(response_format, "model_json_schema"):
            schema = response_format.model_json_schema()
            schema_prompt = (
                f"\n\nReturn a JSON response matching this exact schema: {json.dumps(schema)}"
            )
            messages[-1]["content"] = prompt + schema_prompt
            params["response_format"] = {"type": "json_object"}

        # Send the request
        raw_response = self.api_client.chat.complete(**params)

        return self._create_response_from_raw(raw_response, model, response_format)

    def _create_response_from_raw(
        self, raw_response: Any, model: str, response_format: Optional[Any]
    ) -> LLMResponse:
        """
        Create LLMResponse from Mistral raw response.

        Args:
            raw_response: Raw Mistral response object
            model: Model identifier
            response_format: Pydantic model if structured output was requested

        Returns:
            LLMResponse object
        """
        choice = raw_response.choices[0]
        text = choice.message.content if hasattr(choice.message, "content") else ""
        parsed_data = None

        # If response_format was provided, try to validate
        if response_format and hasattr(response_format, "model_json_schema") and text:
            try:
                json_data = json.loads(text)
                validated = response_format(**json_data)
                text = validated.model_dump_json()
                parsed_data = json_data
            except Exception as e:
                logger.warning(f"Failed to validate Mistral structured response: {e}")
                # Keep original text

        usage = Usage()
        if hasattr(raw_response, "usage") and raw_response.usage:
            usage = Usage(
                input_tokens=raw_response.usage.prompt_tokens,
                output_tokens=raw_response.usage.completion_tokens,
                total_tokens=raw_response.usage.total_tokens,
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

        finish_reason = choice.finish_reason if hasattr(choice, "finish_reason") else "unknown"

        return LLMResponse(
            text=text,
            model=model,
            provider=self.PROVIDER_ID,
            finish_reason=finish_reason,
            usage=usage,
            raw_response=raw_response,
            parsed=parsed_data,
        )

    def get_model_list(self) -> List[Tuple[str, Optional[str]]]:
        """
        Get a list of available models from Mistral.

        Returns:
            List of tuples (model_id, created_date)
        """
        if self.api_client is None:
            raise ValueError("Mistral client is not initialized.")

        model_list = []
        raw_list = self.api_client.models.list()

        for model in raw_list:
            model_list.append((model.id, None))

        return model_list
