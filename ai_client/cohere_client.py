"""
Cohere-specific implementation of the BaseAIClient.

This module provides the CohereClient class, which implements the BaseAIClient
interface specifically for Cohere's API, supporting both text-only and multimodal
interactions with vision models.
"""

import base64
import json
import logging
from typing import List, Tuple, Any, Optional

import cohere

from .base_client import BaseAIClient
from .response import LLMResponse, Usage
from .pricing import calculate_cost

logger = logging.getLogger(__name__)


class CohereClient(BaseAIClient):
    """
    Cohere-specific implementation of the BaseAIClient.

    This class implements the BaseAIClient interface for Cohere's API,
    supporting text-only and multimodal requests via the chat API.

    Key features:
    - Integration with Cohere's chat API (ClientV2)
    - Support for multimodal content (text + images) with vision models
    - Support for Cohere-specific parameters (temperature, max_tokens, etc.)
    - Structured output via JSON mode
    """

    PROVIDER_ID = "cohere"
    SUPPORTS_MULTIMODAL = True  # Cohere supports images with vision models

    def _init_client(self):
        """Initialize the Cohere client with the provided API key."""
        self.api_client = cohere.ClientV2(api_key=self.api_key)

    def _prepare_content_with_images(self, prompt: str, images: List[str]) -> List[dict]:
        """
        Prepare Cohere content with text and images.

        Args:
            prompt: The text prompt
            images: List of image paths/URLs

        Returns:
            List of content blocks for Cohere API
        """
        content = [{"type": "text", "text": prompt}]

        # Add images if any
        for resource in images:
            try:
                if self.is_url(resource):
                    # For URLs, use directly
                    content.append(
                        {"type": "image_url", "image_url": {"url": resource, "detail": "auto"}}
                    )
                else:
                    # For local files, encode as base64
                    with open(resource, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

                        # Detect MIME type from file extension
                        from .utils import detect_image_mime_type

                        mime_type = detect_image_mime_type(resource)

                        data_uri = f"data:{mime_type};base64,{base64_image}"
                        content.append(
                            {"type": "image_url", "image_url": {"url": data_uri, "detail": "auto"}}
                        )
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
        Send a prompt to the Cohere model and get the response.

        Args:
            model: The Cohere model identifier
            prompt: The text prompt to send
            messages: Optional conversation history (multi-turn)
            images: List of image paths/URLs (for vision models)
            system_prompt: System prompt to use (preamble in Cohere terminology)
            response_format: Optional Pydantic model for structured output
            cache: Not used (Cohere doesn't support prompt caching yet)
            file_content: Not used (files already appended to prompt)
            **kwargs: Additional Cohere-specific parameters

        Returns:
            LLMResponse object with the provider's response
        """
        # Prepare content with images if any
        if images:
            content = self._prepare_content_with_images(prompt, images)
        else:
            content = prompt

        # Build messages for chat API
        if messages:
            # Multi-turn conversation - use existing messages
            chat_messages = messages.copy()
            # Update last message with current content
            chat_messages[-1]["content"] = content
        else:
            # Single-turn conversation
            chat_messages = [{"role": "user", "content": content}]

        # Build request parameters
        params = {
            "model": model,
            "messages": chat_messages,
        }

        # Add system prompt (preamble) if provided
        if system_prompt:
            params["preamble"] = system_prompt

        # Extract Cohere-specific settings
        if "temperature" in kwargs:
            params["temperature"] = kwargs["temperature"]
        elif "temperature" in self.settings:
            params["temperature"] = self.settings["temperature"]

        if "max_tokens" in kwargs:
            params["max_tokens"] = kwargs["max_tokens"]
        elif "max_tokens" in self.settings:
            params["max_tokens"] = self.settings["max_tokens"]

        # Handle structured output
        if response_format and hasattr(response_format, "model_json_schema"):
            schema = response_format.model_json_schema()
            # Add schema to prompt
            schema_prompt = (
                f"\n\nReturn a JSON response matching this exact schema: {json.dumps(schema)}"
            )
            # Append to the last user message
            if isinstance(chat_messages[-1]["content"], list):
                # Find text content and append
                for item in chat_messages[-1]["content"]:
                    if item["type"] == "text":
                        item["text"] += schema_prompt
                        break
            else:
                chat_messages[-1]["content"] += schema_prompt

            params["messages"] = chat_messages
            params["response_format"] = {"type": "json_object"}

        # Add any additional Cohere-specific parameters
        cohere_params = ["frequency_penalty", "presence_penalty", "seed", "safety_mode"]
        for param in cohere_params:
            if param in kwargs:
                params[param] = kwargs[param]
            elif param in self.settings:
                params[param] = self.settings[param]

        # Send the request
        raw_response = self.api_client.chat(**params)

        return self._create_response_from_raw(raw_response, model, response_format)

    def _create_response_from_raw(
        self, raw_response: Any, model: str, response_format: Optional[Any]
    ) -> LLMResponse:
        """
        Create LLMResponse from Cohere raw response.

        Args:
            raw_response: Raw Cohere response object
            model: Model identifier
            response_format: Pydantic model if structured output was requested

        Returns:
            LLMResponse object
        """
        # Extract text from response
        text = ""
        if hasattr(raw_response, "message") and hasattr(raw_response.message, "content"):
            # Content is a list of content blocks
            for content_block in raw_response.message.content:
                if hasattr(content_block, "text"):
                    text += content_block.text

        parsed_data = None

        # If response_format was provided, try to validate
        if response_format and hasattr(response_format, "model_json_schema") and text:
            try:
                json_data = json.loads(text)
                validated = response_format(**json_data)
                text = validated.model_dump_json()
                parsed_data = json_data
            except Exception as e:
                logger.warning(f"Failed to validate Cohere structured response: {e}")
                # Keep original text

        # Extract usage information
        usage = Usage()
        if hasattr(raw_response, "usage"):
            usage_info = raw_response.usage

            # Cohere provides tokens and billed_units
            if hasattr(usage_info, "tokens"):
                usage.input_tokens = getattr(usage_info.tokens, "input_tokens", 0)
                usage.output_tokens = getattr(usage_info.tokens, "output_tokens", 0)
                usage.total_tokens = usage.input_tokens + usage.output_tokens

            # Calculate cost if pricing data is available
            costs = calculate_cost(
                self.PROVIDER_ID,
                model,
                usage.input_tokens,
                usage.output_tokens,
            )
            if costs is not None:
                usage.input_cost_usd, usage.output_cost_usd, usage.estimated_cost_usd = costs

        # Extract finish reason
        finish_reason = "stop"
        if hasattr(raw_response, "finish_reason"):
            finish_reason = raw_response.finish_reason.lower()

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
        Get a list of available models from Cohere.

        Returns:
            List of tuples (model_id, created_date)
            Note: Cohere doesn't provide created_date, so it will be None
        """
        if self.api_client is None:
            raise ValueError("Cohere client is not initialized.")

        model_list = []

        try:
            # Use the Client (not ClientV2) for models.list()
            # or access via raw API call
            v1_client = cohere.Client(api_key=self.api_key)
            raw_list = v1_client.models.list()

            # Extract model information
            if hasattr(raw_list, "models"):
                for model in raw_list.models:
                    model_id = model.name if hasattr(model, "name") else str(model)
                    model_list.append((model_id, None))
            else:
                # Fallback if structure is different
                for model in raw_list:
                    model_id = model.name if hasattr(model, "name") else str(model)
                    model_list.append((model_id, None))

        except Exception as e:
            logger.error(f"Error listing Cohere models: {e}")
            # Return empty list on error
            model_list = []

        return model_list
