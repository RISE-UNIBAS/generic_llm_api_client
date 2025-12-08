"""
Google Gemini-specific implementation of the BaseAIClient.

This module provides the GeminiClient class, which implements the BaseAIClient
interface specifically for Google's Gemini API, supporting both text-only and
multimodal (text + images) content.
"""

import base64
import json
import logging
from typing import List, Tuple, Any, Optional

import google.genai as genai
from google.genai.types import GenerateContentConfig, Part
import requests

from .base_client import BaseAIClient
from .response import LLMResponse, Usage
from .pricing import calculate_cost

logger = logging.getLogger(__name__)


class GeminiClient(BaseAIClient):
    """
    Google Gemini-specific implementation of the BaseAIClient.

    This class implements the BaseAIClient interface for Google's Gemini API,
    handling both text-only and multimodal (text + images) requests.

    Key features:
    - Full support for multimodal content via Gemini Pro Vision models
    - Support for Gemini-specific parameters like top_k, top_p
    - Structured output via response schema
    """

    PROVIDER_ID = "genai"
    SUPPORTS_MULTIMODAL = True

    def _init_client(self):
        """Initialize Gemini API client with the provided API key."""
        self.api_client = genai.Client(api_key=self.api_key)

    def _prepare_content_with_images(self, prompt: str, images: List[str]) -> List[Any]:
        """
        Prepare Gemini content with text and images.

        Args:
            prompt: The text prompt
            images: List of image paths/URLs

        Returns:
            List of content parts for Gemini API
        """
        contents = [prompt]

        # Add images if any
        for resource in images:
            try:
                if self.is_url(resource):
                    # For URLs, fetch the image
                    response = requests.get(resource)
                    if response.status_code == 200:
                        image_data = response.content
                    else:
                        logger.error(
                            f"Failed to fetch image from URL {resource}: {response.status_code}"
                        )
                        continue
                else:
                    # For local files, read the image
                    with open(resource, "rb") as f:
                        image_data = f.read()

                # Detect MIME type from file extension
                from .utils import detect_image_mime_type

                mime_type = detect_image_mime_type(resource)

                # Create image part
                image_part = Part(
                    inline_data={
                        "mime_type": mime_type,
                        "data": base64.b64encode(image_data).decode("utf-8"),
                    }
                )
                contents.append(image_part)
            except Exception as e:
                logger.error(f"Error processing image {resource}: {e}")

        return contents

    @staticmethod
    def _remove_defaults_from_schema(schema):
        """Recursively remove default values from JSON schema (for GenAI compatibility)."""
        if isinstance(schema, dict):
            if "default" in schema:
                del schema["default"]
            for key, value in schema.items():
                if isinstance(value, (dict, list)):
                    GeminiClient._remove_defaults_from_schema(value)
        elif isinstance(schema, list):
            for item in schema:
                if isinstance(item, (dict, list)):
                    GeminiClient._remove_defaults_from_schema(item)

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
        Send a prompt to the Gemini model and get the response.

        Args:
            model: The Gemini model identifier (e.g., "gemini-pro", "gemini-pro-vision")
            prompt: The text prompt to send
            messages: Optional conversation history (multi-turn)
            images: List of image paths/URLs
            system_prompt: System prompt (prepended to prompt for Gemini)
            response_format: Optional Pydantic model for structured output
            cache: If True, uses cache_id from kwargs if provided
            file_content: Not used (files already appended to prompt)
            **kwargs: Additional Gemini-specific parameters
                cache_id: Reference to previously created cache

        Returns:
            LLMResponse object with the provider's response
        """
        # Prepend system prompt to user prompt for Gemini
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        # Prepare content with images
        contents = self._prepare_content_with_images(full_prompt, images)

        # Build generation config
        generation_config = {}

        # Extract Gemini-specific parameters
        optional_params = [
            "temperature",
            "top_p",
            "top_k",
            "max_output_tokens",
            "candidate_count",
            "stop_sequences",
        ]
        for param in optional_params:
            value = kwargs.get(param, self.settings.get(param))
            if value is not None:
                generation_config[param] = value

        # Handle structured output
        if response_format and hasattr(response_format, "model_json_schema"):
            schema = response_format.model_json_schema()
            # GenAI doesn't support default values in schema
            self._remove_defaults_from_schema(schema)

            generation_config["response_mime_type"] = "application/json"
            generation_config["response_schema"] = schema

        # Generate content
        # Try different approaches based on what parameters are available
        try:
            if generation_config:
                # Try passing config as a GenerateContentConfig object
                config = GenerateContentConfig(**generation_config)
                raw_response = self.api_client.models.generate_content(
                    model=model, contents=contents, config=config
                )
            else:
                raw_response = self.api_client.models.generate_content(
                    model=model, contents=contents
                )
        except TypeError as e:
            # If config parameter doesn't work, try passing generation_config directly
            if "config" in str(e):
                logger.warning(f"Config parameter not accepted, trying alternative API: {e}")
                if generation_config:
                    raw_response = self.api_client.models.generate_content(
                        model=model, contents=contents, generation_config=generation_config
                    )
                else:
                    raw_response = self.api_client.models.generate_content(
                        model=model, contents=contents
                    )
            else:
                raise

        return self._create_response_from_raw(raw_response, model, response_format)

    def _create_response_from_raw(
        self, raw_response: Any, model: str, response_format: Optional[Any]
    ) -> LLMResponse:
        """
        Create LLMResponse from Gemini raw response.

        Args:
            raw_response: Raw Gemini response object
            model: Model identifier
            response_format: Pydantic model if structured output was requested

        Returns:
            LLMResponse object
        """
        text = raw_response.text if hasattr(raw_response, "text") else ""
        parsed_data = None

        # If response_format was provided, try to validate
        if response_format and hasattr(response_format, "model_json_schema") and text:
            try:
                json_data = json.loads(text)
                validated = response_format(**json_data)
                text = validated.model_dump_json()
                parsed_data = json_data
            except Exception as e:
                logger.warning(f"Failed to validate Gemini structured response: {e}")
                # Keep original text

        usage = Usage()
        if hasattr(raw_response, "usage_metadata"):
            usage = Usage(
                input_tokens=raw_response.usage_metadata.prompt_token_count,
                output_tokens=raw_response.usage_metadata.candidates_token_count,
                total_tokens=raw_response.usage_metadata.total_token_count,
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

        # Determine finish reason
        finish_reason = "unknown"
        if hasattr(raw_response, "candidates") and raw_response.candidates:
            candidate = raw_response.candidates[0]
            if hasattr(candidate, "finish_reason"):
                # Extract the value from the enum (e.g., FinishReason.STOP -> 'STOP')
                fr = candidate.finish_reason
                if hasattr(fr, "value"):
                    finish_reason = fr.value
                elif hasattr(fr, "name"):
                    finish_reason = fr.name
                else:
                    finish_reason = str(fr)

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
        Get a list of available models from Gemini.

        Returns:
            List of tuples (model_id, created_date)
        """
        model_list = []

        try:
            raw_list = self.api_client.models.list()
            for model in raw_list:
                model_name = model.name if hasattr(model, "name") else str(model)
                # Remove 'models/' prefix if present
                if model_name.startswith("models/"):
                    model_name = model_name[7:]
                model_list.append((model_name, None))
        except Exception as e:
            logger.error(f"Error listing Gemini models: {e}")

        return model_list
