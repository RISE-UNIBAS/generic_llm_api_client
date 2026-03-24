"""
DeepSeek-specific client using OpenAI-compatible API.

DeepSeek uses an OpenAI-compatible API, so we can reuse the OpenAIClient
with a custom base URL.
"""

import logging
from typing import Any, List, Optional

from .openai_client import OpenAIClient
from .response import LLMResponse

logger = logging.getLogger(__name__)

# DeepSeek model name fragments that indicate vision/multimodal support
_VISION_MODEL_KEYWORDS = ("vl",)


class DeepSeekClient(OpenAIClient):
    """
    DeepSeek client using OpenAI-compatible API.

    This client simply extends OpenAIClient with a custom base URL.
    Only DeepSeek VL models (names containing 'vl') support image inputs.
    """

    PROVIDER_ID = "deepseek"
    SUPPORTS_MULTIMODAL = True

    def _init_client(self):
        """Initialize the client with DeepSeek's base URL."""
        # Override base_url if not provided
        if not self.base_url:
            self.base_url = "https://api.deepseek.com/v1"

        # Call parent initialization
        super()._init_client()

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
        """Strip images for non-VL DeepSeek models before delegating to OpenAIClient."""
        if images and not any(kw in model.lower() for kw in _VISION_MODEL_KEYWORDS):
            logger.warning(
                f"DeepSeek model '{model}' does not support image inputs "
                f"(only VL models do). Images will be ignored."
            )
            images = []

        return super()._do_prompt(
            model=model,
            prompt=prompt,
            messages=messages,
            images=images,
            system_prompt=system_prompt,
            response_format=response_format,
            cache=cache,
            file_content=file_content,
            **kwargs,
        )
