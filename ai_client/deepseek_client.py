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

# DeepSeek model name fragments that indicate vision/multimodal support. Not every vision
# model says so in its name -- "deepseek-flash" accepts images -- so pass extra fragments
# as the "vision_model_keywords" setting rather than waiting for a release.
_VISION_MODEL_KEYWORDS = ("vl", "vision", "deepseek-flash")


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

    def _supports_images(self, model: str) -> bool:
        """
        Report whether a DeepSeek model accepts image inputs.

        Args:
            model: Model identifier as requested

        Returns:
            True if the model name matches a known vision fragment
        """
        extra = self.settings.get("vision_model_keywords") or ()
        keywords = tuple(extra) + tuple(_VISION_MODEL_KEYWORDS)
        return any(keyword in model.lower() for keyword in keywords)

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
        """Strip images for DeepSeek models that cannot accept them, then delegate."""
        if images and not self._supports_images(model):
            logger.warning(
                f"DeepSeek model '{model}' is not known to support image inputs. Images "
                f"will be ignored; add a matching fragment to the "
                f"'vision_model_keywords' setting if it does."
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
