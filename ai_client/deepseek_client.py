"""
DeepSeek-specific client using OpenAI-compatible API.

DeepSeek uses an OpenAI-compatible API, so we can reuse the OpenAIClient
with a custom base URL.
"""

from .openai_client import OpenAIClient


class DeepSeekClient(OpenAIClient):
    """
    DeepSeek client using OpenAI-compatible API.

    This client simply extends OpenAIClient with a custom base URL.
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
