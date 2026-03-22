"""
Grok-specific client using OpenAI-compatible API.

xAI's Grok uses an OpenAI-compatible API, so we can reuse the OpenAIClient
with a custom base URL.
"""

from .openai_client import OpenAIClient


class XAIClient(OpenAIClient):
    """
    Grok client using xAI's OpenAI-compatible API.

    This client simply extends OpenAIClient with xAI's base URL.
    """

    PROVIDER_ID = "x-ai"
    SUPPORTS_MULTIMODAL = True

    def _init_client(self):
        """Initialize the client with xAI's base URL."""
        if not self.base_url:
            self.base_url = "https://api.x.ai/v1"

        super()._init_client()
