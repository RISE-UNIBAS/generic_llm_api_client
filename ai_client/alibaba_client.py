"""
Alibaba-specific client using OpenAI-compatible API.

Alibaba's models are served via DashScope, which exposes an
OpenAI-compatible endpoint, so we can reuse the OpenAIClient with a custom
base URL.
"""

from .openai_client import OpenAIClient


class AlibabaClient(OpenAIClient):
    """
    Alibaba client using OpenAI-compatible API (DashScope).

    This client simply extends OpenAIClient with Alibaba's DashScope base URL.
    """

    PROVIDER_ID = "alibaba"
    SUPPORTS_MULTIMODAL = True

    def _init_client(self):
        """Initialize the client with Alibaba DashScope's base URL."""
        if not self.base_url:
            self.base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        print(f"Initializing AlibabaClient with base URL: {self.base_url}")

        super()._init_client()
