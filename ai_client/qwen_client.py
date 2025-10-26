"""
Qwen-specific client for OpenAI API.
"""

from openai import OpenAI
from . import OpenAIClient


class QwenClient(OpenAIClient):
    """
    QwenClient class for OpenAI API.
    """
    
    PROVIDER_ID = "qwen"
    SUPPORTS_MULTIMODAL = True
    
    def _init_client(self):
        """Initialize the OpenAI client with the provided API key."""
        self.api_client = OpenAI(api_key=self.api_key,
                                 base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
