"""
Unified AI client package for multiple providers.

This package provides a standardized interface for interacting with various AI models through
their respective APIs, with separate implementation files for each provider.

Key features:
- Abstract base client defining the common interface for all AI providers
- Provider-specific implementations for OpenAI, Google Gemini, Anthropic Claude, Mistral, DeepSeek, and Qwen
- Support for both text-only and multimodal (text + images) content
- Consistent LLMResponse format with detailed usage tracking
- Factory method to create appropriate client based on provider
- Built-in retry logic and rate limiting
- Optional async support for parallel processing

Technical implementation details:
- OpenAI: Uses the chat completions API with multimodal support and structured output
- Google Gemini: Uses the GenerativeModel class with image support
- Anthropic Claude: Uses the messages API with tool-based structured output
- Mistral: Uses the chat completion API with multimodal support
- DeepSeek: OpenAI-compatible API with custom base URL
- Qwen: OpenAI-compatible API with custom base URL
- OpenRouter/sciCORE: Use OpenAI client with custom base URLs
"""

from .base_client import BaseAIClient, create_ai_client
from .openai_client import OpenAIClient
from .gemini_client import GeminiClient
from .claude_client import ClaudeClient
from .mistral_client import MistralClient
from .deepseek_client import DeepSeekClient
from .qwen_client import QwenClient
from .response import LLMResponse, Usage
from .utils import (
    retry_with_exponential_backoff,
    is_rate_limit_error,
    detect_image_mime_type,
    RateLimitError,
    APIError,
)

__version__ = "0.1.2"

__all__ = [
    # Core classes
    "BaseAIClient",
    "create_ai_client",
    # Provider-specific clients
    "OpenAIClient",
    "GeminiClient",
    "ClaudeClient",
    "MistralClient",
    "DeepSeekClient",
    "QwenClient",
    # Response and utility classes
    "LLMResponse",
    "Usage",
    # Utility functions and exceptions
    "retry_with_exponential_backoff",
    "is_rate_limit_error",
    "detect_image_mime_type",
    "RateLimitError",
    "APIError",
    # Version
    "__version__",
]
