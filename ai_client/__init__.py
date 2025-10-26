"""
Unified AI client package for multiple providers including OpenAI, Google Gemini, Anthropic Claude, and Mistral.

This package provides a standardized interface for interacting with various AI models through
their respective APIs, with separate implementation files for each provider.

Key features:
- Abstract base client defining the common interface for all AI providers
- Provider-specific implementations for OpenAI, Google Gemini, Anthropic Claude, and Mistral
- Support for both text-only and multimodal (text + images) content
- Consistent response format and timing information
- Factory method to create appropriate client based on provider

Technical implementation details:
- OpenAI: Uses the chat completions API with multimodal support
- Google Gemini: Uses the GenerativeModel class with image uploads
- Anthropic Claude: Uses the messages API with multimodal support where available
- Mistral: Uses the chat completion API (text-only currently)
"""
from .base_client import BaseAIClient, create_ai_client
from .openai_client import OpenAIClient
from .gemini_client import GeminiClient
from .claude_client import ClaudeClient
from .mistral_client import MistralClient

__all__ = [
    'BaseAIClient',
    'OpenAIClient',
    'GeminiClient',
    'ClaudeClient', 
    'MistralClient',
    'create_ai_client'
]