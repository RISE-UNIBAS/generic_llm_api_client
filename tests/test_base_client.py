"""
Tests for BaseAIClient and create_ai_client factory.
"""

import pytest
from unittest.mock import patch
from ai_client.base_client import create_ai_client
from ai_client import (
    OpenAIClient,
    ClaudeClient,
    GeminiClient,
    MistralClient,
    DeepSeekClient,
    QwenClient,
)


class TestCreateAIClient:
    """Tests for create_ai_client factory function."""

    def test_create_openai_client(self):
        """Test creating OpenAI client."""
        with patch("ai_client.openai_client.OpenAI"):
            client = create_ai_client("openai", api_key="test-key")
            assert isinstance(client, OpenAIClient)
            assert client.PROVIDER_ID == "openai"

    def test_create_claude_client(self):
        """Test creating Claude client."""
        with patch("ai_client.claude_client.Anthropic"):
            client = create_ai_client("anthropic", api_key="test-key")
            assert isinstance(client, ClaudeClient)
            assert client.PROVIDER_ID == "anthropic"

    def test_create_gemini_client(self):
        """Test creating Gemini client."""
        with patch("ai_client.gemini_client.genai"):
            client = create_ai_client("genai", api_key="test-key")
            assert isinstance(client, GeminiClient)
            assert client.PROVIDER_ID == "genai"

    def test_create_mistral_client(self):
        """Test creating Mistral client."""
        with patch("ai_client.mistral_client.Mistral"):
            client = create_ai_client("mistral", api_key="test-key")
            assert isinstance(client, MistralClient)
            assert client.PROVIDER_ID == "mistral"

    def test_create_deepseek_client(self):
        """Test creating DeepSeek client."""
        with patch("ai_client.openai_client.OpenAI"):
            client = create_ai_client("deepseek", api_key="test-key")
            assert isinstance(client, DeepSeekClient)
            assert client.PROVIDER_ID == "deepseek"

    def test_create_qwen_client(self):
        """Test creating Qwen client."""
        with patch("ai_client.openai_client.OpenAI"):
            client = create_ai_client("qwen", api_key="test-key")
            assert isinstance(client, QwenClient)
            assert client.PROVIDER_ID == "qwen"

    def test_create_openrouter_client(self):
        """Test creating OpenRouter client (uses OpenAI)."""
        with patch("ai_client.openai_client.OpenAI"):
            client = create_ai_client(
                "openrouter", api_key="test-key", base_url="https://openrouter.ai/api/v1"
            )
            assert isinstance(client, OpenAIClient)
            assert client.base_url == "https://openrouter.ai/api/v1"

    def test_create_scicore_client(self):
        """Test creating sciCORE client (uses OpenAI)."""
        with patch("ai_client.openai_client.OpenAI"):
            client = create_ai_client(
                "scicore",
                api_key="test-key",
                base_url="https://llm-api-h200.ceda.unibas.ch/litellm/v1",
            )
            assert isinstance(client, OpenAIClient)

    def test_unsupported_provider_raises_error(self):
        """Test that unsupported provider raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported AI provider: invalid_provider"):
            create_ai_client("invalid_provider", api_key="test-key")

    def test_create_with_system_prompt(self):
        """Test creating client with custom system prompt."""
        with patch("ai_client.openai_client.OpenAI"):
            client = create_ai_client(
                "openai", api_key="test-key", system_prompt="You are a helpful assistant"
            )
            assert client.system_prompt == "You are a helpful assistant"

    def test_create_with_settings(self):
        """Test creating client with custom settings."""
        with patch("ai_client.openai_client.OpenAI"):
            client = create_ai_client("openai", api_key="test-key", temperature=0.7, max_tokens=500)
            assert client.settings["temperature"] == 0.7
            assert client.settings["max_tokens"] == 500


class TestBaseAIClient:
    """Tests for BaseAIClient functionality."""

    def test_client_initialization(self):
        """Test basic client initialization."""
        with patch("ai_client.openai_client.OpenAI"):
            client = create_ai_client("openai", api_key="test-key")

            assert client.api_key == "test-key"
            assert client.system_prompt is not None
            assert client.init_time is not None
            assert client.end_time is None

    def test_elapsed_time(self):
        """Test elapsed_time property."""
        with patch("ai_client.openai_client.OpenAI"):
            client = create_ai_client("openai", api_key="test-key")

            import time

            time.sleep(0.1)

            elapsed = client.elapsed_time
            assert elapsed >= 0.1

    def test_end_client(self):
        """Test end_client method."""
        with patch("ai_client.openai_client.OpenAI"):
            client = create_ai_client("openai", api_key="test-key")

            client.end_client()

            assert client.end_time is not None
            assert client.api_client is None

    def test_elapsed_time_after_end(self):
        """Test elapsed_time after end_client."""
        with patch("ai_client.openai_client.OpenAI"):
            client = create_ai_client("openai", api_key="test-key")

            import time

            time.sleep(0.1)

            client.end_client()

            elapsed = client.elapsed_time
            assert elapsed >= 0.1

    def test_is_url(self):
        """Test is_url static method."""
        with patch("ai_client.openai_client.OpenAI"):
            client = create_ai_client("openai", api_key="test-key")

            assert client.is_url("https://example.com/image.jpg") is True
            assert client.is_url("http://example.com/image.jpg") is True
            assert client.is_url("/path/to/image.jpg") is False
            assert client.is_url("image.jpg") is False

    def test_has_multimodal_support(self):
        """Test has_multimodal_support method."""
        with patch("ai_client.openai_client.OpenAI"):
            openai_client = create_ai_client("openai", api_key="test-key")
            assert openai_client.has_multimodal_support() is True

        with patch("ai_client.claude_client.Anthropic"):
            claude_client = create_ai_client("anthropic", api_key="test-key")
            assert claude_client.has_multimodal_support() is True

    def test_str_representation(self):
        """Test __str__ method."""
        with patch("ai_client.openai_client.OpenAI"):
            client = create_ai_client("openai", api_key="test-key")
            assert str(client) == "openai"

        with patch("ai_client.claude_client.Anthropic"):
            client = create_ai_client("anthropic", api_key="test-key")
            assert str(client) == "anthropic"
