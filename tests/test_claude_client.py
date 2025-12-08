"""
Tests for Claude client implementation.
"""

import json
from unittest.mock import Mock, patch
from ai_client import create_ai_client, ClaudeClient
from ai_client.response import LLMResponse


class TestClaudeClient:
    """Tests for ClaudeClient."""

    def test_claude_client_initialization(self):
        """Test Claude client initialization."""
        with patch("ai_client.claude_client.Anthropic") as mock_anthropic:
            client = create_ai_client("anthropic", api_key="test-key")

            assert isinstance(client, ClaudeClient)
            assert client.PROVIDER_ID == "anthropic"
            assert client.SUPPORTS_MULTIMODAL is True
            mock_anthropic.assert_called_once()

    def test_prompt_text_only(self, mock_claude_response):
        """Test text-only prompt."""
        with patch("ai_client.claude_client.Anthropic") as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            mock_client.messages.create.return_value = mock_claude_response

            client = create_ai_client("anthropic", api_key="test-key")
            response = client.prompt("claude-3-5-sonnet-20241022", "Hello!")

            # Check response
            assert isinstance(response, LLMResponse)
            assert response.text == "Hello! I'm Claude."
            assert response.provider == "anthropic"
            assert response.finish_reason == "end_turn"
            assert response.usage.input_tokens == 15
            assert response.usage.output_tokens == 25
            assert response.duration >= 0

            # Check API call
            mock_client.messages.create.assert_called_once()
            call_args = mock_client.messages.create.call_args
            assert call_args.kwargs["model"] == "claude-3-5-sonnet-20241022"
            assert "messages" in call_args.kwargs
            assert "system" in call_args.kwargs

    def test_prompt_with_images(self, mock_claude_response, sample_image_path):
        """Test prompt with images."""
        with patch("ai_client.claude_client.Anthropic") as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            mock_client.messages.create.return_value = mock_claude_response

            client = create_ai_client("anthropic", api_key="test-key")
            response = client.prompt(
                "claude-3-5-sonnet-20241022", "Describe this image", images=[sample_image_path]
            )

            assert isinstance(response, LLMResponse)

            # Check that images were included
            call_args = mock_client.messages.create.call_args
            messages = call_args.kwargs["messages"]
            content = messages[0]["content"]
            assert isinstance(content, list)
            assert any(item["type"] == "image" for item in content)

    def test_prompt_with_structured_output(self, mock_pydantic_model):
        """Test prompt with structured output via tools."""
        with patch("ai_client.claude_client.Anthropic") as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client

            # Mock tool-based response
            mock_response = Mock()
            mock_response.model = "claude-3-5-sonnet-20241022"
            mock_response.content = [Mock()]
            mock_response.content[0].type = "tool_use"
            mock_response.content[0].name = "extract_structured_data"
            mock_response.content[0].input = {"name": "test", "value": 42}
            mock_response.stop_reason = "tool_use"
            mock_response.usage = Mock()
            mock_response.usage.input_tokens = 15
            mock_response.usage.output_tokens = 25

            mock_client.messages.create.return_value = mock_response

            client = create_ai_client("anthropic", api_key="test-key")
            response = client.prompt(
                "claude-3-5-sonnet-20241022", "Extract data", response_format=mock_pydantic_model
            )

            # Check that tools were used
            call_args = mock_client.messages.create.call_args
            assert "tools" in call_args.kwargs
            assert "tool_choice" in call_args.kwargs

            # Check response contains validated JSON
            assert isinstance(response, LLMResponse)
            data = json.loads(response.text)
            assert data["name"] == "test"
            assert data["value"] == 42

            # Check that parsed field is populated
            assert response.parsed is not None
            assert isinstance(response.parsed, dict)
            assert response.parsed["name"] == "test"
            assert response.parsed["value"] == 42

    def test_max_tokens_varies_by_model(self):
        """Test that max_tokens defaults vary by model."""
        with patch("ai_client.claude_client.Anthropic") as mock_anthropic_class:
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].type = "text"
            mock_response.content[0].text = "test"
            mock_response.stop_reason = "end_turn"
            mock_response.usage = Mock()
            mock_response.usage.input_tokens = 10
            mock_response.usage.output_tokens = 10
            mock_client.messages.create.return_value = mock_response

            client = create_ai_client("anthropic", api_key="test-key")

            # Test opus model
            client.prompt("claude-3-opus-20240229", "test")
            call_args = mock_client.messages.create.call_args
            assert call_args.kwargs["max_tokens"] == 4096

            # Test sonnet model
            client.prompt("claude-3-5-sonnet-20241022", "test")
            call_args = mock_client.messages.create.call_args
            assert call_args.kwargs["max_tokens"] == 8192
