"""
Tests for OpenAI client implementation.
"""

import json
from unittest.mock import Mock, patch
from ai_client import create_ai_client, OpenAIClient
from ai_client.response import LLMResponse


class TestOpenAIClient:
    """Tests for OpenAIClient."""

    def test_openai_client_initialization(self):
        """Test OpenAI client initialization."""
        with patch("ai_client.openai_client.OpenAI") as mock_openai:
            client = create_ai_client("openai", api_key="test-key")

            assert isinstance(client, OpenAIClient)
            assert client.PROVIDER_ID == "openai"
            assert client.SUPPORTS_MULTIMODAL is True
            mock_openai.assert_called_once_with(api_key="test-key")

    def test_openai_client_with_base_url(self):
        """Test OpenAI client with custom base URL."""
        with patch("ai_client.openai_client.OpenAI") as mock_openai:
            create_ai_client("openai", api_key="test-key", base_url="https://custom.api.com")

            mock_openai.assert_called_once_with(
                api_key="test-key", base_url="https://custom.api.com"
            )

    def test_prompt_text_only(self, mock_openai_response):
        """Test text-only prompt."""
        with patch("ai_client.openai_client.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_openai_response

            client = create_ai_client("openai", api_key="test-key")
            response = client.prompt("gpt-4", "Hello!")

            # Check response
            assert isinstance(response, LLMResponse)
            assert response.text == "Hello! I'm an AI assistant."
            assert response.model == "gpt-4"
            assert response.provider == "openai"
            assert response.finish_reason == "stop"
            assert response.usage.input_tokens == 10
            assert response.usage.output_tokens == 20
            assert response.usage.total_tokens == 30
            assert response.duration >= 0

            # Check API call
            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["model"] == "gpt-4"
            assert len(call_args.kwargs["messages"]) == 2  # system + user

    def test_prompt_with_images(self, mock_openai_response, sample_image_path):
        """Test prompt with images."""
        with patch("ai_client.openai_client.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_openai_response

            client = create_ai_client("openai", api_key="test-key")
            response = client.prompt("gpt-4o", "Describe this image", images=[sample_image_path])

            assert isinstance(response, LLMResponse)

            # Check that images were included in the API call
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args.kwargs["messages"]
            user_message = messages[1]  # Second message is user
            assert isinstance(user_message["content"], list)
            assert any(item["type"] == "image_url" for item in user_message["content"])

    def test_prompt_with_custom_temperature(self, mock_openai_response):
        """Test prompt with custom temperature."""
        with patch("ai_client.openai_client.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_openai_response

            client = create_ai_client("openai", api_key="test-key")
            client.prompt("gpt-4", "Hello", temperature=0.9)

            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["temperature"] == 0.9

    def test_prompt_with_structured_output(self, mock_pydantic_model):
        """Test prompt with structured output (Pydantic model)."""
        with patch("ai_client.openai_client.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            # Mock parsed response
            mock_response = Mock()
            mock_response.model = "gpt-4"
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.parsed = mock_pydantic_model(name="test", value=42)
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 20
            mock_response.usage.total_tokens = 30
            # Add prompt_tokens_details to prevent Mock comparison errors
            mock_response.usage.prompt_tokens_details = Mock()
            mock_response.usage.prompt_tokens_details.cached_tokens = 0

            mock_client.beta.chat.completions.parse.return_value = mock_response

            client = create_ai_client("openai", api_key="test-key")
            response = client.prompt("gpt-4", "Extract data", response_format=mock_pydantic_model)

            # Check that structured output was used
            mock_client.beta.chat.completions.parse.assert_called_once()

            # Check response contains JSON
            assert isinstance(response, LLMResponse)
            data = json.loads(response.text)
            assert data["name"] == "test"
            assert data["value"] == 42

            # Check that parsed field is populated
            assert response.parsed is not None
            assert isinstance(response.parsed, dict)
            assert response.parsed["name"] == "test"
            assert response.parsed["value"] == 42

    def test_error_response_on_exception(self):
        """Test that errors are handled gracefully."""
        with patch("ai_client.openai_client.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.side_effect = Exception("API Error")

            client = create_ai_client("openai", api_key="test-key")
            response = client.prompt("gpt-4", "Hello")

            # Should return error response, not raise
            assert isinstance(response, LLMResponse)
            assert response.finish_reason == "error"
            assert response.text == ""
            assert "error" in response.raw_response

    def test_get_model_list(self):
        """Test get_model_list method."""
        with patch("ai_client.openai_client.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            # Mock model list
            mock_model1 = Mock()
            mock_model1.id = "gpt-4"
            mock_model1.created = 1678896000  # timestamp

            mock_model2 = Mock()
            mock_model2.id = "gpt-3.5-turbo"
            mock_model2.created = 1678896000

            mock_client.models.list.return_value = [mock_model1, mock_model2]

            client = create_ai_client("openai", api_key="test-key")
            models = client.get_model_list()

            assert len(models) == 2
            assert models[0][0] == "gpt-4"
            assert models[1][0] == "gpt-3.5-turbo"
            # Check that dates are formatted
            assert isinstance(models[0][1], str)
