"""
Tests for Gemini, Mistral, DeepSeek, and Qwen clients.
"""
import pytest
from unittest.mock import Mock, patch
from ai_client import create_ai_client, GeminiClient, MistralClient, DeepSeekClient, QwenClient
from ai_client.response import LLMResponse


class TestGeminiClient:
    """Tests for GeminiClient."""

    def test_gemini_client_initialization(self):
        """Test Gemini client initialization."""
        with patch('ai_client.gemini_client.genai'):
            client = create_ai_client('genai', api_key='test-key')

            assert isinstance(client, GeminiClient)
            assert client.PROVIDER_ID == 'genai'
            assert client.SUPPORTS_MULTIMODAL is True

    def test_prompt_text_only(self, mock_gemini_response):
        """Test text-only prompt."""
        with patch('ai_client.gemini_client.genai') as mock_genai:
            mock_client = Mock()
            mock_genai.Client.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_gemini_response

            client = create_ai_client('genai', api_key='test-key')
            response, duration = client.prompt('gemini-2.0-flash-exp', 'Hello!')

            # Check response
            assert isinstance(response, LLMResponse)
            assert response.text == "Hello! I'm Gemini."
            assert response.provider == "genai"
            assert response.usage.input_tokens == 12
            assert response.usage.output_tokens == 18
            assert duration >= 0

    def test_prompt_with_images(self, mock_gemini_response, sample_image_path):
        """Test prompt with images."""
        with patch('ai_client.gemini_client.genai') as mock_genai:
            mock_client = Mock()
            mock_genai.Client.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_gemini_response

            client = create_ai_client('genai', api_key='test-key')
            response, duration = client.prompt(
                'gemini-2.0-flash-exp',
                'Describe this',
                images=[sample_image_path]
            )

            assert isinstance(response, LLMResponse)


class TestMistralClient:
    """Tests for MistralClient."""

    def test_mistral_client_initialization(self):
        """Test Mistral client initialization."""
        with patch('ai_client.mistral_client.Mistral'):
            client = create_ai_client('mistral', api_key='test-key')

            assert isinstance(client, MistralClient)
            assert client.PROVIDER_ID == 'mistral'
            assert client.SUPPORTS_MULTIMODAL is True

    def test_prompt_text_only(self, mock_mistral_response):
        """Test text-only prompt."""
        with patch('ai_client.mistral_client.Mistral') as mock_mistral_class:
            mock_client = Mock()
            mock_mistral_class.return_value = mock_client
            mock_client.chat.complete.return_value = mock_mistral_response

            client = create_ai_client('mistral', api_key='test-key')
            response, duration = client.prompt('mistral-large-latest', 'Hello!')

            # Check response
            assert isinstance(response, LLMResponse)
            assert response.text == "Hello! I'm Mistral."
            assert response.provider == "mistral"
            assert response.usage.input_tokens == 11
            assert response.usage.output_tokens == 19
            assert duration >= 0

    def test_prompt_with_images(self, mock_mistral_response, sample_image_path):
        """Test prompt with images."""
        with patch('ai_client.mistral_client.Mistral') as mock_mistral_class:
            mock_client = Mock()
            mock_mistral_class.return_value = mock_client
            mock_client.chat.complete.return_value = mock_mistral_response

            client = create_ai_client('mistral', api_key='test-key')
            response, duration = client.prompt(
                'mistral-large-latest',
                'Describe this',
                images=[sample_image_path]
            )

            assert isinstance(response, LLMResponse)


class TestDeepSeekClient:
    """Tests for DeepSeekClient."""

    def test_deepseek_client_initialization(self):
        """Test DeepSeek client initialization."""
        with patch('ai_client.openai_client.OpenAI') as mock_openai:
            client = create_ai_client('deepseek', api_key='test-key')

            assert isinstance(client, DeepSeekClient)
            assert client.PROVIDER_ID == 'deepseek'
            assert client.SUPPORTS_MULTIMODAL is True

            # Should have set custom base URL
            call_args = mock_openai.call_args
            assert call_args.kwargs['base_url'] == 'https://api.deepseek.com/v1'

    def test_deepseek_inherits_openai_functionality(self, mock_openai_response):
        """Test that DeepSeek inherits OpenAI functionality."""
        with patch('ai_client.openai_client.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_openai_response

            client = create_ai_client('deepseek', api_key='test-key')
            response, duration = client.prompt('deepseek-chat', 'Hello!')

            # Should work like OpenAI client
            assert isinstance(response, LLMResponse)
            assert response.text == "Hello! I'm an AI assistant."


class TestQwenClient:
    """Tests for QwenClient."""

    def test_qwen_client_initialization(self):
        """Test Qwen client initialization."""
        with patch('ai_client.openai_client.OpenAI') as mock_openai:
            client = create_ai_client('qwen', api_key='test-key')

            assert isinstance(client, QwenClient)
            assert client.PROVIDER_ID == 'qwen'
            assert client.SUPPORTS_MULTIMODAL is True

            # Should have set custom base URL
            call_args = mock_openai.call_args
            assert 'aliyuncs.com' in call_args.kwargs['base_url']

    def test_qwen_inherits_openai_functionality(self, mock_openai_response):
        """Test that Qwen inherits OpenAI functionality."""
        with patch('ai_client.openai_client.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_openai_response

            client = create_ai_client('qwen', api_key='test-key')
            response, duration = client.prompt('qwen-turbo', 'Hello!')

            # Should work like OpenAI client
            assert isinstance(response, LLMResponse)
            assert response.text == "Hello! I'm an AI assistant."
