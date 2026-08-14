"""
Tests for Gemini, Mistral, Cohere, DeepSeek, Alibaba, and HuggingFace clients.
"""

import pytest
from unittest.mock import Mock, patch
from ai_client import (
    create_ai_client,
    GeminiClient,
    MistralClient,
    CohereClient,
    DeepSeekClient,
    AlibabaClient,
    HuggingFaceClient,
)
from ai_client.response import LLMResponse


class TestGeminiClient:
    """Tests for GeminiClient."""

    def test_gemini_client_initialization(self):
        """Test Gemini client initialization."""
        with patch("ai_client.gemini_client.genai"):
            client = create_ai_client("genai", api_key="test-key")

            assert isinstance(client, GeminiClient)
            assert client.PROVIDER_ID == "genai"
            assert client.SUPPORTS_MULTIMODAL is True

    def test_prompt_text_only(self, mock_gemini_response):
        """Test text-only prompt."""
        with patch("ai_client.gemini_client.genai") as mock_genai:
            mock_client = Mock()
            mock_genai.Client.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_gemini_response

            client = create_ai_client("genai", api_key="test-key")
            response = client.prompt("gemini-2.0-flash-exp", "Hello!")

            # Check response
            assert isinstance(response, LLMResponse)
            assert response.text == "Hello! I'm Gemini."
            assert response.provider == "genai"
            assert response.usage.input_tokens == 12
            assert response.usage.output_tokens == 18
            assert response.duration >= 0

    def test_prompt_with_images(self, mock_gemini_response, sample_image_path):
        """Test prompt with images."""
        with patch("ai_client.gemini_client.genai") as mock_genai:
            mock_client = Mock()
            mock_genai.Client.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_gemini_response

            client = create_ai_client("genai", api_key="test-key")
            response = client.prompt(
                "gemini-2.0-flash-exp", "Describe this", images=[sample_image_path]
            )

            assert isinstance(response, LLMResponse)


class TestMistralClient:
    """Tests for MistralClient."""

    def test_mistral_client_initialization(self):
        """Test Mistral client initialization."""
        with patch("ai_client.mistral_client.Mistral"):
            client = create_ai_client("mistral", api_key="test-key")

            assert isinstance(client, MistralClient)
            assert client.PROVIDER_ID == "mistral"
            assert client.SUPPORTS_MULTIMODAL is True

    def test_prompt_text_only(self, mock_mistral_response):
        """Test text-only prompt."""
        with patch("ai_client.mistral_client.Mistral") as mock_mistral_class:
            mock_client = Mock()
            mock_mistral_class.return_value = mock_client
            mock_client.chat.complete.return_value = mock_mistral_response

            client = create_ai_client("mistral", api_key="test-key")
            response = client.prompt("mistral-large-latest", "Hello!")

            # Check response
            assert isinstance(response, LLMResponse)
            assert response.text == "Hello! I'm Mistral."
            assert response.provider == "mistral"
            assert response.usage.input_tokens == 11
            assert response.usage.output_tokens == 19
            assert response.duration >= 0

    def test_prompt_with_images(self, mock_mistral_response, sample_image_path):
        """Test prompt with images."""
        with patch("ai_client.mistral_client.Mistral") as mock_mistral_class:
            mock_client = Mock()
            mock_mistral_class.return_value = mock_client
            mock_client.chat.complete.return_value = mock_mistral_response

            client = create_ai_client("mistral", api_key="test-key")
            response = client.prompt(
                "mistral-large-latest", "Describe this", images=[sample_image_path]
            )

            assert isinstance(response, LLMResponse)


class TestDeepSeekClient:
    """Tests for DeepSeekClient."""

    def test_deepseek_client_initialization(self):
        """Test DeepSeek client initialization."""
        with patch("ai_client.openai_client.OpenAI") as mock_openai:
            client = create_ai_client("deepseek", api_key="test-key")

            assert isinstance(client, DeepSeekClient)
            assert client.PROVIDER_ID == "deepseek"
            assert client.SUPPORTS_MULTIMODAL is True

            # Should have set custom base URL
            call_args = mock_openai.call_args
            assert call_args.kwargs["base_url"] == "https://api.deepseek.com/v1"

    def test_deepseek_inherits_openai_functionality(self, mock_openai_response):
        """Test that DeepSeek inherits OpenAI functionality."""
        with patch("ai_client.openai_client.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_openai_response

            client = create_ai_client("deepseek", api_key="test-key")
            response = client.prompt("deepseek-chat", "Hello!")

            # Should work like OpenAI client
            assert isinstance(response, LLMResponse)
            assert response.text == "Hello! I'm an AI assistant."


class TestAlibabaClient:
    """Tests for AlibabaClient."""

    def test_alibaba_client_initialization(self):
        """Test Alibaba client initialization."""
        with patch("ai_client.openai_client.OpenAI") as mock_openai:
            client = create_ai_client("alibaba", api_key="test-key")

            assert isinstance(client, AlibabaClient)
            assert client.PROVIDER_ID == "alibaba"
            assert client.SUPPORTS_MULTIMODAL is True

            # Should have set custom base URL
            call_args = mock_openai.call_args
            assert "aliyuncs.com" in call_args.kwargs["base_url"]

    def test_alibaba_inherits_openai_functionality(self, mock_openai_response):
        """Test that Alibaba client inherits OpenAI functionality."""
        with patch("ai_client.openai_client.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_openai_response

            client = create_ai_client("alibaba", api_key="test-key")
            response = client.prompt("qwen-turbo", "Hello!")

            # Should work like OpenAI client
            assert isinstance(response, LLMResponse)
            assert response.text == "Hello! I'm an AI assistant."


class TestHuggingFaceClient:
    """Tests for HuggingFaceClient."""

    ROUTER_URL = "https://router.huggingface.co/v1"
    ENDPOINT_URL = "https://abc123.us-east-1.aws.endpoints.huggingface.cloud/v1"

    def test_huggingface_client_initialization(self):
        """Test HuggingFace client defaults to the Inference Providers router."""
        with patch("ai_client.openai_client.OpenAI") as mock_openai:
            client = create_ai_client("huggingface", api_key="test-key")

            assert isinstance(client, HuggingFaceClient)
            assert client.PROVIDER_ID == "huggingface"
            assert client.SUPPORTS_MULTIMODAL is True

            # Should have set the router base URL
            assert mock_openai.call_args.kwargs["base_url"] == self.ROUTER_URL

    def test_dedicated_endpoint_base_url_is_respected(self):
        """An explicit base_url targets a dedicated Inference Endpoint instead."""
        with patch("ai_client.openai_client.OpenAI") as mock_openai:
            client = create_ai_client("huggingface", api_key="test-key", base_url=self.ENDPOINT_URL)

            assert client.base_url == self.ENDPOINT_URL
            assert mock_openai.call_args.kwargs["base_url"] == self.ENDPOINT_URL

    def test_huggingface_inherits_openai_functionality(self, mock_huggingface_response):
        """Test that HuggingFace inherits OpenAI functionality."""
        with patch("ai_client.openai_client.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_huggingface_response

            client = create_ai_client("huggingface", api_key="test-key")
            response = client.prompt("swiss-ai/Apertus-8B-Instruct-2509", "Hello!")

            assert isinstance(response, LLMResponse)
            assert response.text == "Gruezi! I'm Apertus."
            assert response.provider == "huggingface"

    def test_cost_left_to_downstream_injection(self, mock_huggingface_response):
        """
        The library intentionally leaves HuggingFace cost None and tracks tokens only.

        There is deliberately no 'huggingface' block in pricing.json: a router model is
        served by different partner providers at different prices, and the router echoes a
        normalized model id (e.g. 'swiss-ai/apertus-8b-instruct') that differs from the
        requested id, so a static per-model price would be misleading. Cost is injected
        downstream by the benchmark harness instead.
        """
        with patch("ai_client.openai_client.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_huggingface_response

            client = create_ai_client("huggingface", api_key="test-key")
            response = client.prompt("swiss-ai/Apertus-8B-Instruct-2509", "Hello!")

            # Tokens are tracked...
            assert response.usage.total_tokens == 2_000_000
            # ...but cost is left for downstream injection.
            assert response.usage.estimated_cost_usd is None
            assert response.usage.input_cost_usd is None
            assert response.usage.output_cost_usd is None

    def test_default_user_agent_is_set(self):
        """
        A neutral User-Agent is sent by default.

        Some HF partner providers (publicai, scaleway) sit behind a WAF that 403-blocks the
        OpenAI SDK's default 'OpenAI/Python' User-Agent, so the client overrides it.
        """
        with patch("ai_client.openai_client.OpenAI") as mock_openai:
            create_ai_client("huggingface", api_key="test-key")
            headers = mock_openai.call_args.kwargs["default_headers"]
            assert headers["User-Agent"] == "generic-llm-api-client"

    def test_user_supplied_user_agent_is_preserved(self):
        """A caller-supplied User-Agent is not overridden (case-insensitive), and other
        default headers are preserved alongside it."""
        with patch("ai_client.openai_client.OpenAI") as mock_openai:
            create_ai_client(
                "huggingface",
                api_key="test-key",
                default_headers={"user-agent": "custom/1.0", "X-HF-Bill-To": "my-org"},
            )
            headers = mock_openai.call_args.kwargs["default_headers"]
            assert headers["user-agent"] == "custom/1.0"
            assert "User-Agent" not in headers  # no duplicate added
            assert headers["X-HF-Bill-To"] == "my-org"

    def test_model_list_with_created(self):
        """Router models report a created timestamp, which is formatted as a date."""
        with patch("ai_client.openai_client.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            mock_model = Mock()
            mock_model.id = "swiss-ai/Apertus-8B-Instruct-2509"
            mock_model.created = 1755077423
            mock_client.models.list.return_value = [mock_model]

            client = create_ai_client("huggingface", api_key="test-key")
            models = client.get_model_list()

            assert models == [("swiss-ai/Apertus-8B-Instruct-2509", "2025-08-13")]

    def test_model_list_without_created(self):
        """Dedicated endpoints may omit `created`; the date falls back to None."""
        with patch("ai_client.openai_client.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            # spec restricts the Mock so accessing `.created` raises AttributeError
            mock_model = Mock(spec=["id"])
            mock_model.id = "apertus-8b-xll"
            mock_client.models.list.return_value = [mock_model]

            client = create_ai_client("huggingface", api_key="test-key", base_url=self.ENDPOINT_URL)
            models = client.get_model_list()

            assert models == [("apertus-8b-xll", None)]


class TestCohereClient:
    """Tests for CohereClient."""

    def test_cohere_client_initialization(self):
        """Test Cohere client initialization."""
        with patch("ai_client.cohere_client.cohere"):
            client = create_ai_client("cohere", api_key="test-key")

            assert isinstance(client, CohereClient)
            assert client.PROVIDER_ID == "cohere"
            assert client.SUPPORTS_MULTIMODAL is True

    def test_prompt_text_only(self, mock_cohere_response):
        """Test text-only prompt."""
        with patch("ai_client.cohere_client.cohere") as mock_cohere_module:
            mock_client = Mock()
            mock_cohere_module.ClientV2.return_value = mock_client
            mock_client.chat.return_value = mock_cohere_response

            client = create_ai_client("cohere", api_key="test-key")
            response = client.prompt("command-r", "Hello!")

            # Check response
            assert isinstance(response, LLMResponse)
            assert response.text == "Hello! I'm Cohere."
            assert response.provider == "cohere"
            assert response.usage.input_tokens == 13
            assert response.usage.output_tokens == 17
            assert response.duration >= 0

    def test_prompt_with_images(self, mock_cohere_response, sample_image_path):
        """Test prompt with images (vision model)."""
        with patch("ai_client.cohere_client.cohere") as mock_cohere_module:
            mock_client = Mock()
            mock_cohere_module.ClientV2.return_value = mock_client
            mock_client.chat.return_value = mock_cohere_response

            client = create_ai_client("cohere", api_key="test-key")
            response = client.prompt(
                "command-a-vision-07-2025", "Describe this", images=[sample_image_path]
            )

            assert isinstance(response, LLMResponse)
            assert response.text == "Hello! I'm Cohere."

    def test_prompt_with_custom_parameters(self, mock_cohere_response):
        """Test prompt with custom parameters like temperature and max_tokens."""
        with patch("ai_client.cohere_client.cohere") as mock_cohere_module:
            mock_client = Mock()
            mock_cohere_module.ClientV2.return_value = mock_client
            mock_client.chat.return_value = mock_cohere_response

            client = create_ai_client("cohere", api_key="test-key")
            client.prompt("command-r", "Hello!", temperature=0.7, max_tokens=100)

            # Verify chat was called
            assert mock_client.chat.called
            call_kwargs = mock_client.chat.call_args.kwargs

            # Check that parameters were passed
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["max_tokens"] == 100

    def test_model_list(self):
        """Test getting list of available models."""
        with patch("ai_client.cohere_client.cohere") as mock_cohere_module:
            # Mock both ClientV2 and Client (v1 used for models.list())
            mock_client_v2 = Mock()
            mock_client_v1 = Mock()
            mock_cohere_module.ClientV2.return_value = mock_client_v2
            mock_cohere_module.Client.return_value = mock_client_v1

            # Mock models list response
            mock_model_1 = Mock()
            mock_model_1.name = "command-r"
            mock_model_2 = Mock()
            mock_model_2.name = "command-a-03-2025"

            mock_models_response = Mock()
            mock_models_response.models = [mock_model_1, mock_model_2]
            mock_client_v1.models.list.return_value = mock_models_response

            client = create_ai_client("cohere", api_key="test-key")
            models = client.get_model_list()

            # Check that we got the expected models
            assert len(models) == 2
            assert models[0] == ("command-r", None)
            assert models[1] == ("command-a-03-2025", None)
