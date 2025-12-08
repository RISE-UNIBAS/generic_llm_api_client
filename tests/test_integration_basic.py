"""
Basic integration tests - real API calls for all providers.

These tests make actual API calls and are skipped if API keys are not set.
Run with: pytest -m integration tests/test_integration_basic.py

Set API keys in .env file (see .env.example for template).
"""

import os
import pytest
from dotenv import load_dotenv
from ai_client import create_ai_client
from ai_client.response import LLMResponse

# Load environment variables from .env file
load_dotenv()

# Test prompt
SIMPLE_PROMPT = "Say 'Hello, I am working!' and nothing else."


@pytest.mark.integration
class TestOpenAIIntegration:
    """Integration tests for OpenAI."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip test if API key not set."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    def test_openai_basic_prompt(self):
        """Test basic OpenAI prompt with real API."""
        client = create_ai_client("openai", api_key=os.getenv("OPENAI_API_KEY"))
        response = client.prompt("gpt-4o-mini", SIMPLE_PROMPT)

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.text != ""
        assert response.provider == "openai"
        assert response.model.startswith(
            "gpt-4o-mini"
        )  # API returns versioned names like gpt-4o-mini-2024-07-18
        assert response.finish_reason in ["stop", "end_turn", "length"]

        # Verify usage tracking
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0
        assert response.usage.total_tokens > 0

        # Verify timing
        assert response.duration > 0

        # Verify response content makes sense
        assert "hello" in response.text.lower() or "working" in response.text.lower()


@pytest.mark.integration
class TestClaudeIntegration:
    """Integration tests for Anthropic Claude."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip test if API key not set."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

    def test_claude_basic_prompt(self):
        """Test basic Claude prompt with real API."""
        client = create_ai_client("anthropic", api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.prompt("claude-3-5-haiku-20241022", SIMPLE_PROMPT)

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.text != ""
        assert response.provider == "anthropic"
        assert "claude-3-5-haiku" in response.model.lower()  # Check for model family
        assert response.finish_reason in ["stop", "end_turn", "max_tokens"]

        # Verify usage tracking
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

        # Verify timing
        assert response.duration > 0

        # Verify response content
        assert "hello" in response.text.lower() or "working" in response.text.lower()


@pytest.mark.integration
class TestGeminiIntegration:
    """Integration tests for Google Gemini."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip test if API key not set."""
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set")

    def test_gemini_basic_prompt(self):
        """Test basic Gemini prompt with real API."""
        client = create_ai_client("genai", api_key=os.getenv("GOOGLE_API_KEY"))
        response = client.prompt("gemini-2.0-flash-exp", SIMPLE_PROMPT)

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.text != ""
        assert response.provider == "genai"
        assert "gemini" in response.model.lower()  # Check for model family
        assert response.finish_reason in ["STOP", "stop", "MAX_TOKENS"]

        # Verify usage tracking (Gemini provides token counts)
        assert response.usage.input_tokens >= 0
        assert response.usage.output_tokens >= 0

        # Verify timing
        assert response.duration > 0

        # Verify response content
        assert "hello" in response.text.lower() or "working" in response.text.lower()


@pytest.mark.integration
class TestMistralIntegration:
    """Integration tests for Mistral."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip test if API key not set."""
        if not os.getenv("MISTRAL_API_KEY"):
            pytest.skip("MISTRAL_API_KEY not set")

    def test_mistral_basic_prompt(self):
        """Test basic Mistral prompt with real API."""
        client = create_ai_client("mistral", api_key=os.getenv("MISTRAL_API_KEY"))
        response = client.prompt("mistral-small-latest", SIMPLE_PROMPT)

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.text != ""
        assert response.provider == "mistral"
        assert "mistral" in response.model.lower()  # Check for model family
        assert response.finish_reason in ["stop", "length", "end_turn"]

        # Verify usage tracking
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

        # Verify timing
        assert response.duration > 0

        # Verify response content
        assert "hello" in response.text.lower() or "working" in response.text.lower()


@pytest.mark.integration
class TestDeepSeekIntegration:
    """Integration tests for DeepSeek."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip test if API key not set."""
        if not os.getenv("DEEPSEEK_API_KEY"):
            pytest.skip("DEEPSEEK_API_KEY not set")

    def test_deepseek_basic_prompt(self):
        """Test basic DeepSeek prompt with real API."""
        client = create_ai_client("deepseek", api_key=os.getenv("DEEPSEEK_API_KEY"))
        response = client.prompt("deepseek-chat", SIMPLE_PROMPT)

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.text != ""
        assert response.provider == "deepseek"
        assert "deepseek" in response.model.lower()  # Check for model family
        assert response.finish_reason in ["stop", "length"]

        # Verify usage tracking
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

        # Verify timing
        assert response.duration > 0

        # Verify response content
        assert "hello" in response.text.lower() or "working" in response.text.lower()


@pytest.mark.integration
class TestQwenIntegration:
    """Integration tests for Qwen."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip test if API key not set."""
        if not os.getenv("QWEN_API_KEY"):
            pytest.skip("QWEN_API_KEY not set")

    def test_qwen_basic_prompt(self):
        """Test basic Qwen prompt with real API."""
        client = create_ai_client("qwen", api_key=os.getenv("QWEN_API_KEY"))
        response = client.prompt("qwen-turbo", SIMPLE_PROMPT)

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.text != ""
        assert response.provider == "qwen"
        assert "qwen" in response.model.lower()  # Check for model family
        assert response.finish_reason in ["stop", "length"]

        # Verify usage tracking
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

        # Verify timing
        assert response.duration > 0

        # Verify response content
        assert "hello" in response.text.lower() or "working" in response.text.lower()


@pytest.mark.integration
class TestOpenRouterIntegration:
    """Integration tests for OpenRouter (OpenAI-compatible)."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip test if API key not set."""
        if not os.getenv("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set")

    def test_openrouter_basic_prompt(self):
        """Test basic OpenRouter prompt with real API."""
        client = create_ai_client(
            "openrouter",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
        # Use a cheap model available on OpenRouter
        response = client.prompt("openai/gpt-3.5-turbo", SIMPLE_PROMPT)

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.text != ""
        assert response.provider == "openai"  # OpenRouter uses OpenAI client
        assert response.finish_reason in ["stop", "end_turn", "length"]

        # Verify usage tracking
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

        # Verify timing
        assert response.duration > 0

        # Verify response content
        assert "hello" in response.text.lower() or "working" in response.text.lower()


@pytest.mark.integration
class TestSciCOREIntegration:
    """Integration tests for sciCORE (OpenAI-compatible)."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip test if API key not set."""
        if not os.getenv("SCICORE_API_KEY"):
            pytest.skip("SCICORE_API_KEY not set")

    def test_scicore_basic_prompt(self):
        """Test basic sciCORE prompt with real API."""
        # Note: Update base_url and model according to your sciCORE setup
        base_url = os.getenv("SCICORE_BASE_URL", "https://api.scicore.unibas.ch/v1")
        model = os.getenv("SCICORE_MODEL", "gpt-4")

        client = create_ai_client(
            "scicore", api_key=os.getenv("SCICORE_API_KEY"), base_url=base_url
        )
        response = client.prompt(model, SIMPLE_PROMPT)

        # Verify response structure
        assert isinstance(response, LLMResponse)

        # If response is empty, check if it's an error
        if response.text == "":
            # Provide helpful error message
            if response.finish_reason == "error":
                pytest.skip(
                    f"sciCORE API error (check SCICORE_BASE_URL and SCICORE_MODEL): "
                    f"{response.raw_response.get('error', 'Unknown error')}"
                )
            else:
                pytest.fail(
                    f"Empty response from sciCORE. "
                    f"Check SCICORE_BASE_URL={base_url} and SCICORE_MODEL={model}. "
                    f"Finish reason: {response.finish_reason}"
                )

        assert response.text != ""
        assert response.provider == "openai"  # sciCORE uses OpenAI client
        assert response.finish_reason in ["stop", "end_turn", "length"]

        # Verify usage tracking
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

        # Verify timing
        assert response.duration > 0


@pytest.mark.integration
class TestCohereIntegration:
    """Integration tests for Cohere."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip test if API key not set."""
        if not os.getenv("COHERE_API_KEY"):
            pytest.skip("COHERE_API_KEY not set")

    def test_cohere_basic_prompt(self):
        """Test basic Cohere prompt with real API."""
        client = create_ai_client("cohere", api_key=os.getenv("COHERE_API_KEY"))
        response = client.prompt("command-r", SIMPLE_PROMPT)

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.text != ""
        assert response.provider == "cohere"
        assert response.model == "command-r"
        assert response.finish_reason in ["stop", "complete", "COMPLETE"]

        # Verify usage tracking
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0
        assert response.usage.total_tokens > 0

        # Verify timing
        assert response.duration > 0

        # Verify response content
        assert "hello" in response.text.lower() or "working" in response.text.lower()


@pytest.mark.integration
class TestProviderParity:
    """Tests that verify all providers work similarly."""

    def test_all_available_providers_work(self):
        """Test that all configured providers can make basic calls."""
        providers_to_test = []

        # Check which providers have API keys set
        if os.getenv("OPENAI_API_KEY"):
            providers_to_test.append(("openai", "gpt-4o-mini", os.getenv("OPENAI_API_KEY")))
        if os.getenv("ANTHROPIC_API_KEY"):
            providers_to_test.append(
                ("anthropic", "claude-3-5-haiku-20241022", os.getenv("ANTHROPIC_API_KEY"))
            )
        if os.getenv("GOOGLE_API_KEY"):
            providers_to_test.append(("genai", "gemini-2.0-flash-exp", os.getenv("GOOGLE_API_KEY")))
        if os.getenv("MISTRAL_API_KEY"):
            providers_to_test.append(
                ("mistral", "mistral-small-latest", os.getenv("MISTRAL_API_KEY"))
            )
        if os.getenv("COHERE_API_KEY"):
            providers_to_test.append(("cohere", "command-r", os.getenv("COHERE_API_KEY")))
        if os.getenv("DEEPSEEK_API_KEY"):
            providers_to_test.append(("deepseek", "deepseek-chat", os.getenv("DEEPSEEK_API_KEY")))
        if os.getenv("QWEN_API_KEY"):
            providers_to_test.append(("qwen", "qwen-turbo", os.getenv("QWEN_API_KEY")))

        if not providers_to_test:
            pytest.skip("No API keys configured")

        # Test each provider
        results = []
        for provider_id, model, api_key in providers_to_test:
            client = create_ai_client(provider_id, api_key=api_key)
            response = client.prompt(model, "Reply with just: OK")
            results.append(
                {
                    "provider": provider_id,
                    "success": response.text != "",
                    "duration": response.duration,
                    "tokens": response.usage.total_tokens,
                }
            )

        # Verify all succeeded
        assert all(r["success"] for r in results), f"Some providers failed: {results}"

        # Verify all responses had reasonable timing
        assert all(r["duration"] > 0 for r in results)

        print(f"\nTested {len(results)} providers successfully:")
        for r in results:
            print(f"  {r['provider']}: {r['duration']:.2f}s, {r['tokens']} tokens")
