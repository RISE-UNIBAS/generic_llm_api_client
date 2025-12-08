"""
Multimodal integration tests - real API calls with images.

These tests make actual API calls with images and are skipped if API keys are not set.
Run with: pytest -m integration tests/test_integration_multimodal.py

Set API keys in .env file (see .env.example for template).
"""

import os
import pytest
from dotenv import load_dotenv
from ai_client import create_ai_client
from ai_client.response import LLMResponse

# Load environment variables from .env file
load_dotenv()

# Vision prompt
VISION_PROMPT = "Describe what you see in this image in one sentence."


@pytest.mark.integration
class TestOpenAIVision:
    """Integration tests for OpenAI vision models."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip test if API key not set."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    def test_openai_vision_with_image(self, sample_image_path):
        """Test OpenAI vision model with image."""
        client = create_ai_client("openai", api_key=os.getenv("OPENAI_API_KEY"))
        response = client.prompt("gpt-4o-mini", VISION_PROMPT, images=[sample_image_path])

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.text != ""
        assert response.provider == "openai"
        assert response.finish_reason == "stop"

        # Verify usage tracking (vision uses more tokens)
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

        # Verify timing
        assert response.duration > 0

        print(f"\nOpenAI vision response: {response.text}")


@pytest.mark.integration
class TestClaudeVision:
    """Integration tests for Claude vision models."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip test if API key not set."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

    def test_claude_vision_with_image(self, sample_image_path):
        """Test Claude vision model with image."""
        client = create_ai_client("anthropic", api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.prompt(
            "claude-3-5-sonnet-20241022", VISION_PROMPT, images=[sample_image_path]
        )

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.text != ""
        assert response.provider == "anthropic"
        assert response.finish_reason == "end_turn"

        # Verify usage tracking
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

        # Verify timing
        assert response.duration > 0

        print(f"\nClaude vision response: {response.text}")


@pytest.mark.integration
class TestGeminiVision:
    """Integration tests for Gemini vision models."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip test if API key not set."""
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set")

    def test_gemini_vision_with_image(self, sample_image_path):
        """Test Gemini vision model with image."""
        client = create_ai_client("genai", api_key=os.getenv("GOOGLE_API_KEY"))
        response = client.prompt("gemini-2.0-flash-exp", VISION_PROMPT, images=[sample_image_path])

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.text != ""
        assert response.provider == "genai"

        # Verify timing
        assert response.duration > 0

        print(f"\nGemini vision response: {response.text}")


@pytest.mark.integration
class TestMistralVision:
    """Integration tests for Mistral vision models."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip test if API key not set."""
        if not os.getenv("MISTRAL_API_KEY"):
            pytest.skip("MISTRAL_API_KEY not set")

    def test_mistral_vision_with_image(self, sample_image_path):
        """Test Mistral vision model with image."""
        client = create_ai_client("mistral", api_key=os.getenv("MISTRAL_API_KEY"))
        response = client.prompt("pixtral-12b-2409", VISION_PROMPT, images=[sample_image_path])

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.text != ""
        assert response.provider == "mistral"
        assert response.finish_reason in ["stop", "end_turn"]

        # Verify usage tracking
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

        # Verify timing
        assert response.duration > 0

        print(f"\nMistral vision response: {response.text}")


@pytest.mark.integration
class TestCohereVision:
    """Integration tests for Cohere vision models."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip test if API key not set."""
        if not os.getenv("COHERE_API_KEY"):
            pytest.skip("COHERE_API_KEY not set")

    def test_cohere_vision_with_image(self, sample_image_path):
        """Test Cohere vision model with image."""
        client = create_ai_client("cohere", api_key=os.getenv("COHERE_API_KEY"))
        response = client.prompt(
            "command-a-vision-07-2025", VISION_PROMPT, images=[sample_image_path]
        )

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.text != ""
        assert response.provider == "cohere"
        assert response.finish_reason in ["stop", "complete", "COMPLETE"]

        # Verify usage tracking
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

        # Verify timing
        assert response.duration > 0

        print(f"\nCohere vision response: {response.text}")


@pytest.mark.integration
class TestDeepSeekVision:
    """Integration tests for DeepSeek vision models."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip test if API key not set."""
        if not os.getenv("DEEPSEEK_API_KEY"):
            pytest.skip("DEEPSEEK_API_KEY not set")

    def test_deepseek_vision_with_image(self, sample_image_path):
        """Test DeepSeek vision model with image."""
        client = create_ai_client("deepseek", api_key=os.getenv("DEEPSEEK_API_KEY"))

        # Note: Check if DeepSeek has vision models available
        # Using deepseek-chat as it may support vision
        response = client.prompt("deepseek-chat", VISION_PROMPT, images=[sample_image_path])

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.text != ""
        assert response.provider == "deepseek"

        # Verify timing
        assert response.duration > 0

        print(f"\nDeepSeek vision response: {response.text}")


@pytest.mark.integration
class TestQwenVision:
    """Integration tests for Qwen vision models."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip test if API key not set."""
        if not os.getenv("QWEN_API_KEY"):
            pytest.skip("QWEN_API_KEY not set")

    def test_qwen_vision_with_image(self, sample_image_path):
        """Test Qwen vision model with image."""
        client = create_ai_client("qwen", api_key=os.getenv("QWEN_API_KEY"))

        # Qwen has vision-language models
        response = client.prompt("qwen-vl-max", VISION_PROMPT, images=[sample_image_path])

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.text != ""
        assert response.provider == "qwen"

        # Verify timing
        assert response.duration > 0

        print(f"\nQwen vision response: {response.text}")


@pytest.mark.integration
class TestMultiImageSupport:
    """Test support for multiple images in a single request."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip test if no API keys are set."""
        if not any(
            [
                os.getenv("OPENAI_API_KEY"),
                os.getenv("ANTHROPIC_API_KEY"),
                os.getenv("GOOGLE_API_KEY"),
            ]
        ):
            pytest.skip("No vision-capable API keys set")

    def test_multiple_images_openai(self, sample_image_path):
        """Test OpenAI with multiple images."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        client = create_ai_client("openai", api_key=os.getenv("OPENAI_API_KEY"))

        # Use same image twice for testing (in real use, different images)
        response = client.prompt(
            "gpt-4o-mini",
            "How many images do you see?",
            images=[sample_image_path, sample_image_path],
        )

        assert isinstance(response, LLMResponse)
        assert response.text != ""
        assert response.duration > 0

        # Response should mention multiple images
        assert "2" in response.text or "two" in response.text.lower()

    def test_multiple_images_claude(self, sample_image_path):
        """Test Claude with multiple images."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        client = create_ai_client("anthropic", api_key=os.getenv("ANTHROPIC_API_KEY"))

        response = client.prompt(
            "claude-3-5-sonnet-20241022",
            "How many images do you see?",
            images=[sample_image_path, sample_image_path],
        )

        assert isinstance(response, LLMResponse)
        assert response.text != ""
        assert response.duration > 0

        # Response should mention multiple images
        assert "2" in response.text or "two" in response.text.lower()


@pytest.mark.integration
class TestVisionBenchmarkWorkflow:
    """Simulate a typical benchmark workflow with images."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip test if API key not set."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set for benchmark simulation")

    def test_sequential_image_processing(self, sample_image_path):
        """Simulate processing multiple images sequentially."""
        client = create_ai_client("openai", api_key=os.getenv("OPENAI_API_KEY"))

        # Simulate processing 3 images
        results = []
        for i in range(3):
            response = client.prompt(
                "gpt-4o-mini",
                f"Transcribe any text in this image. Image {i+1}.",
                images=[sample_image_path],
            )
            results.append(
                {
                    "image_num": i + 1,
                    "success": response.text != "",
                    "duration": response.duration,
                    "tokens": response.usage.total_tokens,
                }
            )

        # Verify all succeeded
        assert len(results) == 3
        assert all(r["success"] for r in results)

        total_duration = sum(r["duration"] for r in results)
        total_tokens = sum(r["tokens"] for r in results)

        print("\nBenchmark simulation results:")
        print(f"  Total duration: {total_duration:.2f}s")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Average per image: {total_duration/3:.2f}s, {total_tokens/3:.0f} tokens")
