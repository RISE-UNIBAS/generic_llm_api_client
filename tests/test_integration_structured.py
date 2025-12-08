"""
Structured output integration tests - real API calls with Pydantic models.

These tests verify that structured output works correctly with real APIs.
Run with: pytest -m integration tests/test_integration_structured.py

Set API keys in .env file (see .env.example for template).
"""

import os
import json
import pytest
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
from ai_client import create_ai_client
from ai_client.response import LLMResponse

# Load environment variables from .env file
load_dotenv()


# Test Pydantic models
class PersonInfo(BaseModel):
    """Simple person information model."""

    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age in years")
    occupation: str = Field(description="Person's job or profession")


class ManuscriptMetadata(BaseModel):
    """Model for manuscript metadata extraction."""

    language: str = Field(description="Primary language of the text")
    script: str = Field(description="Writing system used (e.g., Latin, Arabic, Chinese)")
    century: Optional[int] = Field(None, description="Estimated century of creation")
    condition: str = Field(description="Condition: excellent, good, fair, or poor")
    has_illustrations: bool = Field(description="Whether the manuscript contains illustrations")


class DocumentAnalysis(BaseModel):
    """Comprehensive document analysis."""

    title: str = Field(description="Document title or subject")
    summary: str = Field(description="Brief summary in one sentence")
    key_topics: List[str] = Field(description="List of main topics discussed")
    word_count_estimate: int = Field(description="Estimated word count")


@pytest.mark.integration
class TestOpenAIStructuredOutput:
    """Integration tests for OpenAI structured output."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip test if API key not set."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    def test_openai_simple_structured_output(self):
        """Test OpenAI with simple Pydantic model."""
        client = create_ai_client("openai", api_key=os.getenv("OPENAI_API_KEY"))
        response = client.prompt(
            "gpt-4o-mini",
            "Extract information about: John Smith is a 35 year old software engineer.",
            response_format=PersonInfo,
        )

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.text != ""
        assert response.provider == "openai"

        # Parse and verify structured data
        data = json.loads(response.text)
        assert "name" in data
        assert "age" in data
        assert "occupation" in data

        # Verify values make sense
        assert "john" in data["name"].lower() or "smith" in data["name"].lower()
        assert data["age"] == 35
        assert "engineer" in data["occupation"].lower()

        print(f"\nOpenAI structured output: {data}")

    def test_openai_complex_structured_output(self):
        """Test OpenAI with complex nested model."""
        client = create_ai_client("openai", api_key=os.getenv("OPENAI_API_KEY"))
        response = client.prompt(
            "gpt-4o-mini",
            'Analyze this text: "Python Programming Guide. This comprehensive guide covers '
            "object-oriented programming, functional programming, and async programming. "
            'It includes 50 chapters with detailed examples."',
            response_format=DocumentAnalysis,
        )

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.text != ""

        # Parse and verify structured data
        data = json.loads(response.text)
        assert "title" in data
        assert "summary" in data
        assert "key_topics" in data
        assert "word_count_estimate" in data

        # Verify values
        assert isinstance(data["key_topics"], list)
        assert len(data["key_topics"]) > 0
        assert isinstance(data["word_count_estimate"], int)

        print(f"\nOpenAI complex structured output: {data}")


@pytest.mark.integration
class TestClaudeStructuredOutput:
    """Integration tests for Claude structured output (via tools)."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip test if API key not set."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

    def test_claude_simple_structured_output(self):
        """Test Claude with simple Pydantic model using tools."""
        client = create_ai_client("anthropic", api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.prompt(
            "claude-3-5-haiku-20241022",
            "Extract information about: Jane Doe is a 28 year old data scientist.",
            response_format=PersonInfo,
        )

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.text != ""
        assert response.provider == "anthropic"

        # Parse and verify structured data
        data = json.loads(response.text)
        assert "name" in data
        assert "age" in data
        assert "occupation" in data

        # Verify values
        assert "jane" in data["name"].lower() or "doe" in data["name"].lower()
        assert data["age"] == 28
        assert "scientist" in data["occupation"].lower() or "data" in data["occupation"].lower()

        print(f"\nClaude structured output: {data}")

    def test_claude_complex_structured_output(self):
        """Test Claude with complex model."""
        client = create_ai_client("anthropic", api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.prompt(
            "claude-3-5-sonnet-20241022",
            'Analyze this article: "Climate Change Impact Report. This report examines '
            'environmental changes, policy recommendations, and economic impacts across 200 pages."',
            response_format=DocumentAnalysis,
        )

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.text != ""

        # Parse and verify structured data
        data = json.loads(response.text)
        assert "title" in data
        assert "key_topics" in data
        assert isinstance(data["key_topics"], list)

        print(f"\nClaude complex structured output: {data}")


@pytest.mark.integration
class TestGeminiStructuredOutput:
    """Integration tests for Gemini structured output."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip test if API key not set."""
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set")

    def test_gemini_simple_structured_output(self):
        """Test Gemini with simple Pydantic model."""
        client = create_ai_client("genai", api_key=os.getenv("GOOGLE_API_KEY"))
        response = client.prompt(
            "gemini-2.0-flash-exp",
            "Extract information about: Alice Johnson is a 42 year old professor.",
            response_format=PersonInfo,
        )

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.text != ""
        assert response.provider == "genai"

        # Parse and verify structured data
        data = json.loads(response.text)
        assert "name" in data
        assert "age" in data
        assert "occupation" in data

        # Verify values
        assert "alice" in data["name"].lower() or "johnson" in data["name"].lower()
        assert data["age"] == 42
        assert "professor" in data["occupation"].lower()

        print(f"\nGemini structured output: {data}")


@pytest.mark.integration
class TestMistralStructuredOutput:
    """Integration tests for Mistral structured output."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip test if API key not set."""
        if not os.getenv("MISTRAL_API_KEY"):
            pytest.skip("MISTRAL_API_KEY not set")

    def test_mistral_simple_structured_output(self):
        """Test Mistral with simple Pydantic model."""
        client = create_ai_client("mistral", api_key=os.getenv("MISTRAL_API_KEY"))
        response = client.prompt(
            "mistral-small-latest",
            "Extract information about: Bob Wilson is a 50 year old architect.",
            response_format=PersonInfo,
        )

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.text != ""
        assert response.provider == "mistral"

        # Parse and verify structured data
        data = json.loads(response.text)
        assert "name" in data
        assert "age" in data
        assert "occupation" in data

        # Verify values
        assert "bob" in data["name"].lower() or "wilson" in data["name"].lower()
        assert data["age"] == 50
        assert "architect" in data["occupation"].lower()

        print(f"\nMistral structured output: {data}")


@pytest.mark.integration
class TestCohereStructuredOutput:
    """Integration tests for Cohere structured output."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip test if API key not set."""
        if not os.getenv("COHERE_API_KEY"):
            pytest.skip("COHERE_API_KEY not set")

    def test_cohere_simple_structured_output(self):
        """Test Cohere with simple Pydantic model."""
        client = create_ai_client("cohere", api_key=os.getenv("COHERE_API_KEY"))
        response = client.prompt(
            "command-r",
            "Extract information about: Carol Davis is a 35 year old engineer.",
            response_format=PersonInfo,
        )

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.text != ""
        assert response.provider == "cohere"

        # Parse and verify structured data
        data = json.loads(response.text)
        assert "name" in data
        assert "age" in data
        assert "occupation" in data

        # Verify values
        assert "carol" in data["name"].lower() or "davis" in data["name"].lower()
        assert data["age"] == 35
        assert "engineer" in data["occupation"].lower()

        print(f"\nCohere structured output: {data}")


@pytest.mark.integration
class TestStructuredOutputWithVision:
    """Test structured output combined with vision capabilities."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip test if no vision-capable API keys are set."""
        if not any(
            [
                os.getenv("OPENAI_API_KEY"),
                os.getenv("ANTHROPIC_API_KEY"),
                os.getenv("GOOGLE_API_KEY"),
            ]
        ):
            pytest.skip("No vision-capable API keys set")

    def test_openai_vision_structured_output(self, sample_image_path):
        """Test OpenAI vision with structured output for manuscript analysis."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        client = create_ai_client("openai", api_key=os.getenv("OPENAI_API_KEY"))
        response = client.prompt(
            "gpt-4o-mini",
            "Analyze this manuscript image and extract metadata. "
            "If you cannot determine a value, make a reasonable guess.",
            images=[sample_image_path],
            response_format=ManuscriptMetadata,
        )

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.text != ""

        # Parse and verify structured data
        data = json.loads(response.text)
        assert "language" in data
        assert "script" in data
        assert "condition" in data
        assert "has_illustrations" in data

        print(f"\nVision + structured output: {data}")

    def test_claude_vision_structured_output(self, sample_image_path):
        """Test Claude vision with structured output."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        client = create_ai_client("anthropic", api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.prompt(
            "claude-3-5-sonnet-20241022",
            "Analyze this image and provide manuscript metadata. "
            "Make reasonable guesses if needed.",
            images=[sample_image_path],
            response_format=ManuscriptMetadata,
        )

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.text != ""

        # Parse and verify structured data
        data = json.loads(response.text)
        assert "language" in data
        assert "script" in data

        print(f"\nClaude vision + structured: {data}")


@pytest.mark.integration
class TestBenchmarkStructuredWorkflow:
    """Simulate a benchmark workflow with structured output."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip test if API key not set."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set for benchmark simulation")

    def test_batch_structured_extraction(self, sample_image_path):
        """Simulate extracting structured data from multiple images."""
        client = create_ai_client("openai", api_key=os.getenv("OPENAI_API_KEY"))

        # Simulate processing 3 manuscript images
        results = []
        for i in range(3):
            response = client.prompt(
                "gpt-4o-mini",
                f"Analyze manuscript {i+1} and extract metadata.",
                images=[sample_image_path],
                response_format=ManuscriptMetadata,
            )

            data = json.loads(response.text)
            results.append(
                {
                    "image_num": i + 1,
                    "metadata": data,
                    "duration": response.duration,
                    "tokens": response.usage.total_tokens,
                }
            )

        # Verify all succeeded
        assert len(results) == 3
        assert all("language" in r["metadata"] for r in results)

        total_duration = sum(r["duration"] for r in results)
        total_tokens = sum(r["tokens"] for r in results)

        print("\nStructured benchmark results:")
        print(f"  Total duration: {total_duration:.2f}s")
        print(f"  Total tokens: {total_tokens}")
        for r in results:
            print(f"  Image {r['image_num']}: {r['metadata']}")
