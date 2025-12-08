"""
Integration tests for file handling and image resize features - real API calls.

These tests verify that file and image resize features work correctly with real APIs.
Run with: pytest -m integration tests/test_integration_files.py

Set API keys in .env file (see .env.example for template).
"""

import os
import json
import pytest
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
from ai_client import create_ai_client
from ai_client.response import LLMResponse

# Load environment variables from .env file
load_dotenv()


# Test Pydantic models
class DocumentSummary(BaseModel):
    """Model for document summary."""

    title: str = Field(description="Document title or main subject")
    key_points: List[str] = Field(description="List of main points (3-5 items)")
    word_count_estimate: int = Field(description="Estimated word count")


class ImageTextMatch(BaseModel):
    """Model for comparing image to text description."""

    matches: bool = Field(description="Whether the image matches the description")
    confidence: str = Field(description="Confidence level: high, medium, or low")
    explanation: str = Field(description="Brief explanation of the match assessment")


@pytest.fixture
def sample_text_file(tmp_path):
    """Create a temporary text file for testing."""
    file_path = tmp_path / "test_document.txt"
    content = """
    The History of Computing

    Computing has evolved dramatically over the past century. From mechanical
    calculators to modern quantum computers, the field has seen revolutionary
    changes. Key milestones include the invention of the transistor in 1947,
    the development of integrated circuits in the 1960s, and the rise of
    personal computing in the 1980s.

    Today, artificial intelligence and machine learning represent the cutting
    edge of computational research, with applications ranging from natural
    language processing to computer vision.
    """
    file_path.write_text(content, encoding="utf-8")
    return str(file_path)


@pytest.fixture
def sample_multiple_files(tmp_path):
    """Create multiple text files for testing."""
    file1 = tmp_path / "document1.txt"
    file1.write_text("First document: Introduction to AI and machine learning.", encoding="utf-8")

    file2 = tmp_path / "document2.txt"
    file2.write_text("Second document: Deep learning and neural networks.", encoding="utf-8")

    return [str(file1), str(file2)]


@pytest.fixture
def large_test_image(tmp_path):
    """Create a large test image (simulating historical document scan)."""
    image_file = tmp_path / "large_test_image.jpg"

    # Create a 4000x3000 image (simulating a high-res scan)
    img = Image.new("RGB", (4000, 3000), color="white")
    draw = ImageDraw.Draw(img)

    # Add some visual content
    draw.rectangle([100, 100, 3900, 2900], outline="black", width=5)
    draw.text((200, 200), "Historical Document", fill="black")
    draw.rectangle([500, 500, 1500, 1500], fill="red")

    img.save(str(image_file), "JPEG", quality=95)
    return str(image_file)


@pytest.mark.integration
class TestOpenAIWithFiles:
    """Integration tests for OpenAI with text files."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip test if API key not set."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    def test_openai_single_file(self, sample_text_file):
        """Test OpenAI with a single text file."""
        client = create_ai_client("openai", api_key=os.getenv("OPENAI_API_KEY"))

        response = client.prompt(
            "gpt-4o-mini",
            "Summarize this document in one sentence.",
            files=[sample_text_file],
        )

        assert isinstance(response, LLMResponse)
        assert response.text != ""
        assert response.provider == "openai"
        assert response.duration > 0

        # Response should mention computing or history
        assert "comput" in response.text.lower() or "history" in response.text.lower()

        print(f"\nOpenAI with file response: {response.text[:200]}...")

    def test_openai_multiple_files(self, sample_multiple_files):
        """Test OpenAI with multiple text files."""
        client = create_ai_client("openai", api_key=os.getenv("OPENAI_API_KEY"))

        response = client.prompt(
            "gpt-4o-mini",
            "What topics are covered across these documents?",
            files=sample_multiple_files,
        )

        assert isinstance(response, LLMResponse)
        assert response.text != ""

        # Should mention AI/ML topics
        text_lower = response.text.lower()
        assert (
            "ai" in text_lower
            or "machine learning" in text_lower
            or "neural" in text_lower
            or "deep learning" in text_lower
        )

        print(f"\nOpenAI multiple files: {response.text}")

    def test_openai_file_with_structured_output(self, sample_text_file):
        """Test OpenAI with file and structured output."""
        client = create_ai_client("openai", api_key=os.getenv("OPENAI_API_KEY"))

        response = client.prompt(
            "gpt-4o-mini",
            "Analyze this document and provide a structured summary.",
            files=[sample_text_file],
            response_format=DocumentSummary,
        )

        assert isinstance(response, LLMResponse)
        assert response.text != ""

        # Parse structured output
        data = json.loads(response.text)
        assert "title" in data
        assert "key_points" in data
        assert "word_count_estimate" in data
        assert isinstance(data["key_points"], list)
        assert len(data["key_points"]) >= 3

        print(f"\nStructured output from file: {data}")


@pytest.mark.integration
class TestImageResize:
    """Integration tests for automatic image resizing."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip test if API key not set."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    def test_large_image_auto_resized(self, large_test_image):
        """Test that large images are automatically resized."""
        # Create client with max_image_size=2048
        client = create_ai_client(
            "openai",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_image_size=2048,
            image_quality=85,
        )

        response = client.prompt(
            "gpt-4o-mini",
            "What shapes and colors do you see in this image?",
            images=[large_test_image],
        )

        assert isinstance(response, LLMResponse)
        assert response.text != ""

        # Should recognize the shapes/colors
        text_lower = response.text.lower()
        # May mention rectangle, square, red, or black
        has_shape_or_color = any(
            word in text_lower for word in ["rectangle", "square", "red", "black", "shape", "color"]
        )
        assert has_shape_or_color

        print(f"\nLarge image (auto-resized) response: {response.text}")

    def test_resize_disabled(self, large_test_image):
        """Test with resizing disabled (original large image sent)."""
        # Create client with resizing disabled
        client = create_ai_client(
            "openai", api_key=os.getenv("OPENAI_API_KEY"), max_image_size=None
        )

        response = client.prompt(
            "gpt-4o-mini",
            "What do you see in this image?",
            images=[large_test_image],
        )

        assert isinstance(response, LLMResponse)
        assert response.text != ""

        print(f"\nLarge image (no resize) response: {response.text[:200]}...")


@pytest.mark.integration
class TestCombinedFilesAndImages:
    """Integration tests combining text files and images."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip test if API key not set."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    def test_image_and_file_together(self, sample_image_path, tmp_path):
        """Test sending both an image and a text file."""
        # Create a description file
        desc_file = tmp_path / "description.txt"
        desc_file.write_text(
            "This should be a simple geometric image with a red square on white background.",
            encoding="utf-8",
        )

        client = create_ai_client("openai", api_key=os.getenv("OPENAI_API_KEY"))

        response = client.prompt(
            "gpt-4o-mini",
            "Does the image match the description in the text file?",
            images=[sample_image_path],
            files=[str(desc_file)],
        )

        assert isinstance(response, LLMResponse)
        assert response.text != ""

        # Should discuss match/comparison
        text_lower = response.text.lower()
        has_comparison = any(
            word in text_lower
            for word in ["match", "yes", "correct", "accurate", "similar", "square", "red"]
        )
        assert has_comparison

        print(f"\nImage + file comparison: {response.text}")

    def test_image_and_file_with_structured_output(self, sample_image_path, tmp_path):
        """Test image + file with structured output."""
        desc_file = tmp_path / "expected.txt"
        desc_file.write_text("A small red square centered on a white background", encoding="utf-8")

        client = create_ai_client("openai", api_key=os.getenv("OPENAI_API_KEY"))

        response = client.prompt(
            "gpt-4o-mini",
            "Compare the image to the description file and provide a structured assessment.",
            images=[sample_image_path],
            files=[str(desc_file)],
            response_format=ImageTextMatch,
        )

        assert isinstance(response, LLMResponse)
        assert response.text != ""

        # Parse structured output
        data = json.loads(response.text)
        assert "matches" in data
        assert "confidence" in data
        assert "explanation" in data
        assert isinstance(data["matches"], bool)
        assert data["confidence"] in ["high", "medium", "low"]

        print(f"\nStructured image+file assessment: {data}")

    def test_multiple_images_and_files(self, sample_image_path, sample_multiple_files):
        """Test with multiple images and multiple files."""
        client = create_ai_client("openai", api_key=os.getenv("OPENAI_API_KEY"))

        # Use same image twice for testing
        response = client.prompt(
            "gpt-4o-mini",
            "Analyze the images and relate them to the topics in the text files. "
            "Are the images relevant to AI/ML topics?",
            images=[sample_image_path, sample_image_path],
            files=sample_multiple_files,
        )

        assert isinstance(response, LLMResponse)
        assert response.text != ""

        print(f"\nMultiple images + files: {response.text[:300]}...")


@pytest.mark.integration
class TestFilesWithOtherProviders:
    """Test files feature with other providers."""

    def test_claude_with_files(self, sample_text_file):
        """Test Claude with text files."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        client = create_ai_client("anthropic", api_key=os.getenv("ANTHROPIC_API_KEY"))

        response = client.prompt(
            "claude-3-5-haiku-20241022",
            "Summarize this document briefly.",
            files=[sample_text_file],
        )

        assert isinstance(response, LLMResponse)
        assert response.text != ""
        assert response.provider == "anthropic"

        print(f"\nClaude with file: {response.text[:200]}...")

    def test_gemini_with_files(self, sample_text_file):
        """Test Gemini with text files."""
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set")

        client = create_ai_client("genai", api_key=os.getenv("GOOGLE_API_KEY"))

        response = client.prompt(
            "gemini-2.0-flash-exp",
            "What is this document about?",
            files=[sample_text_file],
        )

        assert isinstance(response, LLMResponse)
        assert response.text != ""
        assert response.provider == "genai"

        print(f"\nGemini with file: {response.text[:200]}...")
