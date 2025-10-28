"""
Shared test fixtures and configuration for pytest.
"""
import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    response = Mock()
    response.id = "chatcmpl-123"
    response.model = "gpt-4"
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = "Hello! I'm an AI assistant."
    response.choices[0].finish_reason = "stop"
    response.usage = Mock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 20
    response.usage.total_tokens = 30
    return response


@pytest.fixture
def mock_claude_response():
    """Mock Anthropic Claude API response."""
    response = Mock()
    response.id = "msg_123"
    response.model = "claude-3-5-sonnet-20241022"
    response.content = [Mock()]
    response.content[0].type = "text"
    response.content[0].text = "Hello! I'm Claude."
    response.stop_reason = "end_turn"
    response.usage = Mock()
    response.usage.input_tokens = 15
    response.usage.output_tokens = 25
    return response


@pytest.fixture
def mock_gemini_response():
    """Mock Google Gemini API response."""
    response = Mock()
    response.text = "Hello! I'm Gemini."
    response.usage_metadata = Mock()
    response.usage_metadata.prompt_token_count = 12
    response.usage_metadata.candidates_token_count = 18
    response.usage_metadata.total_token_count = 30
    response.candidates = [Mock()]
    response.candidates[0].finish_reason = "STOP"
    return response


@pytest.fixture
def mock_mistral_response():
    """Mock Mistral API response."""
    response = Mock()
    response.id = "cmpl_123"
    response.model = "mistral-large-latest"
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = "Hello! I'm Mistral."
    response.choices[0].finish_reason = "stop"
    response.usage = Mock()
    response.usage.prompt_tokens = 11
    response.usage.completion_tokens = 19
    response.usage.total_tokens = 30
    return response


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a temporary test image file using Pillow."""
    from PIL import Image, ImageDraw

    # Create a simple 100x100 image with a red square on white background
    image_file = tmp_path / "test_image.png"
    img = Image.new('RGB', (100, 100), color='white')
    draw = ImageDraw.Draw(img)
    draw.rectangle([25, 25, 75, 75], fill='red')
    img.save(str(image_file), 'PNG')

    return str(image_file)


@pytest.fixture
def mock_api_key():
    """Mock API key for testing."""
    return "test-api-key-123"


@pytest.fixture
def mock_pydantic_model():
    """Create a sample Pydantic model for testing structured output."""
    from pydantic import BaseModel

    class TestModel(BaseModel):
        name: str
        value: int

    return TestModel
