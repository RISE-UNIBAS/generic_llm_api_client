"""
Shared test fixtures and configuration for pytest.
"""

import inspect
import json
import pytest
from unittest.mock import Mock

import ai_client.pricing
from ai_client.pricing import set_pricing_file


@pytest.fixture
def stub_pricing(tmp_path):
    """
    Point the global PricingManager at a small pricing table for one test.

    The bundled pricing.json is regenerated from the benchmark repo at every release, so
    asserting on a real model's price would make a test fail whenever a provider changes
    its rates. Call the yielded function with {date: {provider: {model: {...}}}}.

    Resets the module-level manager afterwards; it is cached process-wide and would
    otherwise leak into unrelated tests.
    """

    def _stub(table):
        path = tmp_path / "pricing.json"
        path.write_text(
            json.dumps({"metadata": {"version": "test"}, "pricing": table}), encoding="utf-8"
        )
        set_pricing_file(str(path))
        return path

    yield _stub

    ai_client.pricing._pricing_manager = None


@pytest.fixture
def wire_structured_output():
    """
    Point a mocked structured-output endpoint at a response, or an exception.

    The client reads the body before the SDK validates it, so a test has to wire the
    raw-response surface as well as the plain one. Yields a function taking the mocked
    SDK client and the response or exception it should produce.
    """

    def _wire(api, outcome):
        raw = api.beta.chat.completions.with_raw_response.parse
        plain = api.beta.chat.completions.parse
        if isinstance(outcome, BaseException) or inspect.isfunction(outcome):
            plain.side_effect = outcome
            raw.side_effect = outcome
            return
        plain.return_value = outcome
        raw.side_effect = None
        raw.return_value.parse.return_value = outcome
        raw.return_value.text = "{}"

    return _wire


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    response = Mock()
    response.id = "chatcmpl-123"
    response.model = "gpt-4"
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = "Hello! I'm an AI assistant."
    response.choices[0].message.tool_calls = (
        None  # Explicitly set to None to prevent iteration errors
    )
    response.choices[0].finish_reason = "stop"
    response.usage = Mock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 20
    response.usage.total_tokens = 30
    # Mock prompt_tokens_details for cached tokens (prevents Mock comparison errors)
    response.usage.prompt_tokens_details = Mock()
    response.usage.prompt_tokens_details.cached_tokens = 0
    return response


@pytest.fixture
def mock_huggingface_response():
    """
    Mock HuggingFace Inference Providers router response.

    The router speaks the OpenAI chat-completions format, so this mirrors
    mock_openai_response. `model` is the normalized id the router actually echoes
    back (lowercased, date suffix dropped), which differs from the requested id.
    """
    response = Mock()
    response.id = "chatcmpl-hf-123"
    response.model = "swiss-ai/apertus-8b-instruct"
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = "Gruezi! I'm Apertus."
    response.choices[0].message.tool_calls = None
    response.choices[0].finish_reason = "stop"
    response.usage = Mock()
    response.usage.prompt_tokens = 1_000_000
    response.usage.completion_tokens = 1_000_000
    response.usage.total_tokens = 2_000_000
    response.usage.prompt_tokens_details = Mock()
    response.usage.prompt_tokens_details.cached_tokens = 0
    # The router does not report a per-request cost. Remove the auto-created Mock
    # attribute so the client falls through to the pricing.json lookup.
    del response.usage.cost
    return response


@pytest.fixture
def mock_reasoning_response():
    """
    Mock an OpenAI-shaped response whose reasoning sits outside completion_tokens.

    This is the x-ai shape: total exceeds prompt + completion by the reasoning count, so
    the tokens are billed but invisible to an input/output-only reading.
    """
    response = Mock()
    response.id = "chatcmpl-reasoning-123"
    response.model = "grok-4.6"
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = "Thought about it."
    response.choices[0].message.tool_calls = None
    response.choices[0].finish_reason = "stop"
    response.usage = Mock()
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 200
    response.usage.total_tokens = 500
    response.usage.completion_tokens_details = Mock()
    response.usage.completion_tokens_details.reasoning_tokens = 200
    response.usage.prompt_tokens_details = Mock()
    response.usage.prompt_tokens_details.cached_tokens = 0
    del response.usage.cost
    return response


@pytest.fixture
def mock_openrouter_response():
    """
    Mock an OpenRouter response carrying the cost the provider actually billed.

    OpenRouter routes are unpinned, so its own figure is the only accurate one and must
    never be replaced by a list-price calculation.
    """
    response = Mock()
    response.id = "chatcmpl-or-123"
    response.model = "openai/gpt-4o"
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = "Routed."
    response.choices[0].message.tool_calls = None
    response.choices[0].finish_reason = "stop"
    response.usage = Mock()
    response.usage.prompt_tokens = 1_000_000
    response.usage.completion_tokens = 1_000_000
    response.usage.total_tokens = 2_000_000
    response.usage.cost = 0.0123
    response.usage.prompt_tokens_details = Mock()
    response.usage.prompt_tokens_details.cached_tokens = 0
    return response


@pytest.fixture
def mock_gemini_thinking_response():
    """Mock a Gemini response whose thinking tokens sit outside candidates_token_count."""
    response = Mock()
    response.text = "Thought about it."
    response.usage_metadata = Mock()
    response.usage_metadata.prompt_token_count = 100
    response.usage_metadata.candidates_token_count = 200
    response.usage_metadata.total_token_count = 500
    response.usage_metadata.thoughts_token_count = 200
    response.candidates = [Mock()]
    response.candidates[0].finish_reason = "STOP"
    return response


@pytest.fixture
def mock_gemini_truncated_thinking_response(mock_gemini_thinking_response):
    """
    Mock a Gemini thinking response that stopped early and reported no candidates count.

    Every count on usage_metadata is optional; this is the shape that crashed cost
    calculation before the counts were coerced.
    """
    mock_gemini_thinking_response.usage_metadata.candidates_token_count = None
    return mock_gemini_thinking_response


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
    # Mock cache attributes (prevents Mock comparison errors)
    response.usage.cache_creation_input_tokens = 0
    response.usage.cache_read_input_tokens = 0
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
def mock_cohere_response():
    """Mock Cohere API response."""
    response = Mock()
    response.id = "cohere_123"
    response.finish_reason = "COMPLETE"

    # Mock message content structure
    message = Mock()
    content_block = Mock()
    content_block.text = "Hello! I'm Cohere."
    message.content = [content_block]
    response.message = message

    # Mock usage information
    usage = Mock()
    tokens = Mock()
    # Cohere reports whole numbers as floats.
    tokens.input_tokens = 13.0
    tokens.output_tokens = 17.0
    usage.tokens = tokens

    billed_units = Mock()
    billed_units.input_tokens = 13
    billed_units.output_tokens = 17
    usage.billed_units = billed_units

    response.usage = usage
    return response


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a temporary test image file using Pillow."""
    from PIL import Image, ImageDraw

    # Create a simple 100x100 image with a red square on white background
    image_file = tmp_path / "test_image.png"
    img = Image.new("RGB", (100, 100), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([25, 25, 75, 75], fill="red")
    img.save(str(image_file), "PNG")

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
