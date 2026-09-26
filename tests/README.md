# Test Suite

The full testing guide is available in [TESTING.md](../TESTING.md); this file covers what is specific to
this directory. [INTEGRATION_TESTING.md](../INTEGRATION_TESTING.md) covers the tests that
make real API calls.

```bash
pip install -e ".[test]"
pytest -m "not integration"     # 274 unit tests, no API keys needed
```

## Layout

```
tests/
├── __init__.py                      # Package marker
├── conftest.py                      # Shared fixtures and mocks
├── test_response.py                 # Response dataclasses
├── test_utils.py                    # Utility functions and usage recovery
├── test_base_client.py              # Factory and base functionality
├── test_openai_client.py            # OpenAI client, including parameter adaptation
├── test_claude_client.py            # Claude client
├── test_other_clients.py            # Gemini, Mistral, DeepSeek, Qwen, Cohere, HuggingFace
├── test_cost_accounting.py          # The cost contract, across providers
├── test_pricing.py                  # Cost calculation and billed-cost handling
├── test_reasoning.py                # Reasoning token derivation
├── test_caching.py                  # Prompt caching metrics
├── test_files_and_resize.py         # Text files and image resizing
├── test_async.py                    # Async functionality
├── test_version.py                  # Release metadata
├── test_integration_*.py            # Real API calls; need keys, cost money
└── fixtures/                        # Test images and data
```

## Markers

Registered in `pytest.ini`: `unit`, `integration`, `slow`, `asyncio`. Only `integration`
is currently applied to tests; selecting the other markers collects no tests. Use:

```bash
pytest -m "not integration"     # everything that runs without keys
pytest -m integration           # real API calls
```

## Fixtures

Defined in `conftest.py`.

| Fixture | Purpose |
|---|---|
| `mock_openai_response` | An OpenAI-shaped chat completion (10 input tokens, 20 output tokens) |
| `mock_reasoning_response` | An x-ai-shaped response with reasoning tokens outside the completion count |
| `mock_openrouter_response` | A response containing a provider-billed cost |
| `mock_gemini_response` | A genai response with `usage_metadata` |
| `mock_gemini_thinking_response` | A genai response reporting thinking tokens |
| `mock_gemini_truncated_thinking_response` | A Gemini thinking response with no candidate token count |
| `mock_claude_response` | An Anthropic response with cache counters |
| `mock_mistral_response`, `mock_cohere_response` | Provider-shaped responses |
| `mock_huggingface_response` | A router response echoing a normalised model id |
| `sample_image_path` | Path to a real 100x100 PNG in a temporary directory |
| `mock_api_key`, `mock_pydantic_model` | A test key, and a schema for structured output |
| `stub_pricing` | Configures a test pricing table and resets the manager after the test |
| `wire_structured_output` | Points a mocked structured-output endpoint at a response or an exception |

The following fixtures support cost-accounting tests:

**`stub_pricing`** exists because `ai_client/pricing.json` is regenerated at every release.
Asserting against a real model's rate produces a test that breaks whenever a provider
changes its prices, so use a test table with fixed rates.

**`wire_structured_output`** exists because the client reads the response body before the
SDK validates it, so tests must configure both the raw-response and parsed-response interfaces.

## Writing a Test

```python
import pytest
from unittest.mock import Mock, patch
from ai_client import create_ai_client


def test_my_feature():
    """Test description."""
    with patch('ai_client.openai_client.OpenAI') as mock_openai:
        mock_client = Mock()
        mock_openai.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "test"
        mock_client.chat.completions.create.return_value = mock_response

        client = create_ai_client('openai', api_key='test')
        response = client.prompt('gpt-4', 'test')

        assert response.text == "test"
```

When writing tests, account for the following:

- A failure-path test spends seven seconds in `retry_with_exponential_backoff` unless it
  patches `ai_client.utils.time.sleep`.
- An unrestricted `Mock` creates attributes on access and does not reproduce SDK parsing
  behavior. Use `httpx.MockTransport` with the provider SDK to test missing response fields
  and SDK parsing failures.
- A fixture encodes an assumption about a provider. Cohere returns whole numbers as
  floats, for instance, which a hand-written integer fixture will not reveal.
- Verify that regression tests fail before the fix and pass afterward.
