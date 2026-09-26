# Testing Guide

## Quick Start

```bash
# 1. Install with test dependencies
pip install -e ".[test]"

# 2. Run the unit tests (what CI runs)
pytest -m "not integration"

# 3. Run with coverage
pytest -m "not integration" --cov=ai_client --cov-report=term-missing
```

## Test Suite Overview

**274 unit tests** and **46 integration tests**, covering:

- Response dataclasses (`LLMResponse`, `Usage`) and their serialisation
- Cost accounting: reasoning tokens, discarded attempts, provider-billed costs
- Pricing lookup and cost components
- Utility functions (retry logic, error detection, usage recovery)
- Factory function (`create_ai_client`)
- Base client, including the failure paths
- Every provider client: OpenAI, Claude, Gemini, Mistral, Cohere, DeepSeek, Alibaba,
  xAI, HuggingFace, OpenRouter and sciCORE
- Prompt caching metrics
- Text file inclusion and image resizing
- Async functionality
- Error handling and recovery

## Test Files

```
tests/
├── conftest.py                      # Shared fixtures (mocks, pricing stub, sample data)
├── test_response.py                 # Response dataclass tests
├── test_utils.py                    # Utility function tests
├── test_base_client.py              # Factory and base client tests
├── test_openai_client.py            # OpenAI implementation tests
├── test_claude_client.py            # Claude implementation tests
├── test_other_clients.py            # Gemini, Mistral, DeepSeek, Qwen, Cohere tests
├── test_cost_accounting.py          # Cost contract across providers
├── test_pricing.py                  # Cost calculation and billed-cost handling
├── test_reasoning.py                # Reasoning token derivation
├── test_caching.py                  # Prompt caching metrics
├── test_files_and_resize.py         # Text files and image resizing
├── test_version.py                  # Release metadata
├── test_async.py                    # Async functionality tests
├── test_integration_basic.py        # Basic integration tests (text-only)
├── test_integration_multimodal.py   # Multimodal integration tests (vision)
├── test_integration_structured.py   # Structured output integration tests
├── test_integration_files.py        # File handling integration tests
└── fixtures/                        # Test images and data
    └── README.md                    # Fixtures documentation
```

## Running Tests

### All Tests

```bash
pytest -m "not integration"     # Unit tests only, as CI runs them
pytest -v                       # Verbose output
pytest --tb=short               # Short traceback
```

Plain `pytest` also collects the integration tests. Tests without the required API
keys are skipped. Collecting integration tests adds overhead.

### Specific Tests

```bash
# By file
pytest tests/test_response.py

# By class
pytest tests/test_response.py::TestUsage

# By function
pytest tests/test_response.py::TestUsage::test_usage_creation

# By pattern
pytest -k "openai"              # All tests matching "openai"
pytest -k "reasoning"           # All reasoning token tests
```

### Markers

Declared in `pytest.ini`: `unit`, `integration`, `slow`, `asyncio`. `--strict-markers`
is enabled, so unregistered markers produce an error.

```bash
pytest -m integration           # Real API calls, needs keys
pytest -m "not integration"     # Everything else
```

### With Coverage

```bash
pytest -m "not integration" --cov=ai_client --cov-report=term-missing
pytest -m "not integration" --cov=ai_client --cov-report=html
open htmlcov/index.html
```

## Test Categories

### Unit Tests

Mock the provider SDKs and test internal logic. No API keys required.

Some tests use the **provider SDK with a mocked HTTP transport** to exercise SDK parsing
and validation, including content-filter and schema-mismatch failures. These tests verify
that billed usage is preserved when the SDK rejects a response.

### Integration Tests

Make real API calls to verify actual compatibility:

```bash
pytest -m integration
pytest -m integration tests/test_integration_basic.py
```

**Note:** Integration tests require API keys and incur provider charges. They are skipped
automatically if keys are not set. See [INTEGRATION_TESTING.md](INTEGRATION_TESTING.md).

## Expected Output

```
============== test session starts ==============
tests/test_async.py ......                        [   2%]
tests/test_base_client.py .....................   [  10%]
tests/test_caching.py .................           [  16%]
tests/test_claude_client.py ........              [  19%]
tests/test_cost_accounting.py ..................  [  36%]
tests/test_openai_client.py ....................  [  45%]
tests/test_other_clients.py ....................  [  56%]
tests/test_pricing.py ...........                 [  60%]
tests/test_reasoning.py ........................  [  69%]
tests/test_response.py ......................     [  79%]
tests/test_utils.py ...................           [  99%]
tests/test_version.py ..                          [ 100%]

======== 274 passed, 46 deselected in 25s ========
```

## Coverage

Overall coverage is approximately **74%**. Provider client coverage ranges from 58% to 82%;
many multimodal and file-handling branches are exercised by integration tests. `ai_client/tools/` is currently untested.

Cost-accounting modules have higher coverage targets because undetected errors can
affect the accuracy of stored records:

| Module | Coverage |
|---|---|
| `reasoning.py` | 100% |
| `response.py` | 95% |
| `utils.py` | 88% |
| `base_client.py` | 82% |
| `pricing.py` | 80% |

## What Is Tested

### Cost Accounting

The following accounting rules apply to stored records:

- Reasoning tokens billed outside `output_tokens` are recorded, and
  `input + output + reasoning == total` wherever a total was reported
- `0` (the provider reported none) stays distinct from `None` (it reported no total)
- Costs are calculated using the requested model ID
- Provider-billed costs are preserved across all response paths
- Discarded fallback attempts are reported separately from the successful call
- Usage and cost are preserved when a request or local response conversion fails
- Each billed call is counted exactly once, including when several fail in one request

### Provider Clients

For each provider: initialisation, text-only prompts, multimodal prompts, custom
parameters, structured output, error handling and response formatting.

Parameter adaptation tests cover model-family differences in token-limit parameter names
and temperature support.

### Concurrency

Concurrency tests use one client instance across multiple threads, matching the benchmark
harness. They verify that request-specific state remains isolated.

## What Is Not Tested

- Streaming — not implemented in the library
- `ai_client/tools/` — the tool registry and built-in tools have no unit tests, though
  tool *calling* through a client is covered
- Network timeout and connection-failure scenarios
- Very long prompts and large file handling

## Adding New Tests

### Example Test Template

```python
import pytest
from unittest.mock import Mock, patch
from ai_client import create_ai_client

def test_new_feature():
    """Test description."""
    # 1. Setup mocks
    with patch('ai_client.provider_client.ProviderSDK') as mock_sdk:
        mock_client = Mock()
        mock_sdk.return_value = mock_client

        # 2. Setup response
        mock_response = Mock()
        mock_response.text = "test response"
        mock_client.some_method.return_value = mock_response

        # 3. Create client and test
        client = create_ai_client('provider', api_key='test')
        result = client.some_method()

        # 4. Assertions
        assert result == "expected"
        mock_client.some_method.assert_called_once()
```

### Using Fixtures

Fixtures are defined in `conftest.py`:

```python
def test_with_fixtures(mock_openai_response, sample_image_path):
    """Fixtures are automatically injected."""
    # mock_openai_response is ready to use
    # sample_image_path points to a valid temp image
    pass
```

`stub_pricing` configures a small pricing table for each test and resets the pricing manager
afterward. Use this fixture for deterministic assertions: `ai_client/pricing.json` is
regenerated at each release, so tests based on bundled rates can fail when prices change.

`wire_structured_output` points a mocked structured-output endpoint at a response or an
exception, covering both SDK interfaces used by the client.

### Cost Assertions

Verify that each accounting regression test fails against the code before the fix and
passes after the fix. This confirms that the test detects the reported defect.

## Continuous Integration

`.github/workflows/tests.yml` runs on pushes and pull requests to `main`, `master` and
`develop`:

- **test**: Python 3.10, 3.11 and 3.12 on Ubuntu, Windows and macOS, running
  `pytest -m "not integration"` with coverage, uploaded to Codecov from Ubuntu + 3.11
- **lint**: `black --check ai_client tests`, which is **blocking**, and `ruff check`,
  which is not

Integration tests are excluded from CI; they require API keys and incur provider charges.

Run Black before pushing. CI requires Black formatting, which can differ from
`ruff format`. If Black reports an interpreter compatibility issue on a Python 3.12 build,
run it in a virtual environment with a compatible Python version.

## Troubleshooting

### Tests Fail to Import

```bash
# Solution: Install package in editable mode
pip install -e .
```

### pytest Command Not Found

```bash
# Solution: Install pytest
pip install pytest
```

### Async Tests Fail

```bash
# Solution: Install pytest-asyncio
pip install pytest-asyncio
```

### Coverage Tool Missing

```bash
# Solution: Install pytest-cov
pip install pytest-cov
```

### A Test Is Unexpectedly Slow

`retry_with_exponential_backoff` retries every exception three times, sleeping 1s, 2s and
4s. A test that exercises a failure path spends seven seconds there unless it patches the
sleep:

```python
with patch("ai_client.utils.time.sleep"):
    ...
```

## Test Performance

The unit suite runs in approximately 25 seconds. Tests that exercise real retry timing
account for most of the runtime.

If tests are slower than that:

1. Check for unmocked API calls
2. Check for failure-path tests that do not patch the retry sleep
3. Profile with `pytest --durations=10`
