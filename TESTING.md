# Testing Guide for Generic LLM API Client v0.1.0

## Quick Start

```bash
# 1. Install with test dependencies
pip install -e ".[test]"

# 2. Run all tests
pytest

# 3. Run with coverage
pytest --cov=ai_client --cov-report=term-missing
```

## Test Suite Overview

The test suite includes **70+ unit tests** and **18+ integration tests** covering:

- ✅ Response dataclasses (LLMResponse, Usage)
- ✅ Utility functions (retry logic, error detection)
- ✅ Factory function (create_ai_client)
- ✅ Base client functionality
- ✅ OpenAI client implementation
- ✅ Claude client implementation
- ✅ Gemini client implementation
- ✅ Mistral client implementation
- ✅ DeepSeek client (OpenAI-compatible)
- ✅ Qwen client (OpenAI-compatible)
- ✅ Async functionality
- ✅ Error handling

## Test Files

```
tests/
├── conftest.py                      # Shared fixtures (mocks, sample data)
├── test_response.py                 # Response dataclass tests
├── test_utils.py                    # Utility function tests
├── test_base_client.py              # Factory and base client tests
├── test_openai_client.py            # OpenAI implementation tests
├── test_claude_client.py            # Claude implementation tests
├── test_other_clients.py            # Gemini, Mistral, DeepSeek, Qwen tests
├── test_async.py                    # Async functionality tests
├── test_integration_basic.py        # Basic integration tests (text-only)
├── test_integration_multimodal.py   # Multimodal integration tests (vision)
├── test_integration_structured.py   # Structured output integration tests
└── fixtures/                        # Test images and data
    └── README.md                    # Fixtures documentation
```

## Running Tests

### All Tests

```bash
pytest                          # Run all tests
pytest -v                       # Verbose output
pytest -vv                      # Extra verbose
pytest --tb=short               # Short traceback
```

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
pytest -k "async"               # All async tests
```

### With Coverage

```bash
# Terminal report
pytest --cov=ai_client

# Terminal + missing lines
pytest --cov=ai_client --cov-report=term-missing

# HTML report
pytest --cov=ai_client --cov-report=html
open htmlcov/index.html
```

### Async Tests

```bash
# All async tests
pytest tests/test_async.py

# Or by marker
pytest -m asyncio
```

## Test Categories

### Unit Tests (All Current Tests)

Mock external APIs, test internal logic:

```bash
pytest tests/
```

No API keys required - all providers are mocked.

### Integration Tests

Make real API calls to verify actual compatibility:

```bash
# Run all integration tests (requires API keys)
pytest -m integration

# Run specific integration test file
pytest -m integration tests/test_integration_basic.py

# See INTEGRATION_TESTING.md for full guide
```

**Note:** Integration tests require API keys and cost money. They are automatically skipped if keys are not set. See [INTEGRATION_TESTING.md](INTEGRATION_TESTING.md) for detailed setup and usage guide.

## Expected Output

### Successful Run

```
============== test session starts ==============
tests/test_response.py ............          [  15%]
tests/test_utils.py ................         [  38%]
tests/test_base_client.py ..........         [  53%]
tests/test_openai_client.py ........         [  65%]
tests/test_claude_client.py ......           [  74%]
tests/test_other_clients.py ........         [  85%]
tests/test_async.py .........                [ 100%]

============== 70 passed in 2.45s ==============
```

### With Coverage

```
---------- coverage: platform linux, python 3.11 -----------
Name                              Stmts   Miss  Cover   Missing
---------------------------------------------------------------
ai_client/__init__.py                12      0   100%
ai_client/base_client.py             95      2    98%   145, 172
ai_client/response.py                32      0   100%
ai_client/utils.py                   45      1    98%   87
ai_client/openai_client.py          112      3    97%   ...
ai_client/claude_client.py           98      2    98%   ...
ai_client/gemini_client.py          102      4    96%   ...
ai_client/mistral_client.py          87      2    98%   ...
ai_client/deepseek_client.py          8      0   100%
ai_client/qwen_client.py              8      0   100%
---------------------------------------------------------------
TOTAL                               599     14    98%
```

## What's Tested

### Response System ✅

- Usage dataclass creation and methods
- LLMResponse dataclass creation and methods
- Dictionary conversion
- String representation
- Optional fields handling

### Utilities ✅

- Retry with exponential backoff
- Rate limit error detection
- Retry delay extraction
- Custom exceptions

### Factory Function ✅

- Creating clients for all providers
- Passing API keys
- Custom system prompts
- Custom settings
- Custom base URLs
- Error handling for invalid providers

### Base Client ✅

- Initialization
- Elapsed time tracking
- Client lifecycle (end_client)
- URL detection
- Multimodal support checking
- String representation

### Provider Clients ✅

For each provider (OpenAI, Claude, Gemini, Mistral, DeepSeek, Qwen):

- Client initialization
- Text-only prompts
- Multimodal prompts (text + images)
- Custom parameters (temperature, max_tokens, etc.)
- Structured output (Pydantic models)
- Error handling
- Response formatting

### Async Functionality ✅

- Basic async prompts
- Parallel async execution
- Async with images
- Async with custom parameters
- Async error handling
- Benchmark simulation (batched processing)

## What's NOT Tested (Yet)

### Not Implemented

- ❌ Real API calls (integration tests)
- ❌ Streaming functionality (not implemented)
- ❌ Tool use / function calling (not implemented)
- ❌ Conversation history (not implemented)

### Could Add

- Edge cases for very long prompts
- Network timeout scenarios
- Malformed response handling
- Rate limit retry behavior (real timing)
- Large file handling

## Adding New Tests

### Example Test Template

```python
import pytest
from unittest.mock import Mock, patch
from ai_client import create_ai_client

def test_new_feature():
    \"\"\"Test description.\"\"\"
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
    \"\"\"Fixtures are automatically injected.\"\"\"
    # mock_openai_response is ready to use
    # sample_image_path points to a valid temp image
    pass
```

## Continuous Integration

Example GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11', '3.12']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -e ".[test]"

    - name: Run tests
      run: |
        pytest --cov=ai_client --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

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

## Test Performance

All tests should complete in < 5 seconds:

- Unit tests are mocked (no real API calls)
- Minimal delays (retry tests use 0.01s delays)
- Async tests use small batches

If tests are slow:
1. Check for unmocked API calls
2. Check for unnecessary sleeps
3. Profile with `pytest --durations=10`

## Coverage Goals

Target: **>95% coverage** for all modules

Current coverage should show:
- ✅ `response.py`: 100%
- ✅ `utils.py`: >95%
- ✅ `base_client.py`: >95%
- ✅ `*_client.py`: >95% each

Check with:
```bash
pytest --cov=ai_client --cov-report=term-missing
```

## Next Steps

1. **Run the tests:**
   ```bash
   pip install -e ".[test]"
   pytest
   ```

2. **Check coverage:**
   ```bash
   pytest --cov=ai_client --cov-report=html
   open htmlcov/index.html
   ```

3. **Add integration tests** (optional):
   - Create `.env` with API keys
   - Add `test_integration.py` with real API calls
   - Mark with `@pytest.mark.integration`

4. **Set up CI/CD:**
   - Add GitHub Actions workflow
   - Run tests on every commit
   - Upload coverage to Codecov

## Success Criteria

Before publishing to PyPI:

- ✅ All tests pass
- ✅ Coverage >95%
- ✅ No warnings
- ✅ Tests run in < 5 seconds
- ✅ Documentation complete

Test and verify:

```bash
# Run full test suite
pytest -v --cov=ai_client --cov-report=term-missing

# Should see:
# - All tests passing
# - Coverage >95%
# - No errors or warnings
```

If all checks pass: **Ready for PyPI publication!** 🚀
