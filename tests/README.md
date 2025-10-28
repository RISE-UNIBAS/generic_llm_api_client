# Test Suite for Generic LLM API Client

This directory contains the test suite for the generic-llm-api-client package.

## Test Structure

```
tests/
├── __init__.py                 # Package marker
├── conftest.py                 # Shared fixtures and mocks
├── test_response.py            # Tests for response dataclasses
├── test_utils.py               # Tests for utility functions
├── test_base_client.py         # Tests for factory and base functionality
├── test_openai_client.py       # Tests for OpenAI client
├── test_claude_client.py       # Tests for Claude client
├── test_async.py               # Tests for async functionality
└── README.md                   # This file
```

## Running Tests

### Install Test Dependencies

```bash
# Install package with test dependencies
pip install -e ".[test]"

# Or install with all dev dependencies
pip install -e ".[dev]"
```

### Run All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=ai_client --cov-report=term-missing
```

### Run Specific Tests

```bash
# Run tests in a specific file
pytest tests/test_response.py

# Run a specific test class
pytest tests/test_response.py::TestUsage

# Run a specific test function
pytest tests/test_response.py::TestUsage::test_usage_creation

# Run tests matching a pattern
pytest -k "test_openai"
```

### Run Tests by Marker

```bash
# Run only unit tests
pytest -m unit

# Run only async tests
pytest -m asyncio

# Exclude slow tests
pytest -m "not slow"
```

## Test Types

### Unit Tests (Default)

These tests mock external dependencies (API clients) and test internal logic:

```bash
pytest -m unit
```

All current tests are unit tests that don't require API keys.

### Integration Tests (Future)

Integration tests that make real API calls (require API keys):

```bash
# Not yet implemented
pytest -m integration
```

To add integration tests in the future, create a `.env` file:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GENAI_API_KEY=...
MISTRAL_API_KEY=...
```

## Code Coverage

Generate a detailed HTML coverage report:

```bash
pytest --cov=ai_client --cov-report=html

# Open the report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## Writing New Tests

### Example Test

```python
import pytest
from unittest.mock import Mock, patch
from ai_client import create_ai_client

def test_my_feature():
    \"\"\"Test description.\"\"\"
    with patch('ai_client.openai_client.OpenAI') as mock_openai:
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "test"
        mock_client.chat.completions.create.return_value = mock_response

        # Test
        client = create_ai_client('openai', api_key='test')
        response, duration = client.prompt('gpt-4', 'test')

        assert response.text == "test"
```

### Using Fixtures

Fixtures are defined in `conftest.py`:

```python
def test_with_fixture(mock_openai_response, sample_image_path):
    \"\"\"Test using predefined fixtures.\"\"\"
    # mock_openai_response and sample_image_path are automatically provided
    assert sample_image_path.endswith('.jpg')
```

### Testing Async Code

```python
@pytest.mark.asyncio
async def test_async_feature():
    \"\"\"Test async functionality.\"\"\"
    result = await some_async_function()
    assert result == expected
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -e ".[test]"
    pytest --cov=ai_client --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Test Best Practices

1. **Mock External Dependencies**: All API clients should be mocked
2. **Test One Thing**: Each test should verify one behavior
3. **Clear Names**: Test names should describe what they test
4. **Use Fixtures**: Reuse common setup via fixtures
5. **Fast Tests**: Keep unit tests fast (< 1 second each)
6. **Independent Tests**: Tests should not depend on each other

## Troubleshooting

### Import Errors

```bash
# Make sure package is installed in editable mode
pip install -e .
```

### Missing Dependencies

```bash
# Install test dependencies
pip install -e ".[test]"
```

### Async Test Issues

```bash
# Make sure pytest-asyncio is installed
pip install pytest-asyncio
```

## Current Test Coverage

Run `pytest --cov=ai_client` to see current coverage. Goal is >90% coverage for core functionality.

### Coverage Status

- `response.py`: ✅ Full coverage
- `utils.py`: ✅ Full coverage
- `base_client.py`: ✅ Full coverage
- `openai_client.py`: ✅ Full coverage
- `claude_client.py`: ✅ Full coverage
- `gemini_client.py`: ⚠️ Needs tests
- `mistral_client.py`: ⚠️ Needs tests
- `deepseek_client.py`: ⚠️ Needs tests (extends OpenAI)
- `qwen_client.py`: ⚠️ Needs tests (extends OpenAI)

## Adding Integration Tests

To add integration tests that use real APIs:

1. Create `test_integration.py`
2. Mark with `@pytest.mark.integration`
3. Use environment variables for API keys
4. Add to CI only for scheduled runs, not PRs

Example:

```python
import pytest
import os

@pytest.mark.integration
def test_real_openai_call():
    \"\"\"Integration test with real OpenAI API.\"\"\"
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    client = create_ai_client('openai', api_key=api_key)
    response, duration = client.prompt('gpt-4', 'Say hello')

    assert len(response.text) > 0
    assert response.usage.total_tokens > 0
```
