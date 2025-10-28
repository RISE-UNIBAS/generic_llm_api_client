# Integration Testing Guide

This guide covers running **integration tests** that make real API calls to verify actual compatibility with LLM providers.

## Overview

**Integration tests vs Unit tests:**

- **Unit tests** (70+ tests): Mock all external APIs, test internal logic, run fast (~2s), no API keys needed
- **Integration tests** (18+ tests): Make real API calls, verify actual compatibility, require API keys, cost money

## Quick Start

```bash
# 1. Set up environment
cp .env.example .env
# Edit .env and add your API keys

# 2. Install test dependencies
pip install -e ".[test]"

# 3. Run integration tests
pytest -m integration

# 4. Run specific test file
pytest -m integration tests/test_integration_basic.py
```

## Setting Up API Keys

### Create .env File

```bash
cp .env.example .env
```

Edit `.env` and add keys for providers you want to test:

```bash
# Required for basic tests
OPENAI_API_KEY=sk-...

# Optional - add any you have
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
MISTRAL_API_KEY=...
DEEPSEEK_API_KEY=...
QWEN_API_KEY=...
```

**Note**: You don't need ALL keys. Tests automatically skip providers without keys.

### Security Warning

⚠️ **Never commit `.env` to git!**

The `.env` file is already in `.gitignore`. Double-check:

```bash
git status  # Should NOT show .env
```

## Running Integration Tests

### All Integration Tests

```bash
# Run all integration tests
pytest -m integration

# Verbose output
pytest -m integration -v

# Very verbose (show print statements)
pytest -m integration -vv -s
```

### By Test File

```bash
# Basic text-only tests (cheapest, fastest)
pytest -m integration tests/test_integration_basic.py

# Multimodal tests (uses vision models, more expensive)
pytest -m integration tests/test_integration_multimodal.py

# Structured output tests (Pydantic models)
pytest -m integration tests/test_integration_structured.py
```

### By Provider

```bash
# Test only OpenAI
pytest -m integration -k "openai"

# Test only Claude
pytest -m integration -k "claude"

# Test only vision models
pytest -m integration -k "vision"

# Test structured output
pytest -m integration -k "structured"
```

### Exclude Integration Tests

```bash
# Run only unit tests (default, no API keys needed)
pytest -m "not integration"

# Or just run pytest normally (integration tests require explicit -m flag)
pytest
```

## Test Categories

### 1. Basic Integration Tests (`test_integration_basic.py`)

**What it tests:**
- Text-only prompts
- Basic response structure
- Usage tracking (tokens)
- Error handling

**Providers tested:**
- OpenAI (gpt-4o-mini)
- Claude (claude-3-5-haiku-20241022)
- Gemini (gemini-2.0-flash-exp)
- Mistral (mistral-small-latest)
- DeepSeek (deepseek-chat)
- Qwen (qwen-turbo)

**Cost estimate:** ~$0.01-0.05 per full run

**Example:**
```bash
pytest -m integration tests/test_integration_basic.py
```

### 2. Multimodal Integration Tests (`test_integration_multimodal.py`)

**What it tests:**
- Image + text prompts
- Vision model compatibility
- Multiple images per request
- Benchmark workflow simulation

**Providers tested:**
- OpenAI (gpt-4o-mini)
- Claude (claude-3-5-sonnet-20241022)
- Gemini (gemini-2.0-flash-exp)
- Mistral (pixtral-12b-2409)
- DeepSeek (if vision supported)
- Qwen (qwen-vl-max)

**Cost estimate:** ~$0.05-0.20 per full run (vision is more expensive)

**Example:**
```bash
pytest -m integration tests/test_integration_multimodal.py -v
```

### 3. Structured Output Tests (`test_integration_structured.py`)

**What it tests:**
- Pydantic model support
- JSON schema generation
- Structured output parsing
- Vision + structured output combination

**Providers tested:**
- OpenAI (native structured output)
- Claude (tool-based structured output)
- Gemini (schema-based)
- Mistral (JSON mode)

**Cost estimate:** ~$0.02-0.10 per full run

**Example:**
```bash
pytest -m integration tests/test_integration_structured.py -v
```

## Understanding Test Results

### Successful Run

```
tests/test_integration_basic.py::TestOpenAIIntegration::test_openai_basic_prompt PASSED
tests/test_integration_basic.py::TestClaudeIntegration::test_claude_basic_prompt PASSED
tests/test_integration_basic.py::TestGeminiIntegration::test_gemini_basic_prompt PASSED

OpenAI vision response: I see a red square on a white background.

Tested 3 providers successfully:
  openai: 1.23s, 45 tokens
  anthropic: 0.89s, 38 tokens
  genai: 1.45s, 42 tokens

============== 12 passed, 6 skipped in 15.67s ==============
```

### Skipped Tests

Tests are automatically skipped if API keys are not set:

```
tests/test_integration_basic.py::TestMistralIntegration::test_mistral_basic_prompt SKIPPED
Reason: MISTRAL_API_KEY not set
```

This is **normal** - you only need keys for providers you want to test.

### Failed Tests

If a test fails, check:

1. **API key valid?**
   ```bash
   echo $ANTHROPIC_API_KEY  # Check if set correctly
   ```

2. **Correct model name?** Model names change over time. Check provider docs:
   - OpenAI: https://platform.openai.com/docs/models
   - Anthropic: https://docs.anthropic.com/en/docs/models-overview
   - Gemini: https://ai.google.dev/models/gemini

3. **Rate limits?** Wait a minute and retry:
   ```bash
   pytest -m integration --lf  # Re-run last failed
   ```

4. **Insufficient credits?** Check your provider account balance

## Cost Management

### Minimize Costs

**Use cheapest models:**
```python
# Already configured in tests
OpenAI: gpt-4o-mini (~$0.00015 per 1K input tokens)
Claude: claude-3-5-haiku-20241022 (~$0.00025 per 1K input tokens)
Gemini: gemini-2.0-flash-exp (free tier available)
Mistral: mistral-small-latest (~$0.0002 per 1K input tokens)
```

**Run selectively:**
```bash
# Run only one provider
pytest -m integration -k "openai"

# Run only basic tests (skip expensive vision tests)
pytest -m integration tests/test_integration_basic.py
```

**Avoid running in CI/CD** (unless you set up billing):
```yaml
# In GitHub Actions - skip integration tests by default
- name: Run tests
  run: pytest -m "not integration"
```

### Estimate Costs

**Rough estimates per full run:**
- Basic tests only: $0.01 - $0.05
- All tests (including vision): $0.10 - $0.50
- 10 full runs: ~$1-5
- 100 full runs: ~$10-50

**For humanities benchmarks:**
- 100 manuscript images (OpenAI gpt-4o-mini): ~$1-3
- 1000 manuscript images: ~$10-30
- 10,000 images: ~$100-300

Use async batch processing to optimize costs and speed.

## CI/CD Integration

### Run in GitHub Actions (Optional)

⚠️ **Integration tests cost money!** Only run if you set up billing.

```yaml
name: Integration Tests

on:
  workflow_dispatch:  # Manual trigger only
  schedule:
    - cron: '0 0 * * 0'  # Weekly, Sunday midnight

jobs:
  integration:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -e ".[test]"

    - name: Run integration tests
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
      run: |
        pytest -m integration --tb=short
```

**Add secrets in GitHub:**
Settings → Secrets and variables → Actions → New repository secret

## Benchmark Simulation

Integration tests include benchmark workflow simulations:

```bash
# Simulate processing 3 manuscript images
pytest -m integration -k "benchmark_workflow" -v -s
```

This tests:
- Sequential image processing
- Token usage tracking
- Response time measurement
- Structured output extraction

Output shows realistic benchmark metrics:

```
Benchmark simulation results:
  Total duration: 4.56s
  Total tokens: 3,245
  Average per image: 1.52s, 1082 tokens

Structured benchmark results:
  Total duration: 5.23s
  Total tokens: 3,890
  Image 1: {'language': 'Latin', 'script': 'Latin', 'condition': 'good', ...}
  Image 2: {'language': 'Latin', 'script': 'Latin', 'condition': 'fair', ...}
  Image 3: {'language': 'Greek', 'script': 'Greek', 'condition': 'poor', ...}
```

## Troubleshooting

### python-dotenv Not Found

```bash
pip install python-dotenv
# Or install with test extras
pip install -e ".[test]"
```

### API Key Not Loading

Check `.env` file format (no quotes needed):
```bash
# ❌ Wrong
OPENAI_API_KEY="sk-..."

# ✅ Correct
OPENAI_API_KEY=sk-...
```

Verify loading:
```python
from dotenv import load_dotenv
import os
load_dotenv()
print(os.getenv('OPENAI_API_KEY'))  # Should print your key
```

### Rate Limit Errors

The client has built-in retry logic, but if you hit rate limits:

```bash
# Add delays between tests
pytest -m integration --maxfail=1  # Stop on first failure

# Or run fewer tests
pytest -m integration -k "basic"
```

### Model Not Found Errors

Model names change over time. Update test files if needed:

```python
# In test_integration_basic.py, update model names
# Example: Update to latest Claude model
response, duration = client.prompt('claude-3-5-sonnet-20241022', ...)
```

Check provider documentation for current model names.

### Tests Hang

If tests hang indefinitely:
1. Check network connection
2. Verify API keys are valid
3. Check provider status pages
4. Kill and retry: Ctrl+C, then `pytest --lf`

## Best Practices

### Before Release

Run full integration test suite:

```bash
# 1. Update .env with valid keys
cp .env.example .env
vim .env

# 2. Run all integration tests
pytest -m integration -v

# 3. Verify all passed (or only skipped due to missing keys)
# 4. Check no unexpected failures
```

### During Development

Run targeted tests:

```bash
# Test only what you changed
pytest -m integration -k "openai" -v

# Or test specific feature
pytest -m integration -k "structured" -v
```

### For Humanities Benchmarks

Before running large benchmark:

```bash
# 1. Test with one image first
pytest -m integration -k "benchmark_workflow" -v -s

# 2. Verify token counts and timing are reasonable

# 3. Calculate cost for full benchmark:
#    (tokens per image) * (number of images) * (cost per token)
#    Example: 1000 tokens * 500 images * $0.00015 = $75

# 4. Consider using cheaper models or async batching
```

## Adding New Integration Tests

### Template

```python
import os
import pytest
from dotenv import load_dotenv
from ai_client import create_ai_client

load_dotenv()

@pytest.mark.integration
class TestNewFeature:
    """Integration tests for new feature."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip if API key not set."""
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("OPENAI_API_KEY not set")

    def test_new_feature(self):
        """Test new feature with real API."""
        client = create_ai_client('openai', api_key=os.getenv('OPENAI_API_KEY'))
        response, duration = client.prompt('gpt-4o-mini', 'Test prompt')

        assert response.text != ""
        assert duration > 0
```

### Guidelines

1. **Always mark with `@pytest.mark.integration`**
2. **Always skip if API key missing**
3. **Use cheapest models possible**
4. **Keep prompts short** (to minimize costs)
5. **Test real behavior**, not just "does it work"
6. **Add print statements** for debugging (use `-s` flag to see them)

## Summary

**Run integration tests to:**
- ✅ Verify real API compatibility before release
- ✅ Test new features with actual providers
- ✅ Validate format assumptions
- ✅ Simulate benchmark workflows

**Remember:**
- Unit tests (default): Fast, free, no API keys needed
- Integration tests (explicit): Slow, costs money, requires API keys
- Only run integration tests when you need to verify real API behavior

**Quick commands:**
```bash
# Unit tests (always run these)
pytest

# Integration tests (run before release)
pytest -m integration

# Specific provider
pytest -m integration -k "openai"

# Skip expensive tests
pytest -m integration tests/test_integration_basic.py
```

For questions or issues, see the main [TESTING.md](TESTING.md) guide.
