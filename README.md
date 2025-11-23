# Generic LLM API Client

A unified, provider-agnostic Python client for multiple LLM APIs. Query any LLM (OpenAI, Anthropic Claude, Google Gemini, Mistral, DeepSeek, Qwen, OpenRouter, and more) through a single, consistent interface.

**Perfect for**: Research workflows, benchmarking studies, automated testing, and applications that need to work with multiple LLM providers without dealing with their individual APIs.

## Important Note

This package is a **convenience wrapper** for working with multiple LLM providers through a unified interface. It is **not intended as a replacement** for the official provider libraries (openai, anthropic, google-genai, etc.).

### Use this package when:
- You need to query multiple LLM providers in the same project
- You're building benchmarking or comparison tools
- You want a consistent interface across providers
- You need provider-agnostic code for research workflows

### Use the official libraries when:
- You need cutting-edge features on day one of release
- You require provider-specific advanced features
- You only work with a single provider

**Update pace:** This package is maintained by a small team and may not immediately support every new feature from upstream providers. We prioritize stability and cross-provider compatibility over bleeding-edge feature coverage.

## Features

- **Provider-Agnostic**: Single interface for OpenAI, Anthropic, Google, Mistral, DeepSeek, Qwen, and OpenRouter
- **Multimodal Support**: Text + images across all supporting providers
- **Text File Support**: Automatically include text files in prompts for document analysis
- **Automatic Image Resizing**: Reduce API costs by auto-resizing large images
- **Structured Output**: Unified Pydantic model support across providers
- **Rich Response Objects**: Detailed token usage, costs, timing, and metadata
- **Async Support**: Parallel processing for faster benchmarks
- **Built-in Retry Logic**: Automatic exponential backoff for rate limits
- **Custom Base URLs**: Easy integration with OpenRouter, sciCORE, and other OpenAI-compatible APIs

## Installation

```bash
pip install generic-llm-api-client
```

## Quick Start

```python
from ai_client import create_ai_client

# Create a client for any provider
client = create_ai_client('openai', api_key='sk-...')

# Send a prompt
response, duration = client.prompt('gpt-4', 'What is 2+2?')

print(f"Response: {response.text}")
print(f"Tokens used: {response.usage.total_tokens}")
print(f"Time: {duration:.2f}s")
```

## Supported Providers

| Provider | ID | Multimodal | Structured Output |
|----------|-----|-----------|-------------------|
| OpenAI | `openai` | Yes | Yes |
| Anthropic Claude | `anthropic` | Yes | Yes (via tools) |
| Google Gemini | `genai` | Yes | Yes |
| Mistral | `mistral` | Yes | Yes |
| DeepSeek | `deepseek` | Yes | Yes |
| Qwen | `qwen` | Yes | Yes |
| OpenRouter | `openrouter` | Yes | Yes |
| sciCORE | `scicore` | Yes | Yes |

## Usage Examples

### Basic Text Prompt

```python
from ai_client import create_ai_client

client = create_ai_client('anthropic', api_key='sk-ant-...')
response, duration = client.prompt(
    'claude-3-5-sonnet-20241022',
    'Explain quantum computing in simple terms'
)

print(response.text)
```

### Multimodal (Text + Images)

```python
from ai_client import create_ai_client

client = create_ai_client('openai', api_key='sk-...')

response, duration = client.prompt(
    'gpt-4o',
    'Describe this image in detail',
    images=['path/to/image.jpg']
)

print(response.text)
```

### Multiple Images

```python
response, duration = client.prompt(
    'gpt-4o',
    'Compare these two images',
    images=['image1.jpg', 'image2.jpg']
)
```

### Text Files (NEW in v0.2.0)

Include text files in your prompts for document analysis:

```python
from ai_client import create_ai_client

client = create_ai_client('openai', api_key='sk-...')

# Analyze a single text file
response, duration = client.prompt(
    'gpt-4o',
    'Summarize this historical document',
    files=['manuscript_transcription.txt']
)

# Analyze multiple documents
response, duration = client.prompt(
    'gpt-4o',
    'Compare these two texts and identify common themes',
    files=['document1.txt', 'document2.txt']
)

print(response.text)
```

### Automatic Image Resizing (NEW in v0.2.0)

Reduce API costs by automatically resizing large images:

```python
from ai_client import create_ai_client

# Enable auto-resize (default: 2048px max dimension)
client = create_ai_client(
    'openai',
    api_key='sk-...',
    max_image_size=2048,  # Images larger than this will be resized
    image_quality=85       # JPEG quality for resized images
)

# This 4000x3000 image will be automatically resized to 2048x1536
response, duration = client.prompt(
    'gpt-4o',
    'Analyze this high-resolution historical manuscript',
    images=['huge_manuscript_scan.jpg']  # Original file is never modified
)

# Disable resizing if needed
client = create_ai_client('openai', api_key='sk-...', max_image_size=None)
```

### Combining Files and Images (NEW in v0.2.0)

Perfect for humanities research - compare visual and textual sources:

```python
from ai_client import create_ai_client

client = create_ai_client('openai', api_key='sk-...')

# Compare image to text description
response, duration = client.prompt(
    'gpt-4o',
    'Does this manuscript image match the catalog description?',
    images=['manuscript_photo.jpg'],
    files=['catalog_entry.txt']
)

# Analyze multiple sources together
response, duration = client.prompt(
    'gpt-4o',
    'Which of these paintings best matches the art historical description?',
    images=['painting_a.jpg', 'painting_b.jpg', 'painting_c.jpg'],
    files=['art_history_text.txt']
)

print(response.text)
```

### Structured Output with Pydantic

```python
from pydantic import BaseModel
from ai_client import create_ai_client

class Person(BaseModel):
    name: str
    age: int
    occupation: str

client = create_ai_client('openai', api_key='sk-...')

response, duration = client.prompt(
    'gpt-4',
    'Extract: John Smith is a 35-year-old software engineer',
    response_format=Person
)

# Parse the response
import json
person_data = json.loads(response.text)
person = Person(**person_data)

print(f"{person.name}, {person.age}, {person.occupation}")
```

### Async for Parallel Processing

```python
import asyncio
from ai_client import create_ai_client

async def process_batch():
    client = create_ai_client('openai', api_key='sk-...')

    # Process multiple prompts in parallel
    tasks = [
        client.prompt_async('gpt-4', f'Tell me about {topic}')
        for topic in ['Python', 'JavaScript', 'Rust']
    ]

    results = await asyncio.gather(*tasks)

    for response, duration in results:
        print(f"({duration:.2f}s) {response.text[:100]}...")

asyncio.run(process_batch())
```

### Custom Base URLs (OpenRouter, sciCORE)

```python
from ai_client import create_ai_client

# OpenRouter - access to 100+ models
client = create_ai_client(
    'openrouter',
    api_key='sk-or-...',
    base_url='https://openrouter.ai/api/v1',
    default_headers={
        "HTTP-Referer": "https://your-site.com",
        "X-Title": "Your App"
    }
)

response, _ = client.prompt('anthropic/claude-3-opus', 'Hello!')

# sciCORE (University HPC)
client = create_ai_client(
    'scicore',
    api_key='your-key',
    base_url='https://llm-api-h200.ceda.unibas.ch/litellm/v1'
)

response, _ = client.prompt('deepseek/deepseek-chat', 'Hello!')
```

### Accessing Response Metadata

```python
response, duration = client.prompt('gpt-4', 'Hello')

# Response text
print(response.text)

# Token usage
print(f"Input tokens: {response.usage.input_tokens}")
print(f"Output tokens: {response.usage.output_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")

# Metadata
print(f"Model: {response.model}")
print(f"Provider: {response.provider}")
print(f"Finish reason: {response.finish_reason}")
print(f"Duration: {response.duration}s")

# Raw provider response (for detailed analysis)
raw = response.raw_response

# Convert to dict (for JSON serialization)
response_dict = response.to_dict()
```

## Configuration

### Provider-Specific Settings

```python
from ai_client import create_ai_client

# OpenAI
client = create_ai_client(
    'openai',
    api_key='sk-...',
    temperature=0.7,
    max_tokens=500,
    frequency_penalty=0.5
)

# Claude
client = create_ai_client(
    'anthropic',
    api_key='sk-ant-...',
    temperature=1.0,
    max_tokens=4096,
    top_k=40
)

# Settings can also be passed per-request
response, _ = client.prompt(
    'gpt-4',
    'Hello',
    temperature=0.9,
    max_tokens=100
)
```

### Custom System Prompts

```python
from ai_client import create_ai_client

client = create_ai_client(
    'openai',
    api_key='sk-...',
    system_prompt="You are a helpful coding assistant specialized in Python."
)

# Override for specific request
response, _ = client.prompt(
    'gpt-4',
    'Write a haiku',
    system_prompt="You are a poetic assistant."
)
```

## Use Case: Benchmarking

Perfect for research workflows that need to evaluate multiple models:

```python
from ai_client import create_ai_client
import asyncio

async def benchmark_models():
    providers = [
        ('openai', 'gpt-4'),
        ('anthropic', 'claude-3-5-sonnet-20241022'),
        ('genai', 'gemini-2.0-flash-exp'),
    ]

    prompt = 'Explain quantum entanglement'

    for provider_id, model in providers:
        client = create_ai_client(provider_id, api_key=f'{provider_id}_key')

        response = await client.prompt_async(model, prompt)

        print(f"\n=== {provider_id}/{model} ===")
        print(f"Duration: {response.duration:.2f}s")
        print(f"Tokens: {response.usage.total_tokens}")
        print(f"Response: {response.text[:200]}...")

asyncio.run(benchmark_models())
```

## Error Handling

The package includes built-in retry logic with exponential backoff:

```python
from ai_client import create_ai_client, RateLimitError, APIError

client = create_ai_client('openai', api_key='sk-...')

try:
    response, duration = client.prompt('gpt-4', 'Hello')
    # Automatically retries up to 3 times on rate limit errors
except RateLimitError as e:
    print(f"Rate limited after retries: {e}")
except APIError as e:
    print(f"API error: {e}")
except Exception as e:
    print(f"Unknown error: {e}")
```

## Advanced Features

### Get Available Models

```python
from ai_client import create_ai_client

client = create_ai_client('openai', api_key='sk-...')
models = client.get_model_list()

for model_id, created_date in models:
    print(f"{model_id} (created: {created_date})")
```

### Check Multimodal Support

```python
client = create_ai_client('openai', api_key='sk-...')

if client.has_multimodal_support():
    print("This provider supports images!")
```

## Package Structure

```
ai_client/
  __init__.py           # Package exports
  base_client.py        # BaseAIClient + factory
  response.py           # LLMResponse, Usage dataclasses
  utils.py              # Retry logic, exceptions, utilities
  openai_client.py      # OpenAI implementation
  claude_client.py      # Anthropic Claude
  gemini_client.py      # Google Gemini
  mistral_client.py     # Mistral AI
  deepseek_client.py    # DeepSeek
  qwen_client.py        # Qwen
```

## Requirements

- Python >=3.9
- anthropic ~=0.71.0
- openai ~=2.6.1
- mistralai ~=1.9.11
- google-genai ~=1.46.0
- requests ~=2.32.5

## Development

```bash
# Clone the repository
git clone https://github.com/RISE-UNIBAS/generic-llm-api-client.git
cd generic-llm-api-client

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run integration tests (requires API keys)
pytest -m integration

# Format code
black ai_client tests

# Type checking
mypy ai_client/
```

## Documentation

- **[EXAMPLES.md](EXAMPLES.md)** - Comprehensive usage examples
- **[PUBLISHING.md](PUBLISHING.md)** - Guide for maintainers on publishing releases

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{generic_llm_api_client,
  author = {Sorin Marti},
  title = {Generic LLM API Client: A Unified Interface for Multiple LLM Providers},
  year = {2025},
  url = {https://github.com/RISE-UNIBAS/generic-llm-api-client}
}
```

## Support

- GitHub Issues: [Report bugs or request features](https://github.com/RISE-UNIBAS/generic-llm-api-client/issues)
- Documentation: [Full documentation](https://github.com/RISE-UNIBAS/generic-llm-api-client#readme)

## Roadmap

- [ ] Tool use / function calling support
- [ ] Streaming support
- [ ] Conversation history management
- [ ] More providers (Cohere, AI21, etc.)
- [ ] Cost estimation utilities
- [ ] Prompt caching support
