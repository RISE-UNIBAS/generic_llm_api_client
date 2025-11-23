# Usage Examples

## Basic Usage

### Simple Text Prompt

```python
from ai_client import create_ai_client

# Create a client
client = create_ai_client('openai', api_key='sk-...')

# Send a prompt
response = client.prompt('gpt-4', 'What is 2+2?')

print(f"Response: {response.text}")
print(f"Took: {response.duration:.2f}s")
print(f"Tokens used: {response.usage.total_tokens}")
```

### Multimodal (Text + Images)

```python
client = create_ai_client('openai', api_key='sk-...')

response = client.prompt(
    'gpt-4o',
    'Describe this image in detail',
    images=['path/to/image.jpg']
)

print(response.text)
```

### Multiple Images

```python
response = client.prompt(
    'gpt-4o',
    'Compare these two images',
    images=['image1.jpg', 'image2.jpg']
)
```

## Provider-Specific Examples

### OpenAI

```python
client = create_ai_client('openai', api_key='sk-...')
response = client.prompt(
    'gpt-4-turbo',
    'Explain quantum computing',
    temperature=0.7,
    max_tokens=500
)
```

### Anthropic Claude

```python
client = create_ai_client('anthropic', api_key='sk-ant-...')
response = client.prompt(
    'claude-3-5-sonnet-20241022',
    'Write a poem about code',
    temperature=1.0,
    max_tokens=1024
)
```

### Google Gemini

```python
client = create_ai_client('genai', api_key='...')
response = client.prompt(
    'gemini-2.0-flash-exp',
    'Explain machine learning',
    temperature=0.5
)
```

### Mistral

```python
client = create_ai_client('mistral', api_key='...')
response = client.prompt(
    'mistral-large-latest',
    'Summarize this text...'
)
```

### OpenRouter (Multi-Model Access)

```python
client = create_ai_client(
    'openrouter',
    api_key='sk-or-...',
    base_url='https://openrouter.ai/api/v1',
    default_headers={
        "HTTP-Referer": "https://your-site.com",
        "X-Title": "Your App Name"
    }
)

response = client.prompt(
    'anthropic/claude-3-opus',  # Use any model via OpenRouter
    'Hello!'
)
```

### sciCORE (University HPC)

```python
client = create_ai_client(
    'scicore',
    api_key='your-scicore-key',
    base_url='https://llm-api-h200.ceda.unibas.ch/litellm/v1'
)

response = client.prompt('deepseek/deepseek-chat', 'Hello!')
```

## Structured Output (Pydantic)

### Simple Schema

```python
from pydantic import BaseModel
from ai_client import create_ai_client

class Person(BaseModel):
    name: str
    age: int
    occupation: str

client = create_ai_client('openai', api_key='sk-...')

response = client.prompt(
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

### Complex Schema

```python
from pydantic import BaseModel, Field
from typing import List

class Reference(BaseModel):
    title: str
    year: int
    authors: List[str]

class ResearchPaper(BaseModel):
    title: str
    abstract: str
    keywords: List[str]
    references: List[Reference] = Field(default_factory=list)

client = create_ai_client('anthropic', api_key='sk-ant-...')

response = client.prompt(
    'claude-3-5-sonnet-20241022',
    'Extract structured data from this paper abstract: ...',
    response_format=ResearchPaper
)

# Claude uses tools API for structured output
paper = ResearchPaper(**json.loads(response.text))
```

## Async/Parallel Processing

### Process Multiple Prompts in Parallel

```python
import asyncio
from ai_client import create_ai_client

async def main():
    client = create_ai_client('openai', api_key='sk-...')

    # Create multiple tasks
    tasks = [
        client.prompt_async('gpt-4', f'Tell me about {topic}')
        for topic in ['Python', 'JavaScript', 'Rust', 'Go', 'TypeScript']
    ]

    # Run them all concurrently
    results = await asyncio.gather(*tasks)

    for i, response in enumerate(results):
        print(f"\n=== Result {i+1} ({response.duration:.2f}s) ===")
        print(response.text[:200])

asyncio.run(main())
```

### Benchmark Multiple Images

```python
import asyncio
import os
from ai_client import create_ai_client

async def process_image(client, image_path, prompt):
    """Process a single image."""
    response = await client.prompt_async(
        'gpt-4o',
        prompt,
        images=[image_path]
    )
    return {
        'image': os.path.basename(image_path),
        'response': response.text,
        'tokens': response.usage.total_tokens
    }

async def benchmark_images():
    client = create_ai_client('openai', api_key='sk-...')
    image_dir = 'path/to/images'
    prompt = 'Transcribe all text visible in this image'

    # Get all images
    images = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith(('.jpg', '.png'))
    ]

    # Process in batches of 5 to avoid rate limits
    results = []
    for i in range(0, len(images), 5):
        batch = images[i:i+5]
        batch_tasks = [process_image(client, img, prompt) for img in batch]
        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)

        print(f"Processed batch {i//5 + 1}/{(len(images)-1)//5 + 1}")

    # Summary
    total_tokens = sum(r['tokens'] for r in results)
    total_duration = sum(r['duration'] for r in results)

    print(f"\n=== Summary ===")
    print(f"Total images: {len(results)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Total duration: {total_duration:.2f}s")
    print(f"Avg per image: {total_duration/len(results):.2f}s")

    return results

# Run
results = asyncio.run(benchmark_images())
```

## Working with Response Data

### Extract Detailed Usage Information

```python
response = client.prompt('gpt-4', 'Hello')

print("=== Response Metadata ===")
print(f"Model: {response.model}")
print(f"Provider: {response.provider}")
print(f"Finish reason: {response.finish_reason}")
print(f"Timestamp: {response.timestamp}")
print(f"Duration: {response.duration}s")

print("\n=== Token Usage ===")
print(f"Input tokens: {response.usage.input_tokens}")
print(f"Output tokens: {response.usage.output_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")

if response.usage.cached_tokens:
    print(f"Cached tokens: {response.usage.cached_tokens}")

if response.usage.estimated_cost_usd:
    print(f"Estimated cost: ${response.usage.estimated_cost_usd:.6f}")
```

### Save to JSON

```python
import json

response = client.prompt('gpt-4', 'Hello')

# Convert to dict (excludes non-JSON-serializable raw_response)
response_dict = response.to_dict()

# Save
with open('response.json', 'w') as f:
    json.dump(response_dict, f, indent=2)

# Raw response is still available if needed
raw = response.raw_response  # Original provider-specific object
```

## Error Handling

### With Built-in Retry

```python
from ai_client import create_ai_client

# Retry is built-in with exponential backoff
client = create_ai_client('openai', api_key='sk-...')

try:
    response = client.prompt('gpt-4', 'Hello')
    # Will automatically retry up to 3 times on rate limit errors
except Exception as e:
    print(f"Failed after retries: {e}")
```

### Custom Error Handling

```python
from ai_client import create_ai_client, RateLimitError, APIError

client = create_ai_client('openai', api_key='sk-...')

try:
    response = client.prompt('gpt-4', 'Hello')
except RateLimitError as e:
    print(f"Rate limited: {e}")
except APIError as e:
    print(f"API error: {e}")
except Exception as e:
    print(f"Unknown error: {e}")
```

## Cost Tracking

```python
from ai_client import create_ai_client

client = create_ai_client('openai', api_key='sk-...')

total_input_tokens = 0
total_output_tokens = 0
total_cost = 0.0

prompts = [
    'What is Python?',
    'Explain async/await',
    'What are decorators?'
]

for prompt_text in prompts:
    response = client.prompt('gpt-4', prompt_text)

    total_input_tokens += response.usage.input_tokens
    total_output_tokens += response.usage.output_tokens

    if response.usage.estimated_cost_usd:
        total_cost += response.usage.estimated_cost_usd

print(f"\n=== Cost Summary ===")
print(f"Total input tokens: {total_input_tokens}")
print(f"Total output tokens: {total_output_tokens}")
print(f"Total cost: ${total_cost:.4f}")
```

## Custom System Prompts

```python
client = create_ai_client(
    'openai',
    api_key='sk-...',
    system_prompt="You are a helpful coding assistant specialized in Python."
)

# Uses the default system prompt
response = client.prompt('gpt-4', 'How do I read a file?')

# Override for specific request
response = client.prompt(
    'gpt-4',
    'Write a haiku about code',
    system_prompt="You are a poetic assistant."
)
```

## Provider Comparison

```python
from ai_client import create_ai_client

providers = [
    ('openai', 'gpt-4'),
    ('anthropic', 'claude-3-5-sonnet-20241022'),
    ('genai', 'gemini-2.0-flash-exp'),
    ('mistral', 'mistral-large-latest')
]

prompt = 'Explain quantum entanglement in simple terms'

for provider, model in providers:
    client = create_ai_client(provider, api_key=f'{provider}_key')

    response = client.prompt(model, prompt)

    print(f"\n=== {provider}/{model} ===")
    print(f"Duration: {response.duration:.2f}s")
    print(f"Tokens: {response.usage.total_tokens}")
    print(f"Response: {response.text[:200]}...")
```
