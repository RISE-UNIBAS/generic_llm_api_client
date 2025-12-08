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
## Prompt Caching

Prompt caching reduces costs and latency by reusing previously processed content. The `cache=True` parameter works across all providers, with each handling it appropriately.

### Generic Caching API

**The beauty of this package**: Use the same API regardless of provider!

```python
from ai_client import create_ai_client

# Works with ANY provider - swap 'openai' for 'anthropic', 'genai', etc.
client = create_ai_client('openai', api_key='sk-...')

# Enable caching with a simple flag
response = client.prompt(
    model='gpt-4o',
    prompt='Analyze this document and list key points',
    files=['research_paper.txt'],  # Large file
    cache=True,  # ✅ Generic - each provider interprets appropriately
)

# Check caching results (works for all providers)
if response.usage.cached_tokens:
    savings = response.usage.get_cache_savings()
    print(f"Cache hit! Saved {savings:.1%} of tokens")
```

### How Each Provider Handles `cache=True`

| Provider | Behavior | Min Tokens | Retention |
|----------|----------|------------|-----------|
| **OpenAI** | Always automatic (1024+ tokens) | 1,024 | 5-10 min or 24h |
| **Claude** | Adds `cache_control` blocks to files | 1,024-4,096 | 5 min (auto-refresh) |
| **Gemini** | Uses `cache_id` if provided | 1,024-4,096 | User-defined |
| **Mistral** | Not supported (ignored) | N/A | N/A |

### Provider-Specific Examples

#### OpenAI: Automatic Caching

OpenAI automatically caches prompts with 1024+ tokens. No special code needed!

```python
client = create_ai_client('openai', api_key='sk-...')

long_document = open('research_paper.txt').read()  # e.g., 10,000 tokens

# First request: No cache hit
response1 = client.prompt(
    'gpt-4o',
    f"Document:\n\n{long_document}\n\nWhat is the main argument?"
)
print(f"Cached: {response1.usage.cached_tokens or 0} tokens")  # 0

# Second request with same prefix: Cache hit!
response2 = client.prompt(
    'gpt-4o',
    f"Document:\n\n{long_document}\n\nList three supporting points."
)
print(f"Cached: {response2.usage.cached_tokens or 0} tokens")  # ~10000
print(f"Savings: {response2.usage.get_cache_savings():.1%}")  # ~95%
```

**Advanced OpenAI Options** (via kwargs):

```python
# Improve cache hit rates and extend retention
response = client.prompt(
    'gpt-4o',
    prompt=long_document + "\n\nYour question...",
    cache=True,  # Informational (always on anyway)
    prompt_cache_key="batch_001",  # Group related requests
    prompt_cache_retention="24h",  # Extended retention
)
```

#### Claude: Cache Control Blocks

Claude requires explicit marking via `cache=True` when using files:

```python
client = create_ai_client('anthropic', api_key='sk-...')

# First request: Create cache
response1 = client.prompt(
    'claude-3-5-sonnet-20241022',
    prompt="What is the main theme of this essay?",
    files=['long_essay.txt'],  # 5,000+ tokens
    cache=True,  # ✅ Adds cache_control blocks
)

conv_id = response1.conversation_id
print(f"Cache created: {response1.usage.cache_creation_tokens} tokens")
print(f"Cost: 125% of input rate")

# Subsequent requests (within 5 min): Cache hit!
response2 = client.prompt(
    'claude-3-5-sonnet-20241022',
    prompt="List three supporting arguments.",
    conversation_id=conv_id,  # Reuse conversation + cache
)

print(f"Cache read: {response2.usage.cache_read_tokens} tokens")
print(f"Cost: 10% of input rate (90% savings!)")
print(f"Savings: {response2.usage.get_cache_savings():.1%}")
```

**Claude Cache Pricing**:
- Write to cache: 125% of base input cost
- Read from cache: **10% of base input cost** (90% discount!)
- Auto-refreshes on access (5-min TTL)

#### Gemini: Explicit Cache (Advanced)

*Note: Gemini requires explicit cache creation. This is more advanced.*

```python
client = create_ai_client('genai', api_key='...')

# For Gemini, you'd need to add cache_id via kwargs after creating a cache
# See Gemini docs for cache creation API
response = client.prompt(
    'gemini-2.0-flash',
    prompt="Analyze this",
    cache=True,
    cache_id="cache_abc123",  # Reference to pre-created cache
)
```

### Conversation Tracking

Track multi-turn conversations automatically:

```python
client = create_ai_client('openai', api_key='sk-...')  # Any provider works!

# First message
response1 = client.prompt(
    'gpt-4o',
    "My name is Alice. What's a good programming language to learn?"
)

conv_id = response1.conversation_id
print(f"Started conversation: {conv_id}")

# Continue conversation
response2 = client.prompt(
    'gpt-4o',
    "What is my name?",  # Model remembers "Alice"
    conversation_id=conv_id,
)

print(response2.text)  # "Your name is Alice."

# View history
history = client.get_conversation_history(conv_id)
for msg in history:
    print(f"{msg['role']}: {msg['content'][:50]}...")

# Clear when done
client.clear_conversation(conv_id)
```

### Combined: Caching + Conversations

Maximum efficiency with both features:

```python
client = create_ai_client('anthropic', api_key='sk-...')

# Establish context with caching
response1 = client.prompt(
    'claude-3-5-sonnet-20241022',
    prompt="I've provided a research paper. Please read it.",
    files=['research_paper.txt'],  # 10,000 tokens
    cache=True,  # ✅ Cache the file content
)

conv_id = response1.conversation_id
print(f"Cache created: {response1.usage.cache_creation_tokens} tokens")

# Ask multiple questions (cache reused, conversation maintained)
questions = [
    "What is the main hypothesis?",
    "What methodology was used?",
    "What were the key findings?",
]

for question in questions:
    response = client.prompt(
        'claude-3-5-sonnet-20241022',
        prompt=question,
        conversation_id=conv_id,  # ✅ Reuses both cache AND history
    )
    
    print(f"\nQ: {question}")
    print(f"A: {response.text[:150]}...")
    print(f"Cache read: {response.usage.cache_read_tokens} tokens (90% savings!)")
```

### Best Practices

#### 1. Structure Prompts for Caching

**✅ Good**: Static content first
```python
# Put unchanging content at the beginning
prompt = f"Reference:\n\n{large_document}\n\nUser question: {user_question}"
response = client.prompt(model, prompt, cache=True)
```

**❌ Bad**: Dynamic content first
```python
# Cache misses because prefix changes each time
prompt = f"User {user_id}: {user_question}\n\nReference: {large_document}"
```

#### 2. Monitor Cache Performance

```python
response = client.prompt(model, prompt, files=['doc.txt'], cache=True)

# Generic - works for all providers
savings = response.usage.get_cache_savings()
if savings > 0:
    print(f"Cache hit! Saved {savings:.1%}")
else:
    print("Cache miss - first request or expired")

# Provider-specific details
if response.provider == 'anthropic':
    print(f"Cache created: {response.usage.cache_creation_tokens}")
    print(f"Cache read: {response.usage.cache_read_tokens}")
elif response.provider == 'openai':
    print(f"Cached tokens: {response.usage.cached_tokens}")
```

#### 3. Use Conversations for Context

```python
# Start with large context
response1 = client.prompt(
    model,
    "Here's the document...",
    files=['large_file.txt'],
    cache=True,
)

# Ask follow-ups (reuses cache via conversation)
for question in follow_up_questions:
    response = client.prompt(
        model,
        question,
        conversation_id=response1.conversation_id,  # ✅ Maintains context
    )
```

### Provider Comparison

| Feature | OpenAI | Claude | Gemini |
|---------|--------|---------|--------|
| **Setup** | None (automatic) | `cache=True` + `files` | Manual cache creation |
| **Cost** | Free | Write: 125%, Read: 10% | Varies |
| **Retention** | 5-10 min (or 24h) | 5 min (auto-refresh) | User-defined |
| **Best For** | High-volume apps | Multi-turn conversations | Research sessions |
| **API** | `cache=True` (optional) | `cache=True` (required for files) | `cache=True` + `cache_id` |

**Key Takeaway**: Use `cache=True` everywhere, and it just works! Each provider handles it optimally.
