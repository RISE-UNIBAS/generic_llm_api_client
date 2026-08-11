# Pricing Feature

The package now includes automatic cost calculation for all API requests based on actual provider pricing.

## Features

- **Automatic Cost Calculation**: Costs are calculated automatically for every request
- **Separate Input/Output Costs**: Track input tokens, output tokens, and total costs separately
- **Up-to-date Pricing**: Bundled pricing snapshot, refreshed from the benchmark repo at each release (see below)
- **Injectable Pricing**: Update pricing data externally when needed

## Cost Fields in Usage

Every `LLMResponse` includes a `Usage` object with the following cost fields:

```python
response.usage.input_cost_usd     # Cost for input tokens (in USD)
response.usage.output_cost_usd    # Cost for output tokens (in USD)
response.usage.estimated_cost_usd # Total cost (in USD)
```

## Example Usage

### Basic Usage (Automatic)

```python
from ai_client import create_ai_client

client = create_ai_client('openai', api_key='your-key')
response = client.prompt('gpt-4o', 'Hello!')

# Access cost information
print(f"Input cost: ${response.usage.input_cost_usd:.6f}")
print(f"Output cost: ${response.usage.output_cost_usd:.6f}")
print(f"Total cost: ${response.usage.estimated_cost_usd:.6f}")
```

### Output:
```
Input cost: $0.000005
Output cost: $0.000010
Total cost: $0.000015
```

## Updating Pricing Data

When the package pricing data becomes outdated, you can inject your own pricing file:

```python
from ai_client import set_pricing_file, create_ai_client

# Set your custom pricing file
set_pricing_file('/path/to/your/pricing.json')

# Now all requests use your updated pricing
client = create_ai_client('openai', api_key='your-key')
response = client.prompt('gpt-4o', 'Hello!')
```

## Pricing JSON Format

The pricing data follows this format:

```json
{
  "metadata": {
    "created": "2025-09-29T14:20:54.795632",
    "version": "1.3",
    "last_updated": "2025-10-20T00:00:00.000000"
  },
  "pricing": {
    "2025-10-20": {
      "openai": {
        "gpt-4o": {
          "input_price": 2.5,
          "output_price": 10.0,
          "source_url": "...",
          "added": "2025-09-29T21:09:07.468231"
        }
      }
    }
  }
}
```

**Prices are per million tokens** (e.g., `input_price: 2.5` means $2.50 per 1M input tokens).

## Supported Providers

Pricing data is included for:
- **OpenAI**: gpt-4o, gpt-4o-mini, gpt-5, gpt-5-mini, etc.
- **Anthropic Claude**: claude-3-5-sonnet, claude-3-5-haiku, claude-opus-4, etc.
- **Google Gemini**: gemini-2.0-flash, gemini-2.5-pro, gemini-2.5-flash, etc.
- **Mistral**: mistral-large, mistral-medium, pixtral-large, etc.
- **OpenRouter**: Various models (qwen, llama, grok, etc.)
- **sciCORE**: Academic/research models
- **HuggingFace**: prices for selected Inference Providers models (see the note below)

### A note on HuggingFace pricing

HuggingFace cost is **not** resolved at runtime — `usage.estimated_cost_usd` is `None` for
HuggingFace responses; cost is injected downstream by the benchmark harness. The `huggingface`
entries here exist for that step, keyed to canonical model ids (matching the benchmark's
`model_aliases.json`).

It cannot be automatic for two reasons: the router echoes a *normalized* model id (lowercased,
date suffix dropped) that will not match the canonical key, and a model may be served by several
providers at different prices (a bare id routes to the fastest). Pin a provider with a
`:<provider>` suffix when an exact price matters.

## Cost Calculation Details

1. The pricing manager loads pricing data from `ai_client/pricing.json`
2. For each request, it looks up the model's pricing (most recent date first)
3. Calculates:
   - `input_cost = (input_tokens / 1,000,000) * input_price_per_million`
   - `output_cost = (output_tokens / 1,000,000) * output_price_per_million`
   - `total_cost = input_cost + output_cost`

## When Pricing is Not Available

If pricing data is not available for a model:
- All cost fields will be `None`
- Token counts are still tracked
- No errors are raised

You can check availability:

```python
if response.usage.estimated_cost_usd is not None:
    print(f"Cost: ${response.usage.estimated_cost_usd:.6f}")
else:
    print("Pricing not available for this model")
```

## Updating the Package Pricing

`ai_client/pricing.json` is a **generated snapshot**, not a hand-edited file. Pricing is
maintained in the public benchmark repo (`RISE-UNIBAS/humanities_data_benchmark`,
`scripts/data/pricing.json`), the single source of truth. To refresh the bundled snapshot
before a release:

```bash
python scripts/update_pricing.py
```

Do not hand-edit `ai_client/pricing.json`; make pricing changes in the benchmark repo. See
PUBLISHING.md for the full release checklist.

## Notes

- Prices are sourced from official provider documentation
- Archive.org URLs are included for verification
- Costs are estimates and may vary slightly from actual billing
- Some providers (like OpenRouter) include actual costs in their API response, which are used directly
