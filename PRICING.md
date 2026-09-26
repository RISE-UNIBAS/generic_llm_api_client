# Pricing and Cost Accounting

The package calculates request costs using bundled or custom pricing data and preserves
provider-billed costs when available.

## Features

- **Automatic Cost Calculation**: Costs are calculated automatically when pricing data is available
- **Separate Cost Components**: Track input, output and reasoning costs, and the total, separately
- **Bundled Pricing**: Bundled pricing snapshot, refreshed from the benchmark repository at each release (see below)
- **Injectable Pricing**: Update pricing data externally when needed

## Cost Fields in Usage

Every `LLMResponse` includes a `Usage` object with the following cost fields:

```python
response.usage.input_cost_usd     # Cost for input tokens (in USD)
response.usage.output_cost_usd    # Cost for output tokens (in USD)
response.usage.reasoning_cost_usd # Cost for reasoning tokens (in USD)
response.usage.estimated_cost_usd # Total cost, covering all three (in USD)
```

### Reasoning tokens

`usage.reasoning_tokens` counts only reasoning a provider billed **outside**
`output_tokens`. Provider accounting conventions differ: Gemini and xAI report it
separately, while the OpenAI family already counts it inside the completion. Recording the
uncounted remainder maintains the following relationship when a total is reported:

```
input_tokens + output_tokens + reasoning_tokens == total_tokens
```

`reasoning_tokens` is `0` when a provider reported no reasoning, and `None` when it reported
no total, so the uncounted reasoning cannot be determined. These values remain distinct.
Reasoning is charged at the model's **output** rate.

### Provider-billed costs are never recalculated

Some providers, including OpenRouter, return the billed cost. Routed requests may use
backends with different prices, so the client preserves the reported amount. A billed total
is identified by `input_cost_usd` and `output_cost_usd` being `None` and is not replaced
with a list-price estimate.

When component costs are available, they sum to `estimated_cost_usd`. A provider-billed
total without component costs is preserved.

### Discarded Billable Attempts

A structured-output request that fails and falls back is charged twice. The discarded
attempt is reported separately rather than folded into the successful response:

```python
response.usage.attempts                 # billable attempts behind this response
response.usage.discarded_input_tokens   # tokens billed by attempts that were thrown away
response.usage.discarded_output_tokens
response.usage.discarded_reasoning_tokens
response.usage.discarded_cost_usd      # what the provider billed, where it reported one
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

### HuggingFace Pricing

HuggingFace cost lookup requires a model ID that **pins a provider**, because the
`huggingface` entries are keyed that way:

```python
client.prompt('swiss-ai/Apertus-v1.5-8B:publicai', '...')  # priced
client.prompt('swiss-ai/Apertus-v1.5-8B', '...')           # estimated_cost_usd is None
```

A router model ID without a provider suffix routes to whichever partner provider is fastest, and the same model is
served by several of them at different prices, so there is no single price to report. Pin a
provider with a `:<provider>` suffix to select the corresponding pricing entry; otherwise tokens are
tracked and cost can be supplied downstream by the benchmark harness.

## Cost Calculation Details

1. The pricing manager loads pricing data from `ai_client/pricing.json`
2. For each request, it looks up the model's pricing (most recent date first)
3. Calculates:
   - `input_cost = (input_tokens / 1,000,000) * input_price_per_million`
   - `output_cost = (output_tokens / 1,000,000) * output_price_per_million`
   - `reasoning_cost = (reasoning_tokens / 1,000,000) * output_price_per_million`
   - `total_cost = input_cost + output_cost + reasoning_cost`

Pricing lookup uses the **requested model ID** because routers may return a normalized
or rewritten ID that does not match the pricing table. `response.model` retains the model
ID returned by the provider.

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

`ai_client/pricing.json` is a **generated snapshot**, rather than a manually maintained file. Pricing is
maintained in the public benchmark repository (`RISE-UNIBAS/humanities_data_benchmark`,
`scripts/data/pricing.json`), the single source of truth. To refresh the bundled snapshot
before a release:

```bash
python scripts/update_pricing.py
```

Do not manually edit `ai_client/pricing.json`; make pricing changes in the benchmark repository. See
PUBLISHING.md for the full release checklist.

## Notes

- Prices are sourced from official provider documentation
- Archive.org URLs are included for verification
- Costs are estimates and may vary slightly from actual billing
- Some providers (like OpenRouter) include actual costs in their API response, which are used directly
