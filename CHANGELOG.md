# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.5.2] - 2026-09-29

### Fixed

- Claude models that reject forced `tool_choice` now receive the original schema through
  an optional tool. Fable 5.1, Mythos 5.1, Opus 5.5, and Sonnet 5.5 use this route directly.
  Failed optional-tool requests or detected wrapper keys trigger a text fallback, with
  reported usage retained. Schemas permitting extra keys or declaring property patterns
  skip key checks.

## [v0.5.1] - 2026-09-26

### Added

- `input_cost_usd` and `output_cost_usd` are now recorded for provider-billed costs where
  the provider itemises the bill. For OpenRouter these come from
  `cost_details.upstream_inference_prompt_cost` and `upstream_inference_completions_cost`.
  They are used only when both are numbers and together equal the billed `cost`. Otherwise
  both stay `None`: a per-request fee that belongs to neither part is not divided between
  them, and a bring-your-own-key route (`is_byok: true`), whose total is OpenRouter's fee
  alone, is not split. `estimated_cost_usd` remains the billed `cost`, unchanged. This
  applies to successful responses and to failed requests that reported usage.
- `utils.billed_components`, beside `utils.billed_cost`, and `pricing.record_costs`, which
  fills a `Usage` from a billed cost where one was reported and from list prices otherwise.

### Fixed

- The JSON-mode prompt no longer invites a model to answer with the schema itself. Asked
  for JSON "matching this exact schema", some models returned the schema, which parses and
  validates against a model with optional fields while carrying no data. The OpenAI
  fallback, Mistral and Cohere now ask for an instance of the schema filled with values.
- A JSON-mode response that repeats the schema (`$defs`, `properties` or `$schema` at the
  top level where the schema declares no such field) is now treated as a failed parse:
  `parsed` is `None`, the text is kept as returned, and a warning is logged.

### Notes for consumers

- A billed total can now arrive with components beside it. Code that took "components
  present" to mean "list-price estimate" must stop doing so. `reasoning_cost_usd` stays
  `None` for billed costs, and the input and output parts sum to the total.

## [v0.5.0] - 2026-09-25

This release extends cost accounting to include previously omitted reasoning tokens,
discarded fallback attempts, and usage from failed requests. Reasoning tokens accounted
for the largest omission and could cause costs to be understated by several times.

### Added

- `Usage.reasoning_tokens` and `Usage.reasoning_cost_usd`. `reasoning_tokens` counts only
  reasoning a provider billed *outside* `output_tokens`, so
  `input_tokens + output_tokens + reasoning_tokens == total_tokens` holds wherever a total
  was reported. `None` means no total was reported, so the uncounted reasoning is unknown;
  this remains distinct from `0`.
- `Usage.attempts`, `Usage.discarded_input_tokens`, `Usage.discarded_output_tokens`,
  `Usage.discarded_reasoning_tokens` and `Usage.discarded_cost_usd`, recording attempts that
  were billed but whose responses were not returned. They are reported beside the successful
  response and never folded into its own counts, and the token components account for the
  cost exactly as they do for a successful call.
- `ai_client.reasoning`, providing a shared reasoning-token calculation across
  providers. Discrepancies between reported reasoning counts and the derived gap
  are logged as warnings.
- `pricing.calculate_cost_components` and `CostComponents`, pricing reasoning at the output
  rate, plus `pricing.apply_costs`, which fills a `Usage` in place.
- `utils.usage_counts` and `utils.usage_from_error`, recovering token counts from a provider
  usage object or from a failed request's exception.
- Request parameters are adapted to what a model accepts, so a caller can pass the same
  settings to every model. The gpt-5 series requires `max_completion_tokens` in place
  of `max_tokens`; gpt-5, its mini and nano variants and the o-series accept only their
  default temperature, while gpt-5.1 and newer accept a temperature setting. Newer Claude
  models have removed `temperature` support and reject the parameter. The two rules are kept
  separate, because discarding a temperature a model would have honoured silently
  changes sampling. Known families are corrected before the call, and other parameter
  rejections trigger one retry with adjusted settings. This also supports model families
  not explicitly listed in the client.

### Changed

- **`estimated_cost_usd` now covers input, output and reasoning.** It previously covered
  input and output only. Input plus output remains available as
  `input_cost_usd + output_cost_usd`.
- Cost is calculated using the **requested** model ID rather than the ID returned by the provider.
  Routers normalise or rewrite model ids, and a rewritten ID may not match the pricing table; this
  resulted in missing costs for 7,922 of 7,996 stored HuggingFace requests. The *recorded* `LLMResponse.model` is
  unchanged and retains the model ID returned by the provider.
- HuggingFace cost now resolves for a **provider-pinned** model id such as
  `swiss-ai/Apertus-v1.5-8B:publicai`. A bare router id still yields `None`, because it
  routes to whichever partner is fastest and has no single price.
- Failed requests preserve available usage from the error payload.
- Provider-billed costs are preserved across successful responses, failed requests, and
  discarded fallback attempts. These totals are identified by the absence of component
  costs. Routed requests can use backends with different prices, so the provider's billed
  amount takes precedence over a list-price estimate.

### Fixed

- Three crashes from provider fields typed as optional, each of which turned a successful,
  billed response into an empty error response: Gemini's `candidates_token_count` (commonly
  `None` on a thinking response that stopped early), Anthropic's two cache counters, and
  non-numeric `cached_tokens` in one of the four OpenAI response builders. Token counts are
  now coerced in `Usage.__post_init__`.
- A response reporting a total but no completion count now has that count recovered from
  the total, rather than charging the whole remainder as reasoning. genai omits it on a
  thinking response that stopped early. This applies on failure paths as well, and usage
  recovery understands each provider's shape, including counts nested below the usage
  block as Cohere reports them.
- The structured-output fallback lost the first attempt's tokens entirely. Both the OpenAI
  `.parse()` fallback and the Claude dropped-tool retry now report them. The response body
  is read before the SDK validates it, so the tokens survive a content filter or a schema
  mismatch as well -- these may raise exceptions without usage data even when the
  request was billed.
- A tool-calling request whose second call failed discarded the first call's usage and cost
  along with it.
- A response that arrived and was billed but then failed local conversion, such as one
  carrying malformed tool arguments, became an error response reporting zero tokens. The
  billed usage is now recorded together with usage from any earlier discarded attempts
  within the same request. This applies to every
  provider, not only the OpenAI-compatible ones.
- The OpenRouter billed-cost branch was selected with `hasattr`, so a `cost` of `None`
  counted as a billed total and every mocked usage object took the branch.
- DeepSeek images are no longer dropped for `deepseek-flash`, which accepts them although
  its name does not match the existing vision-model keywords. The fragment list can be extended per
  client with the `vision_model_keywords` setting, allowing support for additional vision models
  without a library release.
- Listing models no longer fails for providers that report no creation timestamp.
  DeepSeek returns `None`, which previously raised `TypeError` and prevented model listing.
- Token counts are stored as integers regardless of the numeric type returned by the provider. Cohere returns whole
  numbers as floats, which reached stored records as `187.0` and prevented the usage-recovery code
  from reading those counts.
- `pyproject.toml` and `ai_client/__init__.py` now report the same version. Version 0.4.6 reported
  `0.4.5` from `ai_client.__version__`.

### Notes for consumers

- The new field names are fixed from this release on; consumers rebuilding `Usage` from an
  explicit field list must include them to preserve the values on reload.
- Code that fills missing cost components must preserve provider-billed totals.
  Where components exist they sum to the total; where a billed total exists without
  components, it remains unchanged.
- Discarded-attempt totals cover the final attempt of a request. The retry wrapper re-enters
  the provider call on failure, and the accumulator starts fresh each time.

[v0.5.2]: https://github.com/RISE-UNIBAS/generic_llm_api_client/releases/tag/v0.5.2
[v0.5.1]: https://github.com/RISE-UNIBAS/generic_llm_api_client/releases/tag/v0.5.1
[v0.5.0]: https://github.com/RISE-UNIBAS/generic_llm_api_client/releases/tag/v0.5.0
