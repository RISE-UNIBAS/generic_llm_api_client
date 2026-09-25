# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.5.0] - 2026-09-25

A cost accounting release. The library tracked tokens and cost, but discarded several things
providers had already billed for: reasoning tokens, attempts thrown away by a fallback, and
the usage of any request that failed. Reasoning was the largest gap — on a reasoning model it
can be the majority of a response, so cost was understated several-fold.

### Added

- `Usage.reasoning_tokens` and `Usage.reasoning_cost_usd`. `reasoning_tokens` counts only
  reasoning a provider billed *outside* `output_tokens`, so
  `input_tokens + output_tokens + reasoning_tokens == total_tokens` holds wherever a total
  was reported. `None` means no total was reported and the provider told us nothing, which
  is deliberately distinct from `0`.
- `Usage.attempts`, `Usage.discarded_input_tokens`, `Usage.discarded_output_tokens`,
  `Usage.discarded_reasoning_tokens` and `Usage.discarded_cost_usd`, recording attempts that
  were billed but whose responses were not returned. They are reported beside the successful
  response and never folded into its own counts, and the token components account for the
  cost exactly as they do for a successful call.
- `ai_client.reasoning`, holding the single definition of the reasoning derivation, so every
  provider is measured the same way. A provider's own reasoning figure disagreeing with the
  derived gap is logged as a warning rather than silently resolved.
- `pricing.calculate_cost_components` and `CostComponents`, pricing reasoning at the output
  rate, plus `pricing.apply_costs`, which fills a `Usage` in place.
- `utils.usage_counts` and `utils.usage_from_error`, recovering token counts from a provider
  usage object or from a failed request's exception.
- `max_completion_tokens` support. gpt-5 and newer reject `max_tokens` on the chat endpoint;
  known model families are renamed before the call, and anything else recovers by retrying
  once when the provider rejects the parameter.

### Changed

- **`estimated_cost_usd` now covers input, output and reasoning.** It previously covered
  input and output only. Input plus output remains available as
  `input_cost_usd + output_cost_usd`.
- Cost is priced on the **requested** model rather than the id the response echoed back.
  Routers normalise or rewrite model ids, and a rewritten id misses the pricing table; this
  zeroed 7,922 of 7,996 stored HuggingFace requests. The *recorded* `LLMResponse.model` is
  unchanged and still reports what actually answered.
- HuggingFace cost now resolves for a **provider-pinned** model id such as
  `swiss-ai/Apertus-v1.5-8B:publicai`. A bare router id still yields `None`, because it
  routes to whichever partner is fastest and has no single price.
- Failed requests report whatever usage the failure carried, instead of all zeros.
- A cost the provider billed directly is never recalculated. Such a total arrives with no
  component costs beside it, which is how it is recognised; OpenRouter routes are unpinned,
  so the provider's own figure is the only accurate one. This holds on failure paths too: a
  billed cost on a failed request, or on an attempt a fallback discarded, is preserved
  rather than replaced by a list-price estimate.

### Fixed

- Three crashes from provider fields typed as optional, each of which turned a successful,
  billed response into an empty error response: Gemini's `candidates_token_count` (commonly
  `None` on a thinking response that stopped early), Anthropic's two cache counters, and
  non-numeric `cached_tokens` in one of the four OpenAI response builders. Token counts are
  now coerced in `Usage.__post_init__`.
- Gemini responses missing `candidates_token_count` now recover the completion count from
  the total rather than charging the whole remainder as reasoning.
- The structured-output fallback lost the first attempt's tokens entirely. Both the OpenAI
  `.parse()` fallback and the Claude dropped-tool retry now report them.
- A tool-calling request whose second call failed discarded the first call's usage and cost
  along with it.
- The OpenRouter billed-cost branch was selected with `hasattr`, so a `cost` of `None`
  counted as a billed total and every mocked usage object took the branch.
- `pyproject.toml` and `ai_client/__init__.py` no longer disagree. 0.4.6 shipped reporting
  `0.4.5` from `ai_client.__version__`.

### Notes for consumers

- The new field names are fixed from this release on; consumers rebuilding `Usage` from an
  explicit field list need to learn them or the values will be dropped on reload.
- Anything that fills a missing cost component must leave a provider-billed total alone.
  Where components exist they sum to the total; where a billed total exists without
  components, it stands as it is.
- Discarded-attempt totals cover the final attempt of a request. The retry wrapper re-enters
  the provider call on failure, and the accumulator starts fresh each time.

[v0.5.0]: https://github.com/RISE-UNIBAS/generic_llm_api_client/releases/tag/v0.5.0