"""
Tests for cost accounting across providers.

Covers the contract the benchmark corpus depends on: reasoning tokens a provider billed
outside output_tokens are recorded, costs are priced on the model that was requested rather
than the one the response echoed back, and a cost the provider billed is never replaced by
a list-price estimate.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import pytest

from ai_client import create_ai_client


def _failed_attempt(
    input_tokens, output_tokens, total_tokens=None, cost=None, anthropic_naming=False
):
    """Build an exception carrying the usage a failed attempt was still billed for."""
    error = Exception("attempt failed")
    error.completion = Mock()
    error.completion.usage = Mock()
    if anthropic_naming:
        error.completion.usage.input_tokens = input_tokens
        error.completion.usage.output_tokens = output_tokens
    else:
        error.completion.usage.prompt_tokens = input_tokens
        error.completion.usage.completion_tokens = output_tokens
    error.completion.usage.total_tokens = (
        input_tokens + output_tokens if total_tokens is None else total_tokens
    )
    if cost is None:
        del error.completion.usage.cost
    else:
        error.completion.usage.cost = cost
    return error


# 1 USD per million in, 10 per million out, for ids that differ from what a router echoes.
TABLE = {
    "2026-01-01": {
        "openai": {"requested-model": {"input_price": 1.0, "output_price": 10.0}},
        "x-ai": {"grok-4.6": {"input_price": 1.0, "output_price": 10.0}},
        "genai": {"gemini-thinking": {"input_price": 1.0, "output_price": 10.0}},
        "huggingface": {
            "swiss-ai/Apertus-v1.5-8B:publicai": {"input_price": 1.0, "output_price": 10.0}
        },
    }
}


@pytest.fixture
def openai_client():
    """Yield an OpenAI client whose SDK object is a mock."""
    with patch("ai_client.openai_client.OpenAI") as openai_class:
        api = Mock()
        openai_class.return_value = api
        yield create_ai_client("openai", api_key="test-key"), api


@pytest.fixture
def xai_client():
    """Yield an xAI client, which inherits every OpenAI response builder unchanged."""
    with patch("ai_client.openai_client.OpenAI") as openai_class:
        api = Mock()
        openai_class.return_value = api
        yield create_ai_client("x-ai", api_key="test-key"), api


class TestReasoningTokensRecorded:
    """Tests that reasoning billed outside output_tokens reaches Usage."""

    def test_openai_shaped_reasoning_is_recorded(self, openai_client, mock_reasoning_response):
        """Test an x-ai style response reports its reasoning tokens."""
        client, api = openai_client
        api.chat.completions.create.return_value = mock_reasoning_response

        response = client.prompt("grok-4.6", "Hello")

        assert response.usage.reasoning_tokens == 200

    def test_openai_shaped_invariant_holds(self, openai_client, mock_reasoning_response):
        """Test input + output + reasoning equals the reported total."""
        client, api = openai_client
        api.chat.completions.create.return_value = mock_reasoning_response
        usage = client.prompt("grok-4.6", "Hello").usage

        total = usage.input_tokens + usage.output_tokens + usage.reasoning_tokens

        assert total == usage.total_tokens == 500

    def test_reasoning_is_billed_at_the_output_rate(
        self, xai_client, mock_reasoning_response, stub_pricing
    ):
        """Test the 200 reasoning tokens cost the same as 200 output tokens."""
        stub_pricing(TABLE)
        client, api = xai_client
        api.chat.completions.create.return_value = mock_reasoning_response

        usage = client.prompt("grok-4.6", "Hello").usage

        assert usage.reasoning_cost_usd == pytest.approx(200 / 1_000_000 * 10.0)
        assert usage.estimated_cost_usd == pytest.approx(
            usage.input_cost_usd + usage.output_cost_usd + usage.reasoning_cost_usd
        )

    def test_gemini_thinking_tokens_are_recorded(self, mock_gemini_thinking_response):
        """Test Gemini's thoughts_token_count reaches Usage."""
        with patch("ai_client.gemini_client.genai") as genai:
            api = Mock()
            genai.Client.return_value = api
            api.models.generate_content.return_value = mock_gemini_thinking_response

            client = create_ai_client("genai", api_key="test-key")
            usage = client.prompt("gemini-thinking", "Hello").usage

        assert usage.reasoning_tokens == 200
        assert usage.input_tokens + usage.output_tokens + usage.reasoning_tokens == 500

    def test_gemini_truncated_thinking_recovers_the_output_count(
        self, mock_gemini_truncated_thinking_response
    ):
        """Test a missing candidates count is derived instead of charged as reasoning."""
        with patch("ai_client.gemini_client.genai") as genai:
            api = Mock()
            genai.Client.return_value = api
            api.models.generate_content.return_value = mock_gemini_truncated_thinking_response

            client = create_ai_client("genai", api_key="test-key")
            response = client.prompt("gemini-thinking", "Hello")

        assert response.finish_reason != "error"
        assert response.usage.output_tokens == 200
        assert response.usage.reasoning_tokens == 200
        assert response.usage.input_tokens + 200 + 200 == 500


class TestProvidersWithoutSeparateReasoning:
    """Tests that providers billing reasoning inside output_tokens report a known zero."""

    def test_claude_reports_zero_not_unknown(self, mock_claude_response):
        """Test Claude, which counts thinking inside output_tokens, reports 0."""
        with patch("ai_client.claude_client.Anthropic") as anthropic_class:
            api = Mock()
            anthropic_class.return_value = api
            api.messages.create.return_value = mock_claude_response

            client = create_ai_client("anthropic", api_key="test-key")
            usage = client.prompt("claude-3-5-sonnet-20241022", "Hi").usage

        assert usage.reasoning_tokens == 0
        assert usage.input_tokens + usage.output_tokens == usage.total_tokens

    def test_mistral_reports_zero_not_unknown(self, mock_mistral_response):
        """Test a provider whose total already matches input plus output reports 0."""
        with patch("ai_client.mistral_client.Mistral") as mistral_class:
            api = Mock()
            mistral_class.return_value = api
            api.chat.complete.return_value = mock_mistral_response

            client = create_ai_client("mistral", api_key="test-key")
            usage = client.prompt("mistral-large-latest", "Hi").usage

        assert usage.reasoning_tokens == 0

    def test_cohere_reports_zero_not_unknown(self, mock_cohere_response):
        """Test Cohere, whose total is computed from the components, reports 0."""
        with patch("ai_client.cohere_client.cohere") as cohere_module:
            api = Mock()
            cohere_module.ClientV2.return_value = api
            api.chat.return_value = mock_cohere_response

            client = create_ai_client("cohere", api_key="test-key")
            usage = client.prompt("command-r", "Hi").usage

        assert usage.reasoning_tokens == 0


class TestPricedOnRequestedModel:
    """Tests that the pricing lookup uses the requested model, not the echoed one."""

    def test_cost_uses_requested_model(self, openai_client, mock_openai_response, stub_pricing):
        """Test a response echoing a different id is still priced from the request."""
        stub_pricing(TABLE)
        client, api = openai_client
        mock_openai_response.model = "echoed-by-the-router"
        api.chat.completions.create.return_value = mock_openai_response

        response = client.prompt("requested-model", "Hello")

        assert response.usage.input_cost_usd == pytest.approx(10 / 1_000_000 * 1.0)
        assert response.usage.output_cost_usd == pytest.approx(20 / 1_000_000 * 10.0)

    def test_recorded_model_remains_the_echoed_one(
        self, openai_client, mock_openai_response, stub_pricing
    ):
        """Test only pricing changed; the recorded model still reflects what answered."""
        stub_pricing(TABLE)
        client, api = openai_client
        mock_openai_response.model = "echoed-by-the-router"
        api.chat.completions.create.return_value = mock_openai_response

        response = client.prompt("requested-model", "Hello")

        assert response.model == "echoed-by-the-router"

    def test_huggingface_prices_a_pinned_id(self, mock_huggingface_response, stub_pricing):
        """Test a provider-pinned HuggingFace id now resolves a cost."""
        stub_pricing(TABLE)
        with patch("ai_client.openai_client.OpenAI") as openai_class:
            api = Mock()
            openai_class.return_value = api
            api.chat.completions.create.return_value = mock_huggingface_response

            client = create_ai_client("huggingface", api_key="test-key")
            usage = client.prompt("swiss-ai/Apertus-v1.5-8B:publicai", "Hello").usage

        assert usage.input_cost_usd == pytest.approx(1.0)
        assert usage.output_cost_usd == pytest.approx(10.0)

    def test_huggingface_bare_router_id_stays_unpriced(
        self, mock_huggingface_response, stub_pricing
    ):
        """Test an unpinned id still has no single price, so cost stays None."""
        stub_pricing(TABLE)
        with patch("ai_client.openai_client.OpenAI") as openai_class:
            api = Mock()
            openai_class.return_value = api
            api.chat.completions.create.return_value = mock_huggingface_response

            client = create_ai_client("huggingface", api_key="test-key")
            usage = client.prompt("swiss-ai/Apertus-8B-Instruct-2509", "Hello").usage

        assert usage.estimated_cost_usd is None
        assert usage.total_tokens == 2_000_000


class TestBilledCostPreserved:
    """Tests that a provider-billed total is never recomputed."""

    def test_openrouter_billed_total_survives(
        self, openai_client, mock_openrouter_response, stub_pricing
    ):
        """Test the provider's own figure is kept even when the model is priceable."""
        stub_pricing(TABLE)
        client, api = openai_client
        api.chat.completions.create.return_value = mock_openrouter_response

        usage = client.prompt("requested-model", "Hello").usage

        assert usage.estimated_cost_usd == 0.0123

    def test_openrouter_billed_total_has_no_components(
        self, openai_client, mock_openrouter_response, stub_pricing
    ):
        """Test a billed total is recognisable by having no component costs beside it."""
        stub_pricing(TABLE)
        client, api = openai_client
        api.chat.completions.create.return_value = mock_openrouter_response

        usage = client.prompt("requested-model", "Hello").usage

        assert usage.input_cost_usd is None
        assert usage.output_cost_usd is None


class TestDiscardedAttempts:
    """Tests that tokens billed on a discarded attempt are salvaged, not lost."""

    def test_discarded_tokens_are_recorded(
        self, openai_client, mock_openai_response, mock_pydantic_model, wire_structured_output
    ):
        """Test a failed structured-output attempt reports what it was billed."""
        client, api = openai_client
        wire_structured_output(api, _failed_attempt(50, 60))
        api.chat.completions.create.return_value = mock_openai_response

        usage = client.prompt("gpt-4", "Hi", response_format=mock_pydantic_model).usage

        assert usage.discarded_input_tokens == 50
        assert usage.discarded_output_tokens == 60
        assert usage.attempts == 2

    def test_successful_counts_are_not_inflated(
        self, openai_client, mock_openai_response, mock_pydantic_model, wire_structured_output
    ):
        """Test the response still reports only what the successful call used."""
        client, api = openai_client
        wire_structured_output(api, _failed_attempt(50, 60))
        api.chat.completions.create.return_value = mock_openai_response

        usage = client.prompt("gpt-4", "Hi", response_format=mock_pydantic_model).usage

        assert usage.input_tokens == 10
        assert usage.output_tokens == 20
        assert usage.total_tokens == 30

    def test_discarded_cost_is_priced(
        self,
        openai_client,
        mock_openai_response,
        mock_pydantic_model,
        stub_pricing,
        wire_structured_output,
    ):
        """Test the wasted attempt is costed at the requested model's rates."""
        stub_pricing(TABLE)
        client, api = openai_client
        wire_structured_output(api, _failed_attempt(1_000_000, 1_000_000))
        api.chat.completions.create.return_value = mock_openai_response

        usage = client.prompt("requested-model", "Hi", response_format=mock_pydantic_model).usage

        assert usage.discarded_cost_usd == pytest.approx(11.0)

    def test_unbilled_failure_does_not_count_an_attempt(
        self, openai_client, mock_openai_response, mock_pydantic_model, wire_structured_output
    ):
        """Test a failure that generated nothing leaves the record untouched."""
        client, api = openai_client
        wire_structured_output(api, Exception("bad request"))
        api.chat.completions.create.return_value = mock_openai_response

        usage = client.prompt("gpt-4", "Hi", response_format=mock_pydantic_model).usage

        assert usage.attempts == 1
        assert usage.discarded_input_tokens == 0

    def test_claude_tool_retry_salvages_usage(self, mock_claude_response, mock_pydantic_model):
        """Test Claude's dropped-tool retry reports the first attempt's tokens."""
        with patch("ai_client.claude_client.Anthropic") as anthropic_class:
            api = Mock()
            anthropic_class.return_value = api
            api.messages.create.side_effect = [
                _failed_attempt(50, 60, anthropic_naming=True),
                mock_claude_response,
            ]

            client = create_ai_client("anthropic", api_key="test-key")
            usage = client.prompt(
                "claude-3-5-sonnet-20241022", "Hi", response_format=mock_pydantic_model
            ).usage

        assert usage.discarded_input_tokens == 50
        assert usage.discarded_output_tokens == 60
        assert usage.input_tokens == 15
        assert usage.output_tokens == 25

    def test_concurrent_fallback_does_not_leak_between_threads(
        self, openai_client, mock_openai_response, mock_pydantic_model, wire_structured_output
    ):
        """Test one thread's discarded tokens never land on another thread's response.

        The benchmark runs many workers against a single client object, so an accumulator
        held on the client rather than passed down would cross-contaminate. The barrier
        guarantees both calls are genuinely in flight at once.
        """
        client, api = openai_client
        barrier = threading.Barrier(2, timeout=10)

        def parse(**kwargs):
            barrier.wait()
            raise _failed_attempt(50, 60)

        def create(**kwargs):
            # Only the plain call waits; the fallback carries a response_format.
            if "response_format" not in kwargs:
                barrier.wait()
            return mock_openai_response

        wire_structured_output(api, parse)
        api.chat.completions.create.side_effect = create

        with ThreadPoolExecutor(max_workers=2) as pool:
            structured = pool.submit(
                client.prompt, "gpt-4", "Hi", response_format=mock_pydantic_model
            )
            plain = pool.submit(client.prompt, "gpt-4", "Hi")
            structured_usage = structured.result().usage
            plain_usage = plain.result().usage

        assert structured_usage.discarded_input_tokens == 50
        assert structured_usage.attempts == 2
        assert plain_usage.discarded_input_tokens == 0
        assert plain_usage.attempts == 1


@pytest.fixture
def instant_retries():
    """Remove the retry backoff, so failure paths do not spend seconds sleeping."""
    with patch("ai_client.utils.time.sleep"):
        yield


class TestFailedRequestsRecordUsage:
    """Tests that a failed request still accounts for what it was billed."""

    def test_usage_recovered_from_the_exception(self, openai_client, instant_retries):
        """Test tokens generated before a failure are recorded, not zeroed."""
        client, api = openai_client
        api.chat.completions.create.side_effect = _failed_attempt(50, 60)

        usage = client.prompt("gpt-4", "Hi").usage

        assert usage.input_tokens == 50
        assert usage.output_tokens == 60

    def test_recovered_usage_is_priced(self, openai_client, instant_retries, stub_pricing):
        """Test a failed request reports what it cost."""
        stub_pricing(TABLE)
        client, api = openai_client
        api.chat.completions.create.side_effect = _failed_attempt(1_000_000, 1_000_000)

        usage = client.prompt("requested-model", "Hi").usage

        assert usage.estimated_cost_usd == pytest.approx(11.0)

    def test_error_without_usage_stays_zero(self, openai_client, instant_retries):
        """Test a failure that generated nothing still reports zeros."""
        client, api = openai_client
        api.chat.completions.create.side_effect = Exception("connection reset")

        usage = client.prompt("gpt-4", "Hi").usage

        assert usage.input_tokens == 0
        assert usage.estimated_cost_usd is None

    def test_error_response_contract_is_unchanged(self, openai_client, instant_retries):
        """Test the existing error shape still holds for callers that depend on it."""
        client, api = openai_client
        api.chat.completions.create.side_effect = _failed_attempt(50, 60)

        response = client.prompt("gpt-4", "Hi")

        assert response.finish_reason == "error"
        assert response.text == ""
        assert "error" in response.raw_response

    def test_discarded_tokens_survive_a_failed_fallback(
        self, openai_client, instant_retries, mock_pydantic_model, wire_structured_output
    ):
        """Test two billed attempts are reported when neither returned a response."""
        client, api = openai_client
        wire_structured_output(api, _failed_attempt(50, 60))
        api.chat.completions.create.side_effect = Exception("gateway timeout")

        usage = client.prompt("gpt-4", "Hi", response_format=mock_pydantic_model).usage

        assert usage.discarded_input_tokens == 50
        assert usage.discarded_output_tokens == 60

    def test_claude_reports_both_billed_attempts_when_neither_returns(
        self, instant_retries, mock_pydantic_model
    ):
        """Test Claude's tool attempt is still accounted for when the retry also fails."""

        def messages_create(**kwargs):
            if "tools" in kwargs:
                raise _failed_attempt(50, 60, anthropic_naming=True)
            raise Exception("gateway timeout")

        with patch("ai_client.claude_client.Anthropic") as anthropic_class:
            api = Mock()
            anthropic_class.return_value = api
            api.messages.create.side_effect = messages_create

            client = create_ai_client("anthropic", api_key="test-key")
            response = client.prompt(
                "claude-3-5-sonnet-20241022", "Hi", response_format=mock_pydantic_model
            )

        assert response.finish_reason == "error"
        assert response.usage.discarded_input_tokens == 50
        assert response.usage.discarded_output_tokens == 60

    def test_tool_path_preserves_the_first_call_tokens(self, openai_client, instant_retries):
        """Test a failed second call does not erase the first call that was billed."""
        from ai_client.response import LLMResponse, Usage

        client, _api = openai_client
        first = LLMResponse(
            text="",
            model="gpt-4",
            provider="openai",
            finish_reason="tool_calls",
            usage=Usage(input_tokens=70, output_tokens=80, total_tokens=150),
            raw_response={},
            tool_calls=[{"name": "t", "arguments": {}}],
        )
        client.tool_registry = Mock()
        client.tool_registry.get_tool.return_value = {"name": "t", "parameters": {}}

        with (
            patch.object(
                client,
                "_do_prompt_with_retry",
                side_effect=[first, Exception("second call failed")],
            ),
            patch.object(client, "_execute_tools", return_value=[]),
        ):
            usage = client.prompt("gpt-4", "Hi", tool="t").usage

        assert usage.discarded_input_tokens == 70
        assert usage.discarded_output_tokens == 80
        assert usage.attempts == 2


class TestNoDoubleCounting:
    """Tests that each billed call is reported exactly once."""

    def test_terminal_failure_is_not_counted_twice(
        self, openai_client, instant_retries, stub_pricing
    ):
        """Test the failure that ended the request is not also a discarded attempt.

        Its usage becomes the response's own; adding it to the discarded fields as well
        would make a consumer summing the two report double the real spend.
        """
        stub_pricing(TABLE)
        client, api = openai_client
        api.chat.completions.create.side_effect = _failed_attempt(1_000_000, 1_000_000)

        usage = client.prompt("requested-model", "Hi").usage

        assert usage.attempts == 1
        assert usage.discarded_input_tokens == 0
        assert usage.discarded_output_tokens == 0
        assert usage.discarded_cost_usd is None
        assert usage.estimated_cost_usd == pytest.approx(11.0)

    def test_two_billed_failures_are_each_counted_once(
        self, openai_client, instant_retries, mock_pydantic_model, wire_structured_output
    ):
        """Test a billed parse failure and a billed fallback failure are kept apart."""
        client, api = openai_client
        wire_structured_output(api, _failed_attempt(50, 60))
        api.chat.completions.create.side_effect = _failed_attempt(100, 200)

        usage = client.prompt("gpt-4", "Hi", response_format=mock_pydantic_model).usage

        assert usage.input_tokens == 100
        assert usage.output_tokens == 200
        assert usage.discarded_input_tokens == 50
        assert usage.discarded_output_tokens == 60
        assert usage.attempts == 2


class TestDiscardedReasoningIsPriced:
    """Tests that a discarded attempt's reasoning is charged, not dropped."""

    def test_discarded_reasoning_tokens_are_recorded(
        self,
        openai_client,
        instant_retries,
        mock_openai_response,
        mock_pydantic_model,
        wire_structured_output,
    ):
        """Test reasoning billed outside the completion count survives the fallback."""
        client, api = openai_client
        wire_structured_output(api, _failed_attempt(100, 200, total_tokens=500))
        api.chat.completions.create.return_value = mock_openai_response

        usage = client.prompt("gpt-4", "Hi", response_format=mock_pydantic_model).usage

        assert usage.discarded_reasoning_tokens == 200

    def test_discarded_reasoning_is_included_in_the_cost(
        self,
        openai_client,
        instant_retries,
        mock_openai_response,
        mock_pydantic_model,
        stub_pricing,
        wire_structured_output,
    ):
        """Test the discarded cost covers reasoning at the output rate."""
        stub_pricing(TABLE)
        client, api = openai_client
        wire_structured_output(api, _failed_attempt(100, 200, total_tokens=500))
        api.chat.completions.create.return_value = mock_openai_response

        usage = client.prompt("requested-model", "Hi", response_format=mock_pydantic_model).usage

        # 100 in at $1/M, 200 out and 200 reasoning at $10/M.
        assert usage.discarded_cost_usd == pytest.approx(0.0041)


class TestBilledCostSurvivesFailure:
    """Tests that a cost the provider charged is never replaced by an estimate."""

    def test_billed_cost_survives_error_recovery(
        self, openai_client, instant_retries, stub_pricing
    ):
        """Test a failed routed request keeps the provider's own figure."""
        stub_pricing(TABLE)
        client, api = openai_client
        api.chat.completions.create.side_effect = _failed_attempt(100, 200, cost=0.0123)

        usage = client.prompt("requested-model", "Hi").usage

        assert usage.estimated_cost_usd == 0.0123
        assert usage.input_cost_usd is None
        assert usage.output_cost_usd is None

    def test_billed_cost_survives_without_a_price_table_entry(
        self, openai_client, instant_retries, stub_pricing
    ):
        """Test an unpriced model still reports what the provider charged."""
        stub_pricing(TABLE)
        client, api = openai_client
        api.chat.completions.create.side_effect = _failed_attempt(100, 200, cost=0.0123)

        usage = client.prompt("not-in-the-table", "Hi").usage

        assert usage.estimated_cost_usd == 0.0123

    def test_billed_cost_of_zero_is_preserved(self, openai_client, instant_retries, stub_pricing):
        """Test a free route reports zero rather than falling back to list prices."""
        stub_pricing(TABLE)
        client, api = openai_client
        api.chat.completions.create.side_effect = _failed_attempt(1_000_000, 1_000_000, cost=0.0)

        usage = client.prompt("requested-model", "Hi").usage

        assert usage.estimated_cost_usd == 0.0

    def test_billed_cost_is_preferred_for_a_discarded_attempt(
        self,
        openai_client,
        instant_retries,
        mock_openai_response,
        mock_pydantic_model,
        stub_pricing,
        wire_structured_output,
    ):
        """Test a discarded attempt reports what it was charged, not an estimate."""
        stub_pricing(TABLE)
        client, api = openai_client
        wire_structured_output(api, _failed_attempt(100, 200, cost=0.0123))
        api.chat.completions.create.return_value = mock_openai_response

        usage = client.prompt("requested-model", "Hi", response_format=mock_pydantic_model).usage

        assert usage.discarded_cost_usd == 0.0123


class TestBilledCostOnEveryPath:
    """Tests that a provider-billed cost survives whichever builder ran."""

    def test_structured_path_keeps_the_billed_total(
        self,
        openai_client,
        mock_openrouter_response,
        mock_pydantic_model,
        stub_pricing,
        wire_structured_output,
    ):
        """Test the structured builder does not replace a billed cost with a list price."""
        stub_pricing(TABLE)
        client, api = openai_client
        mock_openrouter_response.choices[0].message.parsed = None
        wire_structured_output(api, mock_openrouter_response)

        usage = client.prompt("requested-model", "Hi", response_format=mock_pydantic_model).usage

        assert usage.estimated_cost_usd == 0.0123
        assert usage.input_cost_usd is None

    def test_legacy_completions_path_keeps_the_billed_total(self, openai_client, stub_pricing):
        """Test the legacy builder does not replace a billed cost either."""
        stub_pricing(TABLE)
        client, api = openai_client
        raw = Mock()
        raw.model = "echoed"
        raw.choices = [Mock()]
        raw.choices[0].text = "Hi"
        raw.choices[0].finish_reason = "stop"
        raw.usage = Mock()
        raw.usage.prompt_tokens = 1_000_000
        raw.usage.completion_tokens = 1_000_000
        raw.usage.total_tokens = 2_000_000
        raw.usage.cost = 0.0123
        api.completions.create.return_value = raw

        usage = client.prompt("requested-model", "Hi", api_style="completions").usage

        assert usage.estimated_cost_usd == 0.0123


class TestEveryOpenAIBuilder:
    """Tests that all four response builders price on the requested model."""

    def test_chat_completions_builder(self, openai_client, mock_openai_response, stub_pricing):
        """Test _create_response_from_raw prices the requested model."""
        stub_pricing(TABLE)
        client, api = openai_client
        mock_openai_response.model = "echoed"
        api.chat.completions.create.return_value = mock_openai_response

        usage = client.prompt("requested-model", "Hello").usage

        assert usage.estimated_cost_usd is not None

    def test_parsed_builder(
        self,
        openai_client,
        mock_openai_response,
        mock_pydantic_model,
        stub_pricing,
        wire_structured_output,
    ):
        """Test _create_response_from_parsed prices the requested model."""
        stub_pricing(TABLE)
        client, api = openai_client
        mock_openai_response.model = "echoed"
        mock_openai_response.choices[0].message.parsed = None
        wire_structured_output(api, mock_openai_response)

        usage = client.prompt("requested-model", "Hello", response_format=mock_pydantic_model).usage

        assert usage.estimated_cost_usd is not None

    def test_legacy_completions_builder(self, openai_client, stub_pricing):
        """Test _create_response_from_completions prices the requested model."""
        stub_pricing(TABLE)
        client, api = openai_client
        raw = Mock()
        raw.model = "echoed"
        raw.choices = [Mock()]
        raw.choices[0].text = "Hi"
        raw.choices[0].finish_reason = "stop"
        raw.usage = Mock()
        raw.usage.prompt_tokens = 10
        raw.usage.completion_tokens = 20
        raw.usage.total_tokens = 30
        del raw.usage.cost
        api.completions.create.return_value = raw

        usage = client.prompt("requested-model", "Hello", api_style="completions").usage

        assert usage.estimated_cost_usd is not None

    def test_responses_builder(self, openai_client, stub_pricing):
        """Test _create_response_from_responses prices the requested model."""
        stub_pricing(TABLE)
        client, api = openai_client
        raw = Mock()
        raw.model = "echoed"
        raw.output_text = "Hi"
        raw.status = "completed"
        raw.usage = Mock()
        raw.usage.input_tokens = 10
        raw.usage.output_tokens = 20
        raw.usage.total_tokens = 30
        del raw.usage.cost
        api.responses.create.return_value = raw

        usage = client.prompt("requested-model", "Hello", api_style="responses").usage

        assert usage.estimated_cost_usd is not None


class TestStructuredOutputParseFailures:
    """Tests that a response billed then rejected by the SDK is still accounted for.

    These drive the real OpenAI SDK over a mocked transport. A Mock client cannot raise
    the SDK's own parse failures, and those are exactly the paths that used to lose the
    tokens: only LengthFinishReasonError carries the completion, while a content filter
    and a schema mismatch raise exceptions holding nothing.
    """

    @staticmethod
    def _completion(finish_reason, content):
        """A chat completion body reporting 100 input and 200 output tokens."""
        return {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
        }

    def _usage_after_failed_parse(self, first_body, mock_pydantic_model):
        """Run a structured request whose first attempt fails, and return the usage."""
        import httpx
        from openai import OpenAI

        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            body = first_body if calls["n"] == 1 else self._completion("stop", '{"name":"x"}')
            return httpx.Response(200, json=body)

        sdk = OpenAI(
            api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler))
        )
        with (
            patch("ai_client.openai_client.OpenAI", return_value=sdk),
            patch("ai_client.utils.time.sleep"),
        ):
            client = create_ai_client("openai", api_key="test")
            usage = client.prompt("gpt-4o-mini", "Hi", response_format=mock_pydantic_model).usage
        assert calls["n"] == 2, "the fallback should have made a second billed request"
        return usage

    def test_content_filter_failure_is_billed(self, mock_pydantic_model):
        """Test a filtered response reports the tokens it was charged for."""
        usage = self._usage_after_failed_parse(
            self._completion("content_filter", None), mock_pydantic_model
        )

        assert usage.attempts == 2
        assert usage.discarded_input_tokens == 100
        assert usage.discarded_output_tokens == 200

    def test_schema_mismatch_failure_is_billed(self, mock_pydantic_model):
        """Test output that fails validation still reports what it cost."""
        usage = self._usage_after_failed_parse(
            self._completion("stop", '{"name":"x","value":"not-an-int"}'), mock_pydantic_model
        )

        assert usage.attempts == 2
        assert usage.discarded_input_tokens == 100

    def test_length_limit_failure_is_billed(self, mock_pydantic_model):
        """Test the one failure mode that already carried its completion still works."""
        usage = self._usage_after_failed_parse(
            self._completion("length", '{"name":"x"'), mock_pydantic_model
        )

        assert usage.attempts == 2
        assert usage.discarded_input_tokens == 100


class TestConversionFailuresKeepTheirTokens:
    """Tests that a response billed then rejected by local conversion is still counted."""

    def test_malformed_tool_arguments_keep_the_billed_tokens(self):
        """Test a response that arrives fine but will not convert is not zeroed.

        The request is paid for by the time the arguments are parsed, so a malformed one
        must not turn a billed response into an empty error.
        """
        import httpx
        from openai import OpenAI

        payload = {
            "id": "c",
            "object": "chat.completion",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "1",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{not json"},
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
        }
        sdk = OpenAI(
            api_key="test",
            http_client=httpx.Client(
                transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
            ),
        )
        with (
            patch("ai_client.openai_client.OpenAI", return_value=sdk),
            patch("ai_client.utils.time.sleep"),
        ):
            client = create_ai_client("openai", api_key="test")
            usage = client.prompt("m", "Hi").usage

        assert usage.input_tokens == 100
        assert usage.output_tokens == 200

    def test_two_failures_in_one_request_are_both_counted(self, mock_pydantic_model, stub_pricing):
        """Test a billed parse failure and a billed conversion failure are both reported.

        The structured attempt is thrown away by the fallback, and the fallback's own
        response then fails to convert. Both were paid for, and each must appear once.
        """
        import httpx
        from openai import OpenAI

        stub_pricing(TABLE)

        def body(cost, message, finish_reason="stop"):
            return {
                "id": "c",
                "object": "chat.completion",
                "created": 1,
                "model": "m",
                "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 200,
                    "total_tokens": 300,
                    "cost": cost,
                },
            }

        parse_failure = body(
            0.0123, {"role": "assistant", "content": '{"name":"x","value":"not-an-int"}'}
        )
        conversion_failure = body(
            0.0456,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{not json"},
                    }
                ],
            },
            finish_reason="tool_calls",
        )
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            body_for_call = parse_failure if calls["n"] % 2 == 1 else conversion_failure
            return httpx.Response(200, json=body_for_call)

        sdk = OpenAI(
            api_key="test",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with (
            patch("ai_client.openai_client.OpenAI", return_value=sdk),
            patch("ai_client.utils.time.sleep"),
        ):
            client = create_ai_client("openai", api_key="test")
            usage = client.prompt(
                "requested-model", "Hi", response_format=mock_pydantic_model
            ).usage

        assert usage.attempts == 2
        assert usage.estimated_cost_usd == 0.0456
        assert usage.discarded_cost_usd == 0.0123
        assert usage.discarded_input_tokens == 100

    def test_other_providers_keep_their_tokens_too(self, stub_pricing):
        """Test the protection is shared, not specific to the OpenAI client.

        Gemini's builder reads candidates[0] for the finish reason; a response arriving
        with none raises there, after the request has already been billed.
        """
        stub_pricing(TABLE)
        raw = Mock()
        raw.text = "hi"
        raw.usage_metadata = Mock()
        raw.usage_metadata.prompt_token_count = 100
        raw.usage_metadata.candidates_token_count = 200
        raw.usage_metadata.total_token_count = 300
        raw.usage_metadata.thoughts_token_count = 0
        type(raw).candidates = property(
            lambda self: (_ for _ in ()).throw(ValueError("no candidates"))
        )

        with patch("ai_client.gemini_client.genai") as genai, patch("ai_client.utils.time.sleep"):
            api = Mock()
            genai.Client.return_value = api
            api.models.generate_content.return_value = raw

            client = create_ai_client("genai", api_key="test-key")
            usage = client.prompt("gemini-thinking", "Hi").usage

        assert usage.input_tokens == 100
        assert usage.output_tokens == 200

    def test_cohere_conversion_failure_keeps_its_nested_counts(self, stub_pricing):
        """Test recovery reads Cohere's counts, which sit below the usage block."""
        stub_pricing({"2026-01-01": {"cohere": {"m": {"input_price": 1.0, "output_price": 10.0}}}})
        raw = Mock()
        raw.finish_reason = "COMPLETE"
        raw.usage = Mock()
        raw.usage.tokens = Mock()
        raw.usage.tokens.input_tokens = 100.0  # Cohere reports floats
        raw.usage.tokens.output_tokens = 200.0
        type(raw).message = property(lambda self: (_ for _ in ()).throw(ValueError("bad message")))

        with (
            patch("ai_client.cohere_client.cohere") as cohere_module,
            patch("ai_client.utils.time.sleep"),
        ):
            api = Mock()
            cohere_module.ClientV2.return_value = api
            api.chat.return_value = raw

            client = create_ai_client("cohere", api_key="test-key")
            usage = client.prompt("m", "Hi").usage

        assert usage.input_tokens == 100
        assert usage.output_tokens == 200
        assert usage.estimated_cost_usd == pytest.approx(0.0021)

    def test_recovery_derives_an_omitted_completion_count(self, stub_pricing):
        """Test a provider reporting a total but no completion count is not under-billed.

        Leaving output at zero would charge the whole remainder as reasoning, or drop it,
        and the recorded tokens would not add up to the reported total.
        """
        stub_pricing(TABLE)
        raw = Mock()
        raw.usage_metadata = Mock()
        raw.usage_metadata.prompt_token_count = 100
        raw.usage_metadata.candidates_token_count = None
        raw.usage_metadata.total_token_count = 500
        raw.usage_metadata.thoughts_token_count = 200
        type(raw).candidates = property(
            lambda self: (_ for _ in ()).throw(ValueError("no candidates"))
        )

        with patch("ai_client.gemini_client.genai") as genai, patch("ai_client.utils.time.sleep"):
            api = Mock()
            genai.Client.return_value = api
            api.models.generate_content.return_value = raw

            client = create_ai_client("genai", api_key="test-key")
            usage = client.prompt("gemini-thinking", "Hi").usage

        assert usage.output_tokens == 200
        assert usage.reasoning_tokens == 200
        assert usage.input_tokens + usage.output_tokens + usage.reasoning_tokens == 500
        assert usage.estimated_cost_usd == pytest.approx(0.0041)
