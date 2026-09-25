"""
Tests for response dataclasses (LLMResponse, Usage).
"""

from datetime import datetime
from ai_client.response import Usage, LLMResponse, DiscardedAttempts


class TestUsage:
    """Tests for Usage dataclass."""

    def test_usage_creation(self):
        """Test basic Usage creation."""
        usage = Usage(input_tokens=100, output_tokens=50, total_tokens=150)

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.cached_tokens is None
        assert usage.estimated_cost_usd is None

    def test_usage_with_optional_fields(self):
        """Test Usage with optional fields."""
        usage = Usage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cached_tokens=20,
            estimated_cost_usd=0.015,
        )

        assert usage.cached_tokens == 20
        assert usage.estimated_cost_usd == 0.015

    def test_usage_to_dict(self):
        """Test Usage.to_dict() method."""
        usage = Usage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cached_tokens=20,
            estimated_cost_usd=0.015,
        )

        result = usage.to_dict()

        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["total_tokens"] == 150
        assert result["cached_tokens"] == 20
        assert result["estimated_cost_usd"] == 0.015

    def test_usage_to_dict_without_optional(self):
        """Test Usage.to_dict() without optional fields."""
        usage = Usage(input_tokens=100, output_tokens=50, total_tokens=150)

        result = usage.to_dict()

        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["total_tokens"] == 150
        assert "cached_tokens" not in result
        assert "estimated_cost_usd" not in result

    def test_none_token_counts_coerced_to_zero(self):
        """Test omitted provider counts become 0 instead of None."""
        usage = Usage(input_tokens=12, output_tokens=None, total_tokens=None)

        assert usage.output_tokens == 0
        assert usage.total_tokens == 0

    def test_mixed_none_cache_tokens_do_not_raise(self):
        """Test a Claude response reporting only one cache counter still sums."""
        usage = Usage(
            input_tokens=15, output_tokens=25, cache_creation_tokens=100, cache_read_tokens=None
        )

        assert usage.get_total_input_tokens() == 115
        assert usage.get_cache_savings() == 0.0

    def test_to_dict_emits_zero_reasoning_tokens(self):
        """Test a reported zero survives, unlike an unknown."""
        result = Usage(reasoning_tokens=0).to_dict()

        assert result["reasoning_tokens"] == 0

    def test_to_dict_omits_unknown_reasoning_tokens(self):
        """Test reasoning is omitted when no total was reported."""
        assert "reasoning_tokens" not in Usage().to_dict()

    def test_to_dict_emits_reasoning_cost(self):
        """Test reasoning cost is emitted alongside the other components."""
        usage = Usage(input_cost_usd=0.1, output_cost_usd=0.2, reasoning_cost_usd=0.3)
        result = usage.to_dict()

        assert result["reasoning_cost_usd"] == 0.3

    def test_to_dict_hides_a_single_attempt(self):
        """Test an ordinary response carries no attempts key."""
        assert "attempts" not in Usage().to_dict()

    def test_to_dict_emits_repeated_attempts(self):
        """Test a response billed more than once reports how many."""
        assert Usage(attempts=2).to_dict()["attempts"] == 2

    def test_to_dict_omits_empty_discarded_fields(self):
        """Test nothing discarded means no discarded keys."""
        result = Usage().to_dict()

        assert "discarded_input_tokens" not in result
        assert "discarded_cost_usd" not in result

    def test_to_dict_emits_discarded_fields(self):
        """Test discarded tokens are reported separately from the successful call."""
        usage = Usage(
            input_tokens=10,
            discarded_input_tokens=5,
            discarded_output_tokens=7,
            discarded_cost_usd=0.02,
        )
        result = usage.to_dict()

        assert result["input_tokens"] == 10
        assert result["discarded_input_tokens"] == 5
        assert result["discarded_output_tokens"] == 7
        assert result["discarded_cost_usd"] == 0.02

    def test_positional_construction_unchanged(self):
        """Test the new fields were appended, so existing positional callers still work."""
        usage = Usage(100, 50, 150, 20, 0, 0, 0.1, 0.2, 0.3)

        assert usage.input_tokens == 100
        assert usage.cached_tokens == 20
        assert usage.estimated_cost_usd == 0.3
        assert usage.reasoning_tokens is None
        assert usage.attempts == 1


class TestDiscardedAttempts:
    """Tests for DiscardedAttempts."""

    def test_starts_empty(self):
        """Test a fresh accumulator has recorded nothing."""
        discarded = DiscardedAttempts()

        assert discarded.attempts == 0
        assert discarded.cost_usd is None

    def test_add_accumulates_tokens(self):
        """Test each discarded attempt adds to the running totals."""
        discarded = DiscardedAttempts()
        discarded.add(input_tokens=10, output_tokens=20, cost_usd=0.5)
        discarded.add(input_tokens=5, output_tokens=None)

        assert discarded.attempts == 2
        assert discarded.input_tokens == 15
        assert discarded.output_tokens == 20

    def test_add_accumulates_only_priced_attempts(self):
        """Test an attempt that could not be priced leaves the cost as it was."""
        discarded = DiscardedAttempts()
        discarded.add(input_tokens=10, cost_usd=0.5)
        discarded.add(input_tokens=10, cost_usd=None)

        assert discarded.cost_usd == 0.5

    def test_apply_to_folds_into_usage(self):
        """Test the totals land on the returned response's usage."""
        discarded = DiscardedAttempts()
        discarded.add(input_tokens=10, output_tokens=20, cost_usd=0.5)
        usage = Usage(input_tokens=1, output_tokens=2, total_tokens=3)

        discarded.apply_to(usage)

        assert usage.attempts == 2
        assert usage.discarded_input_tokens == 10
        assert usage.discarded_output_tokens == 20
        assert usage.discarded_cost_usd == 0.5

    def test_apply_to_leaves_successful_counts_alone(self):
        """Test the successful call keeps reporting only what it used."""
        discarded = DiscardedAttempts()
        discarded.add(input_tokens=10, output_tokens=20)
        usage = Usage(input_tokens=1, output_tokens=2, total_tokens=3)

        discarded.apply_to(usage)

        assert usage.input_tokens == 1
        assert usage.output_tokens == 2
        assert usage.total_tokens == 3

    def test_apply_to_is_a_noop_when_nothing_was_discarded(self):
        """Test an untouched accumulator leaves the record unchanged."""
        usage = Usage(input_tokens=1, output_tokens=2, total_tokens=3)

        DiscardedAttempts().apply_to(usage)

        assert usage.attempts == 1
        assert "attempts" not in usage.to_dict()


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_llm_response_creation(self):
        """Test basic LLMResponse creation."""
        usage = Usage(input_tokens=10, output_tokens=20, total_tokens=30)
        response = LLMResponse(
            text="Hello world",
            model="gpt-4",
            provider="openai",
            finish_reason="stop",
            usage=usage,
            raw_response={"test": "data"},
            duration=1.5,
        )

        assert response.text == "Hello world"
        assert response.model == "gpt-4"
        assert response.provider == "openai"
        assert response.finish_reason == "stop"
        assert response.usage == usage
        assert response.raw_response == {"test": "data"}
        assert response.duration == 1.5
        assert isinstance(response.timestamp, datetime)

    def test_llm_response_str(self):
        """Test LLMResponse.__str__() returns text."""
        usage = Usage(input_tokens=10, output_tokens=20, total_tokens=30)
        response = LLMResponse(
            text="Hello world",
            model="gpt-4",
            provider="openai",
            finish_reason="stop",
            usage=usage,
            raw_response={},
            duration=1.5,
        )

        assert str(response) == "Hello world"

    def test_llm_response_to_dict(self):
        """Test LLMResponse.to_dict() method."""
        usage = Usage(input_tokens=10, output_tokens=20, total_tokens=30)
        response = LLMResponse(
            text="Hello world",
            model="gpt-4",
            provider="openai",
            finish_reason="stop",
            usage=usage,
            raw_response={"test": "data"},
            duration=1.5,
        )

        result = response.to_dict()

        assert result["text"] == "Hello world"
        assert result["model"] == "gpt-4"
        assert result["provider"] == "openai"
        assert result["finish_reason"] == "stop"
        assert result["duration"] == 1.5
        assert "usage" in result
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 20
        assert result["usage"]["total_tokens"] == 30
        # raw_response should not be in dict (not JSON serializable)
        assert "raw_response" not in result
        # timestamp should be ISO format string
        assert isinstance(result["timestamp"], str)

    def test_llm_response_custom_timestamp(self):
        """Test LLMResponse with custom timestamp."""
        usage = Usage(input_tokens=10, output_tokens=20, total_tokens=30)
        custom_time = datetime(2025, 1, 1, 12, 0, 0)
        response = LLMResponse(
            text="Hello",
            model="gpt-4",
            provider="openai",
            finish_reason="stop",
            usage=usage,
            raw_response={},
            duration=1.0,
            timestamp=custom_time,
        )

        assert response.timestamp == custom_time
