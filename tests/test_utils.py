"""
Tests for utility functions.
"""

import pytest
import time
from types import SimpleNamespace
from unittest.mock import Mock
from ai_client.response import DiscardedAttempts
from ai_client.utils import (
    attach_discarded,
    billed_cost,
    error_payload,
    usage_of,
    discarded_from_error,
    usage_counts,
    usage_from_error,
    retry_with_exponential_backoff,
    is_rate_limit_error,
    get_retry_delay_from_error,
    RateLimitError,
    APIError,
)


class TestRetryWithExponentialBackoff:
    """Tests for retry_with_exponential_backoff decorator."""

    def test_successful_call_no_retry(self):
        """Test that successful calls don't retry."""
        call_count = []

        @retry_with_exponential_backoff
        def successful_func():
            call_count.append(1)
            return "success"

        result = successful_func()

        assert result == "success"
        assert len(call_count) == 1

    def test_retry_on_exception(self):
        """Test that function retries on exception."""
        call_count = []

        @retry_with_exponential_backoff
        def failing_func():
            call_count.append(1)
            if len(call_count) < 3:
                raise ValueError("Temporary error")
            return "success"

        result = failing_func()

        assert result == "success"
        assert len(call_count) == 3

    def test_max_retries_exceeded(self):
        """Test that max retries are respected."""
        call_count = []

        def always_failing():
            call_count.append(1)
            raise ValueError("Always fails")

        wrapped = retry_with_exponential_backoff(always_failing, max_retries=2)

        with pytest.raises(ValueError, match="Always fails"):
            wrapped()

        # Should try 3 times total (initial + 2 retries)
        assert len(call_count) == 3

    def test_exponential_backoff_timing(self):
        """Test that exponential backoff delays are applied."""
        call_times = []

        def failing_func():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("Retry me")
            return "done"

        wrapped = retry_with_exponential_backoff(
            failing_func, max_retries=2, initial_delay=0.1, exponential_base=2.0
        )

        wrapped()

        # Check that delays are applied
        assert len(call_times) == 3
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        # Delays should be at least the configured minimums (with tolerance for system overhead)
        # First retry: initial_delay = 0.1s
        # Second retry: initial_delay * exponential_base = 0.2s
        assert delay1 >= 0.08  # At least 80% of initial delay (0.1s)
        assert delay2 >= 0.16  # At least 80% of second delay (0.2s)

        # Total time should be reasonable (sum of delays + small overhead)
        total_time = call_times[2] - call_times[0]
        assert total_time >= 0.25  # At least 0.1s + 0.2s = 0.3s (with tolerance)

    def test_specific_exception_types(self):
        """Test retry only on specific exception types."""
        call_count = []

        def func_with_specific_error():
            call_count.append(1)
            if len(call_count) == 1:
                raise ValueError("Retry this")
            elif len(call_count) == 2:
                raise TypeError("Don't retry this")
            return "success"

        wrapped = retry_with_exponential_backoff(
            func_with_specific_error,
            max_retries=3,
            retryable_exceptions=(ValueError,),
            initial_delay=0.01,
        )

        # Should retry ValueError but fail on TypeError
        with pytest.raises(TypeError, match="Don't retry this"):
            wrapped()

        assert len(call_count) == 2


class TestIsRateLimitError:
    """Tests for is_rate_limit_error function."""

    def test_detects_rate_limit_error(self):
        """Test detection of rate limit errors."""
        rate_limit_messages = [
            "Rate limit exceeded",
            "rate_limit_error",
            "Too many requests",
            "429 error",
            "Quota exceeded",
            "resource_exhausted",
        ]

        for msg in rate_limit_messages:
            error = Exception(msg)
            assert is_rate_limit_error(error) is True

    def test_does_not_detect_other_errors(self):
        """Test that other errors are not detected as rate limit."""
        other_messages = [
            "Connection error",
            "Invalid API key",
            "Model not found",
            "Internal server error",
        ]

        for msg in other_messages:
            error = Exception(msg)
            assert is_rate_limit_error(error) is False


class TestGetRetryDelayFromError:
    """Tests for get_retry_delay_from_error function."""

    def test_extracts_retry_delay(self):
        """Test extraction of retry delay from error message."""
        test_cases = [
            ("Please retry after 5 seconds", 5.0),
            ("Rate limited. Retry in 10 seconds", 10.0),
            ("Wait 30 seconds before retrying", 30.0),
        ]

        for msg, expected_delay in test_cases:
            error = Exception(msg)
            delay = get_retry_delay_from_error(error)
            assert delay == expected_delay

    def test_returns_none_without_delay_info(self):
        """Test returns None when no delay info in error."""
        error = Exception("Rate limit exceeded")
        delay = get_retry_delay_from_error(error)
        assert delay is None


class TestExceptions:
    """Tests for custom exceptions."""

    def test_rate_limit_error(self):
        """Test RateLimitError exception."""
        with pytest.raises(RateLimitError, match="Too many requests"):
            raise RateLimitError("Too many requests")

    def test_api_error(self):
        """Test APIError exception."""
        with pytest.raises(APIError, match="API failed"):
            raise APIError("API failed")

    def test_exceptions_are_exceptions(self):
        """Test that custom exceptions inherit from Exception."""
        assert issubclass(RateLimitError, Exception)
        assert issubclass(APIError, Exception)


class TestUsageCounts:
    """Tests for reading token counts off a provider usage object."""

    def test_openai_naming(self):
        """Test prompt_tokens and completion_tokens are recognised."""
        source = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

        assert usage_counts(source) == (10, 20, 30)

    def test_anthropic_naming(self):
        """Test input_tokens and output_tokens are recognised."""
        source = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}

        assert usage_counts(source) == (10, 20, 30)

    def test_total_is_derived_when_absent(self):
        """Test a missing total falls back to the sum, as Anthropic reports none."""
        assert usage_counts({"input_tokens": 10, "output_tokens": 20}) == (10, 20, 30)

    def test_attributes_are_read_as_well_as_keys(self):
        """Test a live SDK object is accepted, not just a mapping."""
        source = SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)

        assert usage_counts(source) == (10, 20, 30)

    def test_missing_source_returns_none(self):
        """Test no usage at all yields None."""
        assert usage_counts(None) is None

    def test_source_without_counts_returns_none(self):
        """Test an object carrying no recognisable counts yields None."""
        assert usage_counts({"something_else": 1}) is None

    def test_non_integer_values_are_ignored(self):
        """Test a placeholder attribute is not mistaken for a count."""
        assert usage_counts(Mock()) is None


class TestUsageFromError:
    """Tests for recovering the tokens a failed request was billed for."""

    def test_reads_the_attached_completion(self):
        """Test usage on a structured-output error is found."""
        error = Exception("failed")
        error.completion = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        )

        assert usage_from_error(error) == (10, 20, 30)

    def test_reads_the_error_body(self):
        """Test usage in an SDK error body is found."""
        error = Exception("failed")
        error.body = {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}

        assert usage_from_error(error) == (10, 20, 30)

    def test_reads_the_response_payload(self):
        """Test usage in the raw HTTP response is found."""
        error = Exception("failed")
        error.response = Mock()
        error.response.json.return_value = {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}

        assert usage_from_error(error) == (10, 20, 30)

    def test_unreadable_response_is_tolerated(self):
        """Test a body that will not parse yields None instead of raising."""
        error = Exception("failed")
        error.response = Mock()
        error.response.json.side_effect = ValueError("not json")

        assert usage_from_error(error) is None

    def test_error_carrying_nothing_returns_none(self):
        """Test a failure before generation reports no usage."""
        assert usage_from_error(Exception("connection reset")) is None


class TestDiscardedCarrier:
    """Tests for carrying discarded-attempt totals on an exception."""

    def test_round_trip(self):
        """Test what was attached comes back out."""
        discarded = DiscardedAttempts()
        discarded.add(input_tokens=10, output_tokens=20)
        error = Exception("failed")

        attach_discarded(error, discarded)

        assert discarded_from_error(error) is discarded

    def test_absent_returns_none(self):
        """Test an untouched exception carries nothing."""
        assert discarded_from_error(Exception("failed")) is None

    def test_exception_rejecting_attributes_is_tolerated(self):
        """Test an exception that refuses attributes does not break the failure path."""

        class Strict(Exception):
            def __setattr__(self, name, value):
                raise AttributeError(name)

        error = Strict()

        attach_discarded(error, DiscardedAttempts())

        assert discarded_from_error(error) is None


class TestBilledCost:
    """Tests for reading a cost the provider reported charging."""

    def test_numeric_cost_is_returned(self):
        """Test a reported cost is read from a mapping or an object."""
        assert billed_cost({"cost": 0.0123}) == 0.0123
        assert billed_cost(SimpleNamespace(cost=0.0123)) == 0.0123

    def test_zero_is_a_real_cost(self):
        """Test a free route reports zero rather than nothing."""
        assert billed_cost({"cost": 0.0}) == 0.0

    def test_missing_cost_returns_none(self):
        """Test a provider reporting no cost yields None."""
        assert billed_cost({"prompt_tokens": 10}) is None
        assert billed_cost(None) is None

    def test_placeholder_attribute_is_not_a_cost(self):
        """Test an auto-created attribute is not mistaken for a billed figure."""
        assert billed_cost(Mock()) is None


class TestErrorPayload:
    """Tests for locating the response a failed request carried."""

    def test_prefers_the_attached_completion(self):
        """Test the completion is returned whole, so cost and reasoning survive."""
        error = Exception("failed")
        error.completion = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, cost=0.5)
        )

        payload = error_payload(error)

        assert billed_cost(usage_of(payload)) == 0.5

    def test_falls_back_to_the_error_body(self):
        """Test a payload carried in the error body is found."""
        error = Exception("failed")
        error.body = {"usage": {"prompt_tokens": 10, "completion_tokens": 20, "cost": 0.5}}

        assert billed_cost(usage_of(error_payload(error))) == 0.5

    def test_ignores_a_payload_without_usage(self):
        """Test a body carrying no usage is not treated as the payload."""
        error = Exception("failed")
        error.body = {"message": "bad request"}

        assert error_payload(error) is None

    def test_returns_none_when_nothing_was_billed(self):
        """Test a failure before generation carries no payload."""
        assert error_payload(Exception("connection reset")) is None
