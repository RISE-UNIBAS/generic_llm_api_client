"""
Tests for utility functions.
"""

import pytest
import time
from ai_client.utils import (
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
