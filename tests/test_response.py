"""
Tests for response dataclasses (LLMResponse, Usage).
"""

from datetime import datetime
from ai_client.response import Usage, LLMResponse


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
