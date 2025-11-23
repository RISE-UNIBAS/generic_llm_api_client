"""
Response classes for LLM API interactions.

This module defines the response structures returned by all provider implementations,
ensuring consistent access to usage metrics, timing, and raw responses for research
and benchmarking purposes.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Union


@dataclass
class Usage:
    """Token usage and cost information for an LLM request."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: Optional[int] = None
    input_cost_usd: Optional[float] = None
    output_cost_usd: Optional[float] = None
    estimated_cost_usd: Optional[float] = None  # Total cost (for backwards compatibility)

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        result = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.cached_tokens is not None:
            result["cached_tokens"] = self.cached_tokens
        if self.input_cost_usd is not None:
            result["input_cost_usd"] = self.input_cost_usd
        if self.output_cost_usd is not None:
            result["output_cost_usd"] = self.output_cost_usd
        if self.estimated_cost_usd is not None:
            result["estimated_cost_usd"] = self.estimated_cost_usd
        return result


@dataclass
class LLMResponse:
    """
    Unified response object for all LLM providers.

    This class provides a consistent interface while preserving provider-specific
    raw responses for detailed analysis and benchmarking.

    Attributes:
        text: The generated text response
        model: Model identifier used for generation
        provider: Provider name (openai, anthropic, genai, etc.)
        finish_reason: Why generation stopped (stop, length, error, etc.)
        usage: Token usage information
        raw_response: The original provider-specific response object
        duration: Time taken for the request in seconds
        timestamp: When the response was generated
        parsed: Parsed JSON as dict/list when using structured output (None otherwise)
    """

    text: str
    model: str
    provider: str
    finish_reason: str
    usage: Usage
    raw_response: Any
    duration: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    parsed: Optional[Union[dict, list]] = None

    def to_dict(self) -> dict:
        """
        Convert response to dictionary format.

        Note: raw_response is excluded as it may not be JSON-serializable.
        Access it directly when needed for detailed analysis.
        """

        result = {
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "finish_reason": self.finish_reason,
            "usage": self.usage.to_dict(),
            "duration": self.duration,
            "timestamp": (
                self.timestamp.isoformat()
                if isinstance(self.timestamp, datetime)
                else self.timestamp
            ),
        }
        if self.parsed is not None:
            result["parsed"] = self.parsed
        return result

    def __str__(self) -> str:
        """String representation shows the text content."""
        return self.text
