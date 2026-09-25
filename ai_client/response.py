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
    """
    Token usage and cost information for an LLM request.

    Caching metrics are provider-specific:
    - OpenAI & Gemini: Use cached_tokens field
    - Claude: Uses cache_creation_tokens and cache_read_tokens

    reasoning_tokens counts only reasoning a provider billed outside output_tokens, so
    input_tokens + output_tokens + reasoning_tokens equals total_tokens wherever a total was
    reported, and estimated_cost_usd covers all three. A cost the provider billed directly
    arrives without component costs beside it and is never recalculated.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Caching metrics (provider-specific)
    cached_tokens: Optional[int] = None  # OpenAI & Gemini: all cached tokens
    cache_creation_tokens: int = 0  # Claude: tokens written to cache
    cache_read_tokens: int = 0  # Claude: tokens read from cache

    # Cost tracking
    input_cost_usd: Optional[float] = None
    output_cost_usd: Optional[float] = None
    estimated_cost_usd: Optional[float] = None  # Total cost (for backwards compatibility)

    # Reasoning billed outside output_tokens. None means no total was reported, which is
    # not the same as a provider reporting no reasoning.
    reasoning_tokens: Optional[int] = None
    reasoning_cost_usd: Optional[float] = None

    # Billable attempts behind this response, and what the discarded ones cost.
    attempts: int = 1
    discarded_input_tokens: int = 0
    discarded_output_tokens: int = 0
    discarded_reasoning_tokens: int = 0
    discarded_cost_usd: Optional[float] = None

    def __post_init__(self):
        """Coerce omitted token counts to 0.

        Provider SDKs type these as optional: every count on genai's usage_metadata, and
        both of Anthropic's cache counters. A None reaching cost calculation or
        get_total_input_tokens turns a successful response into an error response.
        """
        self.input_tokens = self.input_tokens or 0
        self.output_tokens = self.output_tokens or 0
        self.total_tokens = self.total_tokens or 0
        self.cache_creation_tokens = self.cache_creation_tokens or 0
        self.cache_read_tokens = self.cache_read_tokens or 0

    def get_total_input_tokens(self) -> int:
        """
        Get total input tokens including all cached tokens.

        For Claude: total = cache_read + cache_creation + input_tokens
        For OpenAI/Gemini: cached_tokens already included in input_tokens
        """
        if self.cache_creation_tokens or self.cache_read_tokens:
            # Claude model
            return self.cache_read_tokens + self.cache_creation_tokens + self.input_tokens
        else:
            # OpenAI/Gemini model
            return self.input_tokens

    def get_cache_savings(self) -> float:
        """
        Calculate percentage of tokens that were cached.
        Returns value between 0.0 and 1.0.
        """
        total_input = self.get_total_input_tokens()
        if total_input == 0:
            return 0.0

        if self.cache_read_tokens:
            # Claude: only cache_read_tokens represent savings
            return self.cache_read_tokens / total_input
        elif self.cached_tokens:
            # OpenAI/Gemini
            return self.cached_tokens / total_input

        return 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        result = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }
        # Emitted on `is not None`, so a reported 0 stays distinguishable from a provider
        # that reported no total and told us nothing.
        if self.reasoning_tokens is not None:
            result["reasoning_tokens"] = self.reasoning_tokens
        if self.cached_tokens is not None:
            result["cached_tokens"] = self.cached_tokens
        if self.cache_creation_tokens:
            result["cache_creation_tokens"] = self.cache_creation_tokens
        if self.cache_read_tokens:
            result["cache_read_tokens"] = self.cache_read_tokens
        if self.input_cost_usd is not None:
            result["input_cost_usd"] = self.input_cost_usd
        if self.output_cost_usd is not None:
            result["output_cost_usd"] = self.output_cost_usd
        if self.reasoning_cost_usd is not None:
            result["reasoning_cost_usd"] = self.reasoning_cost_usd
        if self.estimated_cost_usd is not None:
            result["estimated_cost_usd"] = self.estimated_cost_usd
        if self.attempts > 1:
            result["attempts"] = self.attempts
        if self.discarded_input_tokens:
            result["discarded_input_tokens"] = self.discarded_input_tokens
        if self.discarded_output_tokens:
            result["discarded_output_tokens"] = self.discarded_output_tokens
        if self.discarded_reasoning_tokens:
            result["discarded_reasoning_tokens"] = self.discarded_reasoning_tokens
        if self.discarded_cost_usd is not None:
            result["discarded_cost_usd"] = self.discarded_cost_usd
        return result


@dataclass
class DiscardedAttempts:
    """
    Tokens a provider billed for on attempts whose responses were thrown away.

    A structured-output call that fails and falls back to a second request is billed twice.
    Folding those tokens into the successful response's counts would redefine what a stored
    request's tokens mean, so they accumulate separately and are reported alongside.

    Threaded as an explicit argument rather than held on the client: one client object
    serves many worker threads, and shared state would attribute one thread's waste to
    another thread's response.
    """

    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: Optional[float] = None

    @classmethod
    def from_usage(cls, usage: Usage) -> "DiscardedAttempts":
        """
        Build an accumulator from a response that was billed and then thrown away.

        Args:
            usage: Usage of the discarded response

        Returns:
            A DiscardedAttempts carrying that response's totals
        """
        return cls(
            attempts=usage.attempts,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=usage.reasoning_tokens or 0,
            cost_usd=usage.estimated_cost_usd,
        )

    def add(
        self,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
        reasoning_tokens: Optional[int] = None,
    ) -> None:
        """
        Record one discarded attempt.

        Args:
            input_tokens: Prompt tokens the discarded attempt billed
            output_tokens: Completion tokens the discarded attempt billed
            cost_usd: Cost of the discarded attempt, or None if it could not be priced
            reasoning_tokens: Reasoning tokens billed outside output_tokens
        """
        self.attempts += 1
        self.input_tokens += input_tokens or 0
        self.output_tokens += output_tokens or 0
        self.reasoning_tokens += reasoning_tokens or 0
        if cost_usd is not None:
            self.cost_usd = (self.cost_usd or 0.0) + cost_usd

    def apply_to(self, usage: Usage) -> None:
        """
        Fold the accumulated totals into the usage of the response being returned.

        Args:
            usage: Usage of the response that was finally returned
        """
        if not self.attempts:
            return

        usage.attempts += self.attempts
        usage.discarded_input_tokens += self.input_tokens
        usage.discarded_output_tokens += self.output_tokens
        usage.discarded_reasoning_tokens += self.reasoning_tokens
        if self.cost_usd is not None:
            usage.discarded_cost_usd = (usage.discarded_cost_usd or 0.0) + self.cost_usd


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
        conversation_id: ID for multi-turn conversation tracking (None for single-turn)
        cache_ref: Reference to cache object (Gemini-specific, None otherwise)
        tool_calls: List of tool calls made by the LLM (None if no tools called)
        tool_results: List of tool execution results (None if no tools executed)
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
    conversation_id: Optional[str] = None
    cache_ref: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_results: Optional[list[dict[str, Any]]] = None

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
        if self.conversation_id is not None:
            result["conversation_id"] = self.conversation_id
        if self.cache_ref is not None:
            result["cache_ref"] = self.cache_ref
        if self.tool_calls is not None:
            result["tool_calls"] = self.tool_calls
        if self.tool_results is not None:
            result["tool_results"] = self.tool_results
        return result

    def __str__(self) -> str:
        """String representation shows the text content."""
        return self.text
