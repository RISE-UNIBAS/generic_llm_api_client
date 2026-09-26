"""
Reasoning tokens a provider billed for but left out of output_tokens.

``reasoning_tokens`` is the part not already counted in ``output_tokens``. Providers
disagree on where they put it: genai and x-ai report reasoning outside their completion
count, while the OpenAI family reports it inside. Defining the field as the uncounted
remainder makes ``input + output + reasoning == total`` hold wherever a total was reported,
so consumers need no per-provider rules and no cost is counted twice.

The derivation matches scripts/reasoning_tokens.py in the benchmark repo
(RISE-UNIBAS/humanities_data_benchmark), which produced the backfilled corpus. The two must
stay in step for records from either side to remain comparable.
"""

import json
import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Where each provider reports its own reasoning count, relative to the raw response.
REPORTED_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("usage_metadata", "thoughts_token_count"),  # genai
    ("usage", "completion_tokens_details", "reasoning_tokens"),  # x-ai, openai chat
    ("usage", "output_tokens_details", "reasoning_tokens"),  # openai responses
)


def _as_walkable(raw_response: Any) -> Any:
    """Return a value _dig can walk, parsing a JSON string into a dict."""
    if isinstance(raw_response, str):
        try:
            parsed = json.loads(raw_response)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return raw_response


def _dig(node: Any, path: Tuple[str, ...]) -> Any:
    """Follow path through mappings or attributes, returning None at the first gap.

    Covers both shapes a response arrives in: live SDK models here, stored JSON in the
    benchmark repo.
    """
    for key in path:
        if node is None:
            return None
        node = node.get(key) if isinstance(node, dict) else getattr(node, key, None)
    return node


def reported_reasoning(raw_response: Any) -> Optional[int]:
    """
    Read the provider's own reasoning count from a raw response.

    Args:
        raw_response: Provider response as an SDK object, dict, or JSON string

    Returns:
        The reported count, or None if the provider reported none. A value that is not
        a plain integer is treated as absent.
    """
    node = _as_walkable(raw_response)
    if node is None:
        return None

    for path in REPORTED_PATHS:
        value = _dig(node, path)
        # bool subclasses int; a flag is not a token count.
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def derived_output(
    input_tokens: Optional[int], total_tokens: Optional[int], raw_response: Any = None
) -> int:
    """
    Recover a completion count the provider omitted, from the total it did report.

    genai leaves candidates_token_count unset on a thinking response that stopped early,
    while still reporting a total. Deriving the count keeps input + output + reasoning
    equal to that total, instead of charging the whole remainder as reasoning.

    Args:
        input_tokens: Prompt tokens, or None if the provider omitted them
        total_tokens: Provider-reported total
        raw_response: Provider response, consulted for its own reasoning count

    Returns:
        The recovered completion count, never negative
    """
    reported = reported_reasoning(raw_response) or 0
    return max(0, (total_tokens or 0) - (input_tokens or 0) - reported)


def gap_of(
    input_tokens: Optional[int], output_tokens: Optional[int], total_tokens: Optional[int]
) -> Optional[int]:
    """
    Calculate total_tokens minus input_tokens and output_tokens.

    A missing or zero total means unknown, which is not the same as zero: a provider that
    reported no total told us nothing, while one that reported no reasoning told us zero.
    Collapsing the two would lose a distinction the stored corpus depends on.

    Args:
        input_tokens: Prompt tokens, or None if the provider omitted them
        output_tokens: Completion tokens, or None if the provider omitted them
        total_tokens: Provider-reported total, or None if it reported none

    Returns:
        The gap in tokens, or None when no total was reported. None counts are treated as
        zero, since every field on genai's usage_metadata is optional.
    """
    total = total_tokens or 0
    if not total:
        return None
    return total - (input_tokens or 0) - (output_tokens or 0)


def uncounted_reasoning(
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    total_tokens: Optional[int],
    raw_response: Any = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[int]:
    """
    Calculate billed reasoning tokens not already inside output_tokens.

    The result is clamped to the gap, so a provider that reports reasoning inside
    output_tokens yields 0, and one whose own numbers do not add up yields 0 rather than a
    negative count.

    Args:
        input_tokens: Prompt tokens, or None if the provider omitted them
        output_tokens: Completion tokens, or None if the provider omitted them
        total_tokens: Provider-reported total, or None if it reported none
        raw_response: Provider response, consulted for its own reasoning count
        provider: Provider ID, naming the request in a mismatch warning
        model: Model identifier, naming the request in a mismatch warning

    Returns:
        Reasoning tokens outside output_tokens, or None when no total was reported
    """
    gap = gap_of(input_tokens, output_tokens, total_tokens)
    if gap is None:
        return None

    reported = reported_reasoning(raw_response)

    # A reported count against a gap of 0 is the OpenAI family reporting inside
    # output_tokens, not a disagreement. A real mismatch means the provider changed its
    # accounting, so make it visible rather than silently preferring one number.
    if gap > 0 and reported is not None and reported != gap:
        logger.warning(
            f"Reasoning token mismatch for provider={provider!r} model={model!r}: provider "
            f"reported {reported}, but total - input - output is {gap}. Recording "
            f"{max(0, min(reported, gap))}. Check whether the provider changed its usage "
            f"accounting; stored costs may be wrong from here on."
        )

    value = gap if reported is None else reported
    return max(0, min(value, gap))
