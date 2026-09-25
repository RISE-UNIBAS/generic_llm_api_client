"""
Tests for reasoning token derivation.
"""

import json
import logging
from types import SimpleNamespace
from unittest.mock import Mock

from ai_client.reasoning import gap_of, reported_reasoning, uncounted_reasoning

GENAI = {"usage_metadata": {"thoughts_token_count": 30}}
OPENAI_CHAT = {"usage": {"completion_tokens_details": {"reasoning_tokens": 512}}}
OPENAI_RESPONSES = {"usage": {"output_tokens_details": {"reasoning_tokens": 512}}}


class TestGapOf:
    """Tests for gap_of."""

    def test_gap_from_reported_total(self):
        """Test the gap is total minus input and output."""
        assert gap_of(12, 18, 60) == 30

    def test_missing_total_is_unknown(self):
        """Test a missing total yields None rather than 0."""
        assert gap_of(10, 20, None) is None

    def test_zero_total_is_unknown(self):
        """Test a zero total yields None; the provider told us nothing."""
        assert gap_of(10, 20, 0) is None

    def test_none_counts_treated_as_zero(self):
        """Test None input and output coerce to 0 instead of raising."""
        assert gap_of(None, None, 60) == 60
        assert gap_of(12, None, 60) == 48

    def test_negative_gap_is_returned_unclamped(self):
        """Test gap_of reports a provider's inconsistent numbers as they are."""
        assert gap_of(100, 100, 150) == -50


class TestReportedReasoning:
    """Tests for reported_reasoning."""

    def test_genai_thoughts_token_count(self):
        """Test the genai path is read."""
        assert reported_reasoning(GENAI) == 30

    def test_openai_chat_completion_tokens_details(self):
        """Test the x-ai and openai chat path is read."""
        assert reported_reasoning(OPENAI_CHAT) == 512

    def test_openai_responses_output_tokens_details(self):
        """Test the openai responses path is read."""
        assert reported_reasoning(OPENAI_RESPONSES) == 512

    def test_sdk_object_attributes_are_walked(self):
        """Test attributes are walked, not just mapping keys."""
        raw = SimpleNamespace(usage_metadata=SimpleNamespace(thoughts_token_count=30))
        assert reported_reasoning(raw) == 30

    def test_json_string_payload(self):
        """Test a stored JSON string is parsed, as the benchmark repo holds it."""
        assert reported_reasoning(json.dumps(GENAI)) == 30

    def test_missing_field_returns_none(self):
        """Test a response reporting no reasoning yields None."""
        assert reported_reasoning({"usage": {}}) is None

    def test_non_integer_value_treated_as_absent(self):
        """Test a non-numeric attribute does not count as a reported figure."""
        assert reported_reasoning(Mock()) is None

    def test_bool_is_not_a_count(self):
        """Test a bool is rejected even though it subclasses int."""
        assert reported_reasoning({"usage_metadata": {"thoughts_token_count": True}}) is None

    def test_unparseable_string_returns_none(self):
        """Test a string that is not JSON yields None."""
        assert reported_reasoning("not json") is None

    def test_none_response_returns_none(self):
        """Test a missing raw response yields None."""
        assert reported_reasoning(None) is None


class TestUncountedReasoning:
    """Tests for uncounted_reasoning."""

    def test_genai_invariant_holds(self):
        """Test input + output + reasoning equals total for a genai payload."""
        reasoning = uncounted_reasoning(12, 18, 60, GENAI)

        assert reasoning == 30
        assert 12 + 18 + reasoning == 60

    def test_xai_invariant_holds(self):
        """Test input + output + reasoning equals total for an x-ai payload."""
        raw = {"usage": {"completion_tokens_details": {"reasoning_tokens": 40}}}
        reasoning = uncounted_reasoning(10, 20, 70, raw)

        assert reasoning == 40
        assert 10 + 20 + reasoning == 70

    def test_reported_inside_output_tokens_clamps_to_zero(self):
        """Test the openai family, which counts reasoning inside output_tokens, yields 0."""
        assert uncounted_reasoning(10, 20, 30, OPENAI_CHAT) == 0
        assert uncounted_reasoning(10, 20, 30, OPENAI_RESPONSES) == 0

    def test_falls_back_to_gap_when_nothing_reported(self):
        """Test the gap is used when the provider reports no reasoning count."""
        assert uncounted_reasoning(10, 20, 55, {"usage": {}}) == 25

    def test_unknown_total_returns_none(self):
        """Test a missing total yields None rather than 0."""
        assert uncounted_reasoning(10, 20, None, GENAI) is None

    def test_zero_total_returns_none(self):
        """Test a zero total yields None rather than 0."""
        assert uncounted_reasoning(10, 20, 0, GENAI) is None

    def test_negative_gap_clamps_to_zero(self):
        """Test inconsistent provider numbers yield 0, never a negative count."""
        assert uncounted_reasoning(100, 100, 150, {"usage": {}}) == 0

    def test_mismatch_logs_warning(self, caplog):
        """Test a reported count disagreeing with the gap is logged loudly."""
        raw = {"usage_metadata": {"thoughts_token_count": 25}}

        with caplog.at_level(logging.WARNING, logger="ai_client.reasoning"):
            reasoning = uncounted_reasoning(12, 18, 60, raw, provider="genai", model="x")

        assert reasoning == 25
        assert "Reasoning token mismatch" in caplog.text
        assert "genai" in caplog.text

    def test_reported_against_zero_gap_does_not_warn(self, caplog):
        """Test the normal openai case is not reported as a disagreement."""
        with caplog.at_level(logging.WARNING, logger="ai_client.reasoning"):
            reasoning = uncounted_reasoning(10, 20, 30, OPENAI_CHAT, provider="openai")

        assert reasoning == 0
        assert "Reasoning token mismatch" not in caplog.text
