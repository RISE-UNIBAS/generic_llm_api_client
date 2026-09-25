"""
Tests for cost calculation.
"""

from ai_client.pricing import apply_costs, calculate_cost, calculate_cost_components
from ai_client.response import Usage

# Round numbers so an expected cost is obvious: 1 USD per million in, 10 per million out.
TABLE = {
    "2026-01-01": {
        "genai": {"test-model": {"input_price": 1.0, "output_price": 10.0}},
    }
}
MILLION = 1_000_000


class TestCalculateCostComponents:
    """Tests for calculate_cost_components."""

    def test_reasoning_billed_at_the_output_rate(self, stub_pricing):
        """Test reasoning tokens cost the same as output tokens."""
        stub_pricing(TABLE)

        costs = calculate_cost_components("genai", "test-model", MILLION, MILLION, MILLION)

        assert costs.input_cost_usd == 1.0
        assert costs.output_cost_usd == 10.0
        assert costs.reasoning_cost_usd == 10.0

    def test_total_sums_the_components(self, stub_pricing):
        """Test the total covers input, output and reasoning."""
        stub_pricing(TABLE)

        costs = calculate_cost_components("genai", "test-model", MILLION, MILLION, MILLION)

        assert costs.total_cost_usd == 21.0

    def test_none_reasoning_treated_as_zero(self, stub_pricing):
        """Test an unknown reasoning count does not break the calculation."""
        stub_pricing(TABLE)

        costs = calculate_cost_components("genai", "test-model", MILLION, 0, None)

        assert costs.total_cost_usd == 1.0

    def test_unknown_model_returns_none(self, stub_pricing):
        """Test a model missing from the table yields no costs."""
        stub_pricing(TABLE)

        assert calculate_cost_components("genai", "absent", MILLION, MILLION, 0) is None

    def test_calculate_cost_is_unchanged(self, stub_pricing):
        """Test the old entry point still returns a 3-tuple that excludes reasoning."""
        stub_pricing(TABLE)

        assert calculate_cost("genai", "test-model", MILLION, MILLION) == (1.0, 10.0, 11.0)


class TestApplyCosts:
    """Tests for apply_costs."""

    def test_fills_components_and_total(self, stub_pricing):
        """Test every cost field is populated from list prices."""
        stub_pricing(TABLE)
        usage = Usage(
            input_tokens=MILLION,
            output_tokens=MILLION,
            total_tokens=3 * MILLION,
            reasoning_tokens=MILLION,
        )

        apply_costs(usage, "genai", "test-model")

        assert usage.input_cost_usd == 1.0
        assert usage.output_cost_usd == 10.0
        assert usage.reasoning_cost_usd == 10.0
        assert usage.estimated_cost_usd == 21.0

    def test_components_sum_to_the_total(self, stub_pricing):
        """Test the contract consumers rely on: components add up to the total."""
        stub_pricing(TABLE)
        usage = Usage(
            input_tokens=1234, output_tokens=5678, total_tokens=8000, reasoning_tokens=1088
        )

        apply_costs(usage, "genai", "test-model")
        components = usage.input_cost_usd + usage.output_cost_usd + usage.reasoning_cost_usd

        assert components == usage.estimated_cost_usd

    def test_preserves_a_provider_billed_total(self, stub_pricing):
        """Test a billed total with no components beside it is never recalculated."""
        stub_pricing(TABLE)
        usage = Usage(
            input_tokens=MILLION,
            output_tokens=MILLION,
            total_tokens=2 * MILLION,
            estimated_cost_usd=0.0123,
        )

        apply_costs(usage, "genai", "test-model")

        assert usage.estimated_cost_usd == 0.0123
        assert usage.input_cost_usd is None
        assert usage.output_cost_usd is None

    def test_recalculates_a_total_that_has_components(self, stub_pricing):
        """Test an already-calculated cost is refreshed, unlike a billed one."""
        stub_pricing(TABLE)
        usage = Usage(
            input_tokens=MILLION,
            output_tokens=MILLION,
            total_tokens=2 * MILLION,
            reasoning_tokens=0,
            input_cost_usd=99.0,
            output_cost_usd=99.0,
            estimated_cost_usd=198.0,
        )

        apply_costs(usage, "genai", "test-model")

        assert usage.estimated_cost_usd == 11.0

    def test_unknown_reasoning_leaves_its_cost_none(self, stub_pricing):
        """Test an unpriceable reasoning count is not reported as costing zero."""
        stub_pricing(TABLE)
        usage = Usage(input_tokens=MILLION, output_tokens=MILLION, total_tokens=0)

        apply_costs(usage, "genai", "test-model")

        assert usage.reasoning_tokens is None
        assert usage.reasoning_cost_usd is None
        assert usage.input_cost_usd == 1.0

    def test_leaves_costs_none_when_model_is_unpriced(self, stub_pricing):
        """Test a model missing from the table keeps every cost field None."""
        stub_pricing(TABLE)
        usage = Usage(input_tokens=10, output_tokens=10, total_tokens=20, reasoning_tokens=0)

        apply_costs(usage, "genai", "absent")

        assert usage.input_cost_usd is None
        assert usage.output_cost_usd is None
        assert usage.estimated_cost_usd is None
