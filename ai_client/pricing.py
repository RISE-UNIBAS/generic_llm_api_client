"""
Pricing information for LLM API providers.

This module handles loading and querying pricing data from a JSON file
to automatically calculate estimated costs for API requests.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, NamedTuple, Optional, Tuple

from .utils import billed_components, billed_cost

if TYPE_CHECKING:  # avoids importing a sibling module at runtime
    from .response import Usage

logger = logging.getLogger(__name__)


class CostComponents(NamedTuple):
    """
    Cost of a request split by what was billed, in USD.

    total_cost_usd is the sum of the three components, so consumers can report a total
    without knowing which providers bill reasoning separately.
    """

    input_cost_usd: float
    output_cost_usd: float
    reasoning_cost_usd: float
    total_cost_usd: float


class PricingManager:
    """
    Manages pricing data for LLM providers.

    Loads pricing information from a JSON file and provides methods to
    calculate costs based on token usage.
    """

    def __init__(self, pricing_file: Optional[str] = None):
        """
        Initialize the PricingManager.

        Args:
            pricing_file: Path to the pricing JSON file. If None, looks for
                         'pricing.json' in the same directory as this module.
        """
        self.pricing_data: Dict = {}
        self.pricing_file = pricing_file

        if pricing_file is None:
            # Look for pricing.json in the package directory
            package_dir = Path(__file__).parent
            default_file = package_dir / "pricing.json"
            if default_file.exists():
                self.pricing_file = str(default_file)

        if self.pricing_file:
            self.load_pricing_data()

    def load_pricing_data(self):
        """Load pricing data from the JSON file."""
        try:
            with open(self.pricing_file, "r", encoding="utf-8") as f:
                self.pricing_data = json.load(f)
            logger.info(f"Loaded pricing data from {self.pricing_file}")
        except FileNotFoundError:
            logger.warning(f"Pricing file not found: {self.pricing_file}")
            self.pricing_data = {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse pricing file: {e}")
            self.pricing_data = {}
        except Exception as e:
            logger.error(f"Error loading pricing data: {e}")
            self.pricing_data = {}

    def get_model_pricing(self, provider: str, model: str) -> Optional[Tuple[float, float]]:
        """
        Get pricing information for a specific model.

        Args:
            provider: Provider ID (e.g., 'openai', 'anthropic', 'genai')
            model: Model identifier

        Returns:
            Tuple of (input_price_per_million, output_price_per_million) or None
            if pricing is not available
        """
        logger.debug(f"Looking up pricing for provider='{provider}', model='{model}'")

        if not self.pricing_data or "pricing" not in self.pricing_data:
            logger.debug("No pricing data available")
            return None

        # Get all dates and sort them in descending order (most recent first)
        dates = sorted(self.pricing_data["pricing"].keys(), reverse=True)

        # Try exact match first
        for date in dates:
            date_pricing = self.pricing_data["pricing"][date]
            if provider in date_pricing:
                provider_pricing = date_pricing[provider]
                if model in provider_pricing:
                    model_info = provider_pricing[model]
                    input_price = model_info.get("input_price", 0.0)
                    output_price = model_info.get("output_price", 0.0)
                    logger.debug(
                        f"Found pricing (exact match): input=${input_price}, output=${output_price}"
                    )
                    return (input_price, output_price)

        # If no exact match, try stripping date suffixes
        # Pattern: model-YYYY-MM-DD or model-YYYYMMDD
        import re

        base_model = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)  # Remove -YYYY-MM-DD
        base_model = re.sub(r"-\d{8}$", "", base_model)  # Remove -YYYYMMDD

        if base_model != model:
            logger.debug(f"Trying base model: '{base_model}'")
            for date in dates:
                date_pricing = self.pricing_data["pricing"][date]
                if provider in date_pricing:
                    provider_pricing = date_pricing[provider]
                    if base_model in provider_pricing:
                        model_info = provider_pricing[base_model]
                        input_price = model_info.get("input_price", 0.0)
                        output_price = model_info.get("output_price", 0.0)
                        logger.debug(
                            f"Found pricing (base model match): input=${input_price}, output=${output_price}"
                        )
                        return (input_price, output_price)

        # Not found
        logger.warning(
            f"No pricing found for provider='{provider}', model='{model}' or base model '{base_model}'"
        )
        return None

    def calculate_cost(
        self, provider: str, model: str, input_tokens: int, output_tokens: int
    ) -> Optional[Tuple[float, float, float]]:
        """
        Calculate the estimated cost for a request.

        Args:
            provider: Provider ID
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Tuple of (input_cost, output_cost, total_cost) in USD,
            or None if pricing is not available
        """
        pricing = self.get_model_pricing(provider, model)
        if pricing is None:
            return None

        input_price_per_million, output_price_per_million = pricing

        # Calculate cost (prices are per million tokens)
        input_cost = (input_tokens / 1_000_000) * input_price_per_million
        output_cost = (output_tokens / 1_000_000) * output_price_per_million
        total_cost = input_cost + output_cost

        return (input_cost, output_cost, total_cost)

    def calculate_cost_components(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: Optional[int] = 0,
    ) -> Optional[CostComponents]:
        """
        Calculate the estimated cost for a request, including reasoning tokens.

        Reasoning is charged at the output rate. Google prices its thinking tokens in the
        column labelled "Output price (including thinking tokens)", and solving x-ai's
        reported cost per model-month gives the same rate for reasoning as for output.

        Args:
            provider: Provider ID
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            reasoning_tokens: Reasoning tokens billed outside output_tokens

        Returns:
            CostComponents in USD, or None if pricing is not available
        """
        pricing = self.get_model_pricing(provider, model)
        if pricing is None:
            return None

        input_price_per_million, output_price_per_million = pricing

        # Prices are per million tokens.
        input_cost = ((input_tokens or 0) / 1_000_000) * input_price_per_million
        output_cost = ((output_tokens or 0) / 1_000_000) * output_price_per_million
        reasoning_cost = ((reasoning_tokens or 0) / 1_000_000) * output_price_per_million

        return CostComponents(
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            reasoning_cost_usd=reasoning_cost,
            total_cost_usd=input_cost + output_cost + reasoning_cost,
        )

    def normalize_provider_id(self, provider: str) -> str:
        """
        Normalize provider ID to match pricing data format.

        Args:
            provider: Provider ID from the client

        Returns:
            Normalized provider ID for pricing lookup
        """
        # Map internal provider IDs to pricing data provider names
        provider_map = {
            "openai": "openai",
            "anthropic": "anthropic",
            "genai": "genai",
            "mistral": "mistral",
            "deepseek": "deepseek",
            "alibaba": "alibaba",
            "huggingface": "huggingface",
            "openrouter": "openrouter",
            "scicore": "scicore",
        }
        return provider_map.get(provider, provider)


# Global pricing manager instance
_pricing_manager: Optional[PricingManager] = None


def get_pricing_manager() -> PricingManager:
    """
    Get the global PricingManager instance.

    Returns:
        The global PricingManager instance
    """
    global _pricing_manager
    if _pricing_manager is None:
        _pricing_manager = PricingManager()
    return _pricing_manager


def set_pricing_file(pricing_file: str):
    """
    Set a custom pricing file for the global PricingManager.

    Args:
        pricing_file: Path to the pricing JSON file
    """
    global _pricing_manager
    _pricing_manager = PricingManager(pricing_file)


def calculate_cost(
    provider: str, model: str, input_tokens: int, output_tokens: int
) -> Optional[Tuple[float, float, float]]:
    """
    Calculate the estimated cost for a request using the global PricingManager.

    Args:
        provider: Provider ID
        model: Model identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Tuple of (input_cost, output_cost, total_cost) in USD,
        or None if pricing is not available
    """
    manager = get_pricing_manager()
    normalized_provider = manager.normalize_provider_id(provider)
    return manager.calculate_cost(normalized_provider, model, input_tokens, output_tokens)


def calculate_cost_components(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: Optional[int] = 0,
) -> Optional[CostComponents]:
    """
    Calculate the estimated cost of a request using the global PricingManager.

    Args:
        provider: Provider ID
        model: Model identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        reasoning_tokens: Reasoning tokens billed outside output_tokens

    Returns:
        CostComponents in USD, or None if pricing is not available
    """
    manager = get_pricing_manager()
    normalized_provider = manager.normalize_provider_id(provider)
    return manager.calculate_cost_components(
        normalized_provider, model, input_tokens, output_tokens, reasoning_tokens
    )


def record_costs(usage: "Usage", source, provider: str, model: str) -> None:
    """
    Fill a Usage object's cost fields, preferring what the provider billed, in place.

    A billed total is kept as reported and never repriced: OpenRouter routes are unpinned,
    one model name maps to many backends at different prices, so the provider's own figure
    is the only accurate one and a list price would replace a fact with an estimate. Its
    input and output parts are recorded only where the provider itemised them exactly.
    Without a billed total, costs come from list prices.

    Args:
        usage: Usage to fill; reasoning_tokens must already be set
        source: The provider's usage object or mapping, which may carry a billed cost
        provider: Provider ID, used for the pricing lookup
        model: Model identifier as requested, not as the response echoed it
    """
    billed = billed_cost(source)
    if billed is None:
        apply_costs(usage, provider, model)
        return

    # apply_costs is deliberately not called here: it would reprice a total that has
    # components beside it. Components already on the usage are cleared first, so an
    # earlier estimate never sits beside the billed total.
    usage.estimated_cost_usd = billed
    usage.input_cost_usd = usage.output_cost_usd = usage.reasoning_cost_usd = None
    components = billed_components(source)
    if components is not None:
        usage.input_cost_usd, usage.output_cost_usd = components


def apply_costs(usage: "Usage", provider: str, model: str) -> None:
    """
    Fill a Usage object's cost fields from list prices, in place.

    A total already present with no component costs beside it is taken to be one the
    provider billed and is left alone. record_costs is the entry point that handles
    billed costs, including itemised ones; this check keeps a direct caller from
    overwriting a billed total.

    Costs stay None where the model is not in the pricing table, and reasoning cost stays
    None where the reasoning count itself is unknown.

    Args:
        usage: Usage to fill; reasoning_tokens must already be set
        provider: Provider ID, used for the pricing lookup
        model: Model identifier as requested, not as the response echoed it
    """
    if (
        usage.estimated_cost_usd is not None
        and usage.input_cost_usd is None
        and usage.output_cost_usd is None
    ):
        return

    costs = calculate_cost_components(
        provider, model, usage.input_tokens, usage.output_tokens, usage.reasoning_tokens
    )
    if costs is None:
        return

    usage.input_cost_usd = costs.input_cost_usd
    usage.output_cost_usd = costs.output_cost_usd
    usage.reasoning_cost_usd = (
        costs.reasoning_cost_usd if usage.reasoning_tokens is not None else None
    )
    usage.estimated_cost_usd = costs.total_cost_usd
