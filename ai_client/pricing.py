"""
Pricing information for LLM API providers.

This module handles loading and querying pricing data from a JSON file
to automatically calculate estimated costs for API requests.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


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
            "qwen": "qwen",
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
