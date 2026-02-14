"""
GeoNames search tool implementation.

This tool searches for geographical locations using the geonamescache package.
Requires: pip install geonamescache
"""

import unicodedata
from typing import Dict, Any


def normalize_text(text: str) -> str:
    """
    Normalize text by removing accents/diacritics for fuzzy matching.

    Examples:
        "Zürich" -> "zurich"
        "São Paulo" -> "sao paulo"
    """
    # NFD = decompose accented chars into base + combining chars
    # Then filter out combining characters (category Mn)
    nfd = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in nfd if unicodedata.category(char) != "Mn")


def search_geonames(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search for geographical locations by name.

    Args:
        query: Location name to search for
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Dictionary with search results containing:
        - query: The search query
        - result_count: Number of results found
        - results: List of matching locations with geonameid, name, country, etc.

    Example:
        >>> result = search_geonames("Paris", max_results=3)
        >>> print(result["result_count"])
        3
        >>> print(result["results"][0]["name"])
        Paris
    """
    try:
        import geonamescache
    except ImportError:
        return {
            "error": "geonamescache not installed. Install with: pip install geonamescache",
            "query": query,
            "result_count": 0,
            "results": [],
        }

    gc = geonamescache.GeonamesCache()
    cities = gc.get_cities()

    # Search by name with exact and partial matching
    # Note: get_cities() already filters to populated places (P class in GeoNames)
    results = []
    query_lower = query.lower().strip()

    # Extract city name if query is in "City, Country" format
    # e.g., "Zurich, Switzerland" -> "zurich"
    query_parts = [part.strip() for part in query_lower.split(",")]
    primary_query = query_parts[0]  # Use the part before the comma

    # Normalize for accent-insensitive matching (e.g., "Zurich" matches "Zürich")
    query_normalized = normalize_text(query_lower)
    primary_normalized = normalize_text(primary_query)

    for geonameid, city in cities.items():
        city_name_lower = city["name"].lower()
        city_name_normalized = normalize_text(city_name_lower)

        # Calculate match score - check both full query and primary part
        # Use normalized versions for accent-insensitive matching
        if (
            city_name_normalized == query_normalized
            or city_name_normalized == primary_normalized
            or city_name_lower == query_lower
            or city_name_lower == primary_query
        ):
            # Exact match - highest priority
            match_score = 3
        elif (
            city_name_normalized.startswith(query_normalized)
            or city_name_normalized.startswith(primary_normalized)
            or city_name_lower.startswith(query_lower)
            or city_name_lower.startswith(primary_query)
        ):
            # Starts with query - high priority
            match_score = 2
        elif primary_normalized in city_name_normalized:
            # Query is contained in city name (e.g., "Paris" in "Paris 11e Arrondissement")
            match_score = 1
        elif city_name_normalized in query_normalized and " " in city_name_normalized:
            # City name is in query AND city name contains a space (multi-word)
            # e.g., "Lake Zurich" in "zurich" won't match, but "New York" would
            # This prevents single-word substrings like "Rich" matching "Zurich"
            match_score = 1
        else:
            continue

        results.append(
            {
                "geonameid": geonameid,
                "name": city["name"],
                "country": city["countrycode"],
                "population": city["population"],
                "latitude": city["latitude"],
                "longitude": city["longitude"],
                "feature_class": "P",  # Populated place
                "match_score": match_score,
            }
        )

    # Sort by match score (descending), then by population (descending)
    results.sort(key=lambda x: (x["match_score"], x["population"]), reverse=True)

    # Remove match_score from results (internal use only)
    for r in results:
        del r["match_score"]

    # Limit to max_results
    results = results[:max_results]

    return {"query": query, "result_count": len(results), "results": results}
