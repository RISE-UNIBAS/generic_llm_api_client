"""
Wikidata search tool implementation.

This tool searches Wikidata for entities by name.
Uses the Wikidata API (no authentication required).
"""

import requests
from typing import Dict, Any, List


def search_wikidata(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search for entities in Wikidata by name.

    Args:
        query: Entity name to search for
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Dictionary with search results containing:
        - query: The search query
        - result_count: Number of results found
        - results: List of matching entities with ID, label, description, etc.

    Example:
        >>> result = search_wikidata("Paris", max_results=3)
        >>> print(result["result_count"])
        3
        >>> print(result["results"][0]["label"])
        Paris
    """
    try:
        # Wikidata API endpoint
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "search": query,
            "language": "en",
            "limit": max_results,
            "format": "json",
        }

        # Wikidata requires a proper User-Agent header
        # See: https://meta.wikimedia.org/wiki/User-Agent_policy
        headers = {
            "User-Agent": "ai_client/1.0 (https://github.com/your-repo; contact@example.com) python-requests"
        }

        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("search", []):
            results.append(
                {
                    "wikidata_id": item.get("id"),
                    "label": item.get("label"),
                    "description": item.get("description", ""),
                    "url": item.get("concepturi", ""),
                }
            )

        return {"query": query, "result_count": len(results), "results": results}

    except requests.exceptions.RequestException as e:
        return {
            "error": f"Failed to query Wikidata API: {str(e)}",
            "query": query,
            "result_count": 0,
            "results": [],
        }
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}",
            "query": query,
            "result_count": 0,
            "results": [],
        }
