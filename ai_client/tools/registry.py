"""
Tool registry management for ai_client.

This module handles loading and managing tool definitions from a JSON registry file.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List


class ToolRegistry:
    """Manages tool definitions from registry file."""

    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize registry.

        Args:
            registry_path: Path to tools.json file. If None, uses default.

        Raises:
            ToolRegistryError: If registry cannot be loaded
        """
        if registry_path is None:
            # Check environment variable
            registry_path = os.environ.get("AI_CLIENT_TOOLS_REGISTRY")

        if registry_path is None:
            # Use default registry
            registry_path = Path(__file__).parent / "default_tools.json"

        self.registry_path = Path(registry_path)
        self.tools = self._load_registry()

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """
        Load tools from JSON file.

        Returns:
            Dictionary mapping tool names to definitions

        Raises:
            ToolRegistryError: If registry cannot be loaded
        """
        try:
            with open(self.registry_path, "r") as f:
                data = json.load(f)
                return data.get("tools", {})
        except FileNotFoundError:
            from ..utils import ToolRegistryError

            raise ToolRegistryError(f"Tool registry not found: {self.registry_path}")
        except json.JSONDecodeError as e:
            from ..utils import ToolRegistryError

            raise ToolRegistryError(f"Invalid JSON in registry: {e}")

    def get_tool(self, name: str) -> Dict[str, Any]:
        """
        Get tool definition by name.

        Args:
            name: Tool name from registry

        Returns:
            Tool definition dictionary

        Raises:
            ToolRegistryError: If tool not found
        """
        if name not in self.tools:
            from ..utils import ToolRegistryError

            raise ToolRegistryError(
                f"Tool '{name}' not found in registry. "
                f"Available tools: {', '.join(self.list_tools())}"
            )
        return self.tools[name]

    def list_tools(self) -> List[str]:
        """
        List all available tool names.

        Returns:
            List of tool names
        """
        return list(self.tools.keys())
