"""
Tool calling support for ai_client.

This module provides a pluggable architecture for executing tools defined in a registry.
Tools can be Python functions, REST APIs, or MCP servers.
"""

from .registry import ToolRegistry
from .executor_base import ToolExecutor

__all__ = ["ToolRegistry", "ToolExecutor"]
