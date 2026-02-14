"""
Abstract base class for tool executors.

This module defines the executor interface that all tool execution backends
must implement (Python functions, REST APIs, MCP servers, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class ToolExecutor(ABC):
    """Abstract base class for tool executors."""

    def __init__(self, tool_definition: Dict[str, Any]):
        """
        Initialize executor with tool definition.

        Args:
            tool_definition: Tool definition dict from registry
        """
        self.tool_def = tool_definition

    @abstractmethod
    def execute(self, arguments: Dict[str, Any]) -> str:
        """
        Execute tool with given arguments.

        Args:
            arguments: Tool input parameters as dict

        Returns:
            Tool result as string

        Raises:
            ToolExecutionError: If execution fails
        """
        pass

    @staticmethod
    def create(tool_definition: Dict[str, Any]) -> "ToolExecutor":
        """
        Factory method to create appropriate executor based on tool type.

        Args:
            tool_definition: Tool definition dict with 'executor' field

        Returns:
            Appropriate ToolExecutor subclass instance

        Raises:
            ValueError: If executor type is unknown
        """
        executor_type = tool_definition.get("executor")

        if executor_type == "python_function":
            from .executor_python import PythonFunctionExecutor

            return PythonFunctionExecutor(tool_definition)
        elif executor_type == "rest_api":
            # Future implementation
            raise NotImplementedError("REST API executor not yet implemented")
        elif executor_type == "mcp":
            # Future implementation
            raise NotImplementedError("MCP executor not yet implemented")
        else:
            raise ValueError(f"Unknown executor type: {executor_type}")
