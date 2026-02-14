"""
Python function executor for tool calling.

This executor loads and executes Python functions defined in the tool registry.
"""

import importlib
import json
from typing import Dict, Any
from .executor_base import ToolExecutor


class PythonFunctionExecutor(ToolExecutor):
    """Execute Python functions as tools."""

    def __init__(self, tool_definition: Dict[str, Any]):
        """
        Initialize Python function executor.

        Args:
            tool_definition: Tool definition with 'module' and 'function' fields

        Raises:
            ToolExecutionError: If function cannot be loaded
        """
        super().__init__(tool_definition)

        # Load function on initialization
        module_name = tool_definition.get("module")
        function_name = tool_definition.get("function")

        if not module_name or not function_name:
            from ..utils import ToolExecutionError

            raise ToolExecutionError(
                f"Python function executor requires 'module' and 'function' fields"
            )

        try:
            module = importlib.import_module(module_name)
            self.function = getattr(module, function_name)
        except (ImportError, AttributeError) as e:
            from ..utils import ToolExecutionError

            raise ToolExecutionError(f"Failed to load tool function: {e}")

    def execute(self, arguments: Dict[str, Any]) -> str:
        """
        Execute Python function with arguments.

        Args:
            arguments: Tool input parameters as dict

        Returns:
            Tool result as JSON string

        Raises:
            ToolExecutionError: If execution fails
        """
        try:
            result = self.function(**arguments)

            # Convert result to JSON string for consistency
            if isinstance(result, str):
                return result
            else:
                return json.dumps(result, indent=2)

        except Exception as e:
            from ..utils import ToolExecutionError

            raise ToolExecutionError(f"Tool execution failed: {e}")
