"""
Tool Executor (Act Module) - External function/tool execution.

Handles execution of external tools such as database queries, API calls,
RAG pipelines, etc. Returns raw results that may be verbose and require
compression.
"""

import time
import traceback
from collections.abc import Callable
from typing import Any

from ..core.config import get_config
from ..models.schemas import ToolDefinition, ToolExecutionResult
from ..utils.logging import get_logger
from ..utils.token_counter import TokenCounter


class ToolExecutor:
    """
    Tool Executor - The "Act" component of the architecture.

    Responsibilities:
    - Register and manage available tools
    - Execute tool calls with validated Pydantic inputs
    - Track execution time and success/failure
    - Return raw results (potentially verbose) for context management
    """

    def __init__(self):
        """Initialize the tool executor."""
        self.config = get_config()
        self.logger = get_logger(__name__)
        self.token_counter = TokenCounter()

        # Registry of available tools: {tool_name: (definition, callable)}
        self._tools: dict[str, tuple[ToolDefinition, Callable]] = {}

    def register_tool(
        self,
        definition: ToolDefinition,
        callable_func: Callable,
    ) -> None:
        """
        Register a tool for execution.

        Args:
            definition: ToolDefinition describing the tool
            callable_func: Python callable implementing the tool

        Raises:
            ValueError: If tool name is already registered
        """
        if definition.name in self._tools:
            raise ValueError(f"Tool '{definition.name}' is already registered")

        self._tools[definition.name] = (definition, callable_func)

        # Register tool-specific compression threshold if specified
        if definition.compression_threshold is not None:
            self.config.register_tool_threshold(definition.name, definition.compression_threshold)

        self.logger.info(
            "tool_registered",
            tool_name=definition.name,
            param_count=len(definition.parameters),
            custom_threshold=definition.compression_threshold,
        )

    def get_tool_definitions(self) -> list[ToolDefinition]:
        """
        Get all registered tool definitions.

        Returns:
            List of ToolDefinition objects
        """
        return [definition for definition, _ in self._tools.values()]

    def get_tool_definition(self, tool_name: str) -> ToolDefinition | None:
        """
        Get definition for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            ToolDefinition if found, None otherwise
        """
        tool_data = self._tools.get(tool_name)
        return tool_data[0] if tool_data else None

    def execute_tool(
        self,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> ToolExecutionResult:
        """
        Execute a registered tool with given parameters.

        Args:
            tool_name: Name of the tool to execute
            parameters: Dictionary of parameters for the tool

        Returns:
            ToolExecutionResult containing raw output and metadata

        Raises:
            ValueError: If tool is not registered
        """
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' is not registered")

        definition, callable_func = self._tools[tool_name]

        self.logger.info(
            "executing_tool",
            tool_name=tool_name,
            parameters=parameters,
        )

        # Track execution time
        start_time = time.perf_counter()

        try:
            # Execute the tool
            raw_output = callable_func(**parameters)

            # Convert output to string if needed
            if not isinstance(raw_output, str):
                raw_output = str(raw_output)

            # Calculate execution time
            execution_time_ms = (time.perf_counter() - start_time) * 1000

            # Count tokens in raw output
            raw_token_count = self.token_counter.count_text(raw_output)

            result = ToolExecutionResult(
                tool_name=tool_name,
                raw_output=raw_output,
                raw_token_count=raw_token_count,
                execution_time_ms=execution_time_ms,
                success=True,
            )

            self.logger.info(
                "tool_executed",
                tool_name=tool_name,
                execution_time_ms=execution_time_ms,
                raw_token_count=raw_token_count,
            )

            return result

        except Exception as e:
            # Handle tool execution errors
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            error_msg = f"{type(e).__name__}: {str(e)}"

            self.logger.error(
                "tool_execution_failed",
                tool_name=tool_name,
                error=error_msg,
                traceback=traceback.format_exc(),
            )

            return ToolExecutionResult(
                tool_name=tool_name,
                raw_output="",
                raw_token_count=0,
                execution_time_ms=execution_time_ms,
                success=False,
                error_message=error_msg,
            )

    def execute_tools_batch(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[ToolExecutionResult]:
        """
        Execute multiple tools in sequence.

        Args:
            tool_calls: List of dicts with 'tool_name' and 'parameters'

        Returns:
            List of ToolExecutionResult objects
        """
        results = []

        for call in tool_calls:
            tool_name = call.get("tool_name")
            parameters = call.get("parameters", {})

            if not tool_name:
                self.logger.warning("batch_call_missing_tool_name", call=call)
                continue

            result = self.execute_tool(tool_name, parameters)
            results.append(result)

        return results

    def validate_tool_parameters(
        self,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """
        Validate parameters against tool definition.

        Args:
            tool_name: Name of the tool
            parameters: Parameters to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        tool_data = self._tools.get(tool_name)
        if not tool_data:
            return False, f"Tool '{tool_name}' not found"

        definition, _ = tool_data

        # Check required parameters
        for param in definition.parameters:
            if param.required and param.name not in parameters:
                return False, f"Missing required parameter: {param.name}"

            # Type validation (basic)
            if param.name in parameters:
                value = parameters[param.name]
                expected_type = param.type

                # Basic type checking
                type_map = {
                    "string": str,
                    "number": (int, float),
                    "boolean": bool,
                    "object": dict,
                    "array": list,
                }

                if expected_type in type_map:
                    expected_python_type = type_map[expected_type]
                    if not isinstance(value, expected_python_type):
                        return False, f"Parameter '{param.name}' has wrong type"

        return True, None

    def get_tool_schemas_for_gemini(self) -> list[dict[str, Any]]:
        """
        Convert tool definitions to Gemini Function Calling format.

        Returns:
            List of function schemas in Gemini format
        """
        schemas = []

        for definition, _ in self._tools.values():
            # Build parameter schema
            parameters_schema = {
                "type": "object",
                "properties": {},
                "required": [],
            }

            for param in definition.parameters:
                param_schema = {
                    "type": param.type,
                    "description": param.description,
                }

                if param.enum:
                    param_schema["enum"] = param.enum

                parameters_schema["properties"][param.name] = param_schema

                if param.required:
                    parameters_schema["required"].append(param.name)

            # Build function schema
            function_schema = {
                "name": definition.name,
                "description": definition.description,
                "parameters": parameters_schema,
            }

            schemas.append(function_schema)

        return schemas
