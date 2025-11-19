"""
Unit tests for Act Module (Tool Executor).

Tests cover:
- Successful tool execution
- Error handling with retries
- Timeout handling
- Parameter validation
- Token estimation
- Tool registration (sync and async)
- Multiple concurrent executions
"""

import asyncio
import json
import time
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from nocp.core.act import ToolExecutor
from nocp.exceptions import ToolExecutionError
from nocp.models.contracts import (
    ToolRequest,
    ToolResult,
    ToolType,
    RetryConfig,
)


class TestToolExecutor:
    """Tests for ToolExecutor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.executor = ToolExecutor()

    def test_executor_initialization(self):
        """Test that executor initializes with empty registries."""
        executor = ToolExecutor()
        # Test through public API only
        assert executor.list_tools() == []
        assert not executor.validate_tool("nonexistent_tool")

    def test_register_tool_decorator(self):
        """Test tool registration using decorator."""
        @self.executor.register_tool("test_tool")
        def test_func(param: str) -> str:
            return f"Result: {param}"

        assert "test_tool" in self.executor.list_tools()
        assert self.executor.validate_tool("test_tool")

    def test_register_multiple_tools(self):
        """Test registering multiple tools."""
        @self.executor.register_tool("tool1")
        def func1(x: int) -> int:
            return x * 2

        @self.executor.register_tool("tool2")
        def func2(x: int) -> int:
            return x + 10

        assert len(self.executor.list_tools()) == 2
        assert "tool1" in self.executor.list_tools()
        assert "tool2" in self.executor.list_tools()

    def test_successful_execution(self):
        """Test successful tool execution."""
        @self.executor.register_tool("fetch_data")
        def fetch_data(user_id: str) -> dict:
            return {"user_id": user_id, "name": "John Doe"}

        request = ToolRequest(
            tool_id="fetch_data",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="fetch_data",
            parameters={"user_id": "123"}
        )

        result = self.executor.execute(request)

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.tool_id == "fetch_data"
        assert result.data == {"user_id": "123", "name": "John Doe"}
        assert result.error is None
        assert result.execution_time_ms >= 0
        assert isinstance(result.timestamp, datetime)
        assert result.token_estimate > 0
        assert result.retry_count == 0

    def test_execution_with_no_parameters(self):
        """Test tool execution with no parameters."""
        @self.executor.register_tool("get_timestamp")
        def get_timestamp() -> str:
            return "2024-01-01"

        request = ToolRequest(
            tool_id="get_timestamp",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="get_timestamp",
            parameters={}
        )

        result = self.executor.execute(request)

        assert result.success is True
        assert result.data == "2024-01-01"

    def test_execution_with_multiple_parameters(self):
        """Test tool execution with multiple parameters."""
        @self.executor.register_tool("calculate")
        def calculate(a: int, b: int, operation: str) -> int:
            if operation == "add":
                return a + b
            elif operation == "multiply":
                return a * b
            return 0

        request = ToolRequest(
            tool_id="calculate",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="calculate",
            parameters={"a": 5, "b": 3, "operation": "multiply"}
        )

        result = self.executor.execute(request)

        assert result.success is True
        assert result.data == 15

    def test_tool_not_found_error(self):
        """Test that missing tool raises ToolExecutionError."""
        request = ToolRequest(
            tool_id="nonexistent_tool",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="nonexistent",
            parameters={}
        )

        with pytest.raises(ToolExecutionError) as exc_info:
            self.executor.execute(request)

        assert "not found in registry" in str(exc_info.value)
        assert exc_info.value.details["tool_id"] == "nonexistent_tool"

    def test_execution_error_without_retry(self):
        """Test that tool execution error is raised without retry."""
        @self.executor.register_tool("failing_tool")
        def failing_tool() -> str:
            raise ValueError("Tool failed")

        request = ToolRequest(
            tool_id="failing_tool",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="failing_tool",
            parameters={},
            retry_config=None  # No retry
        )

        with pytest.raises(ToolExecutionError) as exc_info:
            self.executor.execute(request)

        assert "failed after 1 attempts" in str(exc_info.value)
        assert "Tool failed" in str(exc_info.value.details["last_error"])

    def test_execution_with_retry_eventually_succeeds(self):
        """Test retry logic that eventually succeeds."""
        attempt_count = 0

        @self.executor.register_tool("flaky_tool")
        def flaky_tool() -> str:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError(f"Attempt {attempt_count} failed")
            return "success"

        request = ToolRequest(
            tool_id="flaky_tool",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="flaky_tool",
            parameters={},
            retry_config=RetryConfig(
                max_attempts=3,
                backoff_multiplier=1.5,
                initial_delay_ms=10  # Short delay for tests
            )
        )

        result = self.executor.execute(request)

        assert result.success is True
        assert result.data == "success"
        assert result.retry_count == 2  # Succeeded on third attempt (retry_count=2)
        assert attempt_count == 3

    def test_execution_with_retry_exhausted(self):
        """Test that all retries are exhausted on persistent failure."""
        attempt_count = 0

        @self.executor.register_tool("always_failing")
        def always_failing() -> str:
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError(f"Attempt {attempt_count} failed")

        request = ToolRequest(
            tool_id="always_failing",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="always_failing",
            parameters={},
            retry_config=RetryConfig(
                max_attempts=3,
                backoff_multiplier=1.5,
                initial_delay_ms=10
            )
        )

        with pytest.raises(ToolExecutionError) as exc_info:
            self.executor.execute(request)

        assert "failed after 3 attempts" in str(exc_info.value)
        assert attempt_count == 3

    def test_timeout_handling(self):
        """Test timeout handling for slow tools."""
        @self.executor.register_tool("slow_tool")
        def slow_tool() -> str:
            time.sleep(2)  # Sleep for 2 seconds
            return "completed"

        request = ToolRequest(
            tool_id="slow_tool",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="slow_tool",
            parameters={},
            timeout_seconds=1  # 1 second timeout
        )

        with pytest.raises(TimeoutError) as exc_info:
            self.executor.execute(request)

        assert "exceeded 1s timeout" in str(exc_info.value)

    def test_timeout_with_retry(self):
        """Test that timeout triggers retry logic."""
        attempt_count = 0

        @self.executor.register_tool("timeout_tool")
        def timeout_tool() -> str:
            nonlocal attempt_count
            attempt_count += 1
            # Always sleep longer than timeout to ensure all attempts timeout
            time.sleep(2)
            return "success"

        request = ToolRequest(
            tool_id="timeout_tool",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="timeout_tool",
            parameters={},
            timeout_seconds=1,
            retry_config=RetryConfig(
                max_attempts=2,
                backoff_multiplier=1.0,
                initial_delay_ms=10
            )
        )

        # This should timeout on all attempts
        with pytest.raises(TimeoutError):
            self.executor.execute(request)

        assert attempt_count == 2

    def test_token_estimation_for_string(self):
        """Test token estimation for string results."""
        @self.executor.register_tool("get_text")
        def get_text() -> str:
            # Approximately 20 chars = ~5 tokens
            return "Hello, world! Test."

        request = ToolRequest(
            tool_id="get_text",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="get_text",
            parameters={}
        )

        result = self.executor.execute(request)

        assert result.token_estimate > 0
        assert result.token_estimate == len("Hello, world! Test.") // 4

    def test_token_estimation_for_dict(self):
        """Test token estimation for dictionary results."""
        @self.executor.register_tool("get_dict")
        def get_dict() -> dict:
            return {"key1": "value1", "key2": "value2"}

        request = ToolRequest(
            tool_id="get_dict",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="get_dict",
            parameters={}
        )

        result = self.executor.execute(request)

        # Should estimate based on JSON string length
        expected_json = json.dumps({"key1": "value1", "key2": "value2"})
        assert result.token_estimate == len(expected_json) // 4

    def test_token_estimation_for_list(self):
        """Test token estimation for list results."""
        @self.executor.register_tool("get_list")
        def get_list() -> list:
            return [1, 2, 3, 4, 5]

        request = ToolRequest(
            tool_id="get_list",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="get_list",
            parameters={}
        )

        result = self.executor.execute(request)

        expected_json = json.dumps([1, 2, 3, 4, 5])
        assert result.token_estimate == len(expected_json) // 4

    def test_execution_timing(self):
        """Test that execution time is accurately measured."""
        @self.executor.register_tool("timed_tool")
        def timed_tool() -> str:
            time.sleep(0.1)  # Sleep for 100ms
            return "done"

        request = ToolRequest(
            tool_id="timed_tool",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="timed_tool",
            parameters={}
        )

        result = self.executor.execute(request)

        # Execution time should be at least 100ms
        assert result.execution_time_ms >= 100
        # But not unreasonably high (allow 50ms overhead)
        assert result.execution_time_ms < 200

    def test_validate_tool(self):
        """Test tool validation method."""
        @self.executor.register_tool("valid_tool")
        def valid_tool() -> str:
            return "ok"

        assert self.executor.validate_tool("valid_tool") is True
        assert self.executor.validate_tool("invalid_tool") is False

    def test_list_tools(self):
        """Test listing all registered tools."""
        @self.executor.register_tool("tool_a")
        def tool_a() -> str:
            return "a"

        @self.executor.register_tool("tool_b")
        def tool_b() -> str:
            return "b"

        tools = self.executor.list_tools()
        assert len(tools) == 2
        assert "tool_a" in tools
        assert "tool_b" in tools

    def test_execution_metadata_completeness(self):
        """Test that ToolResult contains all required metadata."""
        @self.executor.register_tool("metadata_tool")
        def metadata_tool(param: str) -> dict:
            return {"result": param}

        request = ToolRequest(
            tool_id="metadata_tool",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="metadata_tool",
            parameters={"param": "test"}
        )

        result = self.executor.execute(request)

        # Verify all required fields are present
        assert hasattr(result, 'tool_id')
        assert hasattr(result, 'success')
        assert hasattr(result, 'data')
        assert hasattr(result, 'error')
        assert hasattr(result, 'execution_time_ms')
        assert hasattr(result, 'timestamp')
        assert hasattr(result, 'token_estimate')
        assert hasattr(result, 'retry_count')
        assert hasattr(result, 'metadata')

    def test_tool_result_to_text(self):
        """Test ToolResult.to_text() conversion."""
        @self.executor.register_tool("text_tool")
        def text_tool() -> dict:
            return {"name": "Alice", "age": 30}

        request = ToolRequest(
            tool_id="text_tool",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="text_tool",
            parameters={}
        )

        result = self.executor.execute(request)
        text = result.to_text()

        # Should be valid JSON
        assert isinstance(text, str)
        parsed = json.loads(text)
        assert parsed == {"name": "Alice", "age": 30}


class TestAsyncToolExecutor:
    """Tests for async tool execution."""

    def setup_method(self):
        """Set up test fixtures."""
        self.executor = ToolExecutor()

    @pytest.mark.asyncio
    async def test_register_async_tool(self):
        """Test async tool registration."""
        @self.executor.register_async_tool("async_tool")
        async def async_tool(param: str) -> str:
            await asyncio.sleep(0.01)
            return f"Async result: {param}"

        assert self.executor.validate_tool("async_tool")

    @pytest.mark.asyncio
    async def test_async_execution_success(self):
        """Test successful async tool execution."""
        @self.executor.register_async_tool("async_fetch")
        async def async_fetch(user_id: str) -> dict:
            await asyncio.sleep(0.01)
            return {"user_id": user_id, "name": "Jane Doe"}

        request = ToolRequest(
            tool_id="async_fetch",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="async_fetch",
            parameters={"user_id": "456"}
        )

        result = await self.executor.execute_async(request)

        assert result.success is True
        assert result.data == {"user_id": "456", "name": "Jane Doe"}
        assert result.error is None

    @pytest.mark.asyncio
    async def test_async_tool_not_found(self):
        """Test that missing async tool raises error."""
        request = ToolRequest(
            tool_id="missing_async_tool",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="missing",
            parameters={}
        )

        with pytest.raises(ToolExecutionError) as exc_info:
            await self.executor.execute_async(request)

        assert "not found in registry" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_execution_with_retry(self):
        """Test async retry logic."""
        attempt_count = 0

        @self.executor.register_async_tool("flaky_async")
        async def flaky_async() -> str:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ValueError(f"Async attempt {attempt_count} failed")
            return "async success"

        request = ToolRequest(
            tool_id="flaky_async",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="flaky_async",
            parameters={},
            retry_config=RetryConfig(
                max_attempts=3,
                backoff_multiplier=1.5,
                initial_delay_ms=10
            )
        )

        result = await self.executor.execute_async(request)

        assert result.success is True
        assert result.data == "async success"
        assert result.retry_count == 1  # Succeeded on second attempt
        assert attempt_count == 2

    @pytest.mark.asyncio
    async def test_async_timeout(self):
        """Test async timeout handling."""
        @self.executor.register_async_tool("slow_async")
        async def slow_async() -> str:
            await asyncio.sleep(2)
            return "completed"

        request = ToolRequest(
            tool_id="slow_async",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="slow_async",
            parameters={},
            timeout_seconds=1
        )

        with pytest.raises(TimeoutError) as exc_info:
            await self.executor.execute_async(request)

        assert "exceeded 1s timeout" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_execution_timing(self):
        """Test that async execution time is measured."""
        @self.executor.register_async_tool("timed_async")
        async def timed_async() -> str:
            await asyncio.sleep(0.1)
            return "done"

        request = ToolRequest(
            tool_id="timed_async",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="timed_async",
            parameters={}
        )

        result = await self.executor.execute_async(request)

        assert result.execution_time_ms >= 100
        assert result.execution_time_ms < 200


class TestToolExecutorEdgeCases:
    """Tests for edge cases and error scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.executor = ToolExecutor()

    def test_tool_returns_none(self):
        """Test tool that returns None."""
        @self.executor.register_tool("none_tool")
        def none_tool() -> None:
            return None

        request = ToolRequest(
            tool_id="none_tool",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="none_tool",
            parameters={}
        )

        result = self.executor.execute(request)

        assert result.success is True
        assert result.data is None

    def test_tool_with_complex_return_type(self):
        """Test tool with nested complex data structures."""
        @self.executor.register_tool("complex_tool")
        def complex_tool() -> dict:
            return {
                "users": [
                    {"id": 1, "name": "Alice", "roles": ["admin", "user"]},
                    {"id": 2, "name": "Bob", "roles": ["user"]}
                ],
                "metadata": {
                    "total": 2,
                    "timestamp": "2024-01-01T00:00:00Z"
                }
            }

        request = ToolRequest(
            tool_id="complex_tool",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="complex_tool",
            parameters={}
        )

        result = self.executor.execute(request)

        assert result.success is True
        assert len(result.data["users"]) == 2
        assert result.data["metadata"]["total"] == 2

    def test_retry_config_validation(self):
        """Test that RetryConfig validates parameters."""
        # Valid config
        config = RetryConfig(
            max_attempts=3,
            backoff_multiplier=2.0,
            initial_delay_ms=100
        )
        assert config.max_attempts == 3

        # Test max_attempts bounds
        with pytest.raises(ValidationError):
            RetryConfig(max_attempts=0)  # Below minimum

        with pytest.raises(ValidationError):
            RetryConfig(max_attempts=6)  # Above maximum

        # Test other parameter bounds
        with pytest.raises(ValidationError):
            RetryConfig(backoff_multiplier=0.5)  # Below minimum (must be >= 1.0)

        with pytest.raises(ValidationError):
            RetryConfig(initial_delay_ms=5)  # Below minimum (must be >= 10)

    def test_tool_type_enum(self):
        """Test ToolType enum values."""
        assert ToolType.PYTHON_FUNCTION == "python_function"
        assert ToolType.DATABASE_QUERY == "database_query"
        assert ToolType.API_CALL == "api_call"
        assert ToolType.RAG_RETRIEVAL == "rag_retrieval"
        assert ToolType.FILE_OPERATION == "file_operation"

    def test_multiple_tools_execution(self):
        """Test that multiple tools can be registered and executed."""
        @self.executor.register_tool("tool1")
        def tool1() -> int:
            return 1

        @self.executor.register_tool("tool2")
        def tool2() -> int:
            return 2

        request1 = ToolRequest(
            tool_id="tool1",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="tool1",
            parameters={}
        )

        request2 = ToolRequest(
            tool_id="tool2",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="tool2",
            parameters={}
        )

        result1 = self.executor.execute(request1)
        result2 = self.executor.execute(request2)

        assert result1.data == 1
        assert result2.data == 2

    def test_exponential_backoff_timing(self):
        """Test that retry backoff follows exponential pattern."""
        attempt_times = []

        @self.executor.register_tool("backoff_tool")
        def backoff_tool() -> str:
            attempt_times.append(time.perf_counter())
            raise ValueError("Always fails")

        request = ToolRequest(
            tool_id="backoff_tool",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="backoff_tool",
            parameters={},
            retry_config=RetryConfig(
                max_attempts=3,
                backoff_multiplier=2.0,
                initial_delay_ms=100
            )
        )

        with pytest.raises(ToolExecutionError):
            self.executor.execute(request)

        # Check that we had 3 attempts
        assert len(attempt_times) == 3

        # Check backoff delays (with some tolerance)
        # First retry: ~100ms delay
        # Second retry: ~200ms delay
        delay1 = (attempt_times[1] - attempt_times[0]) * 1000
        assert 80 <= delay1 <= 150  # Allow some tolerance

        delay2 = (attempt_times[2] - attempt_times[1]) * 1000
        assert 180 <= delay2 <= 250  # ~200ms with tolerance
