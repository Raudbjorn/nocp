"""
Pytest configuration and shared fixtures.
"""

from datetime import datetime

import pytest
from nocp.core import ContextManager, OutputSerializer, ToolExecutor
from nocp.models.contracts import ToolResult


@pytest.fixture
def tool_executor():
    """Create a ToolExecutor with sample tools."""
    executor = ToolExecutor()

    @executor.register_tool("echo")
    def echo(message: str) -> str:
        return message

    @executor.register_tool("uppercase")
    def uppercase(text: str) -> str:
        return text.upper()

    @executor.register_tool("fetch_users")
    def fetch_users(count: int = 10) -> list:
        return [
            {"id": str(i), "name": f"User{i}", "email": f"user{i}@example.com"}
            for i in range(count)
        ]

    return executor


@pytest.fixture
def context_manager():
    """Create a ContextManager for testing."""
    return ContextManager(compression_threshold=1000, enable_litellm=False)  # Disable for testing


@pytest.fixture
def output_serializer():
    """Create an OutputSerializer for testing."""
    return OutputSerializer()


@pytest.fixture
def sample_tool_result():
    """Create a sample ToolResult for testing."""
    return ToolResult(
        tool_id="test_tool",
        success=True,
        data={"result": "test data"},
        error=None,
        execution_time_ms=10.0,
        timestamp=datetime.now(),
        token_estimate=50,
    )
