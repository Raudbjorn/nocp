"""
Pytest configuration and shared fixtures.
"""

import pytest
from datetime import datetime

from nocp.core import ToolExecutor, ContextManager, OutputSerializer
from nocp.models.contracts import ToolResult, ToolType


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (>1 second)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --run-slow is specified."""
    if not config.getoption("--run-slow", default=False):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


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
    return ContextManager(
        compression_threshold=1000,
        enable_litellm=False  # Disable for testing
    )


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
        token_estimate=50
    )
