"""
Pytest configuration and shared fixtures.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest
from nocp.core import ContextManager, OutputSerializer, ToolExecutor
from nocp.core.config import ProxyConfig
from nocp.models.contracts import ContextData, ToolResult


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
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        token_estimate=50,
    )


# ============================================================================
# Enhanced Test Fixtures (D5.3.3)
# ============================================================================


@pytest.fixture
def temp_dir():
    """Temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_metrics_file(temp_dir):
    """Temporary metrics file"""
    metrics_file = temp_dir / "metrics.jsonl"
    yield metrics_file
    # Cleanup handled by temp_dir


@pytest.fixture
def sample_tool_output():
    """Sample tool output for testing"""
    return {
        "users": [
            {"id": f"user_{i}", "name": f"User {i}", "email": f"user{i}@example.com"}
            for i in range(10)
        ],
        "total": 10,
        "page": 1,
    }


@pytest.fixture
def large_tool_output():
    """Large tool output for compression testing"""
    return {
        "records": [
            {
                "id": f"rec_{i}",
                "data": {
                    "field1": f"value_{i}",
                    "field2": i * 100,
                    "field3": ["item1", "item2", "item3"],
                },
                "metadata": {"created": "2024-01-01T00:00:00Z", "updated": "2024-01-01T00:00:00Z"},
            }
            for i in range(100)
        ]
    }


@pytest.fixture
def mock_gemini_response(mocker):
    """Mock Gemini API response"""
    mock_response = mocker.Mock()
    mock_response.text = "Mocked LLM response"
    mock_response.usage_metadata.prompt_token_count = 1000
    mock_response.usage_metadata.candidates_token_count = 50
    return mock_response


@pytest.fixture
def mock_litellm_response(mocker):
    """Mock LiteLLM API response"""
    mock_response = mocker.Mock()
    mock_response.choices = [mocker.Mock()]
    mock_response.choices[0].message.content = "Mocked response"
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage.prompt_tokens = 1000
    mock_response.usage.completion_tokens = 50
    return mock_response


@pytest.fixture
def config_minimal():
    """Minimal valid configuration"""
    return ProxyConfig(gemini_api_key="test-key")


@pytest.fixture
def config_all_features():
    """Configuration with all features enabled"""
    return ProxyConfig(
        gemini_api_key="test-key",
        enable_semantic_pruning=True,
        enable_knowledge_distillation=True,
        enable_history_compaction=True,
        enable_format_negotiation=True,
        default_compression_threshold=1000,
    )


@pytest.fixture
def sample_context_data():
    """Sample ContextData for testing"""
    return ContextData(
        tool_results=[
            ToolResult(
                tool_id="test_tool",
                success=True,
                data={"result": "test"},
                error=None,
                execution_time_ms=100.0,
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                token_estimate=50,
            )
        ],
        transient_context={"query": "test query"},
        max_tokens=10000,
    )
