# Testing Strategy

## 1. Testing Philosophy

### 1.1 Test Pyramid

```
                  ▲
                 ╱ ╲
                ╱ E2E╲         10% - End-to-End Tests
               ╱───────╲           (Full pipeline validation)
              ╱         ╲
             ╱Integration╲    30% - Integration Tests
            ╱─────────────╲       (Module interactions)
           ╱               ╲
          ╱   Unit Tests    ╲ 60% - Unit Tests
         ╱___________________╲    (Individual functions/classes)
```

### 1.2 Coverage Goals
- **Minimum Coverage**: 85%
- **Core Modules**: 95% (Act, Assess, Articulate)
- **Integration Points**: 100% (error paths, fallbacks)

---

## 2. Unit Testing

### 2.1 Act Module Tests

**File**: `tests/core/test_act.py`

```python
import pytest
import time
from nocp.core.act import ToolExecutor, ToolRequest, ToolType, RetryConfig
from nocp.models.contracts import ToolResult

class TestToolExecutor:
    """Unit tests for ToolExecutor."""

    @pytest.fixture
    def executor(self):
        """Create executor with test tools."""
        executor = ToolExecutor()

        @executor.register_tool("echo")
        def echo(message: str) -> str:
            return message

        @executor.register_tool("slow_tool")
        def slow_tool(delay: float) -> str:
            time.sleep(delay)
            return "completed"

        @executor.register_tool("failing_tool")
        def failing_tool(fail_count: int) -> str:
            # Global counter to track retries
            if not hasattr(failing_tool, '_attempts'):
                failing_tool._attempts = 0
            failing_tool._attempts += 1

            if failing_tool._attempts <= fail_count:
                raise ValueError(f"Attempt {failing_tool._attempts} failed")
            return "success"

        return executor

    def test_successful_execution(self, executor):
        """Tool executes successfully on first attempt."""
        request = ToolRequest(
            tool_id="echo",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="echo",
            parameters={"message": "hello"}
        )

        result = executor.execute(request)

        assert result.success is True
        assert result.data == "hello"
        assert result.error is None
        assert result.retry_count == 0
        assert result.token_estimate > 0

    def test_tool_not_found(self, executor):
        """Nonexistent tool returns error."""
        request = ToolRequest(
            tool_id="nonexistent",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="nonexistent",
            parameters={}
        )

        result = executor.execute(request)

        assert result.success is False
        assert "not found" in result.error
        assert result.data is None

    def test_retry_logic_eventual_success(self, executor):
        """Tool succeeds after retries."""
        request = ToolRequest(
            tool_id="failing_tool",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="failing_tool",
            parameters={"fail_count": 2},
            retry_config=RetryConfig(max_attempts=3)
        )

        result = executor.execute(request)

        assert result.success is True
        assert result.data == "success"
        assert result.retry_count == 2  # Succeeded on 3rd attempt

    def test_retry_logic_final_failure(self, executor):
        """Tool fails after all retries exhausted."""
        request = ToolRequest(
            tool_id="failing_tool",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="failing_tool",
            parameters={"fail_count": 5},
            retry_config=RetryConfig(max_attempts=3)
        )

        result = executor.execute(request)

        assert result.success is False
        assert "failed" in result.error
        assert result.retry_count == 3

    def test_timeout_handling(self, executor):
        """Tool execution times out after specified duration."""
        request = ToolRequest(
            tool_id="slow_tool",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="slow_tool",
            parameters={"delay": 5.0},
            timeout_seconds=1
        )

        result = executor.execute(request)

        assert result.success is False
        assert "timeout" in result.error.lower()

    def test_token_estimation(self, executor):
        """Token count estimation is reasonable."""
        large_message = "word " * 1000  # ~4000 chars
        request = ToolRequest(
            tool_id="echo",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="echo",
            parameters={"message": large_message}
        )

        result = executor.execute(request)

        # Rule of thumb: 1 token ≈ 4 chars
        # 4000 chars ≈ 1000 tokens (±20%)
        assert 800 <= result.token_estimate <= 1200
```

### 2.2 Assess Module Tests

**File**: `tests/core/test_assess.py`

```python
import pytest
from nocp.core.assess import ContextManager
from nocp.models.contracts import (
    ContextData,
    ToolResult,
    CompressionMethod,
    ChatMessage
)
from datetime import datetime

class TestContextManager:
    """Unit tests for ContextManager."""

    @pytest.fixture
    def manager(self):
        """Create context manager with test config."""
        return ContextManager(
            compression_threshold=1000,
            target_compression_ratio=0.40
        )

    def test_no_compression_below_threshold(self, manager):
        """Small contexts skip compression."""
        small_result = ToolResult(
            tool_id="test",
            success=True,
            data="short text",
            execution_time_ms=10.0,
            timestamp=datetime.now(),
            token_estimate=50
        )

        context = ContextData(tool_results=[small_result])
        optimized = manager.optimize(context)

        assert optimized.method_used == CompressionMethod.NONE
        assert optimized.compression_ratio == 1.0
        assert optimized.estimated_cost_savings == 0.0

    def test_semantic_pruning_strategy_selection(self, manager):
        """Large structured data triggers semantic pruning."""
        large_list = [{"id": i, "value": f"data_{i}"} for i in range(500)]
        large_result = ToolResult(
            tool_id="db_query",
            success=True,
            data=large_list,
            execution_time_ms=100.0,
            timestamp=datetime.now(),
            token_estimate=5000
        )

        context = ContextData(tool_results=[large_result])
        strategy = manager.select_strategy(context)

        assert strategy == CompressionMethod.SEMANTIC_PRUNING

    def test_history_compaction_strategy(self, manager):
        """Long conversation history triggers compaction."""
        messages = [
            ChatMessage(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}",
                timestamp=datetime.now(),
                tokens=10
            )
            for i in range(15)
        ]

        context = ContextData(
            tool_results=[],
            message_history=messages
        )
        strategy = manager.select_strategy(context)

        assert strategy == CompressionMethod.HISTORY_COMPACTION

    def test_compression_achieves_target_ratio(self, manager):
        """Compression meets target ratio."""
        # Create large context (>10k tokens)
        large_text = " ".join([f"word{i}" for i in range(10000)])
        large_result = ToolResult(
            tool_id="test",
            success=True,
            data=large_text,
            execution_time_ms=50.0,
            timestamp=datetime.now(),
            token_estimate=10000
        )

        context = ContextData(tool_results=[large_result])
        optimized = manager.optimize(context)

        # Should achieve significant compression
        assert optimized.compression_ratio < 0.70  # >30% reduction
        assert optimized.optimized_tokens < optimized.original_tokens
        assert optimized.estimated_cost_savings > 0

    def test_token_estimation_accuracy(self, manager):
        """Token estimates are within acceptable range."""
        test_text = "This is a test sentence. " * 100  # ~600 chars

        estimated_tokens = manager.estimate_tokens(test_text)

        # Expected: ~150 tokens (600 chars / 4)
        # Allow ±20% variance
        assert 120 <= estimated_tokens <= 180

    @pytest.mark.parametrize("method", [
        CompressionMethod.SEMANTIC_PRUNING,
        CompressionMethod.KNOWLEDGE_DISTILLATION,
        CompressionMethod.HISTORY_COMPACTION
    ])
    def test_all_compression_methods_work(self, manager, method):
        """Each compression method executes without errors."""
        # Setup context appropriate for each method
        if method == CompressionMethod.SEMANTIC_PRUNING:
            data = [{"id": i} for i in range(100)]
        elif method == CompressionMethod.KNOWLEDGE_DISTILLATION:
            data = "verbose text " * 1000
        else:  # HISTORY_COMPACTION
            data = "message"

        result = ToolResult(
            tool_id="test",
            success=True,
            data=data,
            execution_time_ms=10.0,
            timestamp=datetime.now(),
            token_estimate=5000
        )

        context = ContextData(
            tool_results=[result],
            message_history=[
                ChatMessage(role="user", content=f"msg{i}", timestamp=datetime.now(), tokens=5)
                for i in range(15)
            ] if method == CompressionMethod.HISTORY_COMPACTION else [],
            compression_strategy=method.value
        )

        optimized = manager.optimize(context)
        assert optimized is not None
```

### 2.3 Articulate Module Tests

**File**: `tests/core/test_articulate.py`

```python
import pytest
from pydantic import BaseModel
from nocp.core.articulate import (
    OutputSerializer,
    TOONEncoder,
    SerializationFormat,
    SerializationRequest
)

class SimpleModel(BaseModel):
    name: str
    age: int

class TabularModel(BaseModel):
    users: list[dict[str, str]]

class NestedModel(BaseModel):
    level1: dict[str, dict[str, str]]

class TestTOONEncoder:
    """Unit tests for TOON encoding."""

    def test_simple_dict_encoding(self):
        """Encode simple dictionary."""
        encoder = TOONEncoder()
        data = {"name": "Alice", "age": 30}

        toon_output = encoder.encode(data)

        expected = "name: Alice\nage: 30"
        assert toon_output == expected

    def test_tabular_encoding(self):
        """Encode uniform list as table."""
        encoder = TOONEncoder()
        data = {
            "users": [
                {"id": "1", "name": "Alice"},
                {"id": "2", "name": "Bob"},
                {"id": "3", "name": "Charlie"}
            ]
        }

        toon_output = encoder.encode(data, length_marker="#")

        assert "users#3" in toon_output
        assert "id,name" in toon_output
        assert "1,Alice" in toon_output

    def test_nested_structure(self):
        """Encode nested objects."""
        encoder = TOONEncoder()
        data = {
            "user": {
                "profile": {
                    "name": "Alice"
                }
            }
        }

        toon_output = encoder.encode(data)

        # Should have proper indentation
        lines = toon_output.split("\n")
        assert lines[0] == "user"
        assert "  profile" in toon_output
        assert "    name: Alice" in toon_output

class TestOutputSerializer:
    """Unit tests for OutputSerializer."""

    @pytest.fixture
    def serializer(self):
        return OutputSerializer()

    def test_format_negotiation_tabular(self, serializer):
        """Tabular data selects TOON format."""
        data = TabularModel(users=[
            {"id": "1", "name": "Alice"},
            {"id": "2", "name": "Bob"},
            {"id": "3", "name": "Charlie"},
            {"id": "4", "name": "Dave"},
            {"id": "5", "name": "Eve"},
            {"id": "6", "name": "Frank"}
        ])

        format_choice = serializer.negotiate_format(data)
        assert format_choice == SerializationFormat.TOON

    def test_format_negotiation_nested(self, serializer):
        """Nested data selects compact JSON."""
        data = NestedModel(level1={"a": {"b": {"c": "value"}}})

        format_choice = serializer.negotiate_format(data)
        assert format_choice == SerializationFormat.COMPACT_JSON

    def test_toon_savings(self, serializer):
        """TOON achieves token savings."""
        data = TabularModel(users=[
            {"id": str(i), "name": f"User{i}"}
            for i in range(10)
        ])

        request = SerializationRequest(data=data, force_format="toon")
        result = serializer.serialize(request)

        assert result.format_used == SerializationFormat.TOON
        assert result.savings_ratio > 0.20  # >20% savings
        assert result.is_valid is True

    def test_compact_json_no_whitespace(self, serializer):
        """Compact JSON has no unnecessary whitespace."""
        data = SimpleModel(name="Alice", age=30)

        request = SerializationRequest(data=data, force_format="compact_json")
        result = serializer.serialize(request)

        assert result.format_used == SerializationFormat.COMPACT_JSON
        assert "\n" not in result.serialized_text
        assert "  " not in result.serialized_text  # No spaces after separators

    def test_validation_round_trip(self, serializer):
        """Serialization is lossless."""
        data = SimpleModel(name="Alice", age=30)

        request = SerializationRequest(data=data, validate_output=True)
        result = serializer.serialize(request)

        assert result.is_valid is True

    @pytest.mark.parametrize("complexity,expected_category", [
        (SimpleModel(name="Alice", age=30), "simple"),
        (TabularModel(users=[{"a": "1"}] * 10), "tabular"),
        (NestedModel(level1={"a": {"b": {"c": "d"}}}), "nested")
    ])
    def test_complexity_assessment(self, serializer, complexity, expected_category):
        """Schema complexity correctly categorized."""
        request = SerializationRequest(data=complexity)
        result = serializer.serialize(request)

        assert expected_category in result.schema_complexity
```

---

## 3. Integration Testing

### 3.1 Module Integration Tests

**File**: `tests/integration/test_pipeline.py`

```python
import pytest
from nocp.core.act import ToolExecutor
from nocp.core.assess import ContextManager
from nocp.core.articulate import OutputSerializer
from nocp.models.contracts import (
    ToolRequest,
    ContextData,
    SerializationRequest
)

class TestModulePipeline:
    """Test interactions between modules."""

    def test_act_to_assess_flow(self):
        """Tool results flow correctly to context manager."""
        # Execute tool
        executor = ToolExecutor()

        @executor.register_tool("sample_data")
        def sample_data() -> list:
            return [{"id": i, "value": f"data_{i}"} for i in range(100)]

        request = ToolRequest(
            tool_id="sample_data",
            tool_type="python_function",
            function_name="sample_data",
            parameters={}
        )
        tool_result = executor.execute(request)

        # Compress context
        manager = ContextManager()
        context = ContextData(tool_results=[tool_result])
        optimized = manager.optimize(context)

        # Verify flow
        assert tool_result.success is True
        assert optimized.original_tokens > 0
        assert optimized.optimized_tokens < optimized.original_tokens

    def test_full_pipeline_act_assess_articulate(self):
        """Complete pipeline from tool execution to serialization."""
        # Step 1: Execute tool
        executor = ToolExecutor()

        @executor.register_tool("fetch_users")
        def fetch_users() -> list:
            return [
                {"id": str(i), "name": f"User{i}", "email": f"user{i}@example.com"}
                for i in range(20)
            ]

        tool_request = ToolRequest(
            tool_id="fetch_users",
            tool_type="python_function",
            function_name="fetch_users",
            parameters={}
        )
        tool_result = executor.execute(tool_request)

        # Step 2: Optimize context
        manager = ContextManager()
        context = ContextData(tool_results=[tool_result])
        optimized = manager.optimize(context)

        # Step 3: Serialize output (simulate LLM response)
        from pydantic import BaseModel

        class UserList(BaseModel):
            users: list[dict[str, str]]

        response = UserList(users=tool_result.data)

        serializer = OutputSerializer()
        serialization_request = SerializationRequest(data=response)
        serialized = serializer.serialize(serialization_request)

        # Verify end-to-end
        assert tool_result.success is True
        assert optimized.compression_ratio < 1.0
        assert serialized.savings_ratio > 0
        assert serialized.is_valid is True
```

---

## 4. End-to-End Testing

### 4.1 Full Agent Tests

**File**: `tests/e2e/test_agent.py`

```python
import pytest
from nocp.agent import HighEfficiencyProxyAgent, ProxyRequest
from nocp.models.contracts import ToolRequest

@pytest.mark.e2e
class TestHighEfficiencyProxyAgent:
    """End-to-end tests for complete agent."""

    @pytest.fixture
    def agent(self):
        """Create agent with test configuration."""
        # Mock LLM responses for testing
        return HighEfficiencyProxyAgent(
            config={
                "default_model": "mock-model",
                "enable_compression": True,
                "enable_toon": True
            }
        )

    def test_successful_request_processing(self, agent):
        """Complete request processes successfully."""
        # This would require mocking LiteLLM responses
        # Omitted for brevity - implement with pytest-mock

    def test_error_handling_graceful_degradation(self, agent):
        """Errors trigger fallbacks, not crashes."""
        # Test compression failure → raw output
        # Test serialization failure → JSON fallback
```

---

## 5. Benchmarking Tests

### 5.1 Performance Benchmarks

**File**: `benchmarks/run_benchmarks.py`

```python
import time
import statistics
from typing import List
from nocp.agent import HighEfficiencyProxyAgent

class BenchmarkSuite:
    """Performance benchmarking for optimization pipeline."""

    def run_compression_benchmark(self, iterations: int = 100) -> dict:
        """Measure compression performance."""
        results = {
            "input_reduction": [],
            "output_reduction": [],
            "latency_overhead": [],
            "cost_savings": []
        }

        for _ in range(iterations):
            # Run optimized pipeline
            # Collect metrics
            pass

        return {
            "input_reduction_mean": statistics.mean(results["input_reduction"]),
            "input_reduction_p95": statistics.quantiles(results["input_reduction"], n=20)[18],
            "output_reduction_mean": statistics.mean(results["output_reduction"]),
            "latency_overhead_p95": statistics.quantiles(results["latency_overhead"], n=20)[18],
            "total_cost_savings": sum(results["cost_savings"])
        }
```

---

## 6. Test Execution

### 6.1 Running Tests

```bash
# Run all tests
./nocp test

# Run specific test file
./nocp test tests/core/test_act.py

# Run with coverage
./nocp test --cov=src/nocp --cov-report=html

# Run only unit tests
./nocp test tests/core/

# Run benchmarks
./nocp benchmark
```

### 6.2 Continuous Integration

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: ./nocp test --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## 7. Success Criteria

### 7.1 Unit Tests
- [ ] >90% line coverage for core modules
- [ ] All edge cases covered (errors, timeouts, retries)
- [ ] Tests run in <10 seconds

### 7.2 Integration Tests
- [ ] Module interfaces validated
- [ ] Data flows correctly between components
- [ ] Error propagation works as expected

### 7.3 End-to-End Tests
- [ ] Complete pipeline executes successfully
- [ ] Performance benchmarks meet KPIs
- [ ] Cost savings validated on real data

---

**Next**: See `06-DEPLOYMENT.md` for deployment and operations guide.
