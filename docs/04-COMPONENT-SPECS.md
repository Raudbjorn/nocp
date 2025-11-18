# Component Implementation Specifications

This document provides detailed implementation guidance for each module.

---

## 1. Act Module: Tool Executor

**File**: `src/nocp/core/act.py`

### 1.1 Core Classes

```python
from typing import Dict, Any, Callable, Optional, List
from pydantic import BaseModel
import time
import asyncio
from ..models.contracts import ToolRequest, ToolResult, ToolType, RetryConfig
from ..exceptions import ToolExecutionError

class ToolExecutor:
    """
    Manages tool registration and execution with retry logic.
    """

    def __init__(self):
        self._registry: Dict[str, Callable] = {}
        self._async_registry: Dict[str, Callable] = {}

    def register_tool(
        self,
        tool_id: str,
        tool_type: ToolType = ToolType.PYTHON_FUNCTION
    ) -> Callable:
        """
        Decorator to register a synchronous tool.

        Usage:
            executor = ToolExecutor()

            @executor.register_tool("fetch_data")
            def fetch_data(param1: str) -> dict:
                return {"result": param1}
        """
        def decorator(func: Callable) -> Callable:
            self._registry[tool_id] = func
            return func
        return decorator

    def register_async_tool(self, tool_id: str) -> Callable:
        """Decorator to register an async tool."""
        def decorator(func: Callable) -> Callable:
            self._async_registry[tool_id] = func
            return func
        return decorator

    def execute(self, request: ToolRequest) -> ToolResult:
        """
        Execute a registered tool with retry logic.

        Implementation steps:
        1. Validate tool exists in registry
        2. Execute with timeout
        3. Retry on failure if configured
        4. Estimate token count of result
        5. Return ToolResult
        """
        if request.tool_id not in self._registry:
            return ToolResult(
                tool_id=request.tool_id,
                success=False,
                data=None,
                error=f"Tool '{request.tool_id}' not found in registry",
                execution_time_ms=0.0,
                timestamp=datetime.now(),
                token_estimate=0
            )

        retry_config = request.retry_config or RetryConfig()
        last_error = None

        for attempt in range(retry_config.max_attempts):
            try:
                start = time.perf_counter()

                # Execute tool with timeout
                func = self._registry[request.tool_id]
                result = self._execute_with_timeout(
                    func,
                    request.parameters,
                    request.timeout_seconds
                )

                execution_time = (time.perf_counter() - start) * 1000

                # Estimate tokens
                token_estimate = self._estimate_tokens(result)

                return ToolResult(
                    tool_id=request.tool_id,
                    success=True,
                    data=result,
                    error=None,
                    execution_time_ms=execution_time,
                    timestamp=datetime.now(),
                    token_estimate=token_estimate,
                    retry_count=attempt
                )

            except Exception as e:
                last_error = str(e)
                if attempt < retry_config.max_attempts - 1:
                    # Exponential backoff
                    delay = (retry_config.initial_delay_ms / 1000) * \
                            (retry_config.backoff_multiplier ** attempt)
                    time.sleep(delay)
                continue

        # All retries failed
        return ToolResult(
            tool_id=request.tool_id,
            success=False,
            data=None,
            error=last_error,
            execution_time_ms=0.0,
            timestamp=datetime.now(),
            token_estimate=0,
            retry_count=retry_config.max_attempts
        )

    async def execute_async(self, request: ToolRequest) -> ToolResult:
        """Async version for concurrent execution."""
        # Similar implementation with asyncio
        pass

    def _execute_with_timeout(
        self,
        func: Callable,
        params: Dict[str, Any],
        timeout: int
    ) -> Any:
        """Execute function with timeout."""
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Tool execution exceeded {timeout}s")

        # Set timeout alarm (Unix-only, use threading.Timer for cross-platform)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        try:
            result = func(**params)
            signal.alarm(0)  # Cancel alarm
            return result
        except TimeoutError:
            raise
        finally:
            signal.alarm(0)

    def _estimate_tokens(self, data: Any) -> int:
        """
        Rough token estimate using character count.
        Rule of thumb: 1 token ≈ 4 characters for English text.
        """
        if isinstance(data, str):
            return len(data) // 4
        elif isinstance(data, (dict, list)):
            import json
            text = json.dumps(data)
            return len(text) // 4
        else:
            return len(str(data)) // 4
```

### 1.2 Testing Strategy

```python
# tests/core/test_act.py

import pytest
from nocp.core.act import ToolExecutor, ToolRequest, ToolType
from nocp.exceptions import ToolExecutionError

@pytest.fixture
def executor():
    executor = ToolExecutor()

    @executor.register_tool("test_tool")
    def test_tool(value: str) -> dict:
        return {"result": value.upper()}

    @executor.register_tool("failing_tool")
    def failing_tool() -> None:
        raise ValueError("Intentional failure")

    return executor

def test_successful_execution(executor):
    request = ToolRequest(
        tool_id="test_tool",
        tool_type=ToolType.PYTHON_FUNCTION,
        function_name="test_tool",
        parameters={"value": "hello"}
    )
    result = executor.execute(request)

    assert result.success is True
    assert result.data == {"result": "HELLO"}
    assert result.token_estimate > 0
    assert result.retry_count == 0

def test_tool_not_found(executor):
    request = ToolRequest(
        tool_id="nonexistent",
        tool_type=ToolType.PYTHON_FUNCTION,
        function_name="nonexistent",
        parameters={}
    )
    result = executor.execute(request)

    assert result.success is False
    assert "not found" in result.error

def test_retry_logic(executor):
    # Tool should retry 3 times and then fail
    request = ToolRequest(
        tool_id="failing_tool",
        tool_type=ToolType.PYTHON_FUNCTION,
        function_name="failing_tool",
        parameters={}
    )
    result = executor.execute(request)

    assert result.success is False
    assert result.retry_count == 3
```

---

## 2. Assess Module: Context Manager

**File**: `src/nocp/core/assess.py`

### 2.1 Core Classes

```python
from typing import List, Optional
from pydantic import BaseModel
import litellm
from ..models.contracts import (
    ContextData,
    OptimizedContext,
    CompressionMethod,
    ToolResult
)

class ContextManager:
    """
    Optimizes context to reduce token usage before LLM calls.
    """

    def __init__(
        self,
        student_model: str = "openai/gpt-4o-mini",
        compression_threshold: int = 10_000,
        target_compression_ratio: float = 0.40
    ):
        self.student_model = student_model
        self.compression_threshold = compression_threshold
        self.target_compression_ratio = target_compression_ratio

    def optimize(self, context: ContextData) -> OptimizedContext:
        """
        Main entry point for context optimization.

        Decision tree:
        1. Estimate total tokens
        2. If < threshold, return raw (no compression)
        3. Select compression strategy based on data type
        4. Apply compression
        5. Verify cost-benefit
        6. Return OptimizedContext
        """
        # Step 1: Count original tokens
        raw_text = self._context_to_text(context)
        original_tokens = self.estimate_tokens(raw_text)

        # Step 2: Check if compression warranted
        if original_tokens < self.compression_threshold:
            return OptimizedContext(
                optimized_text=raw_text,
                original_tokens=original_tokens,
                optimized_tokens=original_tokens,
                compression_ratio=1.0,
                method_used=CompressionMethod.NONE,
                compression_time_ms=0.0,
                estimated_cost_savings=0.0
            )

        # Step 3: Select strategy
        strategy = self.select_strategy(context)

        # Step 4: Apply compression
        start_time = time.perf_counter()

        if strategy == CompressionMethod.SEMANTIC_PRUNING:
            compressed_text = self._semantic_pruning(context)
        elif strategy == CompressionMethod.KNOWLEDGE_DISTILLATION:
            compressed_text = self._knowledge_distillation(context)
        elif strategy == CompressionMethod.HISTORY_COMPACTION:
            compressed_text = self._history_compaction(context)
        else:
            compressed_text = raw_text

        compression_time = (time.perf_counter() - start_time) * 1000

        # Step 5: Count compressed tokens
        compressed_tokens = self.estimate_tokens(compressed_text)
        compression_ratio = compressed_tokens / original_tokens

        # Step 6: Calculate cost savings
        # Assume $1.00 per 1M input tokens (Gemini 2.0 Flash pricing)
        token_savings = original_tokens - compressed_tokens
        cost_savings = (token_savings / 1_000_000) * 1.00

        return OptimizedContext(
            optimized_text=compressed_text,
            original_tokens=original_tokens,
            optimized_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            method_used=strategy,
            compression_time_ms=compression_time,
            estimated_cost_savings=cost_savings
        )

    def estimate_tokens(self, text: str, model: str = "gpt-4") -> int:
        """
        Estimate token count using litellm's token_counter.
        """
        try:
            # LiteLLM provides model-specific token counting
            tokens = litellm.token_counter(model=model, text=text)
            return tokens
        except Exception:
            # Fallback: rough estimate (1 token ≈ 4 chars)
            return len(text) // 4

    def select_strategy(self, context: ContextData) -> CompressionMethod:
        """
        Auto-detect optimal compression strategy.

        Heuristics:
        - If tool results contain >1000 tokens of structured data (JSON/lists):
          Use SEMANTIC_PRUNING
        - If tool results contain verbose text (logs, descriptions):
          Use KNOWLEDGE_DISTILLATION
        - If message_history has >10 messages:
          Use HISTORY_COMPACTION
        """
        # Check for large structured data
        for result in context.tool_results:
            if isinstance(result.data, (list, dict)):
                if result.token_estimate > 1000:
                    return CompressionMethod.SEMANTIC_PRUNING

        # Check for verbose text
        total_text_tokens = sum(
            r.token_estimate for r in context.tool_results
            if isinstance(r.data, str)
        )
        if total_text_tokens > 5000:
            return CompressionMethod.KNOWLEDGE_DISTILLATION

        # Check conversation history
        if len(context.message_history) > 10:
            return CompressionMethod.HISTORY_COMPACTION

        return CompressionMethod.NONE

    def _semantic_pruning(self, context: ContextData) -> str:
        """
        Extract top-k most relevant chunks from structured data.

        For MVP: Simple implementation that takes first N items.
        For production: Use embedding similarity to user query.
        """
        # Simplified implementation
        pruned_results = []
        target_tokens = int(sum(r.token_estimate for r in context.tool_results) *
                           self.target_compression_ratio)

        current_tokens = 0
        for result in context.tool_results:
            if isinstance(result.data, list):
                # Take first k items until we hit target
                items_to_keep = []
                for item in result.data:
                    item_tokens = len(str(item)) // 4
                    if current_tokens + item_tokens > target_tokens:
                        break
                    items_to_keep.append(item)
                    current_tokens += item_tokens
                pruned_results.append(items_to_keep)
            else:
                pruned_results.append(result.data)

        return str(pruned_results)

    def _knowledge_distillation(self, context: ContextData) -> str:
        """
        Use student summarizer model to compress verbose text.

        Cost-benefit check:
        - Student model cost: ~$0.15 per 1M tokens (GPT-4o-mini)
        - Must save more than this in main model costs
        """
        raw_text = self._context_to_text(context)
        raw_tokens = self.estimate_tokens(raw_text)

        # Cost of summarization
        summarization_cost = (raw_tokens / 1_000_000) * 0.15

        # Expected savings (assume 60% reduction)
        expected_compressed_tokens = int(raw_tokens * 0.40)
        token_savings = raw_tokens - expected_compressed_tokens
        main_model_savings = (token_savings / 1_000_000) * 1.00  # Gemini pricing

        # Only summarize if cost-effective
        if main_model_savings <= summarization_cost:
            return raw_text  # Not worth it

        # Call student model
        try:
            response = litellm.completion(
                model=self.student_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize the following text concisely while preserving all key information."
                    },
                    {"role": "user", "content": raw_text}
                ],
                max_tokens=expected_compressed_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback to raw if summarization fails
            return raw_text

    def _history_compaction(self, context: ContextData) -> str:
        """
        Compress old conversation messages into a summary.

        Strategy: Keep last 5 messages, summarize the rest.
        """
        if len(context.message_history) <= 5:
            return self._context_to_text(context)

        # Keep recent messages
        recent_messages = context.message_history[-5:]

        # Summarize older messages
        old_messages = context.message_history[:-5]
        old_text = "\n".join(msg.content for msg in old_messages)

        # Summarize using student model
        try:
            summary_response = litellm.completion(
                model=self.student_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Provide a concise summary of the conversation history."
                    },
                    {"role": "user", "content": old_text}
                ],
                max_tokens=500
            )
            summary = summary_response.choices[0].message.content

            # Combine summary with recent messages
            combined = f"[Summary of earlier conversation: {summary}]\n\n"
            combined += "\n".join(msg.content for msg in recent_messages)
            return combined
        except Exception:
            return self._context_to_text(context)

    def _context_to_text(self, context: ContextData) -> str:
        """Convert ContextData to text representation."""
        parts = []

        # Add tool results
        for result in context.tool_results:
            parts.append(f"Tool: {result.tool_id}")
            parts.append(result.to_text())

        # Add transient context
        if context.transient_context:
            parts.append(f"Context: {context.transient_context}")

        # Add message history
        for msg in context.message_history:
            parts.append(f"{msg.role}: {msg.content}")

        return "\n\n".join(parts)
```

### 2.2 Testing Strategy

```python
# tests/core/test_assess.py

import pytest
from nocp.core.assess import ContextManager
from nocp.models.contracts import ContextData, ToolResult, CompressionMethod

@pytest.fixture
def manager():
    return ContextManager(compression_threshold=1000)

def test_no_compression_below_threshold(manager):
    """Small contexts should not be compressed."""
    small_result = ToolResult(
        tool_id="test",
        success=True,
        data="short text",
        token_estimate=10,
        # ... other required fields
    )

    context = ContextData(tool_results=[small_result])
    optimized = manager.optimize(context)

    assert optimized.method_used == CompressionMethod.NONE
    assert optimized.compression_ratio == 1.0

def test_semantic_pruning_selection(manager):
    """Large structured data should trigger semantic pruning."""
    large_list = [{"id": i, "value": f"item_{i}"} for i in range(1000)]
    large_result = ToolResult(
        tool_id="db_query",
        success=True,
        data=large_list,
        token_estimate=5000,
        # ... other fields
    )

    context = ContextData(tool_results=[large_result])
    strategy = manager.select_strategy(context)

    assert strategy == CompressionMethod.SEMANTIC_PRUNING

def test_compression_ratio_target(manager):
    """Compression should achieve target ratio."""
    # Create context with known size
    large_text = "word " * 10000  # ~40k tokens
    result = ToolResult(
        tool_id="test",
        success=True,
        data=large_text,
        token_estimate=40000,
        # ... other fields
    )

    context = ContextData(tool_results=[result])
    optimized = manager.optimize(context)

    # Should achieve significant compression
    assert optimized.compression_ratio < 0.70
    assert optimized.estimated_cost_savings > 0
```

---

## 3. Articulate Module: Output Serializer

**File**: `src/nocp/core/articulate.py`

### 3.1 TOON Encoder Implementation

```python
from typing import Any, List, Dict
from pydantic import BaseModel
import json

class TOONEncoder:
    """
    Token-Oriented Object Notation encoder.

    TOON format combines:
    - Indentation-based structure for nested objects (like YAML)
    - CSV-style tabular layout for uniform arrays
    - Length markers for validation

    Example:
        Input (JSON):
        {
          "users": [
            {"id": "1", "name": "Alice", "age": 30},
            {"id": "2", "name": "Bob", "age": 25}
          ]
        }

        Output (TOON):
        users#2
          id,name,age
          1,Alice,30
          2,Bob,25
    """

    def encode(self, data: Any, length_marker: str = "#") -> str:
        """
        Encode data to TOON format.

        Args:
            data: Dictionary or list to encode
            length_marker: Character to use for length annotations

        Returns:
            TOON-formatted string
        """
        if isinstance(data, BaseModel):
            data = data.model_dump()

        return self._encode_value(data, indent=0, length_marker=length_marker)

    def _encode_value(
        self,
        value: Any,
        indent: int,
        length_marker: str
    ) -> str:
        """Recursively encode a value."""
        if isinstance(value, dict):
            return self._encode_dict(value, indent, length_marker)
        elif isinstance(value, list):
            return self._encode_list(value, indent, length_marker)
        else:
            return str(value)

    def _encode_dict(
        self,
        obj: Dict[str, Any],
        indent: int,
        length_marker: str
    ) -> str:
        """Encode dictionary as indented key-value pairs."""
        lines = []
        indent_str = "  " * indent

        for key, value in obj.items():
            if isinstance(value, list) and self._is_uniform_list(value):
                # Use tabular format for uniform arrays
                lines.append(f"{indent_str}{key}{length_marker}{len(value)}")
                lines.append(self._encode_tabular(value, indent + 1))
            elif isinstance(value, (dict, list)):
                lines.append(f"{indent_str}{key}")
                lines.append(self._encode_value(value, indent + 1, length_marker))
            else:
                lines.append(f"{indent_str}{key}: {value}")

        return "\n".join(lines)

    def _encode_list(
        self,
        arr: List[Any],
        indent: int,
        length_marker: str
    ) -> str:
        """Encode list, using tabular format if uniform."""
        if self._is_uniform_list(arr):
            return self._encode_tabular(arr, indent)
        else:
            # Non-uniform list: encode each item
            lines = []
            indent_str = "  " * indent
            for item in arr:
                lines.append(f"{indent_str}- {self._encode_value(item, indent + 1, length_marker)}")
            return "\n".join(lines)

    def _is_uniform_list(self, arr: List[Any]) -> bool:
        """Check if list contains uniform dictionaries (same keys)."""
        if not arr or not isinstance(arr[0], dict):
            return False

        first_keys = set(arr[0].keys())
        return all(isinstance(item, dict) and set(item.keys()) == first_keys for item in arr)

    def _encode_tabular(self, arr: List[Dict[str, Any]], indent: int) -> str:
        """Encode uniform list as CSV-style table."""
        if not arr:
            return ""

        indent_str = "  " * indent
        keys = list(arr[0].keys())

        # Header row
        header = f"{indent_str}{','.join(keys)}"

        # Data rows
        rows = []
        for item in arr:
            row_values = [str(item[key]) for key in keys]
            rows.append(f"{indent_str}{','.join(row_values)}")

        return "\n".join([header] + rows)

    def decode(self, toon_str: str) -> Any:
        """
        Decode TOON string back to Python objects.
        (Simplified implementation for MVP)
        """
        # For MVP: Return JSON parsing fallback
        # Production: Implement full TOON parser
        raise NotImplementedError("TOON decoding coming in Phase 2")
```

### 3.2 Format Negotiation

```python
class OutputSerializer:
    """
    Serializes Pydantic models to token-optimized formats.
    """

    def __init__(self):
        self.toon_encoder = TOONEncoder()

    def serialize(self, request: SerializationRequest) -> SerializedOutput:
        """
        Main serialization entry point with format negotiation.
        """
        # Step 1: Determine optimal format
        if request.force_format:
            format_used = SerializationFormat(request.force_format)
        else:
            format_used = self.negotiate_format(request.data)

        # Step 2: Serialize
        start_time = time.perf_counter()

        if format_used == SerializationFormat.TOON:
            serialized = self.toon_encoder.encode(
                request.data,
                length_marker="#" if request.include_length_markers else ""
            )
        else:  # COMPACT_JSON
            serialized = request.data.model_dump_json(
                indent=None,
                separators=(',', ':')
            )

        serialization_time = (time.perf_counter() - start_time) * 1000

        # Step 3: Calculate savings
        baseline_json = request.data.model_dump_json(indent=2)
        original_tokens = len(baseline_json) // 4
        optimized_tokens = len(serialized) // 4
        savings_ratio = 1.0 - (optimized_tokens / original_tokens)

        # Step 4: Validation
        is_valid = True
        if request.validate_output:
            try:
                # Attempt to deserialize
                if format_used == SerializationFormat.COMPACT_JSON:
                    json.loads(serialized)
                # TOON validation skipped in MVP
            except Exception:
                is_valid = False

        return SerializedOutput(
            serialized_text=serialized,
            format_used=format_used,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            savings_ratio=savings_ratio,
            is_valid=is_valid,
            serialization_time_ms=serialization_time,
            schema_complexity=self._assess_complexity(request.data)
        )

    def negotiate_format(self, model: BaseModel) -> SerializationFormat:
        """
        Analyze Pydantic model to select optimal format.

        Decision logic:
        - If model contains list fields with >5 uniform items: TOON
        - If model is deeply nested (>3 levels): COMPACT_JSON
        - If model has mostly scalar fields: COMPACT_JSON
        - Default: COMPACT_JSON (safe fallback)
        """
        model_dict = model.model_dump()

        # Check for tabular data
        for value in model_dict.values():
            if isinstance(value, list) and len(value) > 5:
                if self._is_uniform_list(value):
                    return SerializationFormat.TOON

        # Check nesting depth
        if self._get_nesting_depth(model_dict) > 3:
            return SerializationFormat.COMPACT_JSON

        return SerializationFormat.COMPACT_JSON  # Safe default

    def _is_uniform_list(self, arr: List[Any]) -> bool:
        """Check if list is uniform (same structure)."""
        if not arr or not isinstance(arr[0], dict):
            return False
        first_keys = set(arr[0].keys())
        return all(isinstance(item, dict) and set(item.keys()) == first_keys for item in arr)

    def _get_nesting_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth."""
        if not isinstance(obj, (dict, list)):
            return current_depth

        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(
                self._get_nesting_depth(v, current_depth + 1)
                for v in obj.values()
            )
        else:  # list
            if not obj:
                return current_depth
            return max(
                self._get_nesting_depth(item, current_depth + 1)
                for item in obj
            )

    def _assess_complexity(self, model: BaseModel) -> str:
        """Categorize schema complexity."""
        model_dict = model.model_dump()
        depth = self._get_nesting_depth(model_dict)

        has_arrays = any(isinstance(v, list) for v in model_dict.values())

        if depth <= 1 and not has_arrays:
            return "simple"
        elif has_arrays and self._has_uniform_arrays(model_dict):
            return "tabular"
        elif depth > 3:
            return "complex"
        else:
            return "nested"

    def _has_uniform_arrays(self, obj: Dict[str, Any]) -> bool:
        """Check if object contains uniform arrays."""
        for value in obj.values():
            if isinstance(value, list) and self._is_uniform_list(value):
                return True
        return False
```

### 3.3 Testing

```python
# tests/core/test_articulate.py

import pytest
from pydantic import BaseModel
from nocp.core.articulate import OutputSerializer, TOONEncoder, SerializationFormat

class UserModel(BaseModel):
    id: str
    name: str
    age: int

class UsersListModel(BaseModel):
    users: list[UserModel]

def test_toon_encoding_tabular_data():
    """TOON should be selected for tabular data."""
    data = UsersListModel(users=[
        UserModel(id="1", name="Alice", age=30),
        UserModel(id="2", name="Bob", age=25),
        UserModel(id="3", name="Charlie", age=35),
    ])

    serializer = OutputSerializer()
    result = serializer.serialize(SerializationRequest(data=data))

    assert result.format_used == SerializationFormat.TOON
    assert result.savings_ratio > 0.2  # At least 20% savings
    assert "users#3" in result.serialized_text  # Length marker

def test_compact_json_for_nested():
    """Deeply nested structures should use compact JSON."""
    class NestedModel(BaseModel):
        level1: dict[str, dict[str, dict[str, str]]]

    data = NestedModel(level1={"a": {"b": {"c": "value"}}})

    serializer = OutputSerializer()
    result = serializer.serialize(SerializationRequest(data=data))

    assert result.format_used == SerializationFormat.COMPACT_JSON
    assert result.is_valid is True

def test_toon_encoding_format():
    """Verify TOON format structure."""
    data = UsersListModel(users=[
        UserModel(id="1", name="Alice", age=30),
        UserModel(id="2", name="Bob", age=25),
    ])

    encoder = TOONEncoder()
    toon_output = encoder.encode(data)

    expected_lines = [
        "users#2",
        "  id,name,age",
        "  1,Alice,30",
        "  2,Bob,25"
    ]

    assert toon_output == "\n".join(expected_lines)
```

---

## 4. Implementation Checklist

### Phase 0: Bootstrap
- [ ] Create `src/nocp/bootstrap.py` for uv auto-installer
- [ ] Create `nocp` executable shell script
- [ ] Test on clean Linux/macOS systems

### Phase 1: Core Modules
- [ ] Implement `ToolExecutor` class
- [ ] Implement retry logic with exponential backoff
- [ ] Implement `ContextManager` with token counting
- [ ] Implement semantic pruning (basic version)
- [ ] Implement knowledge distillation with cost-benefit check
- [ ] Implement `TOONEncoder` class
- [ ] Implement format negotiation logic
- [ ] Write comprehensive unit tests (>85% coverage)

### Phase 2: Integration
- [ ] Implement `LLMClient` wrapper for LiteLLM
- [ ] Implement `HighEfficiencyProxyAgent` orchestrator
- [ ] Create end-to-end example
- [ ] Integration tests

### Phase 3: Optimization
- [ ] Add conversation history compaction
- [ ] Implement structured logging
- [ ] Create benchmarking suite
- [ ] Performance optimization

---

**Next**: See `05-TESTING-STRATEGY.md` for comprehensive testing approach.
