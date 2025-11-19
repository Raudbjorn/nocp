"""
Tests for async modules (AsyncContextManager, AsyncOutputSerializer, ConcurrentToolExecutor).
"""

import asyncio
import time
from datetime import datetime
from typing import List

import pytest
from pydantic import BaseModel, Field

from nocp.core.act import ToolExecutor
from nocp.core.async_modules import (
    AsyncContextManager,
    AsyncOutputSerializer,
    ConcurrentToolExecutor
)
from nocp.models.contracts import (
    ContextData,
    ToolResult,
    ToolRequest,
    ToolType,
    SerializationRequest,
    SerializationFormat,
    CompressionMethod
)


class User(BaseModel):
    """Sample model for testing."""
    id: int
    name: str
    email: str


class UserListResponse(BaseModel):
    """Sample model with list for testing."""
    users: List[User]
    total: int


@pytest.fixture
def sample_users():
    """Create sample user data."""
    users = [
        User(id=i, name=f"User {i}", email=f"user{i}@example.com")
        for i in range(10)
    ]
    return UserListResponse(users=users, total=10)


@pytest.fixture
def sample_tool_results():
    """Create sample tool results for context testing."""
    return [
        ToolResult(
            tool_id=f"tool_{i}",
            success=True,
            data={"data": "x" * 100},
            error=None,
            execution_time_ms=10.0,
            timestamp=datetime.now(),
            token_estimate=25
        )
        for i in range(5)
    ]


class TestAsyncContextManager:
    """Tests for AsyncContextManager."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test AsyncContextManager initialization."""
        manager = AsyncContextManager(
            student_model="openai/gpt-4o-mini",
            compression_threshold=5000,
            target_compression_ratio=0.35
        )

        assert manager.student_model == "openai/gpt-4o-mini"
        assert manager.compression_threshold == 5000
        assert manager.target_compression_ratio == 0.35

    @pytest.mark.asyncio
    async def test_optimize_below_threshold(self, sample_tool_results):
        """Test that context below threshold is not compressed."""
        manager = AsyncContextManager(
            compression_threshold=10000,
            enable_litellm=False
        )

        context = ContextData(
            tool_results=sample_tool_results,
            transient_context={"query": "test"},
            max_tokens=50000
        )

        result = await manager.optimize_async(context)

        assert result.compression_ratio == 1.0
        assert result.method_used == CompressionMethod.NONE
        assert result.original_tokens == result.optimized_tokens

    @pytest.mark.asyncio
    async def test_semantic_pruning(self):
        """Test semantic pruning compression."""
        manager = AsyncContextManager(
            compression_threshold=100,
            target_compression_ratio=0.4,
            enable_litellm=False
        )

        # Create large list data
        large_list = [{"id": i, "data": "x" * 50} for i in range(100)]
        tool_result = ToolResult(
            tool_id="large_data",
            success=True,
            data=large_list,
            error=None,
            execution_time_ms=10.0,
            timestamp=datetime.now(),
            token_estimate=2000
        )

        context = ContextData(
            tool_results=[tool_result],
            max_tokens=50000
        )

        result = await manager.optimize_async(context)

        assert result.method_used == CompressionMethod.SEMANTIC_PRUNING
        assert result.compression_ratio < 1.0
        assert result.optimized_tokens < result.original_tokens

    @pytest.mark.asyncio
    async def test_knowledge_distillation(self):
        """Test knowledge distillation compression (without LiteLLM)."""
        manager = AsyncContextManager(
            compression_threshold=100,
            target_compression_ratio=0.4,
            enable_litellm=False  # Disabled for testing
        )

        # Create verbose text data
        verbose_text = "This is a long verbose text. " * 500
        tool_result = ToolResult(
            tool_id="verbose_data",
            success=True,
            data=verbose_text,
            error=None,
            execution_time_ms=10.0,
            timestamp=datetime.now(),
            token_estimate=6000
        )

        context = ContextData(
            tool_results=[tool_result],
            max_tokens=50000
        )

        result = await manager.optimize_async(context)

        # With litellm disabled, should use truncation
        assert result.method_used == CompressionMethod.KNOWLEDGE_DISTILLATION
        assert result.optimized_tokens < result.original_tokens

    @pytest.mark.asyncio
    async def test_history_compaction(self):
        """Test history compaction compression."""
        manager = AsyncContextManager(
            compression_threshold=100,
            enable_litellm=False
        )

        # Create many messages
        from nocp.models.contracts import ChatMessage

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
            message_history=messages,
            max_tokens=50000
        )

        result = await manager.optimize_async(context)

        assert result.method_used == CompressionMethod.HISTORY_COMPACTION
        # Should keep only last 5 messages + summary
        assert "Earlier conversation" in result.optimized_text

    @pytest.mark.asyncio
    async def test_estimate_tokens(self):
        """Test token estimation."""
        manager = AsyncContextManager(enable_litellm=False)

        text = "This is a test string"
        tokens = manager.estimate_tokens(text)

        # Rough estimate: 1 token ≈ 4 chars
        expected = len(text) // 4
        assert tokens == expected

    @pytest.mark.asyncio
    async def test_select_strategy(self):
        """Test strategy selection."""
        manager = AsyncContextManager(enable_litellm=False)

        # Test semantic pruning selection
        large_list_result = ToolResult(
            tool_id="large_list",
            success=True,
            data=[{"id": i} for i in range(100)],
            error=None,
            execution_time_ms=10.0,
            timestamp=datetime.now(),
            token_estimate=1500
        )

        context = ContextData(tool_results=[large_list_result], max_tokens=50000)
        strategy = manager.select_strategy(context)
        assert strategy == CompressionMethod.SEMANTIC_PRUNING


class TestAsyncOutputSerializer:
    """Tests for AsyncOutputSerializer."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test AsyncOutputSerializer initialization."""
        serializer = AsyncOutputSerializer()
        assert serializer.toon_encoder is not None

    @pytest.mark.asyncio
    async def test_serialize_simple_data(self, sample_users):
        """Test serialization of simple data."""
        serializer = AsyncOutputSerializer()

        request = SerializationRequest(data=sample_users)
        result = await serializer.serialize_async(request)

        assert result.serialized_text is not None
        assert result.format_used in [SerializationFormat.TOON, SerializationFormat.COMPACT_JSON]
        assert result.optimized_tokens <= result.original_tokens
        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_serialize_with_forced_format(self, sample_users):
        """Test serialization with forced format."""
        serializer = AsyncOutputSerializer()

        request = SerializationRequest(
            data=sample_users,
            force_format="compact_json"
        )
        result = await serializer.serialize_async(request)

        assert result.format_used == SerializationFormat.COMPACT_JSON
        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_serialize_tabular_data(self):
        """Test serialization of tabular data (should use TOON)."""
        serializer = AsyncOutputSerializer()

        # Create large uniform list
        users = [
            User(id=i, name=f"User {i}", email=f"user{i}@example.com")
            for i in range(20)  # > 5 items to trigger TOON
        ]
        data = UserListResponse(users=users, total=20)

        request = SerializationRequest(data=data)
        result = await serializer.serialize_async(request)

        # Should select TOON for tabular data
        assert result.format_used == SerializationFormat.TOON
        assert result.schema_complexity in ["tabular", "nested"]

    @pytest.mark.asyncio
    async def test_negotiate_format(self, sample_users):
        """Test format negotiation."""
        serializer = AsyncOutputSerializer()

        format_used = serializer.negotiate_format(sample_users)
        assert format_used in [SerializationFormat.TOON, SerializationFormat.COMPACT_JSON]

    @pytest.mark.asyncio
    async def test_is_uniform_list(self):
        """Test uniform list detection."""
        serializer = AsyncOutputSerializer()

        # Uniform list
        uniform = [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]
        assert serializer._is_uniform_list(uniform) is True

        # Non-uniform list
        non_uniform = [{"id": 1}, {"id": 2, "name": "B"}]
        assert serializer._is_uniform_list(non_uniform) is False

        # Not a list of dicts
        not_dicts = [1, 2, 3]
        assert serializer._is_uniform_list(not_dicts) is False

    @pytest.mark.asyncio
    async def test_assess_complexity(self, sample_users):
        """Test complexity assessment."""
        serializer = AsyncOutputSerializer()

        complexity = serializer._assess_complexity(sample_users)
        assert complexity in ["simple", "tabular", "nested", "complex"]


class TestConcurrentToolExecutor:
    """Tests for ConcurrentToolExecutor."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test ConcurrentToolExecutor initialization."""
        executor = ToolExecutor()
        concurrent = ConcurrentToolExecutor(executor, max_concurrent=5)

        assert concurrent.tool_executor == executor
        assert concurrent.semaphore._value == 5

    @pytest.mark.asyncio
    async def test_execute_one(self):
        """Test executing a single tool."""
        executor = ToolExecutor()

        @executor.register_async_tool("test_tool")
        async def test_tool(value: int) -> dict:
            return {"result": value * 2}

        concurrent = ConcurrentToolExecutor(executor)

        request = ToolRequest(
            tool_id="test_tool",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="test_tool",
            parameters={"value": 5}
        )

        result = await concurrent.execute_one(request)

        assert result.success is True
        assert result.data["result"] == 10

    @pytest.mark.asyncio
    async def test_execute_many(self):
        """Test executing multiple tools concurrently."""
        executor = ToolExecutor()

        @executor.register_async_tool("compute")
        async def compute(value: int) -> dict:
            await asyncio.sleep(0.01)  # Simulate async work
            return {"result": value * value}

        concurrent = ConcurrentToolExecutor(executor, max_concurrent=5)

        requests = [
            ToolRequest(
                tool_id="compute",
                tool_type=ToolType.PYTHON_FUNCTION,
                function_name="compute",
                parameters={"value": i}
            )
            for i in range(10)
        ]

        start = time.perf_counter()
        results = await concurrent.execute_many(requests)
        elapsed = time.perf_counter() - start

        # All should succeed
        assert len(results) == 10
        assert all(not isinstance(r, Exception) for r in results)

        # Should be faster than sequential (10 * 0.01s = 0.1s)
        # With concurrency=5, should be ~0.02s (2 batches of 5)
        assert elapsed < 0.05  # Some margin for overhead

    @pytest.mark.asyncio
    async def test_execute_many_ordered(self):
        """Test that results are returned in order."""
        executor = ToolExecutor()

        @executor.register_async_tool("identity")
        async def identity(value: int) -> dict:
            await asyncio.sleep(0.01)
            return {"value": value}

        concurrent = ConcurrentToolExecutor(executor)

        requests = [
            ToolRequest(
                tool_id="identity",
                tool_type=ToolType.PYTHON_FUNCTION,
                function_name="identity",
                parameters={"value": i}
            )
            for i in range(5)
        ]

        results = await concurrent.execute_many_ordered(requests)

        # Results should be in same order as requests
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.data["value"] == i

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """Test that semaphore properly limits concurrency."""
        executor = ToolExecutor()

        # Track concurrent executions
        concurrent_count = 0
        max_concurrent = 0
        lock = asyncio.Lock()

        @executor.register_async_tool("concurrent_test")
        async def concurrent_test(value: int) -> dict:
            nonlocal concurrent_count, max_concurrent

            async with lock:
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)

            await asyncio.sleep(0.02)  # Simulate work

            async with lock:
                concurrent_count -= 1

            return {"value": value}

        concurrent = ConcurrentToolExecutor(executor, max_concurrent=3)

        requests = [
            ToolRequest(
                tool_id="concurrent_test",
                tool_type=ToolType.PYTHON_FUNCTION,
                function_name="concurrent_test",
                parameters={"value": i}
            )
            for i in range(10)
        ]

        await concurrent.execute_many(requests)

        # Max concurrent should not exceed limit
        assert max_concurrent <= 3

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in concurrent execution."""
        executor = ToolExecutor()

        @executor.register_async_tool("failing_tool")
        async def failing_tool(value: int) -> dict:
            if value == 5:
                raise ValueError("Intentional error")
            return {"value": value}

        concurrent = ConcurrentToolExecutor(executor)

        requests = [
            ToolRequest(
                tool_id="failing_tool",
                tool_type=ToolType.PYTHON_FUNCTION,
                function_name="failing_tool",
                parameters={"value": i}
            )
            for i in range(10)
        ]

        results = await concurrent.execute_many(requests)

        # Should have 1 exception and 9 successes
        exceptions = [r for r in results if isinstance(r, Exception)]
        successes = [r for r in results if not isinstance(r, Exception)]

        assert len(exceptions) == 1
        assert len(successes) == 9


class TestAsyncPerformance:
    """Performance tests comparing async vs sync."""

    @pytest.mark.asyncio
    async def test_concurrent_faster_than_sequential(self):
        """Test that concurrent execution is faster than sequential."""
        executor = ToolExecutor()

        @executor.register_async_tool("slow_tool")
        async def slow_tool(value: int) -> dict:
            await asyncio.sleep(0.05)  # 50ms delay
            return {"value": value}

        # Sequential execution
        requests = [
            ToolRequest(
                tool_id="slow_tool",
                tool_type=ToolType.PYTHON_FUNCTION,
                function_name="slow_tool",
                parameters={"value": i}
            )
            for i in range(10)
        ]

        start = time.perf_counter()
        for request in requests:
            await executor.execute_async(request)
        sequential_time = time.perf_counter() - start

        # Concurrent execution
        concurrent = ConcurrentToolExecutor(executor, max_concurrent=10)

        start = time.perf_counter()
        await concurrent.execute_many(requests)
        concurrent_time = time.perf_counter() - start

        # Concurrent should be significantly faster
        # Sequential: ~500ms (10 * 50ms)
        # Concurrent: ~50ms (all at once)
        speedup = sequential_time / concurrent_time
        assert speedup > 2.0  # At least 2x faster

    @pytest.mark.asyncio
    async def test_async_context_performance(self):
        """Test async context manager performance."""
        from nocp.core.assess import ContextManager

        tool_results = [
            ToolResult(
                tool_id=f"tool_{i}",
                success=True,
                data={"data": "x" * 200},
                error=None,
                execution_time_ms=10.0,
                timestamp=datetime.now(),
                token_estimate=50
            )
            for i in range(10)
        ]

        context = ContextData(
            tool_results=tool_results,
            max_tokens=50000
        )

        # Sync version
        sync_manager = ContextManager(compression_threshold=100, enable_litellm=False)

        start = time.perf_counter()
        for _ in range(10):
            sync_manager.optimize(context)
        sync_time = time.perf_counter() - start

        # Async version
        async_manager = AsyncContextManager(compression_threshold=100, enable_litellm=False)

        start = time.perf_counter()
        for _ in range(10):
            await async_manager.optimize_async(context)
        async_time = time.perf_counter() - start

        # Both should complete successfully
        # Async might be slightly slower due to overhead for this simple case
        assert async_time < sync_time * 2  # Should not be more than 2x slower
