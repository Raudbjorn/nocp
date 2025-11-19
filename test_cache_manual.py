#!/usr/bin/env python3
"""Manual test for caching layer."""

import os
import sys
import time
from datetime import datetime
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
from nocp.core.cache import LRUCache, CacheConfig
from nocp.core.act import ToolExecutor
from nocp.models.contracts import ToolRequest, ToolResult, ToolType


def test_basic_cache():
    """Test basic LRU cache operations."""
    print("Testing basic LRU cache operations...")

    cache = LRUCache(max_size=100, default_ttl=3600)

    result = ToolResult(
        tool_id="test_tool",
        success=True,
        data={"result": "test"},
        error=None,
        execution_time_ms=100.0,
        timestamp=datetime.now(),
        token_estimate=10
    )

    # Test set and get
    cache.set("key1", result)
    retrieved = cache.get("key1")

    assert retrieved is not None
    assert retrieved.tool_id == "test_tool"
    assert retrieved.data == {"result": "test"}

    # Test stats
    stats = cache.stats()
    print(f"Cache stats after get: {stats}")
    assert stats["hits"] == 1
    assert stats["misses"] == 0

    # Test cache miss
    missing = cache.get("nonexistent")
    assert missing is None

    stats = cache.stats()
    assert stats["misses"] == 1

    print("✓ Basic cache operations passed")


def test_lru_eviction():
    """Test LRU eviction."""
    print("\nTesting LRU eviction...")

    cache = LRUCache(max_size=3)

    # Add 4 items
    for i in range(4):
        result = ToolResult(
            tool_id=f"tool_{i}",
            success=True,
            data={"result": f"test_{i}"},
            error=None,
            execution_time_ms=100.0,
            timestamp=datetime.now(),
            token_estimate=10
        )
        cache.set(f"key_{i}", result)

    # First item should be evicted
    assert cache.get("key_0") is None
    assert cache.get("key_3") is not None

    stats = cache.stats()
    assert stats["evictions"] == 1
    assert stats["size"] == 3

    print("✓ LRU eviction passed")


def test_tool_executor_with_cache():
    """Test ToolExecutor with caching."""
    print("\nTesting ToolExecutor with caching...")

    cache = LRUCache()
    executor = ToolExecutor(cache=cache)

    # Register a tool
    @executor.register_tool("test_tool")
    def test_func(param1: str) -> dict:
        return {"result": param1, "timestamp": time.time()}

    request = ToolRequest(
        tool_id="test_tool",
        tool_type=ToolType.PYTHON_FUNCTION,
        function_name="test_func",
        parameters={"param1": "test"}
    )

    # First execution (cache miss)
    result1 = executor.execute(request)
    assert result1.success is True

    # Second execution (cache hit - should return same result)
    result2 = executor.execute(request)
    assert result2.success is True

    # Results should be identical (same timestamp proves it's cached)
    assert result1.data["timestamp"] == result2.data["timestamp"]

    # Cache should have one hit
    stats = cache.stats()
    print(f"Cache stats after executor test: {stats}")
    assert stats["hits"] >= 1

    print("✓ ToolExecutor with cache passed")


def test_cache_config():
    """Test CacheConfig."""
    print("\nTesting CacheConfig...")

    # Test memory backend
    config = CacheConfig(backend="memory", max_size=500, default_ttl=1800)
    backend = config.create_backend()

    assert backend is not None
    assert isinstance(backend, LRUCache)
    assert backend._max_size == 500

    # Test disabled cache
    config = CacheConfig(enabled=False)
    backend = config.create_backend()
    assert backend is None

    print("✓ CacheConfig passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Manual Cache Testing")
    print("=" * 60)

    try:
        test_basic_cache()
        test_lru_eviction()
        test_cache_config()
        test_tool_executor_with_cache()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
