"""
Tests for the caching layer (LRU cache and ChromaDB cache).
"""

import time
from datetime import datetime

import pytest

from nocp.core.cache import CacheConfig, ChromaDBCache, LRUCache
from nocp.models.contracts import ToolRequest, ToolResult, ToolType


class TestLRUCache:
    """Tests for in-memory LRU cache."""

    def test_cache_initialization(self):
        """Test cache can be initialized with custom parameters."""
        cache = LRUCache(max_size=500, default_ttl=1800)
        assert cache._max_size == 500
        assert cache._default_ttl == 1800
        assert len(cache._cache) == 0

    def test_basic_set_and_get(self):
        """Test basic cache set and get operations."""
        cache = LRUCache()
        result = ToolResult(
            tool_id="test_tool",
            success=True,
            data={"result": "test"},
            error=None,
            execution_time_ms=100.0,
            timestamp=datetime.now(),
            token_estimate=10,
        )

        cache.set("key1", result)
        retrieved = cache.get("key1")

        assert retrieved is not None
        assert retrieved.tool_id == "test_tool"
        assert retrieved.data == {"result": "test"}

    def test_cache_miss(self):
        """Test cache returns None for missing keys."""
        cache = LRUCache()
        result = cache.get("nonexistent_key")
        assert result is None

    def test_cache_hit_statistics(self):
        """Test cache hit/miss statistics are tracked correctly."""
        cache = LRUCache()
        result = ToolResult(
            tool_id="test_tool",
            success=True,
            data={"result": "test"},
            error=None,
            execution_time_ms=100.0,
            timestamp=datetime.now(),
            token_estimate=10,
        )

        # Initial stats
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0

        # Add item
        cache.set("key1", result)

        # Cache hit
        cache.get("key1")
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0

        # Cache miss
        cache.get("key2")
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

        # Calculate hit rate
        assert stats["hit_rate"] == 0.5

    def test_lru_eviction(self):
        """Test LRU eviction when max size is exceeded."""
        cache = LRUCache(max_size=3)

        # Add 4 items (should evict oldest)
        for i in range(4):
            result = ToolResult(
                tool_id=f"tool_{i}",
                success=True,
                data={"result": f"test_{i}"},
                error=None,
                execution_time_ms=100.0,
                timestamp=datetime.now(),
                token_estimate=10,
            )
            cache.set(f"key_{i}", result)

        # First item should be evicted
        assert cache.get("key_0") is None
        # Others should still exist
        assert cache.get("key_1") is not None
        assert cache.get("key_2") is not None
        assert cache.get("key_3") is not None

        stats = cache.stats()
        assert stats["evictions"] == 1
        assert stats["size"] == 3

    def test_lru_ordering(self):
        """Test that accessing an item moves it to the end (most recently used)."""
        cache = LRUCache(max_size=3)

        # Add 3 items
        for i in range(3):
            result = ToolResult(
                tool_id=f"tool_{i}",
                success=True,
                data={"result": f"test_{i}"},
                error=None,
                execution_time_ms=100.0,
                timestamp=datetime.now(),
                token_estimate=10,
            )
            cache.set(f"key_{i}", result)

        # Access key_0 (moves it to end)
        cache.get("key_0")

        # Add new item (should evict key_1, not key_0)
        result = ToolResult(
            tool_id="tool_3",
            success=True,
            data={"result": "test_3"},
            error=None,
            execution_time_ms=100.0,
            timestamp=datetime.now(),
            token_estimate=10,
        )
        cache.set("key_3", result)

        # key_1 should be evicted
        assert cache.get("key_1") is None
        # key_0 should still exist (was recently accessed)
        assert cache.get("key_0") is not None

    def test_ttl_expiration(self):
        """Test that items expire after TTL."""
        cache = LRUCache(default_ttl=1)  # 1 second TTL

        result = ToolResult(
            tool_id="test_tool",
            success=True,
            data={"result": "test"},
            error=None,
            execution_time_ms=100.0,
            timestamp=datetime.now(),
            token_estimate=10,
        )

        cache.set("key1", result)

        # Should be available immediately
        assert cache.get("key1") is not None

        # Wait for expiration
        time.sleep(1.5)

        # Should be expired
        assert cache.get("key1") is None

    def test_custom_ttl(self):
        """Test setting custom TTL per item."""
        cache = LRUCache(default_ttl=3600)  # Default 1 hour

        result = ToolResult(
            tool_id="test_tool",
            success=True,
            data={"result": "test"},
            error=None,
            execution_time_ms=100.0,
            timestamp=datetime.now(),
            token_estimate=10,
        )

        # Set with custom short TTL
        cache.set("key1", result, ttl_seconds=1)

        # Should be available immediately
        assert cache.get("key1") is not None

        # Wait for expiration
        time.sleep(1.5)

        # Should be expired
        assert cache.get("key1") is None

    def test_no_expiration(self):
        """Test that items with no TTL never expire."""
        cache = LRUCache(default_ttl=None)

        result = ToolResult(
            tool_id="test_tool",
            success=True,
            data={"result": "test"},
            error=None,
            execution_time_ms=100.0,
            timestamp=datetime.now(),
            token_estimate=10,
        )

        cache.set("key1", result)

        # Should still be available after some time
        time.sleep(0.5)
        assert cache.get("key1") is not None

    def test_cache_by_request(self):
        """Test caching using ToolRequest."""
        cache = LRUCache()

        request = ToolRequest(
            tool_id="test_tool",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="test_func",
            parameters={"param1": "value1"},
        )

        result = ToolResult(
            tool_id="test_tool",
            success=True,
            data={"result": "test"},
            error=None,
            execution_time_ms=100.0,
            timestamp=datetime.now(),
            token_estimate=10,
        )

        # Cache by request
        cache.set_by_request(request, result)

        # Retrieve by request
        retrieved = cache.get_by_request(request)
        assert retrieved is not None
        assert retrieved.data == {"result": "test"}

    def test_cache_key_determinism(self):
        """Test that same request generates same cache key."""
        cache = LRUCache()

        request1 = ToolRequest(
            tool_id="test_tool",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="test_func",
            parameters={"param1": "value1", "param2": "value2"},
        )

        request2 = ToolRequest(
            tool_id="test_tool",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="test_func",
            parameters={"param2": "value2", "param1": "value1"},  # Different order
        )

        key1 = cache._generate_key(request1)
        key2 = cache._generate_key(request2)

        # Keys should be the same (parameters are sorted)
        assert key1 == key2

    def test_cache_key_uniqueness(self):
        """Test that different requests generate different cache keys."""
        cache = LRUCache()

        request1 = ToolRequest(
            tool_id="test_tool",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="test_func",
            parameters={"param1": "value1"},
        )

        request2 = ToolRequest(
            tool_id="test_tool",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="test_func",
            parameters={"param1": "value2"},  # Different value
        )

        key1 = cache._generate_key(request1)
        key2 = cache._generate_key(request2)

        # Keys should be different
        assert key1 != key2

    def test_delete(self):
        """Test deleting items from cache."""
        cache = LRUCache()

        result = ToolResult(
            tool_id="test_tool",
            success=True,
            data={"result": "test"},
            error=None,
            execution_time_ms=100.0,
            timestamp=datetime.now(),
            token_estimate=10,
        )

        cache.set("key1", result)
        assert cache.get("key1") is not None

        cache.delete("key1")
        assert cache.get("key1") is None

    def test_clear(self):
        """Test clearing all items from cache."""
        cache = LRUCache()

        # Add multiple items
        for i in range(5):
            result = ToolResult(
                tool_id=f"tool_{i}",
                success=True,
                data={"result": f"test_{i}"},
                error=None,
                execution_time_ms=100.0,
                timestamp=datetime.now(),
                token_estimate=10,
            )
            cache.set(f"key_{i}", result)

        assert cache.stats()["size"] == 5

        cache.clear()

        assert cache.stats()["size"] == 0
        assert cache.get("key_0") is None

    @pytest.mark.asyncio
    async def test_async_get_set(self):
        """Test async versions of get/set."""
        cache = LRUCache()

        result = ToolResult(
            tool_id="test_tool",
            success=True,
            data={"result": "test"},
            error=None,
            execution_time_ms=100.0,
            timestamp=datetime.now(),
            token_estimate=10,
        )

        await cache.set_async("key1", result)
        retrieved = await cache.get_async("key1")

        assert retrieved is not None
        assert retrieved.data == {"result": "test"}

    @pytest.mark.asyncio
    async def test_async_by_request(self):
        """Test async versions of get_by_request/set_by_request."""
        cache = LRUCache()

        request = ToolRequest(
            tool_id="test_tool",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="test_func",
            parameters={"param1": "value1"},
        )

        result = ToolResult(
            tool_id="test_tool",
            success=True,
            data={"result": "test"},
            error=None,
            execution_time_ms=100.0,
            timestamp=datetime.now(),
            token_estimate=10,
        )

        await cache.set_by_request_async(request, result)
        retrieved = await cache.get_by_request_async(request)

        assert retrieved is not None
        assert retrieved.data == {"result": "test"}


class TestCacheConfig:
    """Tests for CacheConfig."""

    def test_create_memory_backend(self):
        """Test creating in-memory cache backend."""
        config = CacheConfig(backend="memory", max_size=500, default_ttl=1800)
        backend = config.create_backend()

        assert isinstance(backend, LRUCache)
        assert backend._max_size == 500
        assert backend._default_ttl == 1800

    def test_disabled_cache(self):
        """Test that disabled cache returns None."""
        config = CacheConfig(enabled=False)
        backend = config.create_backend()

        assert backend is None

    def test_invalid_backend(self):
        """Test that invalid backend raises error."""
        config = CacheConfig(backend="invalid")

        with pytest.raises(ValueError, match="Unknown cache backend"):
            config.create_backend()


# ChromaDB tests
class TestChromaDBCache:
    """Tests for ChromaDB cache."""

    @pytest.fixture
    def chromadb_available(self):
        """Check if ChromaDB is available."""
        try:

            return True
        except Exception:
            pytest.skip("ChromaDB not installed")

    @pytest.fixture
    def chromadb_cache(self, chromadb_available, tmp_path):
        """Create ChromaDB cache for testing."""
        # Use tmp_path for persistence to avoid conflicts between tests
        cache = ChromaDBCache(persist_directory=None, collection_name="test_cache")  # In-memory
        cache.clear()  # Clean up before tests
        yield cache
        cache.clear()  # Clean up after tests

    def test_chromadb_basic_set_get(self, chromadb_cache):
        """Test basic ChromaDB cache operations."""
        result = ToolResult(
            tool_id="test_tool",
            success=True,
            data={"result": "test"},
            error=None,
            execution_time_ms=100.0,
            timestamp=datetime.now(),
            token_estimate=10,
        )

        chromadb_cache.set("key1", result)
        retrieved = chromadb_cache.get("key1")

        assert retrieved is not None
        assert retrieved.tool_id == "test_tool"
        assert retrieved.data == {"result": "test"}

    def test_chromadb_ttl(self, chromadb_cache):
        """Test ChromaDB TTL functionality."""
        result = ToolResult(
            tool_id="test_tool",
            success=True,
            data={"result": "test"},
            error=None,
            execution_time_ms=100.0,
            timestamp=datetime.now(),
            token_estimate=10,
        )

        chromadb_cache.set("key1", result, ttl_seconds=1)

        # Should be available immediately
        assert chromadb_cache.get("key1") is not None

        # Wait for expiration
        time.sleep(1.5)

        # Should be expired
        assert chromadb_cache.get("key1") is None

    def test_chromadb_by_request(self, chromadb_cache):
        """Test ChromaDB cache with ToolRequest."""
        request = ToolRequest(
            tool_id="test_tool",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="test_func",
            parameters={"param1": "value1"},
        )

        result = ToolResult(
            tool_id="test_tool",
            success=True,
            data={"result": "test"},
            error=None,
            execution_time_ms=100.0,
            timestamp=datetime.now(),
            token_estimate=10,
        )

        chromadb_cache.set_by_request(request, result)
        retrieved = chromadb_cache.get_by_request(request)

        assert retrieved is not None
        assert retrieved.data == {"result": "test"}

    @pytest.mark.asyncio
    async def test_chromadb_async_operations(self, chromadb_cache):
        """Test ChromaDB async operations."""
        result = ToolResult(
            tool_id="test_tool",
            success=True,
            data={"result": "test"},
            error=None,
            execution_time_ms=100.0,
            timestamp=datetime.now(),
            token_estimate=10,
        )

        await chromadb_cache.set_async("key1", result)
        retrieved = await chromadb_cache.get_async("key1")

        assert retrieved is not None
        assert retrieved.data == {"result": "test"}

    def test_chromadb_stats(self, chromadb_cache):
        """Test ChromaDB statistics."""
        stats = chromadb_cache.stats()

        assert stats["backend"] == "chromadb"
        assert "size" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats


class TestCacheIntegration:
    """Integration tests with ToolExecutor."""

    def test_tool_executor_with_cache(self):
        """Test that ToolExecutor uses cache correctly."""
        from nocp.core.act import ToolExecutor

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
            parameters={"param1": "test"},
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
        assert stats["hits"] == 1

    def test_cache_bypass(self):
        """Test that use_cache=False bypasses cache."""
        from nocp.core.act import ToolExecutor

        cache = LRUCache()
        executor = ToolExecutor(cache=cache)

        @executor.register_tool("test_tool")
        def test_func(param1: str) -> dict:
            return {"result": param1, "timestamp": time.time()}

        request = ToolRequest(
            tool_id="test_tool",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="test_func",
            parameters={"param1": "test"},
        )

        # Execute with cache disabled
        result1 = executor.execute(request, use_cache=False)
        time.sleep(0.01)  # Small delay to ensure different timestamp
        result2 = executor.execute(request, use_cache=False)

        # Results should have different timestamps
        assert result1.data["timestamp"] != result2.data["timestamp"]

        # Cache should have no hits
        stats = cache.stats()
        assert stats["hits"] == 0

    @pytest.mark.asyncio
    async def test_async_tool_executor_with_cache(self):
        """Test async ToolExecutor with cache."""
        from nocp.core.act import ToolExecutor

        cache = LRUCache()
        executor = ToolExecutor(cache=cache)

        @executor.register_async_tool("async_tool")
        async def async_func(param1: str) -> dict:
            return {"result": param1, "timestamp": time.time()}

        request = ToolRequest(
            tool_id="async_tool",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="async_func",
            parameters={"param1": "test"},
        )

        # First execution (cache miss)
        result1 = await executor.execute_async(request)
        assert result1.success is True

        # Second execution (cache hit)
        result2 = await executor.execute_async(request)
        assert result2.success is True

        # Results should be identical (proves it's cached)
        assert result1.data["timestamp"] == result2.data["timestamp"]
