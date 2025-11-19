#!/usr/bin/env python3
"""
Test script for ChromaDB cache implementation.
"""
import os
import sys
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from nocp.core.cache import CacheConfig, ChromaDBCache
from nocp.models.contracts import ToolRequest, ToolResult, ToolType


def test_chromadb_basic():
    """Test basic ChromaDB cache operations."""
    print("\n=== Testing Basic ChromaDB Cache Operations ===")

    # Create cache (in-memory)
    cache = ChromaDBCache(persist_directory=None, collection_name="test_cache")

    # Create test result
    result = ToolResult(
        tool_id="test_tool",
        success=True,
        data={"message": "Hello from ChromaDB!"},
        error=None,
        execution_time_ms=100.0,
        timestamp=datetime.now(),
        token_estimate=10,
    )

    # Test set/get
    print("Setting cache entry...")
    cache.set("test_key", result)

    print("Getting cache entry...")
    retrieved = cache.get("test_key")

    if retrieved and retrieved.data == result.data:
        print("✓ Basic set/get works!")
    else:
        print("✗ Basic set/get failed!")
        return False

    # Test stats
    stats = cache.stats()
    print(f"Cache stats: {stats}")
    print(f"  Backend: {stats['backend']}")
    print(f"  Size: {stats['size']}")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")

    cache.clear()
    return True


def test_chromadb_ttl():
    """Test TTL functionality."""
    print("\n=== Testing ChromaDB TTL ===")

    cache = ChromaDBCache(persist_directory=None, collection_name="test_ttl")

    result = ToolResult(
        tool_id="test_tool",
        success=True,
        data={"message": "TTL test"},
        error=None,
        execution_time_ms=100.0,
        timestamp=datetime.now(),
        token_estimate=10,
    )

    # Set with 1 second TTL
    print("Setting cache with 1s TTL...")
    cache.set("ttl_key", result, ttl_seconds=1)

    # Should be available immediately
    retrieved = cache.get("ttl_key")
    if retrieved:
        print("✓ Cache entry available immediately")
    else:
        print("✗ Cache entry not available immediately!")
        return False

    # Wait for expiration
    print("Waiting for expiration...")
    time.sleep(1.5)

    # Should be expired
    retrieved = cache.get("ttl_key")
    if not retrieved:
        print("✓ Cache entry expired correctly")
    else:
        print("✗ Cache entry did not expire!")
        return False

    cache.clear()
    return True


def test_chromadb_by_request():
    """Test caching by ToolRequest."""
    print("\n=== Testing ChromaDB by ToolRequest ===")

    cache = ChromaDBCache(persist_directory=None, collection_name="test_request")

    request = ToolRequest(
        tool_id="test_tool",
        tool_type=ToolType.PYTHON_FUNCTION,
        function_name="test_func",
        parameters={"param1": "value1", "param2": "value2"},
    )

    result = ToolResult(
        tool_id="test_tool",
        success=True,
        data={"result": "cached_by_request"},
        error=None,
        execution_time_ms=100.0,
        timestamp=datetime.now(),
        token_estimate=10,
    )

    # Cache by request
    print("Caching by ToolRequest...")
    cache.set_by_request(request, result)

    # Retrieve by request
    print("Retrieving by ToolRequest...")
    retrieved = cache.get_by_request(request)

    if retrieved and retrieved.data == result.data:
        print("✓ Cache by request works!")
    else:
        print("✗ Cache by request failed!")
        return False

    # Test that different parameters generate different keys
    request2 = ToolRequest(
        tool_id="test_tool",
        tool_type=ToolType.PYTHON_FUNCTION,
        function_name="test_func",
        parameters={"param1": "value1", "param2": "different"},
    )

    retrieved2 = cache.get_by_request(request2)
    if not retrieved2:
        print("✓ Different parameters generate different cache keys")
    else:
        print("✗ Cache key collision!")
        return False

    cache.clear()
    return True


async def test_chromadb_async():
    """Test async operations."""
    print("\n=== Testing ChromaDB Async Operations ===")

    cache = ChromaDBCache(persist_directory=None, collection_name="test_async")

    result = ToolResult(
        tool_id="test_tool",
        success=True,
        data={"message": "async test"},
        error=None,
        execution_time_ms=100.0,
        timestamp=datetime.now(),
        token_estimate=10,
    )

    # Test async set/get
    print("Testing async set...")
    await cache.set_async("async_key", result)

    print("Testing async get...")
    retrieved = await cache.get_async("async_key")

    if retrieved and retrieved.data == result.data:
        print("✓ Async operations work!")
    else:
        print("✗ Async operations failed!")
        return False

    cache.clear()
    return True


def test_cache_config():
    """Test CacheConfig with ChromaDB."""
    print("\n=== Testing CacheConfig ===")

    config = CacheConfig(
        backend="chromadb",
        chromadb_persist_dir=None,
        chromadb_collection_name="test_config",
        default_ttl=3600,
    )

    cache = config.create_backend()

    if cache and isinstance(cache, ChromaDBCache):
        print("✓ CacheConfig creates ChromaDB backend correctly")
    else:
        print("✗ CacheConfig failed to create ChromaDB backend!")
        return False

    # Test it works
    result = ToolResult(
        tool_id="test_tool",
        success=True,
        data={"message": "config test"},
        error=None,
        execution_time_ms=100.0,
        timestamp=datetime.now(),
        token_estimate=10,
    )

    cache.set("config_key", result)
    retrieved = cache.get("config_key")

    if retrieved and retrieved.data == result.data:
        print("✓ CacheConfig backend works!")
    else:
        print("✗ CacheConfig backend failed!")
        return False

    cache.clear()
    return True


async def main():
    """Run all tests."""
    print("=" * 60)
    print("ChromaDB Cache Implementation Tests")
    print("=" * 60)

    try:
        import chromadb

        print(f"✓ ChromaDB version: {chromadb.__version__}")
    except ImportError:
        print("✗ ChromaDB not installed! Run: pip install chromadb")
        return False

    results = []

    # Run sync tests
    results.append(("Basic Operations", test_chromadb_basic()))
    results.append(("TTL Functionality", test_chromadb_ttl()))
    results.append(("By Request", test_chromadb_by_request()))
    results.append(("Cache Config", test_cache_config()))

    # Run async test

    results.append(("Async Operations", await test_chromadb_async()))

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = 0
    failed = 0

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:.<40} {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    return failed == 0


if __name__ == "__main__":
    import asyncio

    success = asyncio.run(main())
    sys.exit(0 if success else 1)
