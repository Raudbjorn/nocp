"""
Performance benchmarks comparing async vs sync execution.

Measures execution time, throughput, and resource usage for various scenarios.
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from nocp.core.act import ToolExecutor
from nocp.core.async_modules import (
    AsyncContextManager,
    AsyncOutputSerializer,
    ConcurrentToolExecutor
)
from nocp.core.assess import ContextManager
from nocp.core.articulate import OutputSerializer
from nocp.core.cache import LRUCache
from nocp.models.contracts import (
    ToolRequest,
    ToolResult,
    ToolType,
    ContextData,
    SerializationRequest
)
from pydantic import BaseModel, Field


# Sample data models
class User(BaseModel):
    id: int
    name: str
    email: str
    age: int


class UserListResponse(BaseModel):
    users: List[User]
    total: int
    page: int


def create_sample_users(count: int = 100) -> UserListResponse:
    """Create sample user data for benchmarking."""
    users = [
        User(
            id=i,
            name=f"User {i}",
            email=f"user{i}@example.com",
            age=20 + (i % 50)
        )
        for i in range(count)
    ]
    return UserListResponse(users=users, total=count, page=1)


# Benchmark functions
def benchmark_sync_tool_execution(iterations: int = 100):
    """Benchmark synchronous tool execution."""
    print(f"\n{'='*60}")
    print(f"Benchmark: Sync Tool Execution ({iterations} iterations)")
    print('='*60)

    executor = ToolExecutor()

    @executor.register_tool("compute")
    def compute(value: int) -> dict:
        # Simulate some computation
        result = sum(i * i for i in range(value))
        return {"result": result, "input": value}

    request = ToolRequest(
        tool_id="compute",
        tool_type=ToolType.PYTHON_FUNCTION,
        function_name="compute",
        parameters={"value": 1000}
    )

    start = time.perf_counter()
    for _ in range(iterations):
        executor.execute(request, use_cache=False)
    elapsed = time.perf_counter() - start

    print(f"Total time: {elapsed:.3f}s")
    print(f"Average time per execution: {(elapsed/iterations)*1000:.2f}ms")
    print(f"Throughput: {iterations/elapsed:.2f} ops/sec")

    return elapsed


async def benchmark_async_tool_execution(iterations: int = 100):
    """Benchmark asynchronous tool execution."""
    print(f"\n{'='*60}")
    print(f"Benchmark: Async Tool Execution ({iterations} iterations)")
    print('='*60)

    executor = ToolExecutor()

    @executor.register_async_tool("compute_async")
    async def compute_async(value: int) -> dict:
        # Simulate some async computation
        result = sum(i * i for i in range(value))
        return {"result": result, "input": value}

    request = ToolRequest(
        tool_id="compute_async",
        tool_type=ToolType.PYTHON_FUNCTION,
        function_name="compute_async",
        parameters={"value": 1000}
    )

    start = time.perf_counter()
    for _ in range(iterations):
        await executor.execute_async(request, use_cache=False)
    elapsed = time.perf_counter() - start

    print(f"Total time: {elapsed:.3f}s")
    print(f"Average time per execution: {(elapsed/iterations)*1000:.2f}ms")
    print(f"Throughput: {iterations/elapsed:.2f} ops/sec")

    return elapsed


async def benchmark_concurrent_tool_execution(iterations: int = 100, concurrency: int = 10):
    """Benchmark concurrent tool execution."""
    print(f"\n{'='*60}")
    print(f"Benchmark: Concurrent Tool Execution")
    print(f"  Iterations: {iterations}, Concurrency: {concurrency}")
    print('='*60)

    executor = ToolExecutor()

    @executor.register_async_tool("compute_concurrent")
    async def compute_concurrent(value: int) -> dict:
        # Simulate async I/O with computation
        await asyncio.sleep(0.01)  # Simulate I/O delay
        result = sum(i * i for i in range(value))
        return {"result": result, "input": value}

    concurrent_executor = ConcurrentToolExecutor(executor, max_concurrent=concurrency)

    requests = [
        ToolRequest(
            tool_id="compute_concurrent",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="compute_concurrent",
            parameters={"value": 1000}
        )
        for _ in range(iterations)
    ]

    start = time.perf_counter()
    results = await concurrent_executor.execute_many(requests)
    elapsed = time.perf_counter() - start

    successful = sum(1 for r in results if not isinstance(r, Exception))

    print(f"Total time: {elapsed:.3f}s")
    print(f"Average time per execution: {(elapsed/iterations)*1000:.2f}ms")
    print(f"Throughput: {iterations/elapsed:.2f} ops/sec")
    print(f"Successful executions: {successful}/{iterations}")

    return elapsed


def benchmark_sync_context_optimization():
    """Benchmark synchronous context optimization."""
    print(f"\n{'='*60}")
    print("Benchmark: Sync Context Optimization")
    print('='*60)

    manager = ContextManager(
        compression_threshold=1000,
        enable_litellm=False
    )

    # Create sample tool results
    tool_results = [
        ToolResult(
            tool_id=f"tool_{i}",
            success=True,
            data={"data": "x" * 1000},  # 1000 chars of data
            error=None,
            execution_time_ms=10.0,
            timestamp=datetime.now(),
            token_estimate=250
        )
        for i in range(20)
    ]

    context = ContextData(
        tool_results=tool_results,
        transient_context={"query": "test"},
        max_tokens=50000
    )

    iterations = 50
    start = time.perf_counter()
    for _ in range(iterations):
        manager.optimize(context)
    elapsed = time.perf_counter() - start

    print(f"Total time: {elapsed:.3f}s")
    print(f"Average time per optimization: {(elapsed/iterations)*1000:.2f}ms")
    print(f"Throughput: {iterations/elapsed:.2f} ops/sec")

    return elapsed


async def benchmark_async_context_optimization():
    """Benchmark asynchronous context optimization."""
    print(f"\n{'='*60}")
    print("Benchmark: Async Context Optimization")
    print('='*60)

    manager = AsyncContextManager(
        compression_threshold=1000,
        enable_litellm=False
    )

    # Create sample tool results
    tool_results = [
        ToolResult(
            tool_id=f"tool_{i}",
            success=True,
            data={"data": "x" * 1000},
            error=None,
            execution_time_ms=10.0,
            timestamp=datetime.now(),
            token_estimate=250
        )
        for i in range(20)
    ]

    context = ContextData(
        tool_results=tool_results,
        transient_context={"query": "test"},
        max_tokens=50000
    )

    iterations = 50
    start = time.perf_counter()
    for _ in range(iterations):
        await manager.optimize_async(context)
    elapsed = time.perf_counter() - start

    print(f"Total time: {elapsed:.3f}s")
    print(f"Average time per optimization: {(elapsed/iterations)*1000:.2f}ms")
    print(f"Throughput: {iterations/elapsed:.2f} ops/sec")

    return elapsed


def benchmark_sync_serialization():
    """Benchmark synchronous serialization."""
    print(f"\n{'='*60}")
    print("Benchmark: Sync Serialization")
    print('='*60)

    serializer = OutputSerializer()
    data = create_sample_users(count=1000)
    request = SerializationRequest(data=data)

    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        serializer.serialize(request)
    elapsed = time.perf_counter() - start

    print(f"Total time: {elapsed:.3f}s")
    print(f"Average time per serialization: {(elapsed/iterations)*1000:.2f}ms")
    print(f"Throughput: {iterations/elapsed:.2f} ops/sec")

    return elapsed


async def benchmark_async_serialization():
    """Benchmark asynchronous serialization."""
    print(f"\n{'='*60}")
    print("Benchmark: Async Serialization")
    print('='*60)

    serializer = AsyncOutputSerializer()
    data = create_sample_users(count=1000)
    request = SerializationRequest(data=data)

    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        await serializer.serialize_async(request)
    elapsed = time.perf_counter() - start

    print(f"Total time: {elapsed:.3f}s")
    print(f"Average time per serialization: {(elapsed/iterations)*1000:.2f}ms")
    print(f"Throughput: {iterations/elapsed:.2f} ops/sec")

    return elapsed


def benchmark_cache_performance():
    """Benchmark cache performance impact."""
    print(f"\n{'='*60}")
    print("Benchmark: Cache Performance Impact")
    print('='*60)

    # Without cache
    executor_no_cache = ToolExecutor()

    @executor_no_cache.register_tool("compute_no_cache")
    def compute_no_cache(value: int) -> dict:
        result = sum(i * i for i in range(value))
        return {"result": result}

    request = ToolRequest(
        tool_id="compute_no_cache",
        tool_type=ToolType.PYTHON_FUNCTION,
        function_name="compute_no_cache",
        parameters={"value": 5000}
    )

    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        executor_no_cache.execute(request, use_cache=False)
    elapsed_no_cache = time.perf_counter() - start

    print(f"Without cache:")
    print(f"  Total time: {elapsed_no_cache:.3f}s")
    print(f"  Throughput: {iterations/elapsed_no_cache:.2f} ops/sec")

    # With cache
    cache = LRUCache(max_size=1000, default_ttl=3600)
    executor_with_cache = ToolExecutor(cache=cache)

    @executor_with_cache.register_tool("compute_with_cache")
    def compute_with_cache(value: int) -> dict:
        result = sum(i * i for i in range(value))
        return {"result": result}

    request_cached = ToolRequest(
        tool_id="compute_with_cache",
        tool_type=ToolType.PYTHON_FUNCTION,
        function_name="compute_with_cache",
        parameters={"value": 5000}
    )

    start = time.perf_counter()
    for _ in range(iterations):
        executor_with_cache.execute(request_cached, use_cache=True)
    elapsed_with_cache = time.perf_counter() - start

    print(f"\nWith cache (same request repeated):")
    print(f"  Total time: {elapsed_with_cache:.3f}s")
    print(f"  Throughput: {iterations/elapsed_with_cache:.2f} ops/sec")
    print(f"  Speedup: {elapsed_no_cache/elapsed_with_cache:.2f}x")

    stats = cache.stats()
    print(f"  Cache hit rate: {stats['hit_rate']:.2%}")


async def run_all_benchmarks():
    """Run all benchmarks and print summary."""
    print("\n" + "="*60)
    print("NOCP Performance Benchmarks - Async vs Sync")
    print("="*60)

    results = {}

    # Tool execution benchmarks
    results['sync_tools'] = benchmark_sync_tool_execution(iterations=100)
    results['async_tools'] = await benchmark_async_tool_execution(iterations=100)
    results['concurrent_tools'] = await benchmark_concurrent_tool_execution(
        iterations=100,
        concurrency=10
    )

    # Context optimization benchmarks
    results['sync_context'] = benchmark_sync_context_optimization()
    results['async_context'] = await benchmark_async_context_optimization()

    # Serialization benchmarks
    results['sync_serialization'] = benchmark_sync_serialization()
    results['async_serialization'] = await benchmark_async_serialization()

    # Cache benchmarks
    benchmark_cache_performance()

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print('='*60)

    print("\nTool Execution:")
    speedup = results['sync_tools'] / results['async_tools']
    print(f"  Async speedup: {speedup:.2f}x")

    concurrent_speedup = results['sync_tools'] / results['concurrent_tools']
    print(f"  Concurrent speedup (10 concurrent): {concurrent_speedup:.2f}x")

    print("\nContext Optimization:")
    speedup = results['sync_context'] / results['async_context']
    print(f"  Async speedup: {speedup:.2f}x")

    print("\nSerialization:")
    speedup = results['sync_serialization'] / results['async_serialization']
    print(f"  Async speedup: {speedup:.2f}x")

    print("\n" + "="*60)
    print("Benchmark Complete!")
    print("="*60 + "\n")


def main():
    """Main entry point."""
    try:
        asyncio.run(run_all_benchmarks())
        return 0
    except Exception as e:
        print(f"\nError during benchmarks: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
