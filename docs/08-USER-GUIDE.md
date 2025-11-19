# NOCP User Guide

A comprehensive guide to using NOCP (Near-Optimal Context Proxy) for building token-efficient AI applications.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Quick Start Tutorial](#quick-start-tutorial)
3. [Core Concepts](#core-concepts)
4. [Tutorial 1: Basic Tool Execution](#tutorial-1-basic-tool-execution)
5. [Tutorial 2: Caching for Performance](#tutorial-2-caching-for-performance)
6. [Tutorial 3: Context Optimization](#tutorial-3-context-optimization)
7. [Tutorial 4: Async and Concurrent Execution](#tutorial-4-async-and-concurrent-execution)
8. [Tutorial 5: Output Serialization](#tutorial-5-output-serialization)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nocp.git
cd nocp

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

### Requirements

- Python 3.11+
- pydantic >= 2.0
- google-generativeai >= 0.3.0
- litellm >= 1.55.0
- Optional: redis (for distributed caching)

### Environment Setup

Create a `.env` file in your project root:

```env
# API Keys
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key

# Configuration
NOCP_COMPRESSION_THRESHOLD=10000
NOCP_STUDENT_MODEL=openai/gpt-4o-mini
NOCP_TARGET_COMPRESSION_RATIO=0.40
```

---

## Quick Start Tutorial

Here's a minimal example to get you started:

```python
from nocp.core.act import ToolExecutor
from nocp.models.contracts import ToolRequest, ToolType

# Create executor
executor = ToolExecutor()

# Register a tool
@executor.register_tool("greet")
def greet(name: str) -> str:
    return f"Hello, {name}!"

# Execute the tool
request = ToolRequest(
    tool_id="greet",
    tool_type=ToolType.PYTHON_FUNCTION,
    function_name="greet",
    parameters={"name": "World"}
)

result = executor.execute(request)
print(result.data)  # "Hello, World!"
```

---

## Core Concepts

### 1. The Act-Assess-Articulate Pipeline

NOCP follows a three-stage optimization pipeline:

```
┌──────┐      ┌─────────┐      ┌────────────┐
│ ACT  │ ───► │ ASSESS  │ ───► │ ARTICULATE │
└──────┘      └─────────┘      └────────────┘
  Tools        Compress          Serialize
```

- **Act**: Execute tools and collect raw data
- **Assess**: Compress context to reduce tokens
- **Articulate**: Serialize output efficiently

### 2. Token Optimization

NOCP optimizes tokens at every stage:

- **Caching**: Avoid redundant tool executions
- **Compression**: Reduce context size by 50-70%
- **Serialization**: Save 30-60% on output tokens

### 3. Async First

NOCP is designed for async execution:

- Concurrent tool execution
- Non-blocking I/O operations
- Better resource utilization

---

## Tutorial 1: Basic Tool Execution

### Registering Tools

```python
from nocp.core.act import ToolExecutor
from nocp.models.contracts import ToolRequest, ToolType

executor = ToolExecutor()

# Simple tool
@executor.register_tool("add")
def add(a: int, b: int) -> int:
    return a + b

# Tool with complex return type
@executor.register_tool("fetch_users")
def fetch_users(limit: int = 10) -> list:
    return [
        {"id": i, "name": f"User{i}", "email": f"user{i}@example.com"}
        for i in range(limit)
    ]

# Tool with API call
import requests

@executor.register_tool("weather")
def get_weather(city: str) -> dict:
    # Simulated API call
    return {
        "city": city,
        "temperature": 72,
        "condition": "Sunny"
    }
```

### Executing Tools

```python
# Simple execution
request = ToolRequest(
    tool_id="add",
    tool_type=ToolType.PYTHON_FUNCTION,
    function_name="add",
    parameters={"a": 5, "b": 3}
)

result = executor.execute(request)
print(f"Result: {result.data}")  # 8
print(f"Execution time: {result.execution_time_ms}ms")
print(f"Tokens: {result.token_estimate}")
```

### Retry Logic

```python
from nocp.models.contracts import RetryConfig

# Configure retries
request = ToolRequest(
    tool_id="weather",
    tool_type=ToolType.API_CALL,
    function_name="get_weather",
    parameters={"city": "San Francisco"},
    timeout_seconds=10,
    retry_config=RetryConfig(
        max_attempts=3,
        backoff_multiplier=2.0,
        initial_delay_ms=100
    )
)

result = executor.execute(request)
print(f"Retry count: {result.retry_count}")
```

---

## Tutorial 2: Caching for Performance

### In-Memory Caching

```python
from nocp.core.cache import LRUCache
from nocp.core.act import ToolExecutor

# Create cache
cache = LRUCache(max_size=1000, default_ttl=3600)  # 1 hour TTL

# Create executor with cache
executor = ToolExecutor(cache=cache)

@executor.register_tool("expensive_computation")
def expensive_computation(n: int) -> int:
    # Simulate expensive computation
    import time
    time.sleep(1)
    return sum(i * i for i in range(n))

# First call (slow)
request = ToolRequest(
    tool_id="expensive_computation",
    tool_type=ToolType.PYTHON_FUNCTION,
    function_name="expensive_computation",
    parameters={"n": 10000}
)

import time
start = time.time()
result1 = executor.execute(request)
print(f"First call: {time.time() - start:.2f}s")  # ~1s

# Second call (fast - from cache)
start = time.time()
result2 = executor.execute(request)
print(f"Second call: {time.time() - start:.2f}s")  # <0.01s

# Check cache stats
stats = cache.stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Cache size: {stats['size']}/{stats['max_size']}")
```

### Redis Caching (Distributed)

```python
from nocp.core.cache import RedisCache

# Create Redis cache
cache = RedisCache(
    host="localhost",
    port=6379,
    db=0,
    default_ttl=3600
)

executor = ToolExecutor(cache=cache)

# Now cache is shared across processes/servers!
```

### Cache Configuration

```python
from nocp.core.cache import CacheConfig

# Configure via config object
config = CacheConfig(
    backend="memory",  # or "redis"
    max_size=5000,
    default_ttl=7200,
    enabled=True
)

cache = config.create_backend()
executor = ToolExecutor(cache=cache)
```

### Bypassing Cache

```python
# Execute without cache (force fresh execution)
result = executor.execute(request, use_cache=False)
```

---

## Tutorial 3: Context Optimization

### Basic Context Optimization

```python
from nocp.core.assess import ContextManager
from nocp.models.contracts import ContextData, ToolResult
from datetime import datetime

# Create context manager
manager = ContextManager(
    compression_threshold=10_000,  # Compress if >10k tokens
    target_compression_ratio=0.40,  # Target 60% reduction
    enable_litellm=True  # Use LLM for summarization
)

# Create sample tool results
tool_results = [
    ToolResult(
        tool_id="fetch_users",
        success=True,
        data=[{"id": i, "name": f"User{i}"} for i in range(1000)],
        error=None,
        execution_time_ms=100.0,
        timestamp=datetime.now(),
        token_estimate=5000
    )
]

# Create context
context = ContextData(
    tool_results=tool_results,
    transient_context={"query": "Find all active users"},
    max_tokens=50_000
)

# Optimize context
optimized = manager.optimize(context)

print(f"Original tokens: {optimized.original_tokens}")
print(f"Optimized tokens: {optimized.optimized_tokens}")
print(f"Compression ratio: {optimized.compression_ratio:.2%}")
print(f"Method used: {optimized.method_used}")
print(f"Time: {optimized.compression_time_ms:.2f}ms")
```

### Compression Strategies

#### 1. Semantic Pruning (for structured data)

```python
# Automatically selected for large lists/arrays
large_dataset = [{"id": i, "data": "x"*100} for i in range(10000)]

result = ToolResult(
    tool_id="fetch_data",
    success=True,
    data=large_dataset,
    error=None,
    execution_time_ms=100.0,
    timestamp=datetime.now(),
    token_estimate=25000  # Large token count
)

context = ContextData(tool_results=[result])
optimized = manager.optimize(context)

# Will use SEMANTIC_PRUNING to keep top-k relevant items
assert optimized.method_used == CompressionMethod.SEMANTIC_PRUNING
```

#### 2. Knowledge Distillation (for verbose text)

```python
# Automatically selected for large text content
verbose_text = "This is a very long verbose text. " * 1000

result = ToolResult(
    tool_id="fetch_logs",
    success=True,
    data=verbose_text,
    error=None,
    execution_time_ms=100.0,
    timestamp=datetime.now(),
    token_estimate=7500
)

context = ContextData(tool_results=[result])
optimized = manager.optimize(context)

# Will use KNOWLEDGE_DISTILLATION to summarize
assert optimized.method_used == CompressionMethod.KNOWLEDGE_DISTILLATION
```

#### 3. History Compaction (for conversations)

```python
from nocp.models.contracts import ChatMessage

# Create conversation history
messages = [
    ChatMessage(
        role="user" if i % 2 == 0 else "assistant",
        content=f"Message {i}",
        timestamp=datetime.now(),
        tokens=10
    )
    for i in range(20)  # >10 messages triggers compaction
]

context = ContextData(
    tool_results=[],
    message_history=messages
)

optimized = manager.optimize(context)

# Will keep last 5 messages, summarize the rest
assert optimized.method_used == CompressionMethod.HISTORY_COMPACTION
```

### Manual Strategy Selection

```python
# Force specific compression strategy
context = ContextData(
    tool_results=tool_results,
    compression_strategy="semantic_pruning"  # Force this strategy
)

optimized = manager.optimize(context)
```

---

## Tutorial 4: Async and Concurrent Execution

### Async Tool Execution

```python
import asyncio
from nocp.core.act import ToolExecutor

executor = ToolExecutor()

# Register async tool
@executor.register_async_tool("fetch_async")
async def fetch_async(url: str) -> dict:
    # Simulate async HTTP request
    await asyncio.sleep(0.1)
    return {"url": url, "status": 200}

# Execute async
async def main():
    request = ToolRequest(
        tool_id="fetch_async",
        tool_type=ToolType.API_CALL,
        function_name="fetch_async",
        parameters={"url": "https://api.example.com"}
    )

    result = await executor.execute_async(request)
    print(result.data)

asyncio.run(main())
```

### Concurrent Tool Execution

```python
from nocp.core.async_modules import ConcurrentToolExecutor

executor = ToolExecutor()

@executor.register_async_tool("process")
async def process(item_id: int) -> dict:
    await asyncio.sleep(0.1)  # Simulate processing
    return {"id": item_id, "processed": True}

async def main():
    # Create concurrent executor
    concurrent = ConcurrentToolExecutor(
        executor,
        max_concurrent=10  # Process 10 items at a time
    )

    # Create 100 requests
    requests = [
        ToolRequest(
            tool_id="process",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="process",
            parameters={"item_id": i}
        )
        for i in range(100)
    ]

    # Execute all concurrently
    import time
    start = time.time()
    results = await concurrent.execute_many(requests)
    elapsed = time.time() - start

    successful = [r for r in results if not isinstance(r, Exception)]
    print(f"Processed {len(successful)} items in {elapsed:.2f}s")
    # With concurrency=10, processes 100 items in ~1s instead of ~10s

asyncio.run(main())
```

### Async Context Optimization

```python
from nocp.core.async_modules import AsyncContextManager

async def main():
    manager = AsyncContextManager(compression_threshold=5000)

    context = ContextData(tool_results=tool_results)

    # Optimize asynchronously
    optimized = await manager.optimize_async(context)

    print(f"Compression: {optimized.compression_ratio:.2%}")

asyncio.run(main())
```

### Async Serialization

```python
from nocp.core.async_modules import AsyncOutputSerializer
from pydantic import BaseModel

class Response(BaseModel):
    users: list
    total: int

async def main():
    serializer = AsyncOutputSerializer()

    data = Response(users=[...], total=100)
    request = SerializationRequest(data=data)

    # Serialize asynchronously
    result = await serializer.serialize_async(request)

    print(f"Format: {result.format_used}")
    print(f"Savings: {result.savings_ratio:.2%}")

asyncio.run(main())
```

---

## Tutorial 5: Output Serialization

### Basic Serialization

```python
from nocp.core.articulate import OutputSerializer
from nocp.models.contracts import SerializationRequest
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    email: str

class UserList(BaseModel):
    users: list[User]
    total: int

serializer = OutputSerializer()

# Create data
users = [User(id=i, name=f"User{i}", email=f"user{i}@example.com") for i in range(10)]
data = UserList(users=users, total=10)

# Serialize
request = SerializationRequest(data=data)
result = serializer.serialize(request)

print(f"Format used: {result.format_used}")
print(f"Original tokens: {result.original_tokens}")
print(f"Optimized tokens: {result.optimized_tokens}")
print(f"Savings: {result.savings_ratio:.1%}")
print(f"\nSerialized:\n{result.serialized_text}")
```

### TOON Format (Tabular)

For tabular data (uniform arrays), NOCP uses the TOON format:

```python
# Data with uniform structure
users = [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"},
    {"id": 3, "name": "Charlie", "email": "charlie@example.com"},
]

# TOON format output:
# id|name|email
# 1|Alice|alice@example.com
# 2|Bob|bob@example.com
# 3|Charlie|charlie@example.com

# Saves ~30% tokens compared to JSON!
```

### Compact JSON

For nested structures:

```python
# Force compact JSON
request = SerializationRequest(
    data=data,
    force_format="compact_json"
)

result = serializer.serialize(request)
# Removes whitespace and indentation
```

---

## Best Practices

### 1. Use Caching Strategically

```python
# DO: Cache expensive operations
@executor.register_tool("database_query")
def database_query(query: str) -> list:
    # Expensive database operation
    return db.execute(query)

# DON'T: Cache time-sensitive data
@executor.register_tool("current_time")
def current_time() -> str:
    return datetime.now().isoformat()

# Execute with cache disabled for time-sensitive operations
result = executor.execute(request, use_cache=False)
```

### 2. Optimize Context Early

```python
# Compress context before passing to LLM
optimized = manager.optimize(context)

# Use optimized text for LLM input
llm_response = llm.generate(
    prompt=optimized.optimized_text,
    max_tokens=1000
)
```

### 3. Use Async for I/O-Bound Operations

```python
# Good: Async for network calls
@executor.register_async_tool("api_call")
async def api_call(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

# OK: Sync for CPU-bound operations
@executor.register_tool("compute")
def compute(n: int) -> int:
    return sum(i * i for i in range(n))
```

### 4. Monitor Performance

```python
# Track execution metrics
result = executor.execute(request)

print(f"Execution time: {result.execution_time_ms}ms")
print(f"Token estimate: {result.token_estimate}")
print(f"Retry count: {result.retry_count}")

# Track cache performance
stats = cache.stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
print(f"Cache size: {stats['size']}")
print(f"Evictions: {stats['evictions']}")
```

### 5. Handle Errors Gracefully

```python
from nocp.exceptions import ToolExecutionError

try:
    result = executor.execute(request)
except ToolExecutionError as e:
    print(f"Tool execution failed: {e}")
    print(f"Details: {e.details}")
except TimeoutError as e:
    print(f"Tool timed out: {e}")
```

---

## Troubleshooting

### Issue: Cache not working

**Problem:** Tools execute every time despite caching being enabled.

**Solution:**
```python
# Ensure cache is passed to executor
cache = LRUCache()
executor = ToolExecutor(cache=cache)  # ← Important!

# Verify cache is enabled
result = executor.execute(request, use_cache=True)

# Check cache stats
stats = cache.stats()
print(stats)  # Should show hits/misses
```

### Issue: Context compression not happening

**Problem:** Context is not being compressed.

**Solution:**
```python
# Check if context exceeds threshold
manager = ContextManager(compression_threshold=10_000)

# Ensure token count is high enough
context = ContextData(tool_results=results)

# Optimize to check if compression is triggered
optimized = manager.optimize(context)
print(f"Original tokens: {optimized.original_tokens}")
print(f"Method used: {optimized.method_used}")  # Should not be NONE if > threshold
```

### Issue: Async tools not working

**Problem:** Async tools fail with errors.

**Solution:**
```python
# Use register_async_tool for async functions
@executor.register_async_tool("async_tool")  # ← Not register_tool!
async def async_tool(param: str) -> dict:
    await asyncio.sleep(0.1)
    return {"result": param}

# Use execute_async
result = await executor.execute_async(request)  # ← Not execute!
```

### Issue: Redis connection errors

**Problem:** RedisCache fails to connect.

**Solution:**
```python
# Ensure Redis server is running
# docker run -d -p 6379:6379 redis

# Test connection
try:
    cache = RedisCache(host="localhost", port=6379)
    print("Connected!")
except Exception as e:
    print(f"Connection failed: {e}")
```

---

## Next Steps

- Explore [Advanced Examples](../examples/) for real-world use cases
- Read [API Reference](07-API-REFERENCE.md) for complete API documentation
- Check out [Architecture](01-ARCHITECTURE.md) to understand internals
- See [Development Roadmap](03-DEVELOPMENT-ROADMAP.md) for upcoming features

---

For questions or issues, please visit: https://github.com/yourusername/nocp/issues
