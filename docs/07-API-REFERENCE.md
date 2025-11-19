# NOCP API Reference

Complete API documentation for all modules, classes, and functions in NOCP.

## Table of Contents

- [Core Modules](#core-modules)
  - [ToolExecutor](#toolexecutor)
  - [ContextManager](#contextmanager)
  - [OutputSerializer](#outputserializer)
  - [Cache](#cache)
- [Async Modules](#async-modules)
  - [AsyncContextManager](#asynccontextmanager)
  - [AsyncOutputSerializer](#asyncoutputserializer)
  - [ConcurrentToolExecutor](#concurrenttoolexecutor)
- [Data Models](#data-models)
- [Exceptions](#exceptions)

---

## Core Modules

### ToolExecutor

**Location:** `nocp.core.act.ToolExecutor`

Manages tool registration and execution with retry logic and caching support.

#### Constructor

```python
ToolExecutor(cache: Optional[CacheBackend] = None)
```

**Parameters:**
- `cache` (Optional[CacheBackend]): Optional cache backend for caching tool results

#### Methods

##### register_tool

```python
@executor.register_tool(tool_id: str, tool_type: ToolType = ToolType.PYTHON_FUNCTION)
def your_tool(param: type) -> return_type:
    ...
```

Decorator to register a synchronous tool.

**Parameters:**
- `tool_id` (str): Unique identifier for the tool
- `tool_type` (ToolType): Category of tool (default: PYTHON_FUNCTION)

**Returns:** Decorator function

**Example:**
```python
executor = ToolExecutor()

@executor.register_tool("fetch_users")
def fetch_users(count: int = 10) -> list:
    return [{"id": i, "name": f"User{i}"} for i in range(count)]
```

##### register_async_tool

```python
@executor.register_async_tool(tool_id: str)
async def your_async_tool(param: type) -> return_type:
    ...
```

Decorator to register an asynchronous tool.

**Parameters:**
- `tool_id` (str): Unique identifier for the tool

**Returns:** Decorator function

**Example:**
```python
@executor.register_async_tool("fetch_users_async")
async def fetch_users_async(count: int = 10) -> list:
    await asyncio.sleep(0.1)  # Simulate async I/O
    return [{"id": i, "name": f"User{i}"} for i in range(count)]
```

##### execute

```python
execute(request: ToolRequest, use_cache: bool = True) -> ToolResult
```

Execute a registered tool with retry logic and optional caching.

**Parameters:**
- `request` (ToolRequest): Tool request with execution parameters
- `use_cache` (bool): Whether to use cache for this request (default: True)

**Returns:** ToolResult with execution outcome and metadata

**Raises:**
- `ToolExecutionError`: If execution fails after all retries
- `TimeoutError`: If execution exceeds timeout

**Example:**
```python
request = ToolRequest(
    tool_id="fetch_users",
    tool_type=ToolType.PYTHON_FUNCTION,
    function_name="fetch_users",
    parameters={"count": 5}
)

result = executor.execute(request)
print(result.data)  # [{"id": 0, ...}, ...]
```

##### execute_async

```python
async def execute_async(request: ToolRequest, use_cache: bool = True) -> ToolResult
```

Async version of execute() for concurrent execution.

**Parameters:** Same as `execute()`

**Returns:** ToolResult

**Example:**
```python
result = await executor.execute_async(request)
```

##### list_tools

```python
list_tools() -> list[str]
```

Get list of registered tool IDs.

**Returns:** List of tool IDs

##### validate_tool

```python
validate_tool(tool_id: str) -> bool
```

Check if tool is registered and available.

**Parameters:**
- `tool_id` (str): Tool ID to validate

**Returns:** True if tool exists, False otherwise

---

### ContextManager

**Location:** `nocp.core.assess.ContextManager`

Optimizes context to reduce token usage before LLM calls using various compression techniques.

#### Constructor

```python
ContextManager(
    student_model: str = "openai/gpt-4o-mini",
    compression_threshold: int = 10_000,
    target_compression_ratio: float = 0.40,
    enable_litellm: bool = True
)
```

**Parameters:**
- `student_model` (str): Lightweight model for summarization
- `compression_threshold` (int): Only compress if input exceeds this token count
- `target_compression_ratio` (float): Target ratio (0.40 = 60% reduction)
- `enable_litellm` (bool): Enable LiteLLM integration (requires API keys)

#### Methods

##### optimize

```python
optimize(context: ContextData) -> OptimizedContext
```

Main entry point for context optimization.

**Parameters:**
- `context` (ContextData): Context with tool results and conversation history

**Returns:** OptimizedContext with compressed text and metrics

**Example:**
```python
manager = ContextManager(compression_threshold=5000)

context = ContextData(
    tool_results=[large_result],
    transient_context={"query": "Summarize this"},
    max_tokens=50_000
)

optimized = manager.optimize(context)
print(f"Compressed from {optimized.original_tokens} to {optimized.optimized_tokens}")
print(f"Compression ratio: {optimized.compression_ratio:.2%}")
```

##### estimate_tokens

```python
estimate_tokens(text: str, model: str = "gpt-4") -> int
```

Estimate token count using litellm's token_counter or fallback.

**Parameters:**
- `text` (str): Text to count tokens for
- `model` (str): Model tokenizer to use

**Returns:** Estimated token count

##### select_strategy

```python
select_strategy(context: ContextData) -> CompressionMethod
```

Auto-detect optimal compression strategy based on context characteristics.

**Parameters:**
- `context` (ContextData): Context to analyze

**Returns:** Selected CompressionMethod

---

### OutputSerializer

**Location:** `nocp.core.articulate.OutputSerializer`

Serializes Pydantic models to token-optimized formats (TOON or compact JSON).

#### Constructor

```python
OutputSerializer()
```

#### Methods

##### serialize

```python
serialize(request: SerializationRequest) -> SerializedOutput
```

Main serialization entry point with automatic format negotiation.

**Parameters:**
- `request` (SerializationRequest): Request with data and options

**Returns:** SerializedOutput with optimized serialization and metrics

**Example:**
```python
serializer = OutputSerializer()

data = UserListModel(users=[...])
request = SerializationRequest(data=data)
result = serializer.serialize(request)

print(f"Format: {result.format_used}")
print(f"Savings: {result.savings_ratio:.1%}")
print(f"Serialized: {result.serialized_text}")
```

##### negotiate_format

```python
negotiate_format(model: BaseModel) -> SerializationFormat
```

Analyze Pydantic model to select optimal serialization format.

**Parameters:**
- `model` (BaseModel): Pydantic model to analyze

**Returns:** Selected SerializationFormat

---

### Cache

**Location:** `nocp.core.cache`

Caching layer with in-memory LRU and optional Redis support.

#### LRUCache

In-memory LRU (Least Recently Used) cache for tool results.

**Constructor:**
```python
LRUCache(max_size: int = 1000, default_ttl: Optional[int] = 3600)
```

**Parameters:**
- `max_size` (int): Maximum number of items to cache
- `default_ttl` (Optional[int]): Default time-to-live in seconds (None = no expiration)

**Methods:**

##### get / set

```python
get(key: str) -> Optional[ToolResult]
set(key: str, value: ToolResult, ttl_seconds: Optional[int] = None) -> None
```

Get/set cache values.

##### get_by_request / set_by_request

```python
get_by_request(request: ToolRequest) -> Optional[ToolResult]
set_by_request(request: ToolRequest, result: ToolResult, ttl_seconds: Optional[int] = None) -> None
```

Cache operations using ToolRequest as key.

##### stats

```python
stats() -> Dict[str, Any]
```

Get cache statistics including hit rate, size, evictions.

**Example:**
```python
cache = LRUCache(max_size=1000, default_ttl=3600)
executor = ToolExecutor(cache=cache)

# Execute tool (cached automatically)
result = executor.execute(request)

# Check cache stats
stats = cache.stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Cache size: {stats['size']}/{stats['max_size']}")
```

#### RedisCache

Redis-backed distributed cache for tool results.

**Constructor:**
```python
RedisCache(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    password: Optional[str] = None,
    default_ttl: Optional[int] = 3600,
    key_prefix: str = "nocp:cache:"
)
```

**Requirements:** `pip install redis`

**Example:**
```python
cache = RedisCache(host="localhost", port=6379)
executor = ToolExecutor(cache=cache)
```

#### CacheConfig

Configuration helper for creating cache backends.

```python
config = CacheConfig(
    backend="memory",  # or "redis"
    max_size=1000,
    default_ttl=3600,
    redis_host="localhost",
    redis_port=6379,
    enabled=True
)

cache = config.create_backend()
```

---

## Async Modules

### AsyncContextManager

**Location:** `nocp.core.async_modules.AsyncContextManager`

Async version of ContextManager for concurrent context optimization.

#### Constructor

Same as ContextManager.

#### Methods

##### optimize_async

```python
async def optimize_async(context: ContextData) -> OptimizedContext
```

Async version of optimize() for concurrent execution.

**Example:**
```python
manager = AsyncContextManager(compression_threshold=5000)
result = await manager.optimize_async(context)
```

---

### AsyncOutputSerializer

**Location:** `nocp.core.async_modules.AsyncOutputSerializer`

Async version of OutputSerializer for concurrent serialization.

#### Methods

##### serialize_async

```python
async def serialize_async(request: SerializationRequest) -> SerializedOutput
```

Async version of serialize().

**Example:**
```python
serializer = AsyncOutputSerializer()
result = await serializer.serialize_async(request)
```

---

### ConcurrentToolExecutor

**Location:** `nocp.core.async_modules.ConcurrentToolExecutor`

Wrapper for executing multiple tools concurrently with semaphore-based concurrency control.

#### Constructor

```python
ConcurrentToolExecutor(tool_executor: ToolExecutor, max_concurrent: int = 5)
```

**Parameters:**
- `tool_executor` (ToolExecutor): ToolExecutor instance
- `max_concurrent` (int): Maximum number of concurrent tool executions

#### Methods

##### execute_many

```python
async def execute_many(requests: List[ToolRequest]) -> List[ToolResult]
```

Execute multiple tool requests concurrently.

**Parameters:**
- `requests` (List[ToolRequest]): List of tool requests

**Returns:** List of ToolResult objects (or exceptions)

**Example:**
```python
executor = ToolExecutor()
concurrent = ConcurrentToolExecutor(executor, max_concurrent=10)

requests = [create_request(i) for i in range(50)]
results = await concurrent.execute_many(requests)

successful = [r for r in results if not isinstance(r, Exception)]
print(f"Successful: {len(successful)}/{len(requests)}")
```

##### execute_many_ordered

```python
async def execute_many_ordered(requests: List[ToolRequest]) -> List[ToolResult]
```

Execute tools concurrently but return results in the same order as requests.

---

## Data Models

### ToolRequest

```python
class ToolRequest(BaseModel):
    tool_id: str
    tool_type: ToolType
    function_name: str
    parameters: Dict[str, Any] = {}
    timeout_seconds: int = 30
    retry_config: Optional[RetryConfig] = None
```

Input to tool execution.

### ToolResult

```python
class ToolResult(BaseModel):
    tool_id: str
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time_ms: float
    timestamp: datetime
    token_estimate: int
    retry_count: int = 0
    metadata: Dict[str, Any] = {}
```

Output from tool execution.

### ContextData

```python
class ContextData(BaseModel):
    tool_results: List[ToolResult]
    transient_context: Dict[str, Any] = {}
    persistent_context: Optional[str] = None
    message_history: List[ChatMessage] = []
    max_tokens: int = 100_000
    compression_strategy: Optional[str] = None
```

Input to context optimization.

### OptimizedContext

```python
class OptimizedContext(BaseModel):
    optimized_text: str
    original_tokens: int
    optimized_tokens: int
    compression_ratio: float
    method_used: CompressionMethod
    compression_time_ms: float
    semantic_similarity_score: Optional[float] = None
    metadata: Dict[str, Any] = {}
```

Output from context optimization.

### SerializationRequest

```python
class SerializationRequest(BaseModel):
    data: BaseModel
    force_format: Optional[str] = None
    include_length_markers: bool = True
    validate_output: bool = True
```

Input to serialization.

### SerializedOutput

```python
class SerializedOutput(BaseModel):
    serialized_text: str
    format_used: SerializationFormat
    original_tokens: int
    optimized_tokens: int
    savings_ratio: float
    is_valid: bool
    serialization_time_ms: float
    schema_complexity: str
```

Output from serialization.

---

## Exceptions

### ToolExecutionError

Raised when tool execution fails after all retries.

```python
class ToolExecutionError(Exception):
    def __init__(self, message: str, details: Dict[str, Any] = None)
```

### CompressionError

Raised when context compression fails.

```python
class CompressionError(Exception):
    pass
```

### SerializationError

Raised when output serialization fails.

```python
class SerializationError(Exception):
    pass
```

---

## Enumerations

### ToolType

```python
class ToolType(str, Enum):
    DATABASE_QUERY = "database_query"
    API_CALL = "api_call"
    RAG_RETRIEVAL = "rag_retrieval"
    FILE_OPERATION = "file_operation"
    PYTHON_FUNCTION = "python_function"
```

### CompressionMethod

```python
class CompressionMethod(str, Enum):
    SEMANTIC_PRUNING = "semantic_pruning"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    HISTORY_COMPACTION = "history_compaction"
    NONE = "none"
```

### SerializationFormat

```python
class SerializationFormat(str, Enum):
    TOON = "toon"
    COMPACT_JSON = "compact_json"
```

---

## Configuration

### ProxyConfig

Main configuration class for the proxy agent.

```python
from nocp.core.config import get_config, ProxyConfig

config = get_config()

# Access configuration values
print(config.compression_threshold)
print(config.student_model)
```

**Configuration can be set via environment variables or code:**

```python
import os
os.environ['NOCP_COMPRESSION_THRESHOLD'] = '15000'
os.environ['NOCP_STUDENT_MODEL'] = 'openai/gpt-4o-mini'
```

---

## Complete Example

```python
import asyncio
from nocp.core.act import ToolExecutor
from nocp.core.assess import ContextManager
from nocp.core.articulate import OutputSerializer
from nocp.core.cache import LRUCache
from nocp.core.async_modules import ConcurrentToolExecutor
from nocp.models.contracts import ToolRequest, ToolType, ContextData, SerializationRequest
from pydantic import BaseModel

# Setup
cache = LRUCache(max_size=1000)
executor = ToolExecutor(cache=cache)
context_manager = ContextManager()
serializer = OutputSerializer()

# Register tools
@executor.register_async_tool("fetch_data")
async def fetch_data(query: str) -> dict:
    # Simulate API call
    await asyncio.sleep(0.1)
    return {"results": [f"Result for {query}"]}

async def main():
    # Execute tools concurrently
    concurrent = ConcurrentToolExecutor(executor, max_concurrent=5)
    requests = [
        ToolRequest(
            tool_id="fetch_data",
            tool_type=ToolType.API_CALL,
            function_name="fetch_data",
            parameters={"query": f"query_{i}"}
        )
        for i in range(10)
    ]

    results = await concurrent.execute_many(requests)

    # Optimize context
    context = ContextData(tool_results=results)
    optimized = context_manager.optimize(context)

    # Serialize output
    class Response(BaseModel):
        data: list

    response = Response(data=[r.data for r in results if r.success])
    serialization = serializer.serialize(SerializationRequest(data=response))

    print(f"Compression: {optimized.compression_ratio:.2%}")
    print(f"Serialization savings: {serialization.savings_ratio:.2%}")

asyncio.run(main())
```

---

For more examples, see the [User Guide](08-USER-GUIDE.md) and [Examples](../examples/).
