# System Architecture Specification

## 1. Architectural Pattern: Orchestrator-Worker

The LLM Proxy Agent implements a **separation of concerns** architecture where compute-intensive operations are delegated to specialized workers, while the expensive large-context LLM focuses purely on complex reasoning.

### 1.1 Design Rationale

**Problem**: Large-context LLMs (e.g., Gemini 2.5 Flash with 1M+ token windows) charge per token. Sending raw, unprocessed data directly to these models creates unsustainable costs.

**Solution**: Introduce a middleware layer that:
1. Executes tools and collects raw data (Act)
2. Compresses and optimizes input context (Assess)
3. Serializes output in token-efficient formats (Articulate)

### 1.2 Component Interaction Flow

```
┌──────────────────────────────────────────────────────────────┐
│                      Client Application                       │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                   HighEfficiencyProxyAgent                    │
│                        (Orchestrator)                         │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Request Preprocessing                   │    │
│  │  • Parse user intent                                 │    │
│  │  • Identify required tools                           │    │
│  │  • Load transient + persistent context               │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         ▼                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                 ACT: Tool Executor                    │   │
│  │  Input:  ToolRequest (Pydantic)                      │   │
│  │  Output: ToolResult (raw, potentially large)         │   │
│  │                                                       │   │
│  │  • Execute database queries                          │   │
│  │  • Call external APIs                                │   │
│  │  • Run RAG pipelines                                 │   │
│  │  • Execute arbitrary Python functions                │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         ▼                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              ASSESS: Context Manager                  │   │
│  │  Input:  ToolResult + TransientContext               │   │
│  │  Output: OptimizedContext (50-70% smaller)           │   │
│  │                                                       │   │
│  │  ┌────────────────────────────────────────────┐     │   │
│  │  │  1. Token Counting (pre-flight check)      │     │   │
│  │  │     • Use CountTokens API                   │     │   │
│  │  │     • Estimate input cost                   │     │   │
│  │  └────────────────────────────────────────────┘     │   │
│  │  ┌────────────────────────────────────────────┐     │   │
│  │  │  2. Semantic Pruning (RAG outputs)         │     │   │
│  │  │     • Retrieve top-k relevant chunks        │     │   │
│  │  │     • Embed-based similarity filtering      │     │   │
│  │  └────────────────────────────────────────────┘     │   │
│  │  ┌────────────────────────────────────────────┐     │   │
│  │  │  3. Knowledge Distillation (verbose logs)  │     │   │
│  │  │     • Route to Student Summarizer LLM       │     │   │
│  │  │     • Preserve semantic meaning             │     │   │
│  │  └────────────────────────────────────────────┘     │   │
│  │  ┌────────────────────────────────────────────┐     │   │
│  │  │  4. Conversation History Compaction        │     │   │
│  │  │     • Roll-up summarization of old msgs    │     │   │
│  │  │     • Maintain state continuity             │     │   │
│  │  └────────────────────────────────────────────┘     │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         ▼                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            Main LLM (Reasoning Engine)                │   │
│  │  Provider: LiteLLM (multi-cloud)                     │   │
│  │  Input:  OptimizedContext                            │   │
│  │  Output: StructuredResponse (Pydantic via JSON)      │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         ▼                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          ARTICULATE: Output Serializer                │   │
│  │  Input:  StructuredResponse (Pydantic model)         │   │
│  │  Output: CompactPayload (TOON or Compact JSON)       │   │
│  │                                                       │   │
│  │  ┌────────────────────────────────────────────┐     │   │
│  │  │  Format Negotiation Layer                  │     │   │
│  │  │  • Analyze response schema                  │     │   │
│  │  │  • Detect tabular vs nested structure       │     │   │
│  │  │  • Select optimal format (TOON/JSON)        │     │   │
│  │  └────────────────────────────────────────────┘     │   │
│  │  ┌────────────────────────────────────────────┐     │   │
│  │  │  TOON Encoding (tabular data)              │     │   │
│  │  │  • Indentation-based structure              │     │   │
│  │  │  • CSV-style arrays                         │     │   │
│  │  │  • Length markers for validation            │     │   │
│  │  └────────────────────────────────────────────┘     │   │
│  │  ┌────────────────────────────────────────────┐     │   │
│  │  │  Compact JSON (nested structures)          │     │   │
│  │  │  • No whitespace                            │     │   │
│  │  │  • Minimal separators                       │     │   │
│  │  └────────────────────────────────────────────┘     │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         ▼                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Response Post-Processing                 │   │
│  │  • Log token metrics (input/output)                  │   │
│  │  • Calculate cost delta                              │   │
│  │  • Update efficiency monitoring                      │   │
│  └──────────────────────┬───────────────────────────────┘   │
└─────────────────────────┼─────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                   Return to Client                            │
└──────────────────────────────────────────────────────────────┘
```

## 2. Data Flow and Contracts

### 2.1 Pydantic-First Design

All data crossing component boundaries MUST be represented as Pydantic models:

```python
from pydantic import BaseModel
from typing import Any, Dict, List

class ToolRequest(BaseModel):
    """Input to Act module"""
    tool_name: str
    parameters: Dict[str, Any]
    context: Dict[str, Any]

class ToolResult(BaseModel):
    """Output from Act module"""
    success: bool
    data: Any  # Raw output (potentially large)
    metadata: Dict[str, Any]
    token_count: int  # Estimated

class OptimizedContext(BaseModel):
    """Output from Assess module"""
    compressed_data: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    method_used: str  # "semantic_pruning" | "distillation" | "history_compaction"

class StructuredResponse(BaseModel):
    """Output from Main LLM (via LiteLLM)"""
    content: Any  # Validated Pydantic model
    model_used: str
    input_tokens: int
    output_tokens: int
    latency_ms: float

class CompactPayload(BaseModel):
    """Output from Articulate module"""
    serialized_data: str
    format_used: str  # "toon" | "compact_json"
    original_tokens: int
    optimized_tokens: int
    savings_ratio: float
```

### 2.2 Error Handling Strategy

Each component must implement graceful degradation:

```python
class ProxyAgentError(Exception):
    """Base exception for all proxy agent errors"""
    pass

class ToolExecutionError(ProxyAgentError):
    """Act module failures"""
    pass

class CompressionError(ProxyAgentError):
    """Assess module failures - fallback to raw output"""
    pass

class SerializationError(ProxyAgentError):
    """Articulate module failures - fallback to standard JSON"""
    pass
```

**Fallback Rules**:
- If Assess fails: Use raw ToolResult (log warning, track metric)
- If Articulate fails: Return standard JSON (log warning, track metric)
- If Main LLM fails: Implement retry with exponential backoff (max 3 attempts)

## 3. Integration Layer: LiteLLM Gateway

### 3.1 Multi-Cloud Provider Support

LiteLLM provides unified interface to 100+ LLM providers:

```python
from litellm import completion
from pydantic import BaseModel

# Define response schema
class ResponseSchema(BaseModel):
    result: str
    confidence: float

# Unified call format
response = completion(
    model="gemini/gemini-2.0-flash-exp",  # or "anthropic/claude-3-5-sonnet-20241022"
    messages=[{"role": "user", "content": optimized_context}],
    response_format={"type": "json_schema", "schema": ResponseSchema.model_json_schema()}
)
```

### 3.2 Dynamic Routing Strategy

Implement cost-based routing:

```python
class RoutingConfig(BaseModel):
    simple_query_model: str = "openai/gpt-4o-mini"  # $0.75/M tokens
    complex_query_model: str = "google/gemini-2.0-flash-exp"  # $1.00/M tokens
    reasoning_model: str = "anthropic/claude-3-5-sonnet-20241022"  # $15.00/M tokens

def select_model(optimized_context: OptimizedContext, complexity_score: float) -> str:
    """Route to appropriate model based on complexity and cost"""
    if complexity_score < 0.3:
        return config.simple_query_model
    elif complexity_score < 0.7:
        return config.complex_query_model
    else:
        return config.reasoning_model
```

## 4. Observability and Monitoring

### 4.1 Structured Logging Schema

```python
from pydantic import BaseModel
from datetime import datetime

class TransactionLog(BaseModel):
    """Logged for every request"""
    timestamp: datetime
    request_id: str

    # Input metrics
    raw_input_tokens: int
    optimized_input_tokens: int
    input_compression_ratio: float

    # LLM metrics
    model_used: str
    llm_input_tokens: int
    llm_output_tokens: int
    llm_latency_ms: float

    # Output metrics
    raw_output_tokens: int
    optimized_output_tokens: int
    output_compression_ratio: float
    serialization_format: str

    # Cost metrics
    estimated_cost_baseline: float
    estimated_cost_optimized: float
    cost_savings: float

    # Performance
    total_latency_ms: float
    compression_overhead_ms: float
```

### 4.2 Contextual Drift Detection

Monitor the efficiency delta over rolling window:

```python
def detect_drift(recent_logs: List[TransactionLog], window_size: int = 100) -> bool:
    """Alert if compression ratio degrades significantly"""
    avg_compression = sum(log.input_compression_ratio for log in recent_logs) / len(recent_logs)

    # Expected: 0.30-0.50 (representing 50-70% reduction)
    if avg_compression > 0.60:  # Less than 40% reduction
        logger.warning("Contextual drift detected: compression ratio degraded")
        return True
    return False
```

## 5. Scalability Considerations

### 5.1 Caching Strategy

Implement multi-level caching:

```python
# Level 1: In-memory LRU cache for recent tool results
@lru_cache(maxsize=1000)
def cached_tool_execution(tool_name: str, params_hash: str) -> ToolResult:
    pass

# Level 2: Redis cache for compressed contexts (shared across instances)
def get_cached_context(context_hash: str) -> Optional[OptimizedContext]:
    return redis_client.get(f"context:{context_hash}")
```

### 5.2 Async/Concurrent Processing

Enable parallel execution where possible:

```python
async def process_request_async(request: ProxyRequest) -> CompactPayload:
    """Asynchronous request processing"""

    # Run multiple tools concurrently
    tool_results = await asyncio.gather(
        *[execute_tool_async(tool) for tool in request.required_tools]
    )

    # Assess can run in parallel for independent contexts
    compressed_contexts = await asyncio.gather(
        *[assess.compress_async(result) for result in tool_results]
    )

    # LLM call (blocking)
    response = await llm_completion_async(compressed_contexts)

    # Serialization (fast, can be sync)
    return articulate.serialize(response)
```

## 6. Security and Validation

### 6.1 Input Validation

All external inputs validated via Pydantic:

```python
class ProxyRequest(BaseModel):
    """Validated on ingress"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        str_max_length=100_000,
        validate_assignment=True
    )

    user_query: str
    tools: List[str]
    context: Dict[str, Any]
    max_tokens: int = Field(ge=1, le=1_000_000)
```

### 6.2 Secrets Management

Never log or expose API keys:

```python
from pydantic import SecretStr

class LLMConfig(BaseModel):
    api_key: SecretStr
    endpoint: str

    def get_client(self):
        """Use secret without exposing in logs"""
        return LiteLLM(api_key=self.api_key.get_secret_value())
```

---

**Next**: See `02-API-CONTRACTS.md` for detailed API specifications and `04-COMPONENT-SPECS.md` for implementation details.
