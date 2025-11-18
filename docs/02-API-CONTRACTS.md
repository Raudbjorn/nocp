# API Contracts and Data Schemas

## 1. Module Interfaces

### 1.1 Act Module (Tool Executor)

**Purpose**: Execute external tools and functions, returning raw results.

#### Input Contract

```python
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, List
from enum import Enum

class ToolType(str, Enum):
    """Supported tool categories"""
    DATABASE_QUERY = "database_query"
    API_CALL = "api_call"
    RAG_RETRIEVAL = "rag_retrieval"
    FILE_OPERATION = "file_operation"
    PYTHON_FUNCTION = "python_function"

class ToolRequest(BaseModel):
    """Input to Act.execute()"""
    tool_id: str = Field(..., description="Unique identifier for the tool")
    tool_type: ToolType
    function_name: str = Field(..., description="Name of the function to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    retry_config: Optional['RetryConfig'] = None

class RetryConfig(BaseModel):
    """Optional retry configuration"""
    max_attempts: int = Field(default=3, ge=1, le=5)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)
    initial_delay_ms: int = Field(default=100, ge=10)
```

#### Output Contract

```python
from datetime import datetime
from typing import Any, Dict, Optional

class ToolResult(BaseModel):
    """Output from Act.execute()"""
    tool_id: str
    success: bool
    data: Any = Field(..., description="Raw output from the tool")
    error: Optional[str] = None

    # Metadata
    execution_time_ms: float
    timestamp: datetime
    token_estimate: int = Field(..., description="Estimated tokens if serialized to text")

    # Provenance
    retry_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_text(self) -> str:
        """Convert data to text representation for LLM consumption"""
        if isinstance(self.data, str):
            return self.data
        elif isinstance(self.data, (dict, list)):
            import json
            return json.dumps(self.data, indent=2)
        else:
            return str(self.data)
```

#### Interface Specification

```python
from typing import Protocol

class ToolExecutor(Protocol):
    """Interface that all tool executors must implement"""

    def execute(self, request: ToolRequest) -> ToolResult:
        """
        Execute a tool and return the result.

        Raises:
            ToolExecutionError: If execution fails after all retries
            TimeoutError: If execution exceeds timeout
        """
        ...

    async def execute_async(self, request: ToolRequest) -> ToolResult:
        """Async version for concurrent execution"""
        ...

    def validate_tool(self, tool_id: str) -> bool:
        """Check if tool is registered and available"""
        ...
```

---

### 1.2 Assess Module (Context Manager)

**Purpose**: Compress and optimize context to reduce token usage before LLM call.

#### Input Contract

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ContextData(BaseModel):
    """Input to Assess.optimize()"""

    # Primary content to compress
    tool_results: List[ToolResult] = Field(..., description="Raw outputs from Act module")

    # Additional context
    transient_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="User query, current session state, etc."
    )
    persistent_context: Optional[str] = Field(
        None,
        description="System instructions, domain knowledge, etc."
    )

    # Conversation history
    message_history: List['ChatMessage'] = Field(default_factory=list)

    # Optimization hints
    max_tokens: int = Field(default=100_000, ge=1000)
    compression_strategy: Optional[str] = Field(
        None,
        description="Override auto-detection: 'semantic_pruning', 'distillation', 'history_compaction'"
    )

class ChatMessage(BaseModel):
    """Individual message in conversation history"""
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    timestamp: datetime
    tokens: int
```

#### Output Contract

```python
from enum import Enum

class CompressionMethod(str, Enum):
    """Compression techniques applied"""
    SEMANTIC_PRUNING = "semantic_pruning"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    HISTORY_COMPACTION = "history_compaction"
    NONE = "none"  # Fallback if compression not beneficial

class OptimizedContext(BaseModel):
    """Output from Assess.optimize()"""

    # Compressed content
    optimized_text: str = Field(..., description="Token-optimized context for LLM")

    # Metrics
    original_tokens: int
    optimized_tokens: int
    compression_ratio: float = Field(..., description="optimized/original (target: 0.30-0.50)")

    # Provenance
    method_used: CompressionMethod
    compression_time_ms: float

    # Cost analysis
    estimated_cost_savings: float = Field(..., description="In USD")

    # Quality metrics
    semantic_similarity_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Cosine similarity between original and compressed (if computed)"
    )

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

#### Interface Specification

```python
class ContextManager(Protocol):
    """Interface for context optimization"""

    def optimize(self, context: ContextData) -> OptimizedContext:
        """
        Compress context using appropriate strategy.

        Raises:
            CompressionError: If compression fails (fallback to raw)
        """
        ...

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using provider's tokenizer"""
        ...

    def select_strategy(self, context: ContextData) -> CompressionMethod:
        """Auto-detect optimal compression strategy"""
        ...
```

---

### 1.3 Articulate Module (Output Serializer)

**Purpose**: Serialize LLM output in token-efficient format (TOON or compact JSON).

#### Input Contract

```python
from pydantic import BaseModel
from typing import Any, Optional

class SerializationRequest(BaseModel):
    """Input to Articulate.serialize()"""

    # Data to serialize (must be Pydantic model)
    data: BaseModel = Field(..., description="Structured LLM response")

    # Format hints
    force_format: Optional[str] = Field(
        None,
        pattern="^(toon|compact_json)$",
        description="Override format negotiation"
    )

    # Optimization flags
    include_length_markers: bool = Field(
        default=True,
        description="Add length markers for TOON validation"
    )
    validate_output: bool = Field(
        default=True,
        description="Verify serialized output can be deserialized"
    )
```

#### Output Contract

```python
class SerializationFormat(str, Enum):
    """Output format selected"""
    TOON = "toon"
    COMPACT_JSON = "compact_json"

class SerializedOutput(BaseModel):
    """Output from Articulate.serialize()"""

    # Serialized data
    serialized_text: str
    format_used: SerializationFormat

    # Metrics
    original_tokens: int = Field(..., description="If serialized as standard JSON")
    optimized_tokens: int
    savings_ratio: float = Field(..., description="Reduction from baseline")

    # Quality
    is_valid: bool = Field(..., description="Deserialization check passed")
    serialization_time_ms: float

    # Metadata
    schema_complexity: str = Field(
        ...,
        pattern="^(simple|tabular|nested|complex)$",
        description="Why this format was chosen"
    )
```

#### Interface Specification

```python
class OutputSerializer(Protocol):
    """Interface for output serialization"""

    def serialize(self, request: SerializationRequest) -> SerializedOutput:
        """
        Serialize Pydantic model to optimal format.

        Raises:
            SerializationError: Falls back to compact JSON
        """
        ...

    def deserialize(self, serialized: str, format: SerializationFormat, schema: type[BaseModel]) -> BaseModel:
        """Reverse operation for validation"""
        ...

    def negotiate_format(self, model: BaseModel) -> SerializationFormat:
        """Analyze schema to select optimal format"""
        ...
```

---

## 2. Orchestrator Interface

### 2.1 High-Level Request/Response

```python
from typing import List, Optional

class ProxyRequest(BaseModel):
    """Top-level request to HighEfficiencyProxyAgent"""

    # User intent
    query: str = Field(..., min_length=1, max_length=50_000)

    # Tools to execute
    required_tools: List[ToolRequest]

    # Context
    session_id: Optional[str] = None
    user_context: Dict[str, Any] = Field(default_factory=dict)

    # Configuration
    target_model: Optional[str] = Field(
        None,
        description="Override auto-selection (e.g., 'gemini/gemini-2.0-flash-exp')"
    )
    max_total_tokens: int = Field(default=200_000, ge=1000, le=1_000_000)

    # Optimization preferences
    enable_compression: bool = True
    enable_toon: bool = True
    optimization_level: int = Field(default=2, ge=0, le=3, description="0=none, 3=aggressive")

class ProxyResponse(BaseModel):
    """Top-level response from HighEfficiencyProxyAgent"""

    # Response content
    result: SerializedOutput

    # Execution trace
    tool_results: List[ToolResult]
    context_optimization: OptimizedContext
    llm_response: 'LLMResponse'

    # Performance metrics
    total_latency_ms: float
    breakdown: 'LatencyBreakdown'

    # Cost analysis
    cost_analysis: 'CostReport'

class LatencyBreakdown(BaseModel):
    """Detailed timing breakdown"""
    tool_execution_ms: float
    context_compression_ms: float
    llm_inference_ms: float
    output_serialization_ms: float
    overhead_ms: float

class CostReport(BaseModel):
    """Token usage and cost metrics"""
    # Token counts
    raw_input_tokens: int
    optimized_input_tokens: int
    output_tokens: int
    total_tokens: int

    # Cost (USD)
    baseline_cost: float = Field(..., description="If no optimization")
    actual_cost: float
    savings: float
    savings_percentage: float

    # Efficiency
    input_compression_ratio: float
    output_compression_ratio: float

class LLMResponse(BaseModel):
    """Response from main LLM (via LiteLLM)"""
    model: str
    content: BaseModel  # Structured output
    input_tokens: int
    output_tokens: int
    latency_ms: float
    finish_reason: str
```

---

## 3. Configuration Schemas

### 3.1 Agent Configuration

```python
class AgentConfig(BaseModel):
    """Configuration for HighEfficiencyProxyAgent"""

    # LLM settings
    default_model: str = "gemini/gemini-2.0-flash-exp"
    fallback_model: Optional[str] = "openai/gpt-4o-mini"
    student_summarizer_model: str = "openai/gpt-4o-mini"

    # API keys (use environment variables)
    litellm_api_base: Optional[str] = None

    # Optimization thresholds
    compression_threshold_tokens: int = Field(
        default=10_000,
        description="Only compress if input exceeds this"
    )
    toon_threshold_array_size: int = Field(
        default=5,
        description="Use TOON if array has >N uniform items"
    )

    # Performance
    max_concurrent_tools: int = Field(default=5, ge=1, le=20)
    request_timeout_seconds: int = Field(default=60, ge=10, le=300)

    # Caching
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600

    # Monitoring
    enable_logging: bool = True
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR)$")
    enable_metrics: bool = True
```

---

## 4. Error Hierarchy

```python
class ProxyAgentError(Exception):
    """Base exception for all proxy agent errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}

class ToolExecutionError(ProxyAgentError):
    """Tool execution failed"""
    pass

class CompressionError(ProxyAgentError):
    """Context compression failed"""
    pass

class SerializationError(ProxyAgentError):
    """Output serialization failed"""
    pass

class LLMError(ProxyAgentError):
    """LLM API call failed"""
    pass

class ConfigurationError(ProxyAgentError):
    """Invalid configuration"""
    pass

class ValidationError(ProxyAgentError):
    """Pydantic validation failed"""
    pass
```

---

## 5. Type Definitions

```python
from typing import TypeVar, Callable, Awaitable

# Generic types
T = TypeVar('T', bound=BaseModel)
ToolFunction = Callable[[Dict[str, Any]], Any]
AsyncToolFunction = Callable[[Dict[str, Any]], Awaitable[Any]]

# Compression strategies
CompressionStrategy = Callable[[ContextData], OptimizedContext]

# Serialization strategies
SerializationStrategy = Callable[[BaseModel], str]
```

---

**Implementation Note**: All contracts defined here are enforced via Pydantic validation at runtime. Any violation will raise a `ValidationError` with detailed information about the mismatch.

**Next**: See `03-DEVELOPMENT-ROADMAP.md` for phased implementation plan.
