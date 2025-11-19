"""
Pydantic models defining API contracts for all components.
"""

import json
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .enums import OutputFormat

# ============================================================================
# Act Module (Tool Executor) Contracts
# ============================================================================


class ToolType(str, Enum):
    """Supported tool categories."""

    DATABASE_QUERY = "database_query"
    API_CALL = "api_call"
    RAG_RETRIEVAL = "rag_retrieval"
    FILE_OPERATION = "file_operation"
    PYTHON_FUNCTION = "python_function"


class RetryConfig(BaseModel):
    """Optional retry configuration for tool execution."""

    max_attempts: int = Field(default=3, ge=1, le=5)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)
    initial_delay_ms: int = Field(default=100, ge=10)


class ToolRequest(BaseModel):
    """Input to Act.execute()"""

    tool_id: str = Field(..., description="Unique identifier for the tool")
    tool_type: ToolType
    function_name: str = Field(..., description="Name of the function to execute")
    parameters: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    retry_config: RetryConfig | None = None


class ToolResult(BaseModel):
    """Output from Act.execute()"""

    tool_id: str
    success: bool
    data: Any = Field(..., description="Raw output from the tool")
    error: str | None = None

    # Metadata
    execution_time_ms: float
    timestamp: datetime
    token_estimate: int = Field(..., description="Estimated tokens if serialized to text")

    # Provenance
    retry_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_text(self) -> str:
        """Convert data to text representation for LLM consumption."""
        if isinstance(self.data, str):
            return self.data
        elif isinstance(self.data, (dict, list)):
            return json.dumps(self.data, indent=2)
        else:
            return str(self.data)


# ============================================================================
# Assess Module (Context Manager) Contracts
# ============================================================================


class ChatMessage(BaseModel):
    """Individual message in conversation history."""

    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    timestamp: datetime
    tokens: int


class CompressionMethod(str, Enum):
    """Compression techniques applied."""

    SEMANTIC_PRUNING = "semantic_pruning"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    HISTORY_COMPACTION = "history_compaction"
    NONE = "none"


class ContextData(BaseModel):
    """Input to Assess.optimize()"""

    # Primary content to compress
    tool_results: list[ToolResult] = Field(..., description="Raw outputs from Act module")

    # Additional context
    transient_context: dict[str, Any] = Field(
        default_factory=dict, description="User query, current session state, etc."
    )
    persistent_context: str | None = Field(
        None, description="System instructions, domain knowledge, etc."
    )

    # Conversation history
    message_history: list[ChatMessage] = Field(default_factory=list)

    # Optimization hints
    max_tokens: int = Field(default=100_000, ge=1000)
    compression_strategy: str | None = Field(
        None,
        description="Override auto-detection: 'semantic_pruning', 'distillation', 'history_compaction'",
    )


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

    # Quality metrics
    semantic_similarity_score: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Cosine similarity between original and compressed (if computed)",
    )

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Articulate Module (Output Serializer) Contracts
# ============================================================================


class SerializationRequest(BaseModel):
    """Input to Articulate.serialize()"""

    # Data to serialize (must be Pydantic model)
    data: BaseModel = Field(..., description="Structured LLM response")

    # Format hints
    force_format: OutputFormat | None = Field(None, description="Override format negotiation")

    # Optimization flags
    include_length_markers: bool = Field(
        default=True, description="Add length markers for TOON validation"
    )
    validate_output: bool = Field(
        default=True, description="Verify serialized output can be deserialized"
    )


class SerializedOutput(BaseModel):
    """Output from Articulate.serialize()"""

    # Serialized data
    serialized_text: str
    format_used: OutputFormat

    # Metrics
    original_tokens: int = Field(..., description="If serialized as standard JSON")
    optimized_tokens: int
    savings_ratio: float = Field(..., description="Reduction from baseline")

    # Quality
    is_valid: bool = Field(..., description="Deserialization check passed")
    serialization_time_ms: float

    # Metadata
    schema_complexity: str = Field(
        ..., pattern="^(simple|tabular|nested|complex)$", description="Why this format was chosen"
    )


# ============================================================================
# LLM Client Contracts
# ============================================================================


class MessageRole(str, Enum):
    """Message roles for chat completion."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class LLMRequest(BaseModel):
    """Request for LLM completion."""

    messages: list[dict[str, str]] = Field(..., description="Chat messages")
    model: str | None = Field(None, description="Model to use (LiteLLM format)")
    max_tokens: int | None = Field(None, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    response_schema: Any | None = Field(None, description="Pydantic model for structured output")
    tools: list[dict[str, Any]] | None = Field(
        None, description="Tool definitions for function calling"
    )


class LLMResponse(BaseModel):
    """Response from LLM completion."""

    content: Any = Field(..., description="Response content (string or structured)")
    model: str = Field(..., description="Model that generated the response")
    input_tokens: int = Field(..., description="Input token count")
    output_tokens: int = Field(..., description="Output token count")
    latency_ms: float = Field(..., description="Request latency in milliseconds")
    finish_reason: str = Field(..., description="Completion finish reason")
    tool_calls: list[dict[str, Any]] = Field(default_factory=list, description="Tool calls if any")
    raw_response: Any | None = Field(None, description="Raw LiteLLM response object")

    class Config:
        arbitrary_types_allowed = True
