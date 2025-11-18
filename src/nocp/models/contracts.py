"""
Pydantic models defining API contracts for all components.
"""

import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    retry_config: Optional[RetryConfig] = None


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
    message_history: List[ChatMessage] = Field(default_factory=list)

    # Optimization hints
    max_tokens: int = Field(default=100_000, ge=1000)
    compression_strategy: Optional[str] = Field(
        None,
        description="Override auto-detection: 'semantic_pruning', 'distillation', 'history_compaction'"
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


# ============================================================================
# Articulate Module (Output Serializer) Contracts
# ============================================================================

class SerializationFormat(str, Enum):
    """Output format selected."""
    TOON = "toon"
    COMPACT_JSON = "compact_json"


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
