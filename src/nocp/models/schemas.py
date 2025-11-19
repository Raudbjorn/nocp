"""
Core Pydantic schemas for the NOCP proxy agent.

These schemas define the data contracts throughout the agent pipeline,
ensuring type safety and enabling seamless integration with Gemini's
structured output and function calling features.
"""

from typing import Any, Dict, List, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

from .enums import OutputFormat


class ToolParameter(BaseModel):
    """Schema for a single tool parameter definition."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (string, number, boolean, object, array)")
    description: str = Field(..., description="Parameter description for LLM")
    required: bool = Field(default=True, description="Whether parameter is required")
    enum: Optional[List[str]] = Field(default=None, description="Allowed values if enumerated")


class ToolDefinition(BaseModel):
    """
    Schema for defining external tools/functions available to the agent.
    Converts to Gemini Function Calling format.
    """

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "name": "search_database",
            "description": "Search the product database for relevant items",
            "parameters": [
                {
                    "name": "query",
                    "type": "string",
                    "description": "Search query string",
                    "required": True
                }
            ]
        }
    })

    name: str = Field(..., description="Unique tool identifier")
    description: str = Field(..., description="Clear description of tool purpose and usage")
    parameters: List[ToolParameter] = Field(default_factory=list)
    compression_threshold: Optional[int] = Field(
        default=None,
        description="Custom compression threshold for this tool's output"
    )
    preferred_compression_method: Optional[Literal["semantic_pruning", "knowledge_distillation", "history_compaction", "none"]] = Field(
        default=None,
        description="Preferred compression method for this tool's output"
    )


class ToolExecutionResult(BaseModel):
    """Result from executing a tool, before compression."""

    tool_name: str
    raw_output: str
    raw_token_count: Optional[int] = None
    execution_time_ms: float
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CompressionResult(BaseModel):
    """Result of context compression operation."""

    original_tokens: int
    compressed_tokens: int
    compression_ratio: float = Field(
        ...,
        description="Ratio of compressed to original (lower is better)"
    )
    compression_method: Literal["semantic_pruning", "knowledge_distillation", "history_compaction", "none"]
    compression_cost_tokens: int = Field(
        default=0,
        description="Tokens consumed by the compression process itself"
    )
    compression_time_ms: float = Field(
        default=0.0,
        description="Actual time spent on compression in milliseconds"
    )
    net_savings: int = Field(
        ...,
        description="Actual token savings after subtracting compression cost"
    )


class ContextMetrics(BaseModel):
    """Comprehensive metrics for a single agent transaction."""

    transaction_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Input metrics
    raw_input_tokens: int
    compressed_input_tokens: int
    input_compression_ratio: float

    # Output metrics
    raw_output_tokens: int
    final_output_format: OutputFormat
    output_token_savings: int

    # Performance metrics
    total_latency_ms: float
    compression_latency_ms: float
    llm_inference_latency_ms: float

    # Metadata
    tools_used: List[str] = Field(default_factory=list)
    compression_operations: List[CompressionResult] = Field(default_factory=list)


class TransactionLog(BaseModel):
    """
    Complete transaction log for observability and monitoring.

    This schema captures all relevant metrics for a single request-response cycle,
    enabling comprehensive analysis of compression effectiveness, cost, and performance.
    """

    # Request identification
    transaction_id: str = Field(..., description="Unique transaction identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Request details
    user_query: str = Field(..., description="Original user query")
    tools_invoked: List[str] = Field(default_factory=list, description="Tools called during execution")

    # Input compression metrics
    raw_input_tokens: int = Field(..., description="Total tokens before compression")
    optimized_input_tokens: int = Field(..., description="Tokens after compression")
    input_compression_ratio: float = Field(..., description="Compression ratio (optimized/raw)")
    input_compression_method: Optional[str] = Field(None, description="Compression method used")
    compression_justified: bool = Field(True, description="Whether compression passed cost-benefit check")

    # LLM metrics
    model_used: str = Field(..., description="LLM model identifier")
    llm_input_tokens: int = Field(..., description="Actual tokens sent to LLM")
    llm_output_tokens: int = Field(..., description="Tokens received from LLM")
    llm_latency_ms: float = Field(..., description="LLM inference time in milliseconds")

    # Output serialization metrics
    raw_output_tokens: int = Field(..., description="Tokens in raw response")
    optimized_output_tokens: int = Field(..., description="Tokens after serialization")
    output_compression_ratio: float = Field(..., description="Output compression ratio")
    serialization_format: OutputFormat = Field(..., description="Final output format")

    # Performance metrics
    total_latency_ms: float = Field(..., description="End-to-end latency")
    compression_latency_ms: float = Field(0.0, description="Time spent on compression")
    serialization_latency_ms: float = Field(0.0, description="Time spent on serialization")

    # Cost metrics
    estimated_cost_usd: Optional[float] = Field(None, description="Estimated API cost")
    cost_savings_usd: Optional[float] = Field(None, description="Cost savings from compression")

    # Efficiency metrics
    efficiency_delta: int = Field(..., description="Net token savings (raw_input - optimized_input)")
    total_token_savings: int = Field(..., description="Total tokens saved (input + output)")
    compression_overhead_tokens: int = Field(0, description="Tokens used by compression itself")

    # Quality metrics
    compression_success: bool = Field(True, description="Whether compression completed successfully")
    error_message: Optional[str] = Field(None, description="Error message if compression failed")

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def calculate_savings_percentage(self) -> float:
        """Calculate total token savings as percentage."""
        total_raw = self.raw_input_tokens + self.raw_output_tokens
        if total_raw == 0:
            return 0.0
        return (self.total_token_savings / total_raw) * 100


class AgentRequest(BaseModel):
    """
    Schema for incoming agent requests.
    Represents the user's query and available context.
    """

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "Find all products matching the search term and summarize pricing",
            "session_id": "user-session-123",
            "available_tools": ["search_database", "analyze_pricing"]
        }
    })

    query: str = Field(..., description="User's natural language query")
    session_id: Optional[str] = Field(default=None, description="Session identifier for context tracking")
    available_tools: List[str] = Field(
        default_factory=list,
        description="Names of tools available for this request"
    )
    context_override: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional context to inject (for testing/overrides)"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Override maximum output tokens"
    )


class AgentResponse(BaseModel):
    """
    Schema for the final agent response.
    This is the structured output from the LLM that gets serialized to TOON.
    """

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "answer": "Found 5 products matching your criteria",
            "tool_results_summary": ["Searched database: 5 items found", "Analyzed pricing: Average $49.99"],
            "confidence": 0.95,
            "follow_up_actions": ["Review pricing details", "Check inventory"]
        }
    })

    answer: str = Field(..., description="The primary answer to the user's query")
    tool_results_summary: List[str] = Field(
        default_factory=list,
        description="Distilled summary of tool execution results"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in the response"
    )
    follow_up_actions: List[str] = Field(
        default_factory=list,
        description="Suggested next steps or actions"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured data"
    )

    # Internal metrics (not serialized to user)
    metrics: Optional[ContextMetrics] = Field(
        default=None,
        description="Performance and cost metrics for monitoring"
    )
