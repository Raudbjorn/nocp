"""
Context management schemas for transient and persistent context.

Implements the separation between short-term (transient) and long-term (persistent)
context as outlined in the architecture.
"""

from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    """A single message in the conversation history."""

    role: Literal["user", "assistant", "system", "tool"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    token_count: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TransientContext(BaseModel):
    """
    Transient context for a single agent turn.

    Includes:
    - Current prompt/query
    - Available tools for this turn
    - Message history for current conversation
    """

    current_query: str
    available_tool_names: List[str] = Field(default_factory=list)
    conversation_history: List[ConversationMessage] = Field(default_factory=list)
    turn_number: int = Field(default=1)

    # Token tracking
    estimated_tokens: Optional[int] = None
    max_history_tokens: int = Field(
        default=50000,
        description="Maximum tokens to allocate for conversation history"
    )

    def get_total_history_tokens(self) -> int:
        """Calculate total tokens in conversation history."""
        return sum(
            msg.token_count or 0
            for msg in self.conversation_history
        )

    def requires_compaction(self) -> bool:
        """Check if history compaction is needed."""
        return self.get_total_history_tokens() > self.max_history_tokens


class PersistentContext(BaseModel):
    """
    Persistent context that spans multiple conversations/sessions.

    Includes:
    - Long-term memory/state
    - System instructions
    - User preferences
    - Life-cycle state
    """

    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    # System configuration
    system_instructions: str = Field(
        default="You are a helpful AI assistant powered by an efficient token optimization layer.",
        description="Core system instructions for the agent"
    )

    # Long-term memory
    conversation_summary: Optional[str] = Field(
        default=None,
        description="Rolled-up summary of past conversations"
    )
    user_preferences: Dict[str, Any] = Field(
        default_factory=dict,
        description="Learned user preferences and context"
    )

    # State tracking
    total_turns: int = Field(default=0)
    total_tokens_processed: int = Field(default=0)
    total_cost_usd: float = Field(default=0.0)

    # Lifecycle
    state: Literal["active", "idle", "archived"] = Field(default="active")

    def update_metrics(self, tokens: int, cost: float) -> None:
        """Update cumulative metrics."""
        self.total_turns += 1
        self.total_tokens_processed += tokens
        self.total_cost_usd += cost
        self.last_updated = datetime.utcnow()


class ContextSnapshot(BaseModel):
    """
    Complete snapshot of agent context at a point in time.
    Combines transient and persistent context for debugging/monitoring.
    """

    transient: TransientContext
    persistent: PersistentContext
    snapshot_time: datetime = Field(default_factory=datetime.utcnow)
    total_context_tokens: int

    model_config = {"arbitrary_types_allowed": True}
