"""
Request Router - Entry point for agent requests.

Parses incoming queries, validates schemas, and prepares initial context
with minimal token overhead.
"""

import uuid

from ..core.config import get_config
from ..models.context import ConversationMessage, PersistentContext, TransientContext
from ..models.schemas import AgentRequest, ToolDefinition
from ..utils.logging import get_logger
from ..utils.token_counter import TokenCounter


class RequestRouter:
    """
    Request Router component - First stage of the agent pipeline.

    Responsibilities:
    - Parse and validate incoming requests
    - Initialize transient context for the current turn
    - Load or create persistent context for the session
    - Minimize static token overhead through efficient formatting
    """

    def __init__(self):
        """Initialize the request router."""
        self.config = get_config()
        self.logger = get_logger(__name__)
        self.token_counter = TokenCounter()

        # In-memory session storage (in production, use Redis/DB)
        self._sessions: dict[str, PersistentContext] = {}

    def route_request(
        self,
        request: AgentRequest,
        available_tools: list[ToolDefinition],
    ) -> tuple[TransientContext, PersistentContext, str]:
        """
        Route an incoming request and prepare contexts.

        Args:
            request: The agent request to route
            available_tools: List of tools available for this request

        Returns:
            Tuple of (transient_context, persistent_context, transaction_id)
        """
        # Generate unique transaction ID
        transaction_id = str(uuid.uuid4())

        self.logger.info(
            "routing_request",
            transaction_id=transaction_id,
            session_id=request.session_id,
            query_length=len(request.query),
        )

        # Get or create persistent context
        persistent_ctx = self._get_or_create_session(request.session_id)

        # Create transient context for this turn
        transient_ctx = self._create_transient_context(
            request=request,
            available_tools=available_tools,
            persistent_ctx=persistent_ctx,
        )

        # Validate token budget
        self._validate_token_budget(transient_ctx)

        return transient_ctx, persistent_ctx, transaction_id

    def _get_or_create_session(self, session_id: str | None) -> PersistentContext:
        """
        Get existing session or create a new one.

        Args:
            session_id: Optional session identifier

        Returns:
            PersistentContext for the session
        """
        # Generate session ID if not provided
        if session_id is None:
            session_id = f"session-{uuid.uuid4()}"

        # Return existing or create new
        if session_id in self._sessions:
            return self._sessions[session_id]

        # Create new persistent context
        persistent_ctx = PersistentContext(session_id=session_id)
        self._sessions[session_id] = persistent_ctx

        self.logger.info("session_created", session_id=session_id)
        return persistent_ctx

    def _create_transient_context(
        self,
        request: AgentRequest,
        available_tools: list[ToolDefinition],
        persistent_ctx: PersistentContext,
    ) -> TransientContext:
        """
        Create transient context for the current turn.

        Args:
            request: The agent request
            available_tools: Available tools
            persistent_ctx: Persistent context

        Returns:
            Initialized TransientContext
        """
        # Create user message
        user_message = ConversationMessage(
            role="user",
            content=request.query,
            token_count=self.token_counter.count_text(request.query),
        )

        # Initialize transient context
        return TransientContext(
            current_query=request.query,
            available_tool_names=[tool.name for tool in available_tools],
            conversation_history=[user_message],
            turn_number=persistent_ctx.total_turns + 1,
        )

    def _validate_token_budget(self, transient_ctx: TransientContext) -> None:
        """
        Validate that the request fits within token budget.

        Args:
            transient_ctx: The transient context to validate

        Raises:
            ValueError: If request exceeds token budget
        """
        total_tokens = transient_ctx.get_total_history_tokens()

        if total_tokens > self.config.max_input_tokens:
            raise ValueError(
                f"Request exceeds maximum token budget: {total_tokens} > {self.config.max_input_tokens}"
            )

        # Check if history compaction is needed
        if transient_ctx.requires_compaction():
            self.logger.warning(
                "history_compaction_needed",
                current_tokens=total_tokens,
                max_tokens=transient_ctx.max_history_tokens,
            )

    def add_tool_result_to_history(
        self,
        transient_ctx: TransientContext,
        tool_name: str,
        result: str,
    ) -> None:
        """
        Add a tool execution result to conversation history.

        Args:
            transient_ctx: The transient context
            tool_name: Name of the tool that was executed
            result: The tool result (potentially compressed)
        """
        tool_message = ConversationMessage(
            role="tool",
            content=result,
            token_count=self.token_counter.count_text(result),
            metadata={"tool_name": tool_name},
        )

        transient_ctx.conversation_history.append(tool_message)

    def finalize_session(
        self,
        persistent_ctx: PersistentContext,
        tokens_used: int,
        cost: float,
    ) -> None:
        """
        Finalize session after request completion.

        Args:
            persistent_ctx: The persistent context to update
            tokens_used: Total tokens used in this turn
            cost: Estimated cost in USD
        """
        persistent_ctx.update_metrics(tokens=tokens_used, cost=cost)
        self.logger.info(
            "session_finalized",
            session_id=persistent_ctx.session_id,
            total_turns=persistent_ctx.total_turns,
            total_tokens=persistent_ctx.total_tokens_processed,
        )
