"""
High-Efficiency Proxy Agent - Main orchestration class.

This is the central component that coordinates all modules (Act-Assess-Articulate)
to deliver optimized LLM interactions with token efficiency as the primary goal.
"""

import time
import uuid
from typing import Any, Callable, Dict, List, Optional
import google.generativeai as genai

from ..models.schemas import (
    AgentRequest,
    AgentResponse,
    ToolDefinition,
    ContextMetrics,
    CompressionResult,
)
from ..models.context import TransientContext, PersistentContext
from ..modules.router import RequestRouter
from ..modules.tool_executor import ToolExecutor
from ..modules.context_manager import ContextManager
from ..modules.output_serializer import OutputSerializer
from ..core.config import get_config
from ..utils.logging import get_logger, log_metrics, get_metrics_logger
from ..utils.token_counter import TokenCounter


class HighEfficiencyProxyAgent:
    """
    High-Efficiency LLM Proxy Agent with Token Optimization.

    Implements the complete Act-Assess-Articulate architecture with:
    - Dynamic context compression (50-70% reduction target)
    - TOON serialization for outputs (30-60% reduction target)
    - Cost-of-Compression Calculus
    - Comprehensive monitoring and drift detection
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize the High-Efficiency Proxy Agent.

        Args:
            api_key: Optional Gemini API key (uses config if not provided)
            model_name: Optional model name (uses config if not provided)
        """
        # Load configuration
        self.config = get_config()

        # Setup logging
        self.logger = get_logger(__name__)
        self.logger.info("initializing_proxy_agent")

        # Initialize token counter
        self.token_counter = TokenCounter(model_name)

        # Initialize Gemini API
        api_key = api_key or self.config.gemini_api_key
        genai.configure(api_key=api_key)

        self.model_name = model_name or self.config.gemini_model
        self.model = genai.GenerativeModel(self.model_name)

        # Initialize core modules
        self.router = RequestRouter()
        self.tool_executor = ToolExecutor()
        self.context_manager = ContextManager(tool_executor=self.tool_executor)
        self.output_serializer = OutputSerializer()

        self.logger.info(
            "proxy_agent_initialized",
            model=self.model_name,
            max_input_tokens=self.config.max_input_tokens,
            max_output_tokens=self.config.max_output_tokens,
        )

    def register_tool(
        self,
        definition: ToolDefinition,
        callable_func: Callable,
    ) -> None:
        """
        Register a tool for the agent to use.

        Args:
            definition: ToolDefinition describing the tool
            callable_func: Python callable implementing the tool
        """
        self.tool_executor.register_tool(definition, callable_func)

    def process_request(
        self,
        request: AgentRequest,
        return_format: Optional[str] = None,
    ) -> tuple[str, ContextMetrics]:
        """
        Process an agent request end-to-end.

        This is the main entry point that orchestrates the complete pipeline:
        1. Route request (Router)
        2. Execute tools (Act)
        3. Compress context (Assess)
        4. Generate response (LLM)
        5. Serialize output (Articulate)
        6. Log metrics

        Args:
            request: AgentRequest to process
            return_format: Optional output format override

        Returns:
            Tuple of (serialized_response, context_metrics)
        """
        start_time = time.perf_counter()
        transaction_id = str(uuid.uuid4())

        self.logger.info(
            "processing_request",
            transaction_id=transaction_id,
            query=request.query[:100],
        )

        # Initialize metrics tracking
        compression_operations: List[CompressionResult] = []
        tools_used: List[str] = []

        try:
            # Step 1: Route request and prepare context
            transient_ctx, persistent_ctx, _ = self.router.route_request(
                request=request,
                available_tools=self.tool_executor.get_tool_definitions(),
            )

            # Track initial context
            raw_input_tokens = transient_ctx.get_total_history_tokens()

            # Step 2: Execute agent loop with tools
            agent_response = self._execute_agent_loop(
                transient_ctx=transient_ctx,
                persistent_ctx=persistent_ctx,
                compression_operations=compression_operations,
                tools_used=tools_used,
            )

            # Track compressed input tokens
            compressed_input_tokens = transient_ctx.get_total_history_tokens()

            # Step 3: Serialize output
            serialization_start = time.perf_counter()

            serialized_output, output_format, token_savings = self.output_serializer.serialize(
                response=agent_response,
                force_format=return_format,
            )

            serialization_time = (time.perf_counter() - serialization_start) * 1000

            # Calculate final metrics
            total_latency_ms = (time.perf_counter() - start_time) * 1000
            compression_latency_ms = sum(
                comp.compression_time_ms
                for comp in compression_operations
            )

            # Count output tokens
            raw_output_tokens = self.token_counter.count_text(
                agent_response.model_dump_json()
            )

            # Calculate costs
            estimated_cost = self.config.calculate_cost(
                input_tokens=compressed_input_tokens,
                output_tokens=raw_output_tokens,
            )
            baseline_cost = self.config.calculate_cost(
                input_tokens=raw_input_tokens,
                output_tokens=raw_output_tokens,
            )
            estimated_savings = baseline_cost - estimated_cost

            # Create metrics
            metrics = ContextMetrics(
                transaction_id=transaction_id,
                raw_input_tokens=raw_input_tokens,
                compressed_input_tokens=compressed_input_tokens,
                input_compression_ratio=compressed_input_tokens / raw_input_tokens if raw_input_tokens > 0 else 1.0,
                raw_output_tokens=raw_output_tokens,
                final_output_format=output_format,
                output_token_savings=token_savings,
                total_latency_ms=total_latency_ms,
                compression_latency_ms=compression_latency_ms,
                llm_inference_latency_ms=total_latency_ms - compression_latency_ms - serialization_time,
                estimated_cost_usd=estimated_cost,
                estimated_savings_usd=estimated_savings,
                tools_used=tools_used,
                compression_operations=compression_operations,
            )

            # Log metrics
            log_metrics(metrics)

            # Update session
            self.router.finalize_session(
                persistent_ctx=persistent_ctx,
                tokens_used=compressed_input_tokens + raw_output_tokens,
                cost=estimated_cost,
            )

            self.logger.info(
                "request_processed",
                transaction_id=transaction_id,
                total_latency_ms=round(total_latency_ms, 2),
                estimated_cost_usd=round(estimated_cost, 6),
                estimated_savings_usd=round(estimated_savings, 6),
            )

            return serialized_output, metrics

        except Exception as e:
            self.logger.error(
                "request_processing_failed",
                transaction_id=transaction_id,
                error=str(e),
            )
            raise

    def _execute_agent_loop(
        self,
        transient_ctx: TransientContext,
        persistent_ctx: PersistentContext,
        compression_operations: List[CompressionResult],
        tools_used: List[str],
    ) -> AgentResponse:
        """
        Execute the main agent loop with tool calling and context management.

        Args:
            transient_ctx: Transient context for this turn
            persistent_ctx: Persistent context
            compression_operations: List to track compression operations
            tools_used: List to track tools used

        Returns:
            AgentResponse from the LLM
        """
        # Compact history if needed
        history_compression = self.context_manager.compact_conversation_history(transient_ctx)
        if history_compression:
            compression_operations.append(history_compression)

        # Prepare messages for LLM
        messages = self._format_messages_for_gemini(transient_ctx, persistent_ctx)

        # Get tool schemas
        tool_schemas = self.tool_executor.get_tool_schemas_for_gemini()

        # Configure generation
        generation_config = genai.GenerationConfig(
            max_output_tokens=self.config.max_output_tokens,
            temperature=0.7,
        )

        # Call LLM with function calling
        response = self.model.generate_content(
            messages,
            tools=tool_schemas or None,
            generation_config=generation_config,
        )

        # Check for function calls
        if (function_call := response.candidates[0].content.parts[0].function_call):
            tool_name = function_call.name
            tool_params = dict(function_call.args)

            tools_used.append(tool_name)

            # Execute tool
            tool_result = self.tool_executor.execute_tool(tool_name, tool_params)

            # Apply context management (compression)
            managed_output, compression_result = self.context_manager.manage_tool_output(
                tool_result
            )

            if compression_result:
                compression_operations.append(compression_result)

            # Add tool result to history
            self.router.add_tool_result_to_history(
                transient_ctx,
                tool_name,
                managed_output,
            )

            # Continue agent loop with tool result (simplified - single turn)
            # In production, this would be recursive for multi-turn tool usage
            final_response = self.model.generate_content(
                self._format_messages_for_gemini(transient_ctx, persistent_ctx),
                generation_config=generation_config,
            )

            response_text = final_response.text
        else:
            response_text = response.text

        # Parse response into AgentResponse schema
        # For now, return a simple response (in production, use structured output)
        return AgentResponse(
            answer=response_text,
            tool_results_summary=[f"Used tool: {t}" for t in tools_used],
            confidence=0.9,
        )

    def _format_messages_for_gemini(
        self,
        transient_ctx: TransientContext,
        persistent_ctx: PersistentContext,
    ) -> List[Dict[str, str]]:
        """
        Format context into messages for Gemini API.

        Args:
            transient_ctx: Transient context
            persistent_ctx: Persistent context

        Returns:
            List of message dictionaries
        """
        # Add system instructions
        messages = [
            {
                "role": "user",
                "parts": [{"text": persistent_ctx.system_instructions}],
            }
        ]

        # Add conversation history
        for msg in transient_ctx.conversation_history:
            # Map roles to Gemini API format
            if msg.role in ["user", "system"]:
                role = "user"
            elif msg.role == "assistant":
                role = "model"
            else:
                # tool role stays as "tool"
                role = msg.role

            messages.append({
                "role": role,
                "parts": [{"text": msg.content}],
            })

        return messages

    def get_metrics_summary(self, window_size: int = 100) -> Dict[str, Any]:
        """
        Get summary metrics and drift analysis.

        Args:
            window_size: Number of recent transactions to analyze

        Returns:
            Metrics summary dictionary
        """
        metrics_logger = get_metrics_logger()
        drift_metrics = metrics_logger.check_drift(window_size)

        return {
            "drift_analysis": drift_metrics,
            "model": self.model_name,
            "compression_enabled": {
                "semantic_pruning": self.config.enable_semantic_pruning,
                "knowledge_distillation": self.config.enable_knowledge_distillation,
                "history_compaction": self.config.enable_history_compaction,
            },
            "output_format": self.config.default_output_format,
        }
