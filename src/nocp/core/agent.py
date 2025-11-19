"""
High-Efficiency Proxy Agent - Main orchestration class.

This is the central component that coordinates all modules (Act-Assess-Articulate)
to deliver optimized LLM interactions with token efficiency as the primary goal.
"""

import time
import uuid
from collections.abc import Callable
from typing import Any

from ..core.config import get_config
from ..llm.client import LLMClient
from ..llm.router import ModelRouter
from ..models.context import PersistentContext, TransientContext
from ..models.schemas import (
    AgentRequest,
    AgentResponse,
    CompressionResult,
    ContextMetrics,
    ToolDefinition,
)
from ..modules.context_manager import ContextManager
from ..modules.output_serializer import OutputSerializer
from ..modules.router import RequestRouter
from ..modules.tool_executor import ToolExecutor
from ..utils.logging import agent_logger, get_logger, get_metrics_logger, log_metrics
from ..utils.rich_logging import console
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
        api_key: str | None = None,
        model_name: str | None = None,
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

        # Print startup banner (if rich console is enabled)
        if self.config.enable_rich_console:
            console.print_banner()
            console.print_config_summary(self.config)

        # Initialize token counter
        self.token_counter = TokenCounter(model_name)

        # Initialize LLM client with LiteLLM
        if self.config.enable_litellm:
            # Use LiteLLM for multi-provider support
            fallback_models = None
            if self.config.litellm_fallback_models:
                fallback_models = [
                    m.strip() for m in self.config.litellm_fallback_models.split(",")
                ]

            # Build provider-specific API keys dictionary
            provider_api_keys = {}
            if self.config.openai_api_key:
                provider_api_keys["openai"] = self.config.openai_api_key
            if self.config.anthropic_api_key:
                provider_api_keys["anthropic"] = self.config.anthropic_api_key

            self.llm_client = LLMClient(
                default_model=model_name or self.config.litellm_default_model,
                api_key=api_key or self.config.gemini_api_key,
                provider_api_keys=provider_api_keys,
                fallback_models=fallback_models,
                max_retries=self.config.litellm_max_retries,
                timeout=self.config.litellm_timeout,
            )
            self.model_name = model_name or self.config.litellm_default_model
        else:
            # Fallback to google.generativeai for backward compatibility
            import google.generativeai as genai

            api_key = api_key or self.config.gemini_api_key
            genai.configure(api_key=api_key)
            self.model_name = model_name or self.config.gemini_model
            self.model = genai.GenerativeModel(self.model_name)
            self.llm_client = None  # type: ignore[assignment]

        # Initialize model router for intelligent model selection
        self.model_router = ModelRouter()

        # Initialize core modules
        self.router = RequestRouter()
        self.tool_executor = ToolExecutor()
        self.context_manager = ContextManager(tool_executor=self.tool_executor)
        self.output_serializer = OutputSerializer()

        self.logger.info(
            "proxy_agent_initialized",
            model=self.model_name,
            litellm_enabled=self.config.enable_litellm,
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
        return_format: str | None = None,
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

        agent_logger.log_operation_start(
            "agent_request",
            {"transaction_id": transaction_id, "query_preview": request.query[:100]},
        )

        self.logger.info(
            "processing_request",
            transaction_id=transaction_id,
            query=request.query[:100],
        )

        # Initialize metrics tracking
        compression_operations: list[CompressionResult] = []
        tools_used: list[str] = []

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
                force_format=return_format,  # type: ignore[arg-type]
            )

            serialization_time = (time.perf_counter() - serialization_start) * 1000

            # Calculate final metrics
            total_latency_ms = (time.perf_counter() - start_time) * 1000
            compression_latency_ms = sum(
                comp.compression_time_ms for comp in compression_operations
            )

            # Count output tokens
            raw_output_tokens = self.token_counter.count_text(agent_response.model_dump_json())

            # Calculate token savings
            input_token_savings = raw_input_tokens - compressed_input_tokens

            # Create metrics
            metrics = ContextMetrics(
                transaction_id=transaction_id,
                raw_input_tokens=raw_input_tokens,
                compressed_input_tokens=compressed_input_tokens,
                input_compression_ratio=(
                    compressed_input_tokens / raw_input_tokens if raw_input_tokens > 0 else 1.0
                ),
                raw_output_tokens=raw_output_tokens,
                final_output_format=output_format,
                output_token_savings=token_savings,
                total_latency_ms=total_latency_ms,
                compression_latency_ms=compression_latency_ms,
                llm_inference_latency_ms=total_latency_ms
                - compression_latency_ms
                - serialization_time,
                tools_used=tools_used,
                compression_operations=compression_operations,
            )

            # Log metrics
            log_metrics(metrics)

            # Print beautiful metrics table (if rich console is enabled)
            if self.config.enable_rich_console:
                console.print_metrics(metrics)

            # Update session
            self.router.finalize_session(
                persistent_ctx=persistent_ctx,
                tokens_used=compressed_input_tokens + raw_output_tokens,
                cost=0.0,  # Cost tracking removed - focus on token efficiency
            )

            self.logger.info(
                "request_processed",
                transaction_id=transaction_id,
                total_latency_ms=round(total_latency_ms, 2),
                input_token_savings=input_token_savings,
                output_token_savings=token_savings,
                compression_ratio=round(metrics.input_compression_ratio, 3),
            )

            # Print success message (if rich console is enabled)
            if self.config.enable_rich_console:
                console.print_success(
                    f"Request processed successfully - saved {input_token_savings + token_savings:,} tokens"
                )

            # Log operation completion with component logger
            agent_logger.log_operation_complete(
                "agent_request",
                duration_ms=total_latency_ms,
                details={
                    "transaction_id": transaction_id,
                    "input_token_savings": input_token_savings,
                    "output_token_savings": token_savings,
                    "compression_ratio": round(metrics.input_compression_ratio, 3),
                    "tools_used": len(tools_used),
                },
            )

            return serialized_output, metrics

        except Exception as e:
            self.logger.error(
                "request_processing_failed",
                transaction_id=transaction_id,
                error=str(e),
            )
            if self.config.enable_rich_console:
                console.print_error(f"Request processing failed: {str(e)}")

            # Log error with component logger
            agent_logger.log_operation_error("agent_request", e, {"transaction_id": transaction_id})
            raise

    def _handle_tool_execution(
        self,
        tool_name: str,
        tool_params: dict[str, Any],
        transient_ctx: TransientContext,
        compression_operations: list[CompressionResult],
        tools_used: list[str],
    ) -> None:
        """
        Execute a tool and manage its output.

        This helper extracts the common tool execution logic used by both
        LiteLLM and Gemini response handlers.

        Args:
            tool_name: Name of the tool to execute
            tool_params: Parameters for the tool
            transient_ctx: Transient context to update
            compression_operations: List to track compression operations
            tools_used: List to track tools used
        """
        tools_used.append(tool_name)

        # Execute tool
        tool_result = self.tool_executor.execute_tool(tool_name, tool_params)

        # Apply context management (compression)
        managed_output, compression_result = self.context_manager.manage_tool_output(tool_result)

        if compression_result:
            compression_operations.append(compression_result)

        # Add tool result to history
        self.router.add_tool_result_to_history(
            transient_ctx,
            tool_name,
            managed_output,
        )

    def _execute_agent_loop(
        self,
        transient_ctx: TransientContext,
        persistent_ctx: PersistentContext,
        compression_operations: list[CompressionResult],
        tools_used: list[str],
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
        agent_logger.log_operation_start("agent_loop")
        # Compact history if needed
        history_compression = self.context_manager.compact_conversation_history(transient_ctx)
        if history_compression:
            compression_operations.append(history_compression)

        # Prepare messages for LLM
        messages = self._format_messages_for_llm(transient_ctx, persistent_ctx)

        # Get tool schemas
        tool_schemas = (
            self.tool_executor.get_tool_schemas_for_gemini()
            if hasattr(self.tool_executor, "get_tool_schemas_for_gemini")
            else None
        )

        # Call LLM with or without LiteLLM
        if self.llm_client:
            # Use LiteLLM client
            if tool_schemas and len(tool_schemas) > 0:
                response = self.llm_client.complete_with_tools(
                    messages=messages,
                    tools=tool_schemas,
                    max_tokens=self.config.max_output_tokens,
                    temperature=0.7,
                )
            else:
                response = self.llm_client.complete(
                    messages=messages,
                    max_tokens=self.config.max_output_tokens,
                    temperature=0.7,
                )
        else:
            # Use genai directly (backward compatibility)
            import google.generativeai as genai

            generation_config = genai.GenerationConfig(
                max_output_tokens=self.config.max_output_tokens,
                temperature=0.7,
            )
            response = self.model.generate_content(
                messages,
                tools=tool_schemas or None,
                generation_config=generation_config,
            )

        # Check for function calls and extract response
        if self.llm_client:
            # Handle LiteLLM response
            if response.tool_calls:
                # Tool calling flow
                tool_call = response.tool_calls[0]
                tool_name = tool_call["name"]
                tool_params = tool_call["arguments"]

                # Execute tool and manage output (shared logic)
                self._handle_tool_execution(
                    tool_name,
                    tool_params,
                    transient_ctx,
                    compression_operations,
                    tools_used,
                )

                # Continue agent loop with tool result
                final_response = self.llm_client.complete(
                    messages=self._format_messages_for_llm(transient_ctx, persistent_ctx),
                    max_tokens=self.config.max_output_tokens,
                    temperature=0.7,
                )
                response_text = str(final_response.content) if final_response.content else ""
            else:
                response_text = str(response.content) if response.content else ""
        else:
            # Handle genai response (backward compatibility)
            if (
                hasattr(response, "candidates")
                and len(response.candidates) > 0
                and hasattr(response.candidates[0], "content")
                and hasattr(response.candidates[0].content, "parts")
                and len(response.candidates[0].content.parts) > 0
                and (function_call := response.candidates[0].content.parts[0].function_call)
            ):
                tool_name = function_call.name
                tool_params = dict(function_call.args)

                # Execute tool and manage output (shared logic)
                self._handle_tool_execution(
                    tool_name,
                    tool_params,
                    transient_ctx,
                    compression_operations,
                    tools_used,
                )

                # Continue agent loop with tool result
                import google.generativeai as genai

                generation_config = genai.GenerationConfig(
                    max_output_tokens=self.config.max_output_tokens,
                    temperature=0.7,
                )
                final_response = self.model.generate_content(
                    self._format_messages_for_llm(transient_ctx, persistent_ctx),
                    generation_config=generation_config,
                )
                response_text = final_response.text
            else:
                response_text = response.text

        # Parse response into AgentResponse schema
        # For now, return a simple response (in production, use structured output)
        agent_logger.log_operation_complete(
            "agent_loop",
            details={"tools_used": len(tools_used), "compressions": len(compression_operations)},
        )

        return AgentResponse(
            answer=response_text,
            tool_results_summary=[f"Used tool: {t}" for t in tools_used],
            confidence=0.9,
        )

    def _format_messages_for_llm(
        self,
        transient_ctx: TransientContext,
        persistent_ctx: PersistentContext,
    ) -> list[dict[str, str]]:
        """
        Format context into messages for LLM API (LiteLLM or Gemini).

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

            messages.append(
                {
                    "role": role,
                    "parts": [{"text": msg.content}],
                }
            )

        return messages  # type: ignore[return-value]

    def get_metrics_summary(self, window_size: int = 100) -> dict[str, Any]:
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
