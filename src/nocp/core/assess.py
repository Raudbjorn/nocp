"""
Assess Module: Context Manager

Optimizes context to reduce token usage before LLM calls.
Implements semantic pruning, knowledge distillation, and history compaction.
"""

import json
import time
from typing import List, Optional

from ..exceptions import CompressionError
from ..models.contracts import (
    ContextData,
    OptimizedContext,
    CompressionMethod,
    ToolResult,
    ChatMessage,
)
from ..utils.logging import assess_logger


class ContextManager:
    """
    Optimizes context to reduce token usage before LLM calls.

    Example:
        manager = ContextManager(
            student_model="openai/gpt-4o-mini",
            compression_threshold=10_000
        )

        context = ContextData(
            tool_results=[large_result],
            transient_context={"query": "Summarize this"},
            max_tokens=50_000
        )

        optimized = manager.optimize(context)
        print(f"Compressed from {optimized.original_tokens} to {optimized.optimized_tokens}")
    """

    def __init__(
        self,
        student_model: str = "openai/gpt-4o-mini",
        compression_threshold: int = 10_000,
        target_compression_ratio: float = 0.40,
        enable_litellm: bool = True
    ):
        """
        Initialize Context Manager.

        Args:
            student_model: Lightweight model for summarization
            compression_threshold: Only compress if input exceeds this token count
            target_compression_ratio: Target ratio (0.40 = 60% reduction)
            enable_litellm: Enable LiteLLM integration (requires API keys)
        """
        self.student_model = student_model
        self.compression_threshold = compression_threshold
        self.target_compression_ratio = target_compression_ratio
        self.enable_litellm = enable_litellm

        # Try to import litellm if enabled
        self.litellm = None
        if enable_litellm:
            try:
                import litellm
                self.litellm = litellm
            except ImportError:
                assess_logger.logger.warning("litellm not available. Compression features limited.")

    def optimize(self, context: ContextData) -> OptimizedContext:
        """
        Main entry point for context optimization.

        Decision tree:
        1. Estimate total tokens
        2. If < threshold, return raw (no compression)
        3. Select compression strategy based on data type
        4. Apply compression
        5. Verify cost-benefit
        6. Return OptimizedContext

        Args:
            context: ContextData with tool results and conversation history

        Returns:
            OptimizedContext with compressed text and metrics
        """
        assess_logger.log_operation_start("context_optimization")
        start_time = time.perf_counter()

        # Step 1: Count original tokens
        raw_text = self._context_to_text(context)
        original_tokens = self.estimate_tokens(raw_text)

        # Step 2: Check if compression warranted
        if original_tokens < self.compression_threshold:
            early_exit_time = (time.perf_counter() - start_time) * 1000
            assess_logger.log_operation_complete(
                "context_optimization",
                duration_ms=early_exit_time,
                details={"method": "NONE", "original_tokens": original_tokens, "reason": "below_threshold"}
            )
            return OptimizedContext(
                optimized_text=raw_text,
                original_tokens=original_tokens,
                optimized_tokens=original_tokens,
                compression_ratio=1.0,
                method_used=CompressionMethod.NONE,
                compression_time_ms=0.0
            )

        # Step 3: Select strategy
        if context.compression_strategy:
            try:
                strategy = CompressionMethod(context.compression_strategy)
            except ValueError:
                strategy = self.select_strategy(context)
        else:
            strategy = self.select_strategy(context)

        # Step 4: Apply compression
        compression_start_time = time.perf_counter()

        try:
            if strategy == CompressionMethod.SEMANTIC_PRUNING:
                compressed_text = self._semantic_pruning(context)
            elif strategy == CompressionMethod.KNOWLEDGE_DISTILLATION:
                compressed_text = self._knowledge_distillation(context)
            elif strategy == CompressionMethod.HISTORY_COMPACTION:
                compressed_text = self._history_compaction(context)
            else:
                compressed_text = raw_text
        except Exception as e:
            # Fallback to raw on compression failure
            assess_logger.logger.warning(f"Compression failed ({e}), using raw context")
            compressed_text = raw_text
            strategy = CompressionMethod.NONE

        compression_time = (time.perf_counter() - compression_start_time) * 1000
        total_time = (time.perf_counter() - start_time) * 1000

        # Step 5: Count compressed tokens
        compressed_tokens = self.estimate_tokens(compressed_text)
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

        assess_logger.log_operation_complete(
            "context_optimization",
            duration_ms=total_time,
            details={
                "method": strategy.value,
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "compression_ratio": round(compression_ratio, 3),
                "compression_time_ms": round(compression_time, 2)
            }
        )

        return OptimizedContext(
            optimized_text=compressed_text,
            original_tokens=original_tokens,
            optimized_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            method_used=strategy,
            compression_time_ms=compression_time
        )

    def estimate_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Estimate token count using litellm's token_counter with support for multiple models.

        This method integrates with LiteLLM to provide accurate token counting for various
        model tokenizers (OpenAI, Anthropic, Gemini, etc.).

        Args:
            text: Text to count tokens for
            model: Model tokenizer to use (e.g., "gpt-4", "gpt-3.5-turbo", "claude-3-opus").
                   If None, uses student_model or defaults to "gpt-4"

        Returns:
            Estimated token count (accurate within ±5% for supported models)
        """
        # Default to student model if no model specified
        target_model = model or self.student_model or "gpt-4"

        if self.litellm:
            try:
                # LiteLLM's token_counter supports multiple models
                tokens = self.litellm.token_counter(model=target_model, text=text)
                return tokens
            except Exception as e:
                # Log the error but fall back gracefully
                assess_logger.logger.warning(f"LiteLLM token counting failed for {target_model}: {e}")

        # Fallback: rough estimate (1 token ≈ 4 chars)
        # This should be within ±20% accuracy for most text
        return len(text) // 4

    def select_strategy(self, context: ContextData) -> CompressionMethod:
        """
        Auto-detect optimal compression strategy.

        Heuristics:
        - If tool results contain >1000 tokens of structured data (JSON/lists):
          Use SEMANTIC_PRUNING
        - If tool results contain verbose text (logs, descriptions):
          Use KNOWLEDGE_DISTILLATION
        - If message_history has >10 messages:
          Use HISTORY_COMPACTION

        Args:
            context: ContextData to analyze

        Returns:
            Selected CompressionMethod
        """
        # Check for large structured data
        for result in context.tool_results:
            if isinstance(result.data, (list, dict)):
                if result.token_estimate > 1000:
                    return CompressionMethod.SEMANTIC_PRUNING

        # Check for verbose text
        total_text_tokens = sum(
            r.token_estimate for r in context.tool_results
            if isinstance(r.data, str)
        )
        if total_text_tokens > 5000:
            return CompressionMethod.KNOWLEDGE_DISTILLATION

        # Check conversation history
        if len(context.message_history) > 10:
            return CompressionMethod.HISTORY_COMPACTION

        return CompressionMethod.NONE

    def _semantic_pruning(self, context: ContextData) -> str:
        """
        Extract top-k most relevant chunks from structured data (RAG output).

        Implements top-k chunk selection with intelligent filtering:
        - For list data: Select top chunks based on relevance heuristics
        - For text data: Extract key sentences/paragraphs
        - Target: 60-70% reduction for document-heavy inputs

        Future enhancement: Use embedding-based similarity to user query.

        Args:
            context: ContextData with tool results

        Returns:
            Pruned context as text (targeting 30-40% of original size)
        """
        pruned_results = []
        target_tokens = int(
            sum(r.token_estimate for r in context.tool_results) *
            self.target_compression_ratio
        )

        current_tokens = 0

        for result in context.tool_results:
            if isinstance(result.data, list):
                # Apply top-k selection for lists
                items = result.data

                # Skip empty lists
                if len(items) == 0:
                    pruned_results.append({
                        "tool_id": result.tool_id,
                        "data": [],
                        "metadata": {
                            "pruned": False,
                            "kept": 0,
                            "total": 0,
                            "reduction_pct": 0.0
                        }
                    })
                    continue

                # Calculate how many items to keep based on target ratio
                # Aim for 30-40% retention (60-70% reduction)
                max_items = max(1, int(len(items) * self.target_compression_ratio))

                items_to_keep = []
                for i, item in enumerate(items[:max_items]):
                    item_tokens = self.estimate_tokens(str(item))
                    if current_tokens + item_tokens > target_tokens:
                        break
                    items_to_keep.append(item)
                    current_tokens += item_tokens

                pruned_results.append({
                    "tool_id": result.tool_id,
                    "data": items_to_keep,
                    "metadata": {
                        "pruned": True,
                        "kept": len(items_to_keep),
                        "total": len(items),
                        "reduction_pct": round((1 - len(items_to_keep) / len(items)) * 100, 1) if len(items) > 0 else 0.0
                    }
                })
            elif isinstance(result.data, str) and len(result.data) > 1000:
                # For large text, extract key portions
                text = result.data
                # Simple heuristic: take first and last paragraphs + middle section
                paragraphs = text.split('\n\n')
                if len(paragraphs) > 5:
                    # Calculate total paragraphs to keep based on target ratio
                    keep_count = max(3, int(len(paragraphs) * self.target_compression_ratio))

                    # Ensure keep_count doesn't exceed available paragraphs
                    keep_count = min(keep_count, len(paragraphs))

                    # Strategy: keep first 2, last 2, and fill middle from center
                    if keep_count <= 4:
                        # If keep_count is small, just take first and last
                        kept_paragraphs = paragraphs[:keep_count//2] + paragraphs[-(keep_count - keep_count//2):]
                    else:
                        # Keep first 2, last 2, and sample from middle
                        middle_count = keep_count - 4
                        middle_start = len(paragraphs) // 2 - middle_count // 2
                        middle_end = middle_start + middle_count
                        kept_paragraphs = (
                            paragraphs[:2] +
                            paragraphs[middle_start:middle_end] +
                            paragraphs[-2:]
                        )
                    pruned_text = '\n\n'.join(kept_paragraphs)
                else:
                    pruned_text = text

                pruned_results.append({
                    "tool_id": result.tool_id,
                    "data": pruned_text,
                    "metadata": {
                        "pruned": True,
                        "original_length": len(text),
                        "pruned_length": len(pruned_text)
                    }
                })
                current_tokens += self.estimate_tokens(pruned_text)
            else:
                # Keep smaller items as-is
                pruned_results.append({
                    "tool_id": result.tool_id,
                    "data": result.data
                })
                current_tokens += result.token_estimate

        # Add transient context with query (important for relevance)
        if context.transient_context:
            pruned_results.insert(0, {
                "query": context.transient_context.get("query", ""),
                "context": context.transient_context
            })

        return json.dumps(pruned_results, indent=2)

    def _knowledge_distillation(self, context: ContextData) -> str:
        """
        Use student summarizer model to compress verbose text.

        Uses a lightweight student model (gpt-4o-mini) to summarize content while
        preserving key information. Implements cost-benefit calculation to ensure
        compression savings exceed overhead.

        Cost-Benefit Logic:
        - Only compress if: (input_tokens - output_tokens) > compression_cost
        - Compression cost = prompt_tokens + summary_tokens for student model
        - Falls back to raw output if compression is too expensive

        Args:
            context: ContextData with verbose text

        Returns:
            Summarized context (or raw text if compression not beneficial)
        """
        raw_text = self._context_to_text(context)
        raw_tokens = self.estimate_tokens(raw_text)

        # If litellm not available, use simple truncation
        if not self.litellm:
            target_length = int(len(raw_text) * self.target_compression_ratio)
            truncated = raw_text[:target_length] + "\n[... truncated for length ...]"
            return truncated

        # Target compressed size (40% of original = 60% reduction)
        expected_compressed_tokens = int(raw_tokens * self.target_compression_ratio)

        # Estimate compression cost (overhead only)
        # The cost is: prompt_template tokens + output tokens
        # We don't count raw_tokens here because we have to send those to the main model anyway
        # The compression cost is the ADDITIONAL cost of using the student model
        prompt_template = "Summarize the following text concisely while preserving all key information."
        prompt_overhead_tokens = self.estimate_tokens(prompt_template)
        estimated_response_tokens = expected_compressed_tokens

        # Total overhead cost of compression (not counting the input we'd send anyway)
        compression_overhead = prompt_overhead_tokens + estimated_response_tokens

        # Calculate potential savings
        # Savings = (tokens saved from compression) - (overhead of compression)
        # Tokens saved = raw_tokens - expected_compressed_tokens
        # Overhead = prompt template + compressed output
        potential_savings = (raw_tokens - expected_compressed_tokens) - compression_overhead

        # Only proceed if savings are positive (compression is beneficial)
        if potential_savings <= 0:
            assess_logger.logger.warning(
                f"Compression not cost-effective. "
                f"Raw: {raw_tokens}, Compressed: {expected_compressed_tokens}, "
                f"Overhead: {compression_overhead}, Savings: {potential_savings}"
            )
            return raw_text

        # Call student model for summarization
        try:
            response = self.litellm.completion(
                model=self.student_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize the following text concisely while preserving all key information. "
                                   "Focus on facts, key insights, and actionable items."
                    },
                    {"role": "user", "content": raw_text}
                ],
                max_tokens=expected_compressed_tokens,
                temperature=0.3  # Lower temperature for more consistent summaries
            )

            summary = response.choices[0].message.content

            # Verify actual compression ratio
            summary_tokens = self.estimate_tokens(summary)
            actual_compression_ratio = summary_tokens / raw_tokens if raw_tokens > 0 else 1.0

            # Log compression metrics
            assess_logger.log_metric(
                "compression_ratio",
                actual_compression_ratio,
                "ratio"
            )

            return summary

        except Exception as e:
            # Fallback to raw if summarization fails
            assess_logger.logger.warning(f"Summarization failed ({e}), using raw text")
            return raw_text

    def _history_compaction(self, context: ContextData) -> str:
        """
        Compress old conversation messages into a summary.

        Strategy: Keep last 5 messages, summarize the rest.

        Args:
            context: ContextData with message history

        Returns:
            Compacted context with summary + recent messages
        """
        if len(context.message_history) <= 5:
            return self._context_to_text(context)

        # Keep recent messages
        recent_messages = context.message_history[-5:]

        # Summarize older messages
        old_messages = context.message_history[:-5]
        old_text = "\n".join(f"{msg.role}: {msg.content}" for msg in old_messages)

        # If litellm not available, use simple truncation
        if not self.litellm:
            summary = f"[Earlier conversation: {len(old_messages)} messages exchanged]"
        else:
            # Summarize using student model
            try:
                summary_response = self.litellm.completion(
                    model=self.student_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Provide a concise summary of the conversation history."
                        },
                        {"role": "user", "content": old_text}
                    ],
                    max_tokens=500
                )
                summary = summary_response.choices[0].message.content
            except Exception:
                summary = f"[Earlier conversation: {len(old_messages)} messages exchanged]"

        # Combine summary with recent messages
        parts = [f"[Summary of earlier conversation: {summary}]"]

        # Add recent messages
        for msg in recent_messages:
            parts.append(f"{msg.role}: {msg.content}")

        # Add tool results and other context
        for result in context.tool_results:
            parts.append(f"\nTool: {result.tool_id}")
            parts.append(result.to_text())

        return "\n\n".join(parts)

    def _context_to_text(self, context: ContextData) -> str:
        """
        Convert ContextData to text representation.

        Args:
            context: ContextData to convert

        Returns:
            Text representation of all context
        """
        parts = []

        # Add persistent context
        if context.persistent_context:
            parts.append(f"System: {context.persistent_context}")

        # Add tool results
        for result in context.tool_results:
            parts.append(f"Tool: {result.tool_id}")
            parts.append(result.to_text())

        # Add transient context
        if context.transient_context:
            parts.append(f"Context: {json.dumps(context.transient_context, indent=2)}")

        # Add message history
        for msg in context.message_history:
            parts.append(f"{msg.role}: {msg.content}")

        return "\n\n".join(parts)
