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
        enable_litellm: bool = True,
        main_model_cost_per_million: float = 1.00,
        student_model_cost_per_million: float = 0.15
    ):
        """
        Initialize Context Manager.

        Args:
            student_model: Lightweight model for summarization
            compression_threshold: Only compress if input exceeds this token count
            target_compression_ratio: Target ratio (0.40 = 60% reduction)
            enable_litellm: Enable LiteLLM integration (requires API keys)
            main_model_cost_per_million: Cost per 1M input tokens for main LLM (default: $1.00 for Gemini 2.0 Flash)
            student_model_cost_per_million: Cost per 1M tokens for student summarizer (default: $0.15 for GPT-4o-mini)
        """
        self.student_model = student_model
        self.compression_threshold = compression_threshold
        self.target_compression_ratio = target_compression_ratio
        self.enable_litellm = enable_litellm
        self.main_model_cost_per_million = main_model_cost_per_million
        self.student_model_cost_per_million = student_model_cost_per_million

        # Try to import litellm if enabled
        self.litellm = None
        if enable_litellm:
            try:
                import litellm
                self.litellm = litellm
            except ImportError:
                print("Warning: litellm not available. Compression features limited.")

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
        # Step 1: Count original tokens
        raw_text = self._context_to_text(context)
        original_tokens = self.estimate_tokens(raw_text)

        # Step 2: Check if compression warranted
        if original_tokens < self.compression_threshold:
            return OptimizedContext(
                optimized_text=raw_text,
                original_tokens=original_tokens,
                optimized_tokens=original_tokens,
                compression_ratio=1.0,
                method_used=CompressionMethod.NONE,
                compression_time_ms=0.0,
                estimated_cost_savings=0.0
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
        start_time = time.perf_counter()

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
            print(f"Warning: Compression failed ({e}), using raw context")
            compressed_text = raw_text
            strategy = CompressionMethod.NONE

        compression_time = (time.perf_counter() - start_time) * 1000

        # Step 5: Count compressed tokens
        compressed_tokens = self.estimate_tokens(compressed_text)
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

        # Step 6: Calculate cost savings using configured pricing
        token_savings = original_tokens - compressed_tokens
        cost_savings = (token_savings / 1_000_000) * self.main_model_cost_per_million

        return OptimizedContext(
            optimized_text=compressed_text,
            original_tokens=original_tokens,
            optimized_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            method_used=strategy,
            compression_time_ms=compression_time,
            estimated_cost_savings=cost_savings
        )

    def estimate_tokens(self, text: str, model: str = "gpt-4") -> int:
        """
        Estimate token count using litellm's token_counter or fallback.

        Args:
            text: Text to count tokens for
            model: Model tokenizer to use

        Returns:
            Estimated token count
        """
        if self.litellm:
            try:
                tokens = self.litellm.token_counter(model=model, text=text)
                return tokens
            except Exception:
                # Fallback to rough estimate
                pass

        # Fallback: rough estimate (1 token ≈ 4 chars)
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
        Extract top-k most relevant chunks from structured data.

        For MVP: Simple implementation that takes first N items.
        For production: Use embedding similarity to user query.

        Args:
            context: ContextData with tool results

        Returns:
            Pruned context as text
        """
        # Simplified implementation
        pruned_results = []
        target_tokens = int(
            sum(r.token_estimate for r in context.tool_results) *
            self.target_compression_ratio
        )

        current_tokens = 0
        for result in context.tool_results:
            if isinstance(result.data, list):
                # Take first k items until we hit target
                items_to_keep = []
                for item in result.data:
                    item_tokens = len(str(item)) // 4
                    if current_tokens + item_tokens > target_tokens:
                        break
                    items_to_keep.append(item)
                    current_tokens += item_tokens

                pruned_results.append({
                    "tool_id": result.tool_id,
                    "data": items_to_keep,
                    "note": f"Showing {len(items_to_keep)} of {len(result.data)} items"
                })
            else:
                # Keep non-list data as-is
                pruned_results.append({
                    "tool_id": result.tool_id,
                    "data": result.data
                })

        # Add transient context
        if context.transient_context:
            pruned_results.append({"context": context.transient_context})

        return json.dumps(pruned_results, indent=2)

    def _knowledge_distillation(self, context: ContextData) -> str:
        """
        Use student summarizer model to compress verbose text.

        Cost-benefit check:
        - Student model cost: ~$0.15 per 1M tokens (GPT-4o-mini)
        - Must save more than this in main model costs

        Args:
            context: ContextData with verbose text

        Returns:
            Summarized context
        """
        raw_text = self._context_to_text(context)
        raw_tokens = self.estimate_tokens(raw_text)

        # If litellm not available, use simple truncation
        if not self.litellm:
            target_length = int(len(raw_text) * self.target_compression_ratio)
            return raw_text[:target_length] + "\n[... truncated for length ...]"

        # Cost of summarization using configured pricing
        summarization_cost = (raw_tokens / 1_000_000) * self.student_model_cost_per_million

        # Expected savings (assume configured compression ratio)
        expected_compressed_tokens = int(raw_tokens * self.target_compression_ratio)
        token_savings = raw_tokens - expected_compressed_tokens
        main_model_savings = (token_savings / 1_000_000) * self.main_model_cost_per_million

        # Only summarize if cost-effective
        if main_model_savings <= summarization_cost:
            return raw_text  # Not worth it

        # Call student model
        try:
            response = self.litellm.completion(
                model=self.student_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize the following text concisely while preserving all key information."
                    },
                    {"role": "user", "content": raw_text}
                ],
                max_tokens=expected_compressed_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback to raw if summarization fails
            print(f"Warning: Summarization failed ({e}), using raw text")
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
