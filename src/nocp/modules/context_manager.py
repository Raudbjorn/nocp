"""
Context Manager (Assess Module) - Dynamic context compression and optimization.

Implements the core token optimization strategies:
1. Semantic Pruning for RAG/document outputs
2. Knowledge Distillation via Student Summarizer
3. Conversation History Compaction

This is the critical component that enforces the dynamic compression policy
and Cost-of-Compression Calculus.
"""

from typing import List, Optional, Literal
import google.generativeai as genai

from ..models.schemas import ToolExecutionResult, CompressionResult
from ..models.context import TransientContext, ConversationMessage
from ..core.config import get_config
from ..utils.logging import get_logger
from ..utils.token_counter import TokenCounter


class ContextManager:
    """
    Context Manager - The "Assess" component of the architecture.

    Responsibilities:
    - Evaluate raw tool outputs using CountTokens API
    - Apply dynamic compression based on T_comp threshold
    - Implement Cost-of-Compression Calculus
    - Manage conversation history compaction
    - Minimize input tokens while preserving semantic content
    """

    def __init__(self, student_model: Optional[str] = None):
        """
        Initialize the context manager.

        Args:
            student_model: Optional model name for student summarizer
        """
        self.config = get_config()
        self.logger = get_logger(__name__)
        self.token_counter = TokenCounter()

        # Initialize student summarizer model
        genai.configure(api_key=self.config.gemini_api_key)
        self.student_model_name = student_model or self.config.student_summarizer_model
        self.student_model = genai.GenerativeModel(self.student_model_name)

    def manage_tool_output(
        self,
        tool_result: ToolExecutionResult,
    ) -> tuple[str, Optional[CompressionResult]]:
        """
        Manage tool output through dynamic compression policy.

        This implements the core Token Gate logic from the architecture.

        Args:
            tool_result: Raw result from tool execution

        Returns:
            Tuple of (processed_output, compression_result)
        """
        tool_name = tool_result.tool_name
        raw_output = tool_result.raw_output
        raw_token_count = tool_result.raw_token_count or 0

        # Get compression threshold for this tool
        t_comp = self.config.get_compression_threshold(tool_name)

        self.logger.info(
            "evaluating_tool_output",
            tool_name=tool_name,
            raw_tokens=raw_token_count,
            threshold=t_comp,
        )

        # Token Gate: Check if compression is needed
        if raw_token_count <= t_comp:
            self.logger.info(
                "compression_skipped",
                tool_name=tool_name,
                reason="below_threshold",
                raw_tokens=raw_token_count,
            )
            return raw_output, None

        # Apply compression strategy based on tool type/content
        compression_method = self._select_compression_method(tool_name, raw_output)

        # Execute compression
        compressed_output, compression_cost = self._apply_compression(
            raw_output=raw_output,
            method=compression_method,
        )

        # Count compressed tokens
        compressed_token_count = self.token_counter.count_text(compressed_output)

        # Calculate metrics
        net_savings = raw_token_count - compressed_token_count - compression_cost

        # Cost-of-Compression Calculus: Verify savings justify the overhead
        if net_savings <= 0:
            self.logger.warning(
                "compression_not_justified",
                tool_name=tool_name,
                raw_tokens=raw_token_count,
                compressed_tokens=compressed_token_count,
                compression_cost=compression_cost,
                net_savings=net_savings,
            )
            # Return raw output if compression wasn't beneficial
            return raw_output, None

        # Create compression result
        compression_result = CompressionResult(
            original_tokens=raw_token_count,
            compressed_tokens=compressed_token_count,
            compression_ratio=compressed_token_count / raw_token_count,
            compression_method=compression_method,
            compression_cost_tokens=compression_cost,
            net_savings=net_savings,
        )

        self.logger.info(
            "compression_successful",
            tool_name=tool_name,
            original_tokens=raw_token_count,
            compressed_tokens=compressed_token_count,
            net_savings=net_savings,
            method=compression_method,
        )

        return compressed_output, compression_result

    def _select_compression_method(
        self,
        tool_name: str,
        content: str,
    ) -> Literal["semantic_pruning", "knowledge_distillation", "history_compaction", "none"]:
        """
        Select appropriate compression method based on content type.

        Args:
            tool_name: Name of the tool
            content: Raw content

        Returns:
            Compression method name
        """
        # Heuristics for method selection
        # In production, this could use metadata from tool definitions

        # Check for RAG/database outputs (typically structured or have clear sections)
        if any(keyword in tool_name.lower() for keyword in ["search", "rag", "database", "query"]):
            if self.config.enable_semantic_pruning:
                return "semantic_pruning"

        # Default to knowledge distillation for unstructured content
        if self.config.enable_knowledge_distillation:
            return "knowledge_distillation"

        return "none"

    def _apply_compression(
        self,
        raw_output: str,
        method: Literal["semantic_pruning", "knowledge_distillation", "history_compaction", "none"],
    ) -> tuple[str, int]:
        """
        Apply selected compression method.

        Args:
            raw_output: Raw output to compress
            method: Compression method to use

        Returns:
            Tuple of (compressed_output, compression_cost_tokens)
        """
        if method == "semantic_pruning":
            return self._apply_semantic_pruning(raw_output)
        elif method == "knowledge_distillation":
            return self._apply_knowledge_distillation(raw_output)
        elif method == "history_compaction":
            return self._apply_history_compaction(raw_output)
        else:
            return raw_output, 0

    def _apply_semantic_pruning(self, content: str) -> tuple[str, int]:
        """
        Apply semantic pruning for document/RAG outputs.

        Strategy: Keep most relevant chunks, discard redundant information.
        This is a simplified implementation; production would use embeddings.

        Args:
            content: Content to prune

        Returns:
            Tuple of (pruned_content, compression_cost)
        """
        # Simplified semantic pruning: Take first 50% of content
        # In production, this would:
        # 1. Split into chunks
        # 2. Compute embeddings for each chunk
        # 3. Rank by similarity to query
        # 4. Keep top-k chunks

        lines = content.split("\n")
        keep_count = max(1, len(lines) // 2)  # Keep top 50%
        pruned = "\n".join(lines[:keep_count])

        # Minimal compression cost (no LLM call)
        compression_cost = 0

        return pruned, compression_cost

    def _apply_knowledge_distillation(self, content: str) -> tuple[str, int]:
        """
        Apply knowledge distillation using Student Summarizer.

        Uses a lightweight model to create a concise, structured summary
        of verbose tool output.

        Args:
            content: Content to summarize

        Returns:
            Tuple of (summary, compression_cost)
        """
        try:
            # Prompt for structured summarization
            prompt = f"""Summarize the following tool output into a concise, structured format.
Focus on key information and actionable insights. Keep it under {self.config.student_summarizer_max_tokens} tokens.

Tool Output:
{content}

Provide a structured summary with:
- Main points (bullet list)
- Key data/metrics
- Action items (if applicable)
"""

            # Call student model
            response = self.student_model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=self.config.student_summarizer_max_tokens,
                    temperature=0.3,  # Lower temperature for factual summarization
                ),
            )

            summary = response.text

            # Calculate compression cost (prompt + response)
            compression_cost = (
                self.token_counter.count_text(prompt) +
                self.token_counter.count_text(summary)
            )

            return summary, compression_cost

        except Exception as e:
            self.logger.error(
                "knowledge_distillation_failed",
                error=str(e),
            )
            # Fallback to simple truncation
            max_chars = self.config.student_summarizer_max_tokens * 4
            return content[:max_chars] + "...", 0

    def _apply_history_compaction(self, content: str) -> tuple[str, int]:
        """
        Apply conversation history compaction.

        Summarizes old messages into a condensed state description.

        Args:
            content: History to compact

        Returns:
            Tuple of (compacted_content, compression_cost)
        """
        # Similar to knowledge distillation but specifically for conversation history
        try:
            prompt = f"""Create a concise summary of the following conversation history.
Preserve key context, decisions made, and current state. Maximum 500 tokens.

Conversation:
{content}

Summary:
"""

            response = self.student_model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=500,
                    temperature=0.3,
                ),
            )

            summary = response.text
            compression_cost = (
                self.token_counter.count_text(prompt) +
                self.token_counter.count_text(summary)
            )

            return summary, compression_cost

        except Exception as e:
            self.logger.error("history_compaction_failed", error=str(e))
            return content, 0

    def compact_conversation_history(
        self,
        transient_ctx: TransientContext,
    ) -> Optional[CompressionResult]:
        """
        Compact conversation history if it exceeds threshold.

        Args:
            transient_ctx: Transient context with conversation history

        Returns:
            CompressionResult if compaction was applied, None otherwise
        """
        if not transient_ctx.requires_compaction():
            return None

        if not self.config.enable_history_compaction:
            self.logger.warning("history_compaction_needed_but_disabled")
            return None

        # Extract old messages (keep recent N messages)
        keep_recent = 5
        if len(transient_ctx.conversation_history) <= keep_recent:
            return None

        old_messages = transient_ctx.conversation_history[:-keep_recent]
        recent_messages = transient_ctx.conversation_history[-keep_recent:]

        # Convert old messages to text
        old_history_text = "\n\n".join(
            f"{msg.role}: {msg.content}"
            for msg in old_messages
        )

        original_tokens = sum(msg.token_count or 0 for msg in old_messages)

        # Apply compaction
        compacted_text, compression_cost = self._apply_history_compaction(old_history_text)
        compacted_tokens = self.token_counter.count_text(compacted_text)

        # Replace old messages with summary
        summary_message = ConversationMessage(
            role="system",
            content=f"[Conversation Summary]: {compacted_text}",
            token_count=compacted_tokens,
            metadata={"type": "compacted_history"},
        )

        transient_ctx.conversation_history = [summary_message] + recent_messages

        # Create compression result
        compression_result = CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=compacted_tokens,
            compression_ratio=compacted_tokens / original_tokens if original_tokens > 0 else 1.0,
            compression_method="history_compaction",
            compression_cost_tokens=compression_cost,
            net_savings=original_tokens - compacted_tokens - compression_cost,
        )

        self.logger.info(
            "history_compacted",
            original_messages=len(old_messages),
            original_tokens=original_tokens,
            compacted_tokens=compacted_tokens,
            net_savings=compression_result.net_savings,
        )

        return compression_result
