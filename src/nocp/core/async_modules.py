"""
Async versions of core modules for concurrent execution.

Provides async equivalents of ContextManager, OutputSerializer, and other core modules
to enable concurrent tool execution and better performance.
"""

import asyncio
import json
import logging
import time
from typing import Any

from pydantic import BaseModel

from ..models.contracts import (
    CompressionMethod,
    ContextData,
    OptimizedContext,
    SerializationRequest,
    SerializedOutput,
    ToolRequest,
)
from ..models.enums import OutputFormat
from ..serializers.toon import TOONEncoder

logger = logging.getLogger(__name__)


class AsyncContextManager:
    """
    Async version of ContextManager for concurrent context optimization.

    Example:
        manager = AsyncContextManager(
            student_model="openai/gpt-4o-mini",
            compression_threshold=10_000
        )

        context = ContextData(tool_results=[large_result], ...)
        optimized = await manager.optimize_async(context)
    """

    def __init__(
        self,
        student_model: str = "openai/gpt-4o-mini",
        compression_threshold: int = 10_000,
        target_compression_ratio: float = 0.40,
        enable_litellm: bool = True,
    ):
        """Initialize Async Context Manager."""
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
                pass

    async def optimize_async(self, context: ContextData) -> OptimizedContext:
        """
        Async version of optimize() for concurrent execution.

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
            )

        # Step 3: Select strategy
        if context.compression_strategy:
            try:
                strategy = CompressionMethod(context.compression_strategy)
            except ValueError:
                strategy = self.select_strategy(context)
        else:
            strategy = self.select_strategy(context)

        # Step 4: Apply compression asynchronously
        start_time = time.perf_counter()

        try:
            if strategy == CompressionMethod.SEMANTIC_PRUNING:
                compressed_text = await self._semantic_pruning_async(context)
            elif strategy == CompressionMethod.KNOWLEDGE_DISTILLATION:
                compressed_text = await self._knowledge_distillation_async(context)
            elif strategy == CompressionMethod.HISTORY_COMPACTION:
                compressed_text = await self._history_compaction_async(context)
            else:
                compressed_text = raw_text
        except Exception as e:
            # Fallback to raw on compression failure
            logger.warning(
                f"Compression strategy '{strategy}' failed: {e}. Falling back to raw context."
            )
            compressed_text = raw_text
            strategy = CompressionMethod.NONE

        compression_time = (time.perf_counter() - start_time) * 1000

        # Step 5: Count compressed tokens
        compressed_tokens = self.estimate_tokens(compressed_text)
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

        return OptimizedContext(
            optimized_text=compressed_text,
            original_tokens=original_tokens,
            optimized_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            method_used=strategy,
            compression_time_ms=compression_time,
        )

    def estimate_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Estimate token count (synchronous operation)."""
        if self.litellm:
            try:
                tokens = self.litellm.token_counter(model=model, text=text)
                return tokens
            except Exception:
                pass

        # Fallback: rough estimate (1 token ≈ 4 chars)
        return len(text) // 4

    def select_strategy(self, context: ContextData) -> CompressionMethod:
        """Auto-detect optimal compression strategy (synchronous)."""
        # Check for large structured data
        for result in context.tool_results:
            if isinstance(result.data, (list, dict)):
                if result.token_estimate > 1000:
                    return CompressionMethod.SEMANTIC_PRUNING

        # Check for verbose text
        total_text_tokens = sum(
            r.token_estimate for r in context.tool_results if isinstance(r.data, str)
        )
        if total_text_tokens > 5000:
            return CompressionMethod.KNOWLEDGE_DISTILLATION

        # Check conversation history
        if len(context.message_history) > 10:
            return CompressionMethod.HISTORY_COMPACTION

        return CompressionMethod.NONE

    async def _semantic_pruning_async(self, context: ContextData) -> str:
        """Async version of semantic pruning."""
        # This operation is mostly synchronous (no I/O), but we wrap it for consistency
        return await asyncio.to_thread(self._semantic_pruning_sync, context)

    def _semantic_pruning_sync(self, context: ContextData) -> str:
        """Synchronous semantic pruning logic."""
        pruned_results = []
        target_tokens = int(
            sum(r.token_estimate for r in context.tool_results) * self.target_compression_ratio
        )

        current_tokens = 0
        for result in context.tool_results:
            if isinstance(result.data, list):
                items_to_keep = []
                for item in result.data:
                    item_tokens = len(str(item)) // 4
                    if current_tokens + item_tokens > target_tokens:
                        break
                    items_to_keep.append(item)
                    current_tokens += item_tokens

                pruned_results.append(
                    {
                        "tool_id": result.tool_id,
                        "data": items_to_keep,
                        "note": f"Showing {len(items_to_keep)} of {len(result.data)} items",
                    }
                )
            else:
                pruned_results.append({"tool_id": result.tool_id, "data": result.data})

        if context.transient_context:
            pruned_results.append({"context": context.transient_context})

        return json.dumps(pruned_results, indent=2)

    async def _knowledge_distillation_async(self, context: ContextData) -> str:
        """Async version of knowledge distillation using student model."""
        raw_text = self._context_to_text(context)
        raw_tokens = self.estimate_tokens(raw_text)

        # If litellm not available, use simple truncation
        if not self.litellm:
            target_length = int(len(raw_text) * self.target_compression_ratio)
            return raw_text[:target_length] + "\n[... truncated for length ...]"

        expected_compressed_tokens = int(raw_tokens * self.target_compression_ratio)

        # Call student model asynchronously
        try:
            # Use native async method for better performance
            response = await self.litellm.acompletion(
                model=self.student_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize the following text concisely while preserving all key information.",
                    },
                    {"role": "user", "content": raw_text},
                ],
                max_tokens=expected_compressed_tokens,
            )
            return response.choices[0].message.content
        except Exception:
            return raw_text

    async def _history_compaction_async(self, context: ContextData) -> str:
        """Async version of history compaction."""
        if len(context.message_history) <= 5:
            return self._context_to_text(context)

        # Keep recent messages
        recent_messages = context.message_history[-5:]

        # Summarize older messages
        old_messages = context.message_history[:-5]
        old_text = "\n".join(f"{msg.role}: {msg.content}" for msg in old_messages)

        # If litellm not available, use simple summary
        if not self.litellm:
            summary = f"[Earlier conversation: {len(old_messages)} messages exchanged]"
        else:
            # Summarize using student model asynchronously
            try:
                # Use native async method for better performance
                summary_response = await self.litellm.acompletion(
                    model=self.student_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Provide a concise summary of the conversation history.",
                        },
                        {"role": "user", "content": old_text},
                    ],
                    max_tokens=500,
                )
                summary = summary_response.choices[0].message.content
            except Exception:
                summary = f"[Earlier conversation: {len(old_messages)} messages exchanged]"

        # Combine summary with recent messages
        parts = [f"[Summary of earlier conversation: {summary}]"]

        for msg in recent_messages:
            parts.append(f"{msg.role}: {msg.content}")

        for result in context.tool_results:
            parts.append(f"\nTool: {result.tool_id}")
            parts.append(result.to_text())

        return "\n\n".join(parts)

    def _context_to_text(self, context: ContextData) -> str:
        """Convert ContextData to text representation."""
        parts = []

        if context.persistent_context:
            parts.append(f"System: {context.persistent_context}")

        for result in context.tool_results:
            parts.append(f"Tool: {result.tool_id}")
            parts.append(result.to_text())

        if context.transient_context:
            parts.append(f"Context: {json.dumps(context.transient_context, indent=2)}")

        for msg in context.message_history:
            parts.append(f"{msg.role}: {msg.content}")

        return "\n\n".join(parts)


class AsyncOutputSerializer:
    """
    Async version of OutputSerializer for concurrent serialization.

    Example:
        serializer = AsyncOutputSerializer()
        result = await serializer.serialize_async(request)
    """

    def __init__(self):
        self.toon_encoder = TOONEncoder()

    async def serialize_async(self, request: SerializationRequest) -> SerializedOutput:
        """
        Async version of serialize() for concurrent execution.

        Args:
            request: SerializationRequest with data and options

        Returns:
            SerializedOutput with optimized serialization and metrics
        """
        # Step 1: Determine optimal format
        if request.force_format:
            format_used = OutputFormat(request.force_format)
        else:
            format_used = self.negotiate_format(request.data)

        # Step 2: Serialize (run in thread pool for CPU-bound work)
        start_time = time.perf_counter()

        try:
            if format_used == OutputFormat.TOON:
                serialized = await asyncio.to_thread(
                    self.toon_encoder.encode,
                    request.data,
                    "#" if request.include_length_markers else "",
                )
            else:  # COMPACT_JSON
                serialized = await asyncio.to_thread(
                    request.data.model_dump_json, indent=None, separators=(",", ":")
                )
        except Exception:
            # Fallback to compact JSON on error
            serialized = await asyncio.to_thread(
                request.data.model_dump_json, indent=None, separators=(",", ":")
            )
            format_used = OutputFormat.COMPACT_JSON

        serialization_time = (time.perf_counter() - start_time) * 1000

        # Step 3: Calculate savings
        baseline_json = await asyncio.to_thread(request.data.model_dump_json, indent=2)
        original_tokens = len(baseline_json) // 4
        optimized_tokens = len(serialized) // 4
        savings_ratio = 1.0 - (optimized_tokens / original_tokens) if original_tokens > 0 else 0.0

        # Step 4: Validation
        is_valid = True
        if request.validate_output:
            try:
                if format_used == OutputFormat.COMPACT_JSON:
                    await asyncio.to_thread(json.loads, serialized)
            except Exception:
                is_valid = False

        return SerializedOutput(
            serialized_text=serialized,
            format_used=format_used,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            savings_ratio=savings_ratio,
            is_valid=is_valid,
            serialization_time_ms=serialization_time,
            schema_complexity=self._assess_complexity(request.data),
        )

    def negotiate_format(self, model: BaseModel) -> OutputFormat:
        """Analyze model to select optimal format (synchronous)."""
        model_dict = model.model_dump()

        # Check for tabular data
        for value in model_dict.values():
            if isinstance(value, list) and len(value) > 5:
                if self._is_uniform_list(value):
                    return OutputFormat.TOON

        # Check nesting depth
        if self._get_nesting_depth(model_dict) > 3:
            return OutputFormat.COMPACT_JSON

        return OutputFormat.COMPACT_JSON

    def _is_uniform_list(self, arr: list[Any]) -> bool:
        """Check if list is uniform."""
        if not arr or not isinstance(arr[0], dict):
            return False
        first_keys = set(arr[0].keys())
        return all(isinstance(item, dict) and set(item.keys()) == first_keys for item in arr)

    def _get_nesting_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth."""
        if not isinstance(obj, (dict, list)):
            return current_depth

        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._get_nesting_depth(v, current_depth + 1) for v in obj.values())
        else:  # list
            if not obj:
                return current_depth
            return max(self._get_nesting_depth(item, current_depth + 1) for item in obj)

    def _assess_complexity(self, model: BaseModel) -> str:
        """Categorize schema complexity."""
        model_dict = model.model_dump()
        depth = self._get_nesting_depth(model_dict)

        has_arrays = any(isinstance(v, list) for v in model_dict.values())

        if depth <= 1 and not has_arrays:
            return "simple"
        elif has_arrays and self._has_uniform_arrays(model_dict):
            return "tabular"
        elif depth > 3:
            return "complex"
        else:
            return "nested"

    def _has_uniform_arrays(self, obj: dict[str, Any]) -> bool:
        """Check if object contains uniform arrays."""
        for value in obj.values():
            if isinstance(value, list) and self._is_uniform_list(value):
                return True
        return False


class ConcurrentToolExecutor:
    """
    Wrapper for executing multiple tools concurrently.

    Example:
        executor = ConcurrentToolExecutor(tool_executor, max_concurrent=5)
        results = await executor.execute_many(requests)
    """

    def __init__(self, tool_executor, max_concurrent: int = 5):
        """
        Initialize concurrent executor.

        Args:
            tool_executor: ToolExecutor instance
            max_concurrent: Maximum number of concurrent tool executions
        """
        self.tool_executor = tool_executor
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_one(self, request: ToolRequest):
        """Execute a single tool request with semaphore control."""
        async with self.semaphore:
            return await self.tool_executor.execute_async(request)

    async def execute_many(self, requests: list[ToolRequest]) -> list:
        """
        Execute multiple tool requests concurrently.

        Args:
            requests: List of ToolRequest objects

        Returns:
            List of ToolResult objects
        """
        tasks = [self.execute_one(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def execute_many_ordered(self, requests: list[ToolRequest]) -> list:
        """
        Execute tools concurrently but return results in order.

        Args:
            requests: List of ToolRequest objects

        Returns:
            List of ToolResult objects in the same order as requests
        """
        tasks = [self.execute_one(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
