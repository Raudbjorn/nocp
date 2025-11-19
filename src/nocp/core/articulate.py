"""
Articulate Module: Output Serializer

Serializes Pydantic models to token-optimized formats (TOON or compact JSON).
Implements format negotiation to select the most efficient serialization strategy.
"""

import json
import time
from typing import Any

from pydantic import BaseModel

from ..models.contracts import (
    SerializationFormat,
    SerializationRequest,
    SerializedOutput,
)
from ..serializers.toon import TOONEncoder
from ..utils.logging import articulate_logger


class OutputSerializer:
    """
    Serializes Pydantic models to token-optimized formats.

    Example:
        serializer = OutputSerializer()

        data = UserListModel(users=[...])
        request = SerializationRequest(data=data)
        result = serializer.serialize(request)

        print(f"Format: {result.format_used}")
        print(f"Savings: {result.savings_ratio:.1%}")
    """

    def __init__(self):
        self.toon_encoder = TOONEncoder()

    def serialize(self, request: SerializationRequest) -> SerializedOutput:
        """
        Main serialization entry point with format negotiation.

        Args:
            request: SerializationRequest with data and options

        Returns:
            SerializedOutput with optimized serialization and metrics
        """
        articulate_logger.log_operation_start("output_serialization")

        # Step 1: Determine optimal format
        if request.force_format:
            format_used = SerializationFormat(request.force_format)
        else:
            format_used = self.negotiate_format(request.data)

        # Step 2: Serialize
        start_time = time.perf_counter()

        try:
            if format_used == SerializationFormat.TOON:
                serialized = self.toon_encoder.encode(
                    request.data, length_marker="#" if request.include_length_markers else ""
                )
            else:  # COMPACT_JSON
                serialized = request.data.model_dump_json(indent=None, separators=(",", ":"))
        except Exception as e:
            # Fallback to compact JSON on error
            articulate_logger.logger.warning(
                f"Serialization failed ({e}), falling back to compact JSON"
            )
            serialized = request.data.model_dump_json(indent=None, separators=(",", ":"))
            format_used = SerializationFormat.COMPACT_JSON

        serialization_time = (time.perf_counter() - start_time) * 1000

        # Step 3: Calculate savings
        baseline_json = request.data.model_dump_json(indent=2)
        original_tokens = len(baseline_json) // 4
        optimized_tokens = len(serialized) // 4
        savings_ratio = 1.0 - (optimized_tokens / original_tokens) if original_tokens > 0 else 0.0

        # Step 4: Validation
        is_valid = True
        if request.validate_output:
            try:
                # Attempt to deserialize
                if format_used == SerializationFormat.COMPACT_JSON:
                    json.loads(serialized)
                # TOON validation skipped in MVP (would need full decoder)
            except Exception:
                is_valid = False

        articulate_logger.log_operation_complete(
            "output_serialization",
            duration_ms=serialization_time,
            details={
                "format": format_used.value,
                "original_tokens": original_tokens,
                "optimized_tokens": optimized_tokens,
                "savings_ratio": round(savings_ratio, 3),
                "is_valid": is_valid,
            },
        )

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

    def negotiate_format(self, model: BaseModel) -> SerializationFormat:
        """
        Analyze Pydantic model to select optimal format.

        Decision logic:
        - If model contains list fields with >5 uniform items: TOON
        - If model is deeply nested (>3 levels): COMPACT_JSON
        - If model has mostly scalar fields: COMPACT_JSON
        - Default: COMPACT_JSON (safe fallback)

        Args:
            model: Pydantic model to analyze

        Returns:
            Selected SerializationFormat
        """
        model_dict = model.model_dump()

        # Check for tabular data
        for value in model_dict.values():
            if isinstance(value, list) and len(value) > 5:
                if self._is_uniform_list(value):
                    return SerializationFormat.TOON

        # Check nesting depth
        if self._get_nesting_depth(model_dict) > 3:
            return SerializationFormat.COMPACT_JSON

        return SerializationFormat.COMPACT_JSON  # Safe default

    def _is_uniform_list(self, arr: list[Any]) -> bool:
        """Check if list is uniform (same structure)."""
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
        """
        Categorize schema complexity.

        Returns:
            One of: "simple", "tabular", "nested", "complex"
        """
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
