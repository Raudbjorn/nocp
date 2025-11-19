"""
Output Serializer (Articulate Module) - TOON-based output optimization.

Converts final LLM responses from Pydantic objects to Token-Oriented Object
Notation (TOON) format for maximum token efficiency (30-60% reduction vs JSON).

Implements the Format Negotiation Layer to intelligently choose between
TOON and compact JSON based on data structure tabularity.
"""

import json
from typing import Any, Literal

from pydantic import BaseModel

from ..core.config import get_config
from ..utils.logging import get_logger
from ..utils.token_counter import TokenCounter


class OutputSerializer:
    """
    Output Serializer - The "Articulate" component of the architecture.

    Responsibilities:
    - Analyze output structure for tabularity
    - Apply Format Negotiation Layer
    - Serialize to TOON or compact JSON
    - Maximize output token efficiency (30-60% reduction target)
    """

    def __init__(self):
        """Initialize the output serializer."""
        self.config = get_config()
        self.logger = get_logger(__name__)
        self.token_counter = TokenCounter()

    def serialize(
        self,
        response: BaseModel,
        force_format: Literal["toon", "compact_json", "json"] | None = None,
    ) -> tuple[str, Literal["toon", "compact_json", "json"], int]:
        """
        Serialize Pydantic response to optimal format.

        Args:
            response: Pydantic model instance to serialize
            force_format: Optional format override

        Returns:
            Tuple of (serialized_string, format_used, token_savings)
        """
        # Convert to dict
        response_dict = response.model_dump(exclude_none=True)

        # Remove internal metrics if present
        if "metrics" in response_dict:
            del response_dict["metrics"]

        # Determine format
        if force_format:
            format_to_use = force_format
        elif self.config.enable_format_negotiation:
            format_to_use = self._negotiate_format(response_dict)
        else:
            format_to_use = self.config.default_output_format

        self.logger.info(
            "serializing_output",
            format=format_to_use,
            negotiated=force_format is None,
        )

        # Serialize using chosen format
        if format_to_use == "toon":
            serialized = self._serialize_to_toon(response_dict)
        elif format_to_use == "compact_json":
            serialized = self._serialize_to_compact_json(response_dict)
        else:
            serialized = self._serialize_to_json(response_dict)

        # Calculate token savings vs baseline compact JSON
        baseline_tokens = self.token_counter.count_text(
            self._serialize_to_compact_json(response_dict)
        )
        actual_tokens = self.token_counter.count_text(serialized)
        token_savings = baseline_tokens - actual_tokens

        self.logger.info(
            "output_serialized",
            format=format_to_use,
            baseline_tokens=baseline_tokens,
            actual_tokens=actual_tokens,
            token_savings=token_savings,
            savings_percent=(
                round(100 * token_savings / baseline_tokens, 1) if baseline_tokens > 0 else 0
            ),
        )

        return serialized, format_to_use, token_savings

    def _negotiate_format(
        self,
        data: dict[str, Any],
    ) -> Literal["toon", "compact_json", "json"]:
        """
        Implement Format Negotiation Layer.

        Analyzes data structure to determine optimal serialization format.

        Args:
            data: Data dictionary to analyze

        Returns:
            Recommended format
        """
        tabularity_score = self._calculate_tabularity(data)

        self.logger.debug(
            "format_negotiation",
            tabularity_score=tabularity_score,
            threshold=self.config.toon_fallback_threshold,
        )

        # If highly tabular, use TOON
        if tabularity_score >= self.config.toon_fallback_threshold:
            return "toon"

        # Otherwise, use compact JSON
        return "compact_json"

    def _calculate_tabularity(self, data: dict[str, Any]) -> float:
        """
        Calculate tabularity score (0.0 to 1.0).

        Higher scores indicate data is more suitable for TOON's tabular format.

        Args:
            data: Data dictionary to analyze

        Returns:
            Tabularity score (0.0 = not tabular, 1.0 = highly tabular)
        """
        if not data:
            return 0.0

        score = 0.0

        # Check 1: Presence of arrays (TOON excels at arrays)
        array_count = sum(isinstance(v, list) for v in data.values())
        if array_count > 0:
            score += 0.4

        # Check 2: Flat structure (not deeply nested)
        max_depth = self._get_max_depth(data)
        if max_depth <= 2:
            score += 0.3

        # Check 3: Uniform arrays (arrays with consistent structure)
        uniform_arrays = self._count_uniform_arrays(data)
        total_arrays = sum(isinstance(v, list) for v in data.values())
        if total_arrays > 0:
            uniformity_ratio = uniform_arrays / total_arrays
            score += 0.3 * uniformity_ratio

        return score

    def _get_max_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth."""
        if not isinstance(obj, (dict, list)):
            return current_depth

        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._get_max_depth(v, current_depth + 1) for v in obj.values())

        if isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._get_max_depth(item, current_depth + 1) for item in obj)

        return current_depth

    def _count_uniform_arrays(self, data: dict[str, Any]) -> int:
        """Count arrays with uniform structure."""
        uniform_count = 0

        for value in data.values():
            if isinstance(value, list) and len(value) > 0:
                # Check if all items have same type and structure
                first_type = type(value[0])
                if all(isinstance(item, first_type) for item in value):
                    # For dicts, check if all have same keys
                    if isinstance(value[0], dict):
                        first_keys = set(value[0].keys())
                        if all(set(item.keys()) == first_keys for item in value):
                            uniform_count += 1
                    else:
                        uniform_count += 1

        return uniform_count

    def _serialize_to_toon(self, data: dict[str, Any]) -> str:
        """
        Serialize to TOON format.

        Uses python-toon library for spec-compliant serialization.

        Args:
            data: Data to serialize

        Returns:
            TOON-formatted string
        """
        try:
            # Try to import python-toon
            import toon

            return toon.dumps(data)

        except ImportError:
            self.logger.warning(
                "toon_library_not_available",
                fallback="compact_json",
            )
            # Fallback to compact JSON if TOON library not available
            return self._serialize_to_compact_json(data)

        except Exception as e:
            self.logger.error(
                "toon_serialization_failed",
                error=str(e),
                fallback="compact_json",
            )
            return self._serialize_to_compact_json(data)

    def _serialize_to_compact_json(self, data: dict[str, Any]) -> str:
        """
        Serialize to compact JSON (no whitespace).

        Args:
            data: Data to serialize

        Returns:
            Compact JSON string
        """
        return json.dumps(data, separators=(",", ":"), ensure_ascii=False)

    def _serialize_to_json(self, data: dict[str, Any]) -> str:
        """
        Serialize to formatted JSON (baseline for comparison).

        Args:
            data: Data to serialize

        Returns:
            Formatted JSON string
        """
        return json.dumps(data, indent=2, ensure_ascii=False)

    def analyze_output_structure(self, response: BaseModel) -> dict[str, Any]:
        """
        Analyze output structure for monitoring/debugging.

        Args:
            response: Pydantic response to analyze

        Returns:
            Analysis dictionary
        """
        data = response.model_dump(exclude_none=True)

        return {
            "total_fields": len(data),
            "array_fields": sum(isinstance(v, list) for v in data.values()),
            "nested_objects": sum(isinstance(v, dict) for v in data.values()),
            "max_depth": self._get_max_depth(data),
            "tabularity_score": self._calculate_tabularity(data),
            "recommended_format": self._negotiate_format(data),
        }
