"""
TOON (Text-Oriented Object Notation) Encoder

A compact serialization format optimized for LLM token efficiency.
Combines indentation-based structure for nested objects with CSV-style
tabular layout for uniform arrays.
"""

from typing import Any, Dict, List
from pydantic import BaseModel


class TOONEncoder:
    """
    Token-Oriented Object Notation encoder.

    Example:
        Input (JSON):
        {
          "users": [
            {"id": "1", "name": "Alice", "age": 30},
            {"id": "2", "name": "Bob", "age": 25}
          ]
        }

        Output (TOON):
        users#2
          id,name,age
          1,Alice,30
          2,Bob,25
    """

    def encode(self, data: Any, length_marker: str = "#") -> str:
        """
        Encode data to TOON format.

        Args:
            data: Dictionary or list to encode (or Pydantic model)
            length_marker: Character to use for length annotations

        Returns:
            TOON-formatted string
        """
        if isinstance(data, BaseModel):
            data = data.model_dump()

        return self._encode_value(data, indent=0, length_marker=length_marker)

    def _encode_value(
        self,
        value: Any,
        indent: int,
        length_marker: str
    ) -> str:
        """Recursively encode a value."""
        if isinstance(value, dict):
            return self._encode_dict(value, indent, length_marker)
        elif isinstance(value, list):
            return self._encode_list(value, indent, length_marker)
        else:
            return str(value)

    def _encode_dict(
        self,
        obj: Dict[str, Any],
        indent: int,
        length_marker: str
    ) -> str:
        """Encode dictionary as indented key-value pairs."""
        lines = []
        indent_str = "  " * indent

        for key, value in obj.items():
            if isinstance(value, list) and self._is_uniform_list(value):
                # Use tabular format for uniform arrays
                lines.append(f"{indent_str}{key}{length_marker}{len(value)}")
                lines.append(self._encode_tabular(value, indent + 1))
            elif isinstance(value, (dict, list)):
                lines.append(f"{indent_str}{key}")
                lines.append(self._encode_value(value, indent + 1, length_marker))
            else:
                lines.append(f"{indent_str}{key}: {value}")

        return "\n".join(lines)

    def _encode_list(
        self,
        arr: List[Any],
        indent: int,
        length_marker: str
    ) -> str:
        """Encode list, using tabular format if uniform."""
        if self._is_uniform_list(arr):
            return self._encode_tabular(arr, indent)
        else:
            # Non-uniform list: encode each item
            lines = []
            indent_str = "  " * indent
            for item in arr:
                lines.append(f"{indent_str}- {self._encode_value(item, indent + 1, length_marker)}")
            return "\n".join(lines)

    def _is_uniform_list(self, arr: List[Any]) -> bool:
        """Check if list contains uniform dictionaries (same keys)."""
        if not arr or not isinstance(arr[0], dict):
            return False

        first_keys = set(arr[0].keys())
        return all(isinstance(item, dict) and set(item.keys()) == first_keys for item in arr)

    def _encode_tabular(self, arr: List[Dict[str, Any]], indent: int) -> str:
        """Encode uniform list as CSV-style table."""
        if not arr:
            return ""

        indent_str = "  " * indent
        keys = list(arr[0].keys())

        # Header row
        header = f"{indent_str}{','.join(keys)}"

        # Data rows
        rows = []
        for item in arr:
            row_values = [self._escape_csv_value(str(item[key])) for key in keys]
            rows.append(f"{indent_str}{','.join(row_values)}")

        return "\n".join([header] + rows)

    def _escape_csv_value(self, value: str) -> str:
        """Escape CSV values containing commas or quotes."""
        if "," in value or '"' in value or "\n" in value:
            # Escape quotes by doubling them
            escaped = value.replace('"', '""')
            return f'"{escaped}"'
        return value

    def decode(self, toon_str: str) -> Any:
        """
        Decode TOON string back to Python objects.

        Note: Full decoding implementation deferred to Phase 2.
        For MVP, validation uses JSON round-trip instead.

        Args:
            toon_str: TOON-formatted string

        Returns:
            Decoded Python object

        Raises:
            NotImplementedError: Full TOON decoding coming in Phase 2
        """
        raise NotImplementedError("TOON decoding coming in Phase 2")
