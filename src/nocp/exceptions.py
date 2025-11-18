"""
Custom exceptions for the nocp proxy agent.
"""

from typing import Any, Dict, Optional


class ProxyAgentError(Exception):
    """Base exception for all proxy agent errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ToolExecutionError(ProxyAgentError):
    """Raised when tool execution fails."""
    pass


class CompressionError(ProxyAgentError):
    """Raised when context compression fails."""
    pass


class SerializationError(ProxyAgentError):
    """Raised when output serialization fails."""
    pass


class LLMError(ProxyAgentError):
    """Raised when LLM API call fails."""
    pass


class ConfigurationError(ProxyAgentError):
    """Raised when configuration is invalid."""
    pass


class ValidationError(ProxyAgentError):
    """Raised when Pydantic validation fails."""
    pass
