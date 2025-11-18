"""
nocp - High-Efficiency LLM Proxy Agent

A Python-based middleware orchestration layer for optimizing token costs
and performance when integrating large-context LLM models.
"""

__version__ = "0.1.0"
__author__ = "nocp developers"

from .exceptions import (
    ProxyAgentError,
    ToolExecutionError,
    CompressionError,
    SerializationError,
    LLMError,
    ConfigurationError,
)

__all__ = [
    "__version__",
    "ProxyAgentError",
    "ToolExecutionError",
    "CompressionError",
    "SerializationError",
    "LLMError",
    "ConfigurationError",
]
