"""Configuration enums for type-safe settings.

This module provides enum types for all configuration choices in NOCP,
enabling IDE autocomplete, preventing typos, and improving type safety.
"""

from enum import Enum


class OutputFormat(str, Enum):
    """Supported output serialization formats.

    Attributes:
        TOON: Token-Oriented Object Notation - optimal for tabular data (30-60% token reduction)
        COMPACT_JSON: Minified JSON without whitespace
        JSON: Standard formatted JSON with indentation
    """
    TOON = "toon"
    COMPACT_JSON = "compact_json"
    JSON = "json"

    def __str__(self) -> str:
        """Return the enum value as a string for serialization."""
        return self.value


class LogLevel(str, Enum):
    """Standard logging levels.

    Attributes:
        DEBUG: Detailed diagnostic information
        INFO: General informational messages
        WARNING: Warning messages for potentially problematic situations
        ERROR: Error messages for serious problems
        CRITICAL: Critical messages for severe errors
    """
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def __str__(self) -> str:
        """Return the enum value as a string for serialization."""
        return self.value


class CompressionStrategy(str, Enum):
    """Available context compression strategies.

    Attributes:
        SEMANTIC_PRUNING: Remove redundant/low-value content from RAG/document outputs
        KNOWLEDGE_DISTILLATION: Use student summarizer model for intelligent compression
        HISTORY_COMPACTION: Compress conversation history while preserving context
        NONE: Disable compression (for debugging or specific use cases)
    """
    SEMANTIC_PRUNING = "semantic_pruning"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    HISTORY_COMPACTION = "history_compaction"
    NONE = "none"

    def __str__(self) -> str:
        """Return the enum value as a string for serialization."""
        return self.value


class LLMProvider(str, Enum):
    """Supported LLM providers via LiteLLM.

    Attributes:
        GEMINI: Google Gemini models (gemini/gemini-2.0-flash-exp, etc.)
        OPENAI: OpenAI models (openai/gpt-4o, openai/gpt-4o-mini, etc.)
        ANTHROPIC: Anthropic Claude models (anthropic/claude-3-5-sonnet, etc.)
        COHERE: Cohere models (cohere/command-r-plus, etc.)
        AZURE: Azure OpenAI Service models (azure/gpt-4, etc.)
    """
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    AZURE = "azure"

    def __str__(self) -> str:
        """Return the enum value as a string for serialization."""
        return self.value


# Export all enums
__all__ = [
    "OutputFormat",
    "LogLevel",
    "CompressionStrategy",
    "LLMProvider",
]
