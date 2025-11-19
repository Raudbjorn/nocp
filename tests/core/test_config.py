"""Tests for configuration and enums."""

import pytest
from pydantic import ValidationError

from nocp.core.config import ProxyConfig
from nocp.models.enums import (
    OutputFormat,
    LogLevel,
    CompressionStrategy,
    LLMProvider,
)


class TestEnums:
    """Test enum definitions and behavior."""

    def test_output_format_values(self):
        """Test OutputFormat enum has correct values."""
        assert OutputFormat.TOON.value == "toon"
        assert OutputFormat.COMPACT_JSON.value == "compact_json"
        assert OutputFormat.JSON.value == "json"

    def test_output_format_string_conversion(self):
        """Test OutputFormat converts to string properly."""
        assert str(OutputFormat.TOON) == "toon"
        assert str(OutputFormat.COMPACT_JSON) == "compact_json"
        assert str(OutputFormat.JSON) == "json"

    def test_log_level_values(self):
        """Test LogLevel enum has correct values."""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"

    def test_compression_strategy_values(self):
        """Test CompressionStrategy enum has correct values."""
        assert CompressionStrategy.SEMANTIC_PRUNING.value == "semantic_pruning"
        assert CompressionStrategy.KNOWLEDGE_DISTILLATION.value == "knowledge_distillation"
        assert CompressionStrategy.HISTORY_COMPACTION.value == "history_compaction"
        assert CompressionStrategy.NONE.value == "none"

    def test_llm_provider_values(self):
        """Test LLMProvider enum has correct values."""
        assert LLMProvider.GEMINI.value == "gemini"
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.COHERE.value == "cohere"
        assert LLMProvider.AZURE.value == "azure"


class TestProxyConfigEnums:
    """Test ProxyConfig integration with enums."""

    def test_default_output_format(self, monkeypatch):
        """Test default_output_format accepts OutputFormat enum."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        config = ProxyConfig()
        assert config.default_output_format == OutputFormat.TOON
        assert isinstance(config.default_output_format, OutputFormat)

    def test_custom_output_format(self, monkeypatch):
        """Test setting custom output format via env var."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("DEFAULT_OUTPUT_FORMAT", "compact_json")
        config = ProxyConfig()
        assert config.default_output_format == OutputFormat.COMPACT_JSON

    def test_invalid_output_format(self, monkeypatch):
        """Test invalid output format raises validation error."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("DEFAULT_OUTPUT_FORMAT", "invalid_format")
        with pytest.raises(ValidationError) as exc_info:
            ProxyConfig()
        assert "default_output_format" in str(exc_info.value)

    def test_default_log_level(self, monkeypatch):
        """Test log_level accepts LogLevel enum."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        config = ProxyConfig()
        assert config.log_level == LogLevel.INFO
        assert isinstance(config.log_level, LogLevel)

    def test_custom_log_level(self, monkeypatch):
        """Test setting custom log level via env var."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        config = ProxyConfig()
        assert config.log_level == LogLevel.DEBUG

    def test_invalid_log_level(self, monkeypatch):
        """Test invalid log level raises validation error."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("LOG_LEVEL", "INVALID")
        with pytest.raises(ValidationError) as exc_info:
            ProxyConfig()
        assert "log_level" in str(exc_info.value)

    def test_default_compression_strategies(self, monkeypatch):
        """Test compression_strategies field has correct defaults."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        config = ProxyConfig()
        assert len(config.compression_strategies) == 3
        assert CompressionStrategy.SEMANTIC_PRUNING in config.compression_strategies
        assert CompressionStrategy.KNOWLEDGE_DISTILLATION in config.compression_strategies
        assert CompressionStrategy.HISTORY_COMPACTION in config.compression_strategies

    def test_compression_strategies_validation(self, monkeypatch):
        """Test compression_strategies validates enum values."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        # This should work with valid strategy names
        monkeypatch.setenv("COMPRESSION_STRATEGIES", '["semantic_pruning", "none"]')
        config = ProxyConfig()
        assert CompressionStrategy.SEMANTIC_PRUNING in config.compression_strategies
        assert CompressionStrategy.NONE in config.compression_strategies
        assert len(config.compression_strategies) == 2

    def test_invalid_compression_strategies(self, monkeypatch):
        """Test that invalid compression strategy names raise validation errors."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("COMPRESSION_STRATEGIES", '["invalid_strategy", "semantic_pruning"]')
        with pytest.raises(ValidationError) as exc_info:
            ProxyConfig()
        assert "compression_strategies" in str(exc_info.value)

    def test_is_strategy_enabled_method(self, monkeypatch):
        """Test is_strategy_enabled helper method."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        config = ProxyConfig()

        # Test enabled strategies
        assert config.is_strategy_enabled(CompressionStrategy.SEMANTIC_PRUNING) is True
        assert config.is_strategy_enabled(CompressionStrategy.KNOWLEDGE_DISTILLATION) is True
        assert config.is_strategy_enabled(CompressionStrategy.HISTORY_COMPACTION) is True

        # Test disabled strategy
        assert config.is_strategy_enabled(CompressionStrategy.NONE) is False

    def test_backward_compatibility_with_boolean_flags(self, monkeypatch):
        """Test that legacy boolean flags correctly sync with compression_strategies."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("ENABLE_SEMANTIC_PRUNING", "false")
        config = ProxyConfig()

        # The boolean flag should still be set correctly
        assert config.enable_semantic_pruning is False

        # The new compression_strategies list should now be synchronized with the flag
        # Thanks to the model_validator, the flag is now the single source of truth
        assert CompressionStrategy.SEMANTIC_PRUNING not in config.compression_strategies

    def test_legacy_flags_sync_multiple_flags(self, monkeypatch):
        """Test that multiple legacy flags are correctly synchronized."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("ENABLE_SEMANTIC_PRUNING", "false")
        monkeypatch.setenv("ENABLE_KNOWLEDGE_DISTILLATION", "true")
        monkeypatch.setenv("ENABLE_HISTORY_COMPACTION", "false")
        config = ProxyConfig()

        # Only KNOWLEDGE_DISTILLATION should remain enabled
        assert CompressionStrategy.SEMANTIC_PRUNING not in config.compression_strategies
        assert CompressionStrategy.KNOWLEDGE_DISTILLATION in config.compression_strategies
        assert CompressionStrategy.HISTORY_COMPACTION not in config.compression_strategies
        assert len(config.compression_strategies) == 1


class TestEnumStringification:
    """Test that enums work correctly with logging and serialization."""

    def test_output_format_in_f_string(self):
        """Test OutputFormat works in f-strings."""
        format_type = OutputFormat.TOON
        message = f"Using format: {format_type}"
        assert message == "Using format: toon"

    def test_log_level_in_f_string(self):
        """Test LogLevel works in f-strings."""
        level = LogLevel.DEBUG
        message = f"Log level set to: {level}"
        assert message == "Log level set to: DEBUG"

    def test_enum_equality_with_strings(self):
        """Test enum can be compared with strings."""
        # Direct comparison with enum
        assert OutputFormat.TOON == OutputFormat.TOON

        # String value comparison
        assert OutputFormat.TOON.value == "toon"

        # Note: enum != string directly, but .value does
        assert OutputFormat.TOON != "toon"  # This is expected behavior
        assert OutputFormat.TOON.value == "toon"  # Use .value for string comparison
