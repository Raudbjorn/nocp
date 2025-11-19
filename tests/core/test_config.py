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

# PyProject.toml Configuration Tests
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from nocp.core.config import load_pyproject_defaults, reset_config


class TestPyProjectLoading:
    """Tests for pyproject.toml configuration loading."""

    def test_load_pyproject_defaults_success(self, tmp_path):
        """Test successful loading of [tool.nocp] configuration."""
        # Create a temporary pyproject.toml
        pyproject_content = """
[tool.nocp]
default_compression_threshold = 8000
enable_semantic_pruning = false
default_output_format = "compact_json"
log_level = "DEBUG"
litellm_default_model = "openai/gpt-4"
"""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_file.write_text(pyproject_content)

        # Change to temp directory
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            defaults = load_pyproject_defaults()

            assert defaults["default_compression_threshold"] == 8000
            assert defaults["enable_semantic_pruning"] is False
            assert defaults["default_output_format"] == "compact_json"
            assert defaults["log_level"] == "DEBUG"
            assert defaults["litellm_default_model"] == "openai/gpt-4"
        finally:
            os.chdir(original_dir)

    def test_load_pyproject_defaults_missing_file(self, tmp_path):
        """Test handling of missing pyproject.toml."""
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            defaults = load_pyproject_defaults()
            assert defaults == {}
        finally:
            os.chdir(original_dir)

    def test_load_pyproject_defaults_missing_tool_section(self, tmp_path):
        """Test handling when [tool.nocp] section is missing."""
        pyproject_content = """
[project]
name = "test-project"
version = "0.1.0"

[tool.other]
setting = "value"
"""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_file.write_text(pyproject_content)

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            defaults = load_pyproject_defaults()
            assert defaults == {}
        finally:
            os.chdir(original_dir)

    def test_load_pyproject_defaults_invalid_toml(self, tmp_path):
        """Test handling of invalid TOML syntax."""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_file.write_text("invalid [[ toml syntax")

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            # Should return empty dict and log warning
            defaults = load_pyproject_defaults()
            assert defaults == {}
        finally:
            os.chdir(original_dir)

    @patch('nocp.core.config.tomllib', None)
    def test_load_pyproject_defaults_no_tomllib(self):
        """Test warning when tomllib/tomli is not available."""
        with pytest.warns(ImportWarning, match="tomli package not installed"):
            defaults = load_pyproject_defaults()
            assert defaults == {}


class TestConfigurationPrecedence:
    """Tests for configuration precedence order."""

    def test_explicit_kwargs_override_pyproject(self, tmp_path, monkeypatch):
        """Test that explicit kwargs override pyproject.toml settings."""
        # Create pyproject.toml with defaults
        pyproject_content = """
[tool.nocp]
default_compression_threshold = 5000
log_level = "INFO"
"""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_file.write_text(pyproject_content)

        # Set minimal required env var
        monkeypatch.setenv("NOCP_GEMINI_API_KEY", "test-key")

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            reset_config()

            # Explicit kwargs should override pyproject.toml
            config = ProxyConfig(
                default_compression_threshold=10000,
                log_level=LogLevel.ERROR
            )

            assert config.default_compression_threshold == 10000
            assert config.log_level == LogLevel.ERROR
        finally:
            os.chdir(original_dir)
            reset_config()

    def test_env_vars_override_pyproject(self, tmp_path, monkeypatch):
        """Test that environment variables override pyproject.toml settings."""
        # Create pyproject.toml with defaults
        pyproject_content = """
[tool.nocp]
default_compression_threshold = 5000
log_level = "INFO"
litellm_default_model = "gemini/gemini-2.0-flash-exp"
"""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_file.write_text(pyproject_content)

        # Set environment variables
        monkeypatch.setenv("NOCP_GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("NOCP_DEFAULT_COMPRESSION_THRESHOLD", "15000")
        monkeypatch.setenv("NOCP_LOG_LEVEL", "DEBUG")

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            reset_config()

            config = ProxyConfig()

            # Env vars should override pyproject.toml
            assert config.default_compression_threshold == 15000
            assert config.log_level == LogLevel.DEBUG
            # But pyproject.toml values are used when env var not set
            assert config.litellm_default_model == "gemini/gemini-2.0-flash-exp"
        finally:
            os.chdir(original_dir)
            reset_config()

    def test_pyproject_overrides_defaults(self, tmp_path, monkeypatch):
        """Test that pyproject.toml settings override hardcoded defaults."""
        # Create pyproject.toml with custom defaults
        pyproject_content = """
[tool.nocp]
default_compression_threshold = 8000
enable_semantic_pruning = false
default_output_format = "compact_json"
"""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_file.write_text(pyproject_content)

        monkeypatch.setenv("NOCP_GEMINI_API_KEY", "test-key")

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            reset_config()

            config = ProxyConfig()

            # pyproject.toml should override defaults
            assert config.default_compression_threshold == 8000
            assert config.enable_semantic_pruning is False
            assert config.default_output_format == OutputFormat.COMPACT_JSON
        finally:
            os.chdir(original_dir)
            reset_config()

    def test_hardcoded_defaults_used_when_no_overrides(self, monkeypatch):
        """Test that hardcoded defaults are used when no overrides present."""
        monkeypatch.setenv("NOCP_GEMINI_API_KEY", "test-key")
        reset_config()

        config = ProxyConfig()

        # Should use hardcoded defaults
        assert config.default_compression_threshold == 5000
        assert config.enable_semantic_pruning is True
        assert config.default_output_format == OutputFormat.TOON
        reset_config()


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing configurations."""

    def test_config_works_without_pyproject(self, monkeypatch):
        """Test that configuration still works without pyproject.toml."""
        monkeypatch.setenv("NOCP_GEMINI_API_KEY", "test-key")
        reset_config()

        # Should work fine without pyproject.toml
        config = ProxyConfig()
        assert config.gemini_api_key == "test-key"
        assert config.default_compression_threshold == 5000
        reset_config()

    def test_existing_methods_still_work(self, monkeypatch):
        """Test that existing config methods still function correctly."""
        monkeypatch.setenv("NOCP_GEMINI_API_KEY", "test-key")
        reset_config()

        config = ProxyConfig()

        # Test tool threshold registration
        config.register_tool_threshold("test_tool", 8000)
        assert config.get_compression_threshold("test_tool") == 8000
        assert config.get_compression_threshold("unknown_tool") == 5000

        # Test ensure_log_directory
        config.ensure_log_directory()
        assert config.metrics_log_file.parent.exists()

        reset_config()
