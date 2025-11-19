"""
Unit tests for configuration validation.

Tests the Pydantic field validators and model validators in ProxyConfig
to ensure configuration errors are caught at startup, and tests enum integrations.
"""

import pytest
import logging
from pydantic import ValidationError

from nocp.core.config import ProxyConfig
from nocp.models.enums import (
    OutputFormat,
    LogLevel,
    CompressionStrategy,
    LLMProvider,
)


@pytest.fixture
def base_config_kwargs():
    """Provides base required arguments for ProxyConfig instantiation."""
    return {"gemini_api_key": "test-key"}


class TestCompressionThresholdValidator:
    """Tests for default_compression_threshold field validator."""

    def test_valid_threshold(self, base_config_kwargs):
        """Test that valid compression thresholds are accepted."""
        config = ProxyConfig(
            **base_config_kwargs,
            default_compression_threshold=5000
        )
        assert config.default_compression_threshold == 5000

    def test_threshold_too_low_raises_error(self, base_config_kwargs):
        """Test that compression threshold below 1000 raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ProxyConfig(
                **base_config_kwargs,
                default_compression_threshold=500
            )
        error_msg = str(exc_info.value)
        assert "too low" in error_msg.lower()
        assert "1000" in error_msg

    def test_threshold_minimum_boundary(self, base_config_kwargs):
        """Test that threshold of exactly 1000 is accepted."""
        config = ProxyConfig(
            **base_config_kwargs,
            default_compression_threshold=1000
        )
        assert config.default_compression_threshold == 1000

    def test_threshold_very_high_logs_warning(self, base_config_kwargs, caplog):
        """Test that very high threshold logs a warning."""
        with caplog.at_level(logging.WARNING):
            config = ProxyConfig(
                **base_config_kwargs,
                default_compression_threshold=150_000
            )
        assert config.default_compression_threshold == 150_000
        assert "Very high default_compression_threshold" in caplog.text


class TestCompressionCostMultiplierValidator:
    """Tests for compression_cost_multiplier field validator."""

    def test_valid_multiplier(self, base_config_kwargs):
        """Test that valid multipliers are accepted."""
        config = ProxyConfig(
            **base_config_kwargs,
            compression_cost_multiplier=1.5
        )
        assert config.compression_cost_multiplier == 1.5

    def test_multiplier_below_one_raises_error(self, base_config_kwargs):
        """Test that multiplier < 1.0 raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ProxyConfig(
                **base_config_kwargs,
                compression_cost_multiplier=0.5
            )
        error_msg = str(exc_info.value)
        assert ">= 1.0" in error_msg

    def test_multiplier_exactly_one(self, base_config_kwargs):
        """Test that multiplier of exactly 1.0 is accepted."""
        config = ProxyConfig(
            **base_config_kwargs,
            compression_cost_multiplier=1.0
        )
        assert config.compression_cost_multiplier == 1.0

    def test_multiplier_very_high_logs_warning(self, base_config_kwargs, caplog):
        """Test that very high multiplier logs a warning."""
        with caplog.at_level(logging.WARNING):
            config = ProxyConfig(
                **base_config_kwargs,
                compression_cost_multiplier=15.0
            )
        assert config.compression_cost_multiplier == 15.0
        assert "Very high compression_cost_multiplier" in caplog.text


class TestToonFallbackThresholdValidator:
    """Tests for toon_fallback_threshold field validator."""

    def test_valid_threshold(self, base_config_kwargs):
        """Test that valid thresholds (0.0-1.0) are accepted."""
        config = ProxyConfig(
            **base_config_kwargs,
            toon_fallback_threshold=0.3
        )
        assert config.toon_fallback_threshold == 0.3

    def test_threshold_below_zero_raises_error(self, base_config_kwargs):
        """Test that threshold < 0.0 raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ProxyConfig(
                **base_config_kwargs,
                toon_fallback_threshold=-0.1
            )
        error_msg = str(exc_info.value)
        assert "0.0-1.0" in error_msg

    def test_threshold_above_one_raises_error(self, base_config_kwargs):
        """Test that threshold > 1.0 raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ProxyConfig(
                **base_config_kwargs,
                toon_fallback_threshold=1.5
            )
        error_msg = str(exc_info.value)
        assert "0.0-1.0" in error_msg

    def test_threshold_boundary_values(self, base_config_kwargs):
        """Test that boundary values 0.0 and 1.0 are accepted."""
        config_zero = ProxyConfig(
            **base_config_kwargs,
            toon_fallback_threshold=0.0
        )
        assert config_zero.toon_fallback_threshold == 0.0

        config_one = ProxyConfig(
            **base_config_kwargs,
            toon_fallback_threshold=1.0
        )
        assert config_one.toon_fallback_threshold == 1.0


class TestStudentMaxTokensValidator:
    """Tests for student_summarizer_max_tokens field validator."""

    def test_valid_max_tokens(self, base_config_kwargs):
        """Test that valid max tokens are accepted."""
        config = ProxyConfig(
            **base_config_kwargs,
            student_summarizer_max_tokens=2000
        )
        assert config.student_summarizer_max_tokens == 2000

    def test_max_tokens_too_low_raises_error(self, base_config_kwargs):
        """Test that max tokens < 100 raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ProxyConfig(
                **base_config_kwargs,
                student_summarizer_max_tokens=50
            )
        error_msg = str(exc_info.value)
        assert "too low" in error_msg.lower()
        assert "100" in error_msg

    def test_max_tokens_very_high_logs_warning(self, base_config_kwargs, caplog):
        """Test that very high max tokens logs a warning."""
        with caplog.at_level(logging.WARNING):
            config = ProxyConfig(
                **base_config_kwargs,
                student_summarizer_max_tokens=15_000
            )
        assert config.student_summarizer_max_tokens == 15_000
        assert "Very high student_summarizer_max_tokens" in caplog.text


class TestTokenLimitsValidator:
    """Tests for max_input_tokens and max_output_tokens field validators."""

    def test_valid_token_limits(self, base_config_kwargs):
        """Test that valid token limits are accepted."""
        config = ProxyConfig(
            **base_config_kwargs,
            max_input_tokens=100_000,
            max_output_tokens=10_000
        )
        assert config.max_input_tokens == 100_000
        assert config.max_output_tokens == 10_000

    def test_zero_token_limit_raises_error(self, base_config_kwargs):
        """Test that token limit of 0 raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ProxyConfig(
                **base_config_kwargs,
                max_input_tokens=0
            )
        error_msg = str(exc_info.value)
        assert "positive" in error_msg.lower()

    def test_negative_token_limit_raises_error(self, base_config_kwargs):
        """Test that negative token limit raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ProxyConfig(
                **base_config_kwargs,
                max_output_tokens=-1000
            )
        error_msg = str(exc_info.value)
        assert "positive" in error_msg.lower()

    def test_very_high_token_limit_logs_warning(self, base_config_kwargs, caplog):
        """Test that very high token limits log a warning."""
        with caplog.at_level(logging.WARNING):
            config = ProxyConfig(
                **base_config_kwargs,
                max_input_tokens=15_000_000
            )
        assert config.max_input_tokens == 15_000_000
        assert "Very high token limit" in caplog.text


class TestCrossFieldValidation:
    """Tests for model_validator cross-field constraints."""

    def test_output_exceeds_input_logs_warning(self, base_config_kwargs, caplog):
        """Test that max_output > max_input logs a warning."""
        with caplog.at_level(logging.WARNING):
            config = ProxyConfig(
                **base_config_kwargs,
                max_input_tokens=50_000,
                max_output_tokens=100_000
            )
        assert config.max_output_tokens > config.max_input_tokens
        assert "max_output_tokens" in caplog.text and "max_input_tokens" in caplog.text

    def test_compression_threshold_exceeds_max_input_raises_error(self, base_config_kwargs):
        """Test that compression threshold > max_input raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ProxyConfig(
                **base_config_kwargs,
                max_input_tokens=50_000,
                default_compression_threshold=100_000
            )
        error_msg = str(exc_info.value)
        assert "exceeds max_input_tokens" in error_msg

    def test_student_exceeds_compression_logs_warning(self, base_config_kwargs, caplog):
        """Test that student max tokens > compression threshold logs warning."""
        with caplog.at_level(logging.WARNING):
            config = ProxyConfig(
                **base_config_kwargs,
                default_compression_threshold=5_000,
                student_summarizer_max_tokens=6_000
            )
        assert config.student_summarizer_max_tokens > config.default_compression_threshold
        assert "student_summarizer_max_tokens" in caplog.text and "default_compression_threshold" in caplog.text

    def test_valid_cross_field_configuration(self, base_config_kwargs):
        """Test that a valid configuration passes all validations."""
        config = ProxyConfig(
            **base_config_kwargs,
            max_input_tokens=1_000_000,
            max_output_tokens=65_535,
            default_compression_threshold=5_000,
            student_summarizer_max_tokens=2_000,
            compression_cost_multiplier=1.5,
            toon_fallback_threshold=0.3,
            log_level=LogLevel.INFO
        )
        assert config.max_input_tokens == 1_000_000
        assert config.default_compression_threshold < config.max_input_tokens
        assert config.student_summarizer_max_tokens < config.default_compression_threshold


class TestConfigDefaults:
    """Tests for default configuration values."""

    def test_default_config_passes_validation(self, base_config_kwargs):
        """Test that the default configuration passes all validators."""
        config = ProxyConfig(**base_config_kwargs)
        assert config.default_compression_threshold >= 1000
        assert config.compression_cost_multiplier >= 1.0
        assert 0.0 <= config.toon_fallback_threshold <= 1.0
        assert config.log_level in {LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR, LogLevel.CRITICAL}
        assert config.student_summarizer_max_tokens >= 100
        assert config.max_input_tokens > 0
        assert config.max_output_tokens > 0


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
