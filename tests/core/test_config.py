"""
Unit tests for configuration validation.

Tests the Pydantic field validators and model validators in ProxyConfig
to ensure configuration errors are caught at startup.
"""

import pytest
import logging
from pydantic import ValidationError

from nocp.core.config import ProxyConfig


class TestCompressionThresholdValidator:
    """Tests for default_compression_threshold field validator."""

    def test_valid_threshold(self):
        """Test that valid compression thresholds are accepted."""
        config = ProxyConfig(
            gemini_api_key="test-key",
            default_compression_threshold=5000
        )
        assert config.default_compression_threshold == 5000

    def test_threshold_too_low_raises_error(self):
        """Test that compression threshold below 1000 raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ProxyConfig(
                gemini_api_key="test-key",
                default_compression_threshold=500
            )
        error_msg = str(exc_info.value)
        assert "too low" in error_msg.lower()
        assert "1000" in error_msg

    def test_threshold_minimum_boundary(self):
        """Test that threshold of exactly 1000 is accepted."""
        config = ProxyConfig(
            gemini_api_key="test-key",
            default_compression_threshold=1000
        )
        assert config.default_compression_threshold == 1000

    def test_threshold_very_high_logs_warning(self, caplog):
        """Test that very high threshold logs a warning."""
        with caplog.at_level(logging.WARNING):
            config = ProxyConfig(
                gemini_api_key="test-key",
                default_compression_threshold=150_000
            )
        assert config.default_compression_threshold == 150_000
        assert any("high" in record.message.lower() for record in caplog.records)


class TestCompressionCostMultiplierValidator:
    """Tests for compression_cost_multiplier field validator."""

    def test_valid_multiplier(self):
        """Test that valid multipliers are accepted."""
        config = ProxyConfig(
            gemini_api_key="test-key",
            compression_cost_multiplier=1.5
        )
        assert config.compression_cost_multiplier == 1.5

    def test_multiplier_below_one_raises_error(self):
        """Test that multiplier < 1.0 raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ProxyConfig(
                gemini_api_key="test-key",
                compression_cost_multiplier=0.5
            )
        error_msg = str(exc_info.value)
        assert ">= 1.0" in error_msg

    def test_multiplier_exactly_one(self):
        """Test that multiplier of exactly 1.0 is accepted."""
        config = ProxyConfig(
            gemini_api_key="test-key",
            compression_cost_multiplier=1.0
        )
        assert config.compression_cost_multiplier == 1.0

    def test_multiplier_very_high_logs_warning(self, caplog):
        """Test that very high multiplier logs a warning."""
        with caplog.at_level(logging.WARNING):
            config = ProxyConfig(
                gemini_api_key="test-key",
                compression_cost_multiplier=15.0
            )
        assert config.compression_cost_multiplier == 15.0
        assert any("high" in record.message.lower() for record in caplog.records)


class TestToonFallbackThresholdValidator:
    """Tests for toon_fallback_threshold field validator."""

    def test_valid_threshold(self):
        """Test that valid thresholds (0.0-1.0) are accepted."""
        config = ProxyConfig(
            gemini_api_key="test-key",
            toon_fallback_threshold=0.3
        )
        assert config.toon_fallback_threshold == 0.3

    def test_threshold_below_zero_raises_error(self):
        """Test that threshold < 0.0 raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ProxyConfig(
                gemini_api_key="test-key",
                toon_fallback_threshold=-0.1
            )
        error_msg = str(exc_info.value)
        assert "0.0-1.0" in error_msg

    def test_threshold_above_one_raises_error(self):
        """Test that threshold > 1.0 raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ProxyConfig(
                gemini_api_key="test-key",
                toon_fallback_threshold=1.5
            )
        error_msg = str(exc_info.value)
        assert "0.0-1.0" in error_msg

    def test_threshold_boundary_values(self):
        """Test that boundary values 0.0 and 1.0 are accepted."""
        config_zero = ProxyConfig(
            gemini_api_key="test-key",
            toon_fallback_threshold=0.0
        )
        assert config_zero.toon_fallback_threshold == 0.0

        config_one = ProxyConfig(
            gemini_api_key="test-key",
            toon_fallback_threshold=1.0
        )
        assert config_one.toon_fallback_threshold == 1.0


class TestLogLevelValidator:
    """Tests for log_level field validator."""

    def test_valid_log_levels(self):
        """Test that valid log levels are accepted and normalized."""
        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            config = ProxyConfig(
                gemini_api_key="test-key",
                log_level=level
            )
            assert config.log_level == level

    def test_lowercase_log_level_normalized(self):
        """Test that lowercase log levels are normalized to uppercase."""
        config = ProxyConfig(
            gemini_api_key="test-key",
            log_level="info"
        )
        assert config.log_level == "INFO"

    def test_invalid_log_level_raises_error(self):
        """Test that invalid log level raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ProxyConfig(
                gemini_api_key="test-key",
                log_level="INVALID"
            )
        error_msg = str(exc_info.value)
        assert "DEBUG" in error_msg or "INFO" in error_msg


class TestStudentMaxTokensValidator:
    """Tests for student_summarizer_max_tokens field validator."""

    def test_valid_max_tokens(self):
        """Test that valid max tokens are accepted."""
        config = ProxyConfig(
            gemini_api_key="test-key",
            student_summarizer_max_tokens=2000
        )
        assert config.student_summarizer_max_tokens == 2000

    def test_max_tokens_too_low_raises_error(self):
        """Test that max tokens < 100 raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ProxyConfig(
                gemini_api_key="test-key",
                student_summarizer_max_tokens=50
            )
        error_msg = str(exc_info.value)
        assert "too low" in error_msg.lower()
        assert "100" in error_msg

    def test_max_tokens_very_high_logs_warning(self, caplog):
        """Test that very high max tokens logs a warning."""
        with caplog.at_level(logging.WARNING):
            config = ProxyConfig(
                gemini_api_key="test-key",
                student_summarizer_max_tokens=15_000
            )
        assert config.student_summarizer_max_tokens == 15_000
        assert any("high" in record.message.lower() for record in caplog.records)


class TestTokenLimitsValidator:
    """Tests for max_input_tokens and max_output_tokens field validators."""

    def test_valid_token_limits(self):
        """Test that valid token limits are accepted."""
        config = ProxyConfig(
            gemini_api_key="test-key",
            max_input_tokens=100_000,
            max_output_tokens=10_000
        )
        assert config.max_input_tokens == 100_000
        assert config.max_output_tokens == 10_000

    def test_zero_token_limit_raises_error(self):
        """Test that token limit of 0 raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ProxyConfig(
                gemini_api_key="test-key",
                max_input_tokens=0
            )
        error_msg = str(exc_info.value)
        assert "positive" in error_msg.lower()

    def test_negative_token_limit_raises_error(self):
        """Test that negative token limit raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ProxyConfig(
                gemini_api_key="test-key",
                max_output_tokens=-1000
            )
        error_msg = str(exc_info.value)
        assert "positive" in error_msg.lower()

    def test_very_high_token_limit_logs_warning(self, caplog):
        """Test that very high token limits log a warning."""
        with caplog.at_level(logging.WARNING):
            config = ProxyConfig(
                gemini_api_key="test-key",
                max_input_tokens=15_000_000
            )
        assert config.max_input_tokens == 15_000_000
        assert any("high" in record.message.lower() for record in caplog.records)


class TestCrossFieldValidation:
    """Tests for model_validator cross-field constraints."""

    def test_output_exceeds_input_logs_warning(self, caplog):
        """Test that max_output > max_input logs a warning."""
        with caplog.at_level(logging.WARNING):
            config = ProxyConfig(
                gemini_api_key="test-key",
                max_input_tokens=50_000,
                max_output_tokens=100_000
            )
        assert config.max_output_tokens > config.max_input_tokens
        assert any("max_output_tokens" in record.message for record in caplog.records)

    def test_compression_threshold_exceeds_max_input_raises_error(self):
        """Test that compression threshold > max_input raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ProxyConfig(
                gemini_api_key="test-key",
                max_input_tokens=50_000,
                default_compression_threshold=100_000
            )
        error_msg = str(exc_info.value)
        assert "exceeds max_input_tokens" in error_msg

    def test_student_exceeds_compression_logs_warning(self, caplog):
        """Test that student max tokens > compression threshold logs warning."""
        with caplog.at_level(logging.WARNING):
            config = ProxyConfig(
                gemini_api_key="test-key",
                default_compression_threshold=5_000,
                student_summarizer_max_tokens=6_000
            )
        assert config.student_summarizer_max_tokens > config.default_compression_threshold
        assert any("student_summarizer_max_tokens" in record.message for record in caplog.records)

    def test_valid_cross_field_configuration(self):
        """Test that a valid configuration passes all validations."""
        config = ProxyConfig(
            gemini_api_key="test-key",
            max_input_tokens=1_000_000,
            max_output_tokens=65_535,
            default_compression_threshold=5_000,
            student_summarizer_max_tokens=2_000,
            compression_cost_multiplier=1.5,
            toon_fallback_threshold=0.3,
            log_level="INFO"
        )
        assert config.max_input_tokens == 1_000_000
        assert config.default_compression_threshold < config.max_input_tokens
        assert config.student_summarizer_max_tokens < config.default_compression_threshold


class TestConfigDefaults:
    """Tests for default configuration values."""

    def test_default_config_passes_validation(self):
        """Test that the default configuration passes all validators."""
        config = ProxyConfig(gemini_api_key="test-key")
        assert config.default_compression_threshold >= 1000
        assert config.compression_cost_multiplier >= 1.0
        assert 0.0 <= config.toon_fallback_threshold <= 1.0
        assert config.log_level in {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        assert config.student_summarizer_max_tokens >= 100
        assert config.max_input_tokens > 0
        assert config.max_output_tokens > 0
