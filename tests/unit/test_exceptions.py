"""Unit tests for enhanced exception classes"""

from datetime import datetime
from nocp.exceptions import (
    ProxyAgentError,
    LLMError,
    ToolExecutionError,
    CompressionError,
    SerializationError,
    ConfigurationError,
    ValidationError,
)


class TestProxyAgentError:
    """Test base ProxyAgentError class"""

    def test_basic_initialization(self):
        """Test basic error initialization"""
        error = ProxyAgentError("Test error")

        assert error.message == "Test error"
        assert error.details == {}
        assert error.retry_after is None
        assert error.recoverable is False
        assert error.user_message == "Test error"
        assert isinstance(error.timestamp, datetime)

    def test_with_details(self):
        """Test error with details dictionary"""
        details = {"key": "value", "count": 42}
        error = ProxyAgentError("Test error", details=details)

        assert error.details == details

    def test_with_retry_after(self):
        """Test error with retry_after"""
        error = ProxyAgentError("Test error", retry_after=60.0)

        assert error.retry_after == 60.0

    def test_recoverable_flag(self):
        """Test recoverable flag"""
        error = ProxyAgentError("Test error", recoverable=True)

        assert error.recoverable is True

    def test_custom_user_message(self):
        """Test custom user-friendly message"""
        error = ProxyAgentError("Technical error details", user_message="User-friendly message")

        assert error.message == "Technical error details"
        assert error.user_message == "User-friendly message"

    def test_str_representation_basic(self):
        """Test string representation without details"""
        error = ProxyAgentError("Test error")

        assert str(error) == "Test error"

    def test_str_representation_with_details(self):
        """Test string representation with details"""
        error = ProxyAgentError("Test error", details={"key": "value"})

        assert "Test error" in str(error)
        assert "key=value" in str(error)

    def test_str_representation_with_retry(self):
        """Test string representation with retry_after"""
        error = ProxyAgentError("Test error", retry_after=60.0)

        assert "Test error" in str(error)
        assert "[retry after 60.0s]" in str(error)

    def test_str_representation_recoverable(self):
        """Test string representation for recoverable error"""
        error = ProxyAgentError("Test error", recoverable=True)

        assert "Test error" in str(error)
        assert "[recoverable]" in str(error)

    def test_str_representation_full(self):
        """Test string representation with all attributes"""
        error = ProxyAgentError(
            "Test error", details={"key": "value"}, retry_after=30.0, recoverable=True
        )

        error_str = str(error)
        assert "Test error" in error_str
        assert "key=value" in error_str
        assert "[retry after 30.0s]" in error_str
        assert "[recoverable]" in error_str

    def test_to_dict(self):
        """Test conversion to dictionary"""
        error = ProxyAgentError(
            "Test error",
            details={"key": "value"},
            retry_after=60.0,
            recoverable=True,
            user_message="User message",
        )

        error_dict = error.to_dict()

        assert error_dict["error_type"] == "ProxyAgentError"
        assert error_dict["message"] == "Test error"
        assert error_dict["details"] == {"key": "value"}
        assert error_dict["retry_after"] == 60.0
        assert error_dict["recoverable"] is True
        assert error_dict["user_message"] == "User message"
        assert "timestamp" in error_dict
        # Verify timestamp is ISO format string
        datetime.fromisoformat(error_dict["timestamp"])


class TestLLMError:
    """Test LLMError class"""

    def test_rate_limit_error(self):
        """Test 429 rate limit error"""
        error = LLMError("Rate limit exceeded", status_code=429, retry_after=60.0)

        assert error.status_code == 429
        assert error.recoverable is True
        assert error.retry_after == 60.0
        assert "API rate limit exceeded" in error.user_message

    def test_auth_error(self):
        """Test 401 authentication error"""
        error = LLMError("Invalid API key", status_code=401)

        assert error.status_code == 401
        assert error.recoverable is False
        assert "API authentication failed" in error.user_message
        assert "API key" in error.user_message

    def test_service_unavailable(self):
        """Test 503 service unavailable error"""
        error = LLMError("Service down", status_code=503)

        assert error.status_code == 503
        assert error.recoverable is False
        assert "Service temporarily unavailable" in error.user_message

    def test_generic_llm_error(self):
        """Test generic LLM error"""
        error = LLMError("Unknown error", status_code=500)

        assert error.status_code == 500
        assert error.recoverable is False
        assert "error occurred while calling the LLM API" in error.user_message

    def test_no_status_code(self):
        """Test LLM error without status code"""
        error = LLMError("Connection failed")

        assert error.status_code is None
        assert error.recoverable is False

    def test_with_details(self):
        """Test LLM error with additional details"""
        details = {"model": "gemini-2.0-flash", "tokens": 10000}
        error = LLMError("API request failed", details=details, status_code=429)

        assert error.details == details
        assert error.status_code == 429


class TestToolExecutionError:
    """Test ToolExecutionError class"""

    def test_basic_tool_error(self):
        """Test basic tool execution error"""
        error = ToolExecutionError("Tool failed", tool_id="search_tool")

        assert error.tool_id == "search_tool"
        assert error.retry_count == 0
        assert error.recoverable is True  # retry_count < 3
        assert "search_tool" in error.user_message

    def test_with_retry_count(self):
        """Test tool error with retry count"""
        error = ToolExecutionError("Tool failed", tool_id="search_tool", retry_count=2)

        assert error.retry_count == 2
        assert error.recoverable is True  # retry_count < 3

    def test_max_retries_exceeded(self):
        """Test tool error when max retries exceeded"""
        error = ToolExecutionError("Tool failed", tool_id="search_tool", retry_count=3)

        assert error.retry_count == 3
        assert error.recoverable is False  # retry_count >= 3

    def test_details_updated_with_tool_info(self):
        """Test that details are updated with tool_id and retry_count"""
        custom_details = {"extra": "info"}
        error = ToolExecutionError(
            "Tool failed", tool_id="search_tool", details=custom_details, retry_count=1
        )

        assert error.details["tool_id"] == "search_tool"
        assert error.details["retry_count"] == 1
        assert error.details["extra"] == "info"


class TestCompressionError:
    """Test CompressionError class"""

    def test_basic_compression_error(self):
        """Test basic compression error"""
        error = CompressionError("Compression failed", strategy="semantic_pruning")

        assert error.strategy == "semantic_pruning"
        assert error.recoverable is True
        assert "Context compression failed" in error.user_message
        assert "uncompressed context" in error.user_message

    def test_details_include_strategy(self):
        """Test that strategy is added to details"""
        error = CompressionError(
            "Compression failed", strategy="semantic_pruning", details={"tokens": 10000}
        )

        assert error.details["strategy"] == "semantic_pruning"
        assert error.details["tokens"] == 10000

    def test_always_recoverable(self):
        """Test that compression errors are always recoverable"""
        error = CompressionError("Failed", strategy="test")

        assert error.recoverable is True


class TestSerializationError:
    """Test SerializationError class"""

    def test_basic_serialization_error(self):
        """Test basic serialization error"""
        error = SerializationError("Serialization failed", format="TOON")

        assert error.format == "TOON"
        assert error.recoverable is True
        assert "Failed to serialize to TOON" in error.user_message
        assert "fallback format" in error.user_message

    def test_details_include_format(self):
        """Test that format is added to details"""
        error = SerializationError("Serialization failed", format="TOON", details={"size": 5000})

        assert error.details["format"] == "TOON"
        assert error.details["size"] == 5000

    def test_always_recoverable(self):
        """Test that serialization errors are always recoverable"""
        error = SerializationError("Failed", format="test")

        assert error.recoverable is True


class TestConfigurationError:
    """Test ConfigurationError class"""

    def test_basic_config_error(self):
        """Test basic configuration error"""
        error = ConfigurationError("Invalid configuration")

        assert error.message == "Invalid configuration"
        assert error.field is None
        assert error.value is None
        assert error.recoverable is False
        assert "Configuration error" in error.user_message

    def test_with_field(self):
        """Test configuration error with field"""
        error = ConfigurationError("Invalid value", field="compression_threshold")

        assert error.field == "compression_threshold"
        assert error.details["field"] == "compression_threshold"

    def test_with_field_and_value(self):
        """Test configuration error with field and value"""
        error = ConfigurationError("Invalid value", field="compression_threshold", value=-100)

        assert error.field == "compression_threshold"
        assert error.value == -100
        assert error.details["field"] == "compression_threshold"
        assert error.details["value"] == -100

    def test_with_zero_value(self):
        """Test that value=0 is included in details"""
        error = ConfigurationError("Invalid value", field="max_retries", value=0)

        assert error.value == 0
        assert error.details["value"] == 0

    def test_never_recoverable(self):
        """Test that configuration errors are never recoverable"""
        error = ConfigurationError("Invalid", field="test", value=123)

        assert error.recoverable is False


class TestValidationError:
    """Test ValidationError class"""

    def test_validation_error_inherits_base(self):
        """Test that ValidationError inherits from ProxyAgentError"""
        error = ValidationError("Validation failed")

        assert isinstance(error, ProxyAgentError)
        assert error.message == "Validation failed"


class TestExceptionUsage:
    """Test practical usage scenarios"""

    def test_catching_and_logging(self):
        """Test catching exception and converting to dict for logging"""
        try:
            raise LLMError(
                "API request failed",
                status_code=429,
                retry_after=60.0,
                details={"model": "gemini-2.0-flash"},
            )
        except LLMError as e:
            log_dict = e.to_dict()

            assert log_dict["error_type"] == "LLMError"
            assert log_dict["message"] == "API request failed"
            assert log_dict["retry_after"] == 60.0
            assert log_dict["recoverable"] is True

    def test_retry_decision(self):
        """Test making retry decision based on exception"""
        error = ToolExecutionError("Tool failed", tool_id="test_tool", retry_count=1)

        # Should retry because recoverable
        assert error.recoverable is True

        error_max_retries = ToolExecutionError("Tool failed", tool_id="test_tool", retry_count=3)

        # Should not retry
        assert error_max_retries.recoverable is False

    def test_user_message_display(self):
        """Test displaying user-friendly messages"""
        errors = [
            LLMError("API failed", status_code=429),
            ToolExecutionError("Failed", tool_id="search"),
            CompressionError("Failed", strategy="semantic"),
            SerializationError("Failed", format="TOON"),
            ConfigurationError("Invalid", field="threshold"),
        ]

        for error in errors:
            # All should have user-friendly messages
            assert error.user_message
            assert len(error.user_message) > 0
            # User message should be different from technical message
            # (except for ConfigurationError which prefixes it)
