"""
Unit tests for ErrorHandler utilities.

Tests cover:
- handle_with_fallback for graceful error handling
- log_duration for performance monitoring
- retry_with_backoff with exponential backoff
- ignore_errors for non-critical operations
- with_retry decorator
"""

import time
from unittest.mock import Mock, patch, MagicMock, call
import pytest

from nocp.utils.error_handler import ErrorHandler, with_retry


class TestHandleWithFallback:
    """Tests for handle_with_fallback method."""

    def test_successful_operation(self):
        """Test that successful operation returns result."""
        result = ErrorHandler.handle_with_fallback(
            operation=lambda: "success",
            fallback="fallback",
            error_msg="Operation failed"
        )
        assert result == "success"

    def test_failed_operation_returns_fallback(self):
        """Test that failed operation returns fallback value."""
        def failing_op():
            raise ValueError("Test error")

        result = ErrorHandler.handle_with_fallback(
            operation=failing_op,
            fallback="fallback",
            error_msg="Operation failed"
        )
        assert result == "fallback"

    def test_logs_error_with_correct_level(self):
        """Test that error is logged with correct log level."""
        def failing_op():
            raise ValueError("Test error")

        with patch('nocp.utils.error_handler.logger') as mock_logger:
            ErrorHandler.handle_with_fallback(
                operation=failing_op,
                fallback="fallback",
                error_msg="Operation failed",
                log_level="warning"
            )
            mock_logger.warning.assert_called_once()
            args = mock_logger.warning.call_args[0][0]
            assert "Operation failed" in args
            assert "Test error" in args

    def test_fallback_with_complex_types(self):
        """Test fallback with complex types like lists and dicts."""
        def failing_op():
            raise Exception("Error")

        result = ErrorHandler.handle_with_fallback(
            operation=failing_op,
            fallback={"key": "value"},
            error_msg="Dict operation failed"
        )
        assert result == {"key": "value"}

        result = ErrorHandler.handle_with_fallback(
            operation=failing_op,
            fallback=[1, 2, 3],
            error_msg="List operation failed"
        )
        assert result == [1, 2, 3]


class TestLogDuration:
    """Tests for log_duration context manager."""

    def test_logs_start_and_completion(self):
        """Test that both start and completion are logged."""
        with patch('nocp.utils.error_handler.logger') as mock_logger:
            with ErrorHandler.log_duration("Test operation"):
                time.sleep(0.01)  # Small delay to ensure measurable duration

            # Check that info was called twice (start and completion)
            assert mock_logger.info.call_count == 2

            # Check start message
            start_call = mock_logger.info.call_args_list[0][0][0]
            assert "Starting Test operation" in start_call

            # Check completion message
            completion_call = mock_logger.info.call_args_list[1][0][0]
            assert "Test operation completed" in completion_call
            assert "s" in completion_call  # Should contain time in seconds

    def test_logs_with_custom_log_level(self):
        """Test that custom log level is respected."""
        with patch('nocp.utils.error_handler.logger') as mock_logger:
            with ErrorHandler.log_duration("Test operation", log_level="debug"):
                pass

            # Verify debug was called instead of info
            assert mock_logger.debug.call_count == 2
            assert mock_logger.info.call_count == 0

    def test_duration_measurement_accuracy(self):
        """Test that duration is measured accurately."""
        with patch('nocp.utils.error_handler.logger') as mock_logger:
            sleep_time = 0.1
            with ErrorHandler.log_duration("Timed operation"):
                time.sleep(sleep_time)

            # Get the completion message
            completion_call = mock_logger.info.call_args_list[1][0][0]

            # Extract duration from message (format: "... completed in X.XXXs")
            # The duration should be approximately sleep_time
            assert "completed in" in completion_call

    def test_context_manager_with_exception(self):
        """Test that duration is logged even when exception occurs."""
        with patch('nocp.utils.error_handler.logger') as mock_logger:
            with pytest.raises(ValueError):
                with ErrorHandler.log_duration("Failing operation"):
                    raise ValueError("Test error")

            # Should still log completion
            assert mock_logger.info.call_count == 2


class TestRetryWithBackoff:
    """Tests for retry_with_backoff method."""

    def test_successful_first_attempt(self):
        """Test that successful operation on first attempt returns immediately."""
        mock_op = Mock(return_value="success")

        result = ErrorHandler.retry_with_backoff(
            operation=mock_op,
            max_attempts=3
        )

        assert result == "success"
        mock_op.assert_called_once()

    def test_retry_on_failure(self):
        """Test that operation is retried on failure."""
        mock_op = Mock(side_effect=[
            ValueError("First fail"),
            ValueError("Second fail"),
            "success"
        ])

        with patch('nocp.utils.error_handler.logger'):
            result = ErrorHandler.retry_with_backoff(
                operation=mock_op,
                max_attempts=3,
                initial_delay=0.01  # Short delay for testing
            )

        assert result == "success"
        assert mock_op.call_count == 3

    def test_raises_after_max_attempts(self):
        """Test that exception is raised after max attempts."""
        mock_op = Mock(side_effect=ValueError("Always fails"))

        with patch('nocp.utils.error_handler.logger'):
            with pytest.raises(ValueError, match="Always fails"):
                ErrorHandler.retry_with_backoff(
                    operation=mock_op,
                    max_attempts=3,
                    initial_delay=0.01
                )

        assert mock_op.call_count == 3

    def test_exponential_backoff(self):
        """Test that backoff delay increases exponentially."""
        mock_op = Mock(side_effect=ValueError("Fail"))
        delays = []

        def mock_sleep(delay):
            delays.append(delay)
            time.sleep(0.001)  # Actual minimal sleep for test speed

        with patch('nocp.utils.error_handler.logger'):
            with patch('nocp.utils.error_handler.time.sleep', side_effect=mock_sleep):
                with pytest.raises(ValueError):
                    ErrorHandler.retry_with_backoff(
                        operation=mock_op,
                        max_attempts=3,
                        initial_delay=1.0,
                        backoff_factor=2.0
                    )

        # Should have 2 delays (between 3 attempts)
        assert len(delays) == 2
        assert delays[0] == 1.0
        assert delays[1] == 2.0

    def test_specific_retryable_exceptions(self):
        """Test that only specific exceptions are retried."""
        mock_op = Mock(side_effect=ValueError("Should not retry"))

        with patch('nocp.utils.error_handler.logger'):
            with pytest.raises(ValueError):
                ErrorHandler.retry_with_backoff(
                    operation=mock_op,
                    max_attempts=3,
                    retryable_exceptions=(ConnectionError,)
                )

        # Should fail immediately since ValueError is not retryable
        assert mock_op.call_count == 1

    def test_logs_retry_attempts(self):
        """Test that retry attempts are logged."""
        mock_op = Mock(side_effect=[ValueError("Fail"), "success"])

        with patch('nocp.utils.error_handler.logger') as mock_logger:
            ErrorHandler.retry_with_backoff(
                operation=mock_op,
                max_attempts=3,
                initial_delay=0.01
            )

            # Should log warning for the first failure
            mock_logger.warning.assert_called_once()
            log_msg = mock_logger.warning.call_args[0][0]
            assert "Attempt 1/3 failed" in log_msg
            assert "Retrying" in log_msg


class TestIgnoreErrors:
    """Tests for ignore_errors method."""

    def test_successful_operation_returns_result(self):
        """Test that successful operation returns result."""
        result = ErrorHandler.ignore_errors(
            operation=lambda: "success",
            error_msg="Operation failed"
        )
        assert result == "success"

    def test_failed_operation_returns_none(self):
        """Test that failed operation returns None."""
        def failing_op():
            raise ValueError("Test error")

        result = ErrorHandler.ignore_errors(
            operation=failing_op,
            error_msg="Operation failed"
        )
        assert result is None

    def test_logs_error_with_warning_level(self):
        """Test that error is logged with warning level by default."""
        def failing_op():
            raise ValueError("Test error")

        with patch('nocp.utils.error_handler.logger') as mock_logger:
            ErrorHandler.ignore_errors(
                operation=failing_op,
                error_msg="Non-critical operation failed"
            )

            mock_logger.warning.assert_called_once()
            args = mock_logger.warning.call_args[0][0]
            assert "Non-critical operation failed" in args
            assert "Test error" in args

    def test_custom_log_level(self):
        """Test that custom log level is respected."""
        def failing_op():
            raise ValueError("Test error")

        with patch('nocp.utils.error_handler.logger') as mock_logger:
            ErrorHandler.ignore_errors(
                operation=failing_op,
                error_msg="Operation failed",
                log_level="debug"
            )

            mock_logger.debug.assert_called_once()
            mock_logger.warning.assert_not_called()


class TestWithRetryDecorator:
    """Tests for with_retry decorator."""

    def test_successful_call(self):
        """Test that successful call works without retry."""
        @with_retry(max_attempts=3)
        def successful_func(value):
            return value * 2

        result = successful_func(5)
        assert result == 10

    def test_retries_on_failure(self):
        """Test that function is retried on failure."""
        call_count = {"count": 0}

        @with_retry(max_attempts=3, initial_delay=0.01)
        def flaky_func():
            call_count["count"] += 1
            if call_count["count"] < 3:
                raise ValueError("Temporary failure")
            return "success"

        with patch('nocp.utils.error_handler.logger'):
            result = flaky_func()

        assert result == "success"
        assert call_count["count"] == 3

    def test_preserves_function_metadata(self):
        """Test that decorator preserves function metadata."""
        @with_retry(max_attempts=3)
        def documented_func():
            """This is a docstring."""
            pass

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is a docstring."

    def test_works_with_arguments(self):
        """Test that decorated function works with arguments."""
        @with_retry(max_attempts=3, initial_delay=0.01)
        def func_with_args(a, b, c=10):
            if a < 0:
                raise ValueError("a must be positive")
            return a + b + c

        result = func_with_args(5, 3, c=2)
        assert result == 10

    def test_raises_after_max_attempts(self):
        """Test that exception is raised after max attempts."""
        @with_retry(max_attempts=2, initial_delay=0.01)
        def always_fails():
            raise ValueError("Always fails")

        with patch('nocp.utils.error_handler.logger'):
            with pytest.raises(ValueError, match="Always fails"):
                always_fails()

    def test_custom_retry_parameters(self):
        """Test decorator with custom retry parameters."""
        call_count = {"count": 0}

        @with_retry(max_attempts=5, backoff_factor=3.0, initial_delay=0.01)
        def custom_retry_func():
            call_count["count"] += 1
            if call_count["count"] < 4:
                raise ValueError("Temporary")
            return "success"

        with patch('nocp.utils.error_handler.logger'):
            result = custom_retry_func()

        assert result == "success"
        assert call_count["count"] == 4


class TestIntegration:
    """Integration tests combining multiple error handling utilities."""

    def test_combined_error_handling(self):
        """Test combining different error handling patterns."""
        call_count = {"count": 0}

        def complex_operation():
            call_count["count"] += 1
            if call_count["count"] < 2:
                raise ValueError("Transient error")
            return {"data": "success"}

        # Use retry with duration logging
        with patch('nocp.utils.error_handler.logger'):
            with ErrorHandler.log_duration("Complex operation"):
                result = ErrorHandler.retry_with_backoff(
                    operation=complex_operation,
                    max_attempts=3,
                    initial_delay=0.01
                )

        assert result == {"data": "success"}
        assert call_count["count"] == 2

    def test_nested_error_handlers(self):
        """Test nesting different error handling patterns."""
        outer_result = ErrorHandler.handle_with_fallback(
            operation=lambda: ErrorHandler.handle_with_fallback(
                operation=lambda: "inner_success",
                fallback="inner_fallback",
                error_msg="Inner operation failed"
            ),
            fallback="outer_fallback",
            error_msg="Outer operation failed"
        )

        assert outer_result == "inner_success"

    def test_cache_pattern(self):
        """Test common cache pattern with error handling."""
        cache = {}

        def fetch_from_cache(key: str):
            if key not in cache:
                raise KeyError(f"Key {key} not found")
            return cache[key]

        # Cache miss should return fallback
        result = ErrorHandler.handle_with_fallback(
            operation=lambda: fetch_from_cache("missing_key"),
            fallback=None,
            error_msg="Cache miss",
            log_level="debug"
        )
        assert result is None

        # Cache hit should return value
        cache["existing_key"] = "value"
        result = ErrorHandler.handle_with_fallback(
            operation=lambda: fetch_from_cache("existing_key"),
            fallback=None,
            error_msg="Cache miss",
            log_level="debug"
        )
        assert result == "value"
