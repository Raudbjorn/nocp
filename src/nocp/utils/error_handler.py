"""Centralized error handling utilities"""
from typing import Callable, TypeVar, Optional
from contextlib import contextmanager
import time
from functools import wraps
from ..utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class ErrorHandler:
    """Centralized error handling with logging and fallbacks"""

    @staticmethod
    def handle_with_fallback(
        operation: Callable[[], T],
        fallback: T,
        error_msg: str,
        log_level: str = "error"
    ) -> T:
        """
        Execute operation with fallback on error.

        Args:
            operation: Function to execute
            fallback: Value to return on error
            error_msg: Error message prefix
            log_level: Log level for errors

        Returns:
            Operation result or fallback value

        Example:
            result = ErrorHandler.handle_with_fallback(
                lambda: fetch_from_cache(),
                fallback=[],
                error_msg="Cache fetch failed"
            )
        """
        try:
            return operation()
        except Exception as e:
            getattr(logger, log_level)(f"{error_msg}: {e}")
            return fallback

    @staticmethod
    @contextmanager
    def log_duration(operation_name: str, log_level: str = "info"):
        """
        Context manager to log operation duration.

        Example:
            with ErrorHandler.log_duration("Database query"):
                result = db.query(...)
        """
        start = time.perf_counter()
        try:
            getattr(logger, log_level)(f"Starting {operation_name}")
            yield
        finally:
            duration = time.perf_counter() - start
            getattr(logger, log_level)(
                f"{operation_name} completed in {duration:.3f}s"
            )

    @staticmethod
    def retry_with_backoff(
        operation: Callable[[], T],
        max_attempts: int = 3,
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
        retryable_exceptions: tuple = (Exception,)
    ) -> T:
        """
        Retry operation with exponential backoff.

        Args:
            operation: Function to execute
            max_attempts: Maximum retry attempts
            backoff_factor: Multiplier for delay between retries
            initial_delay: Initial delay in seconds
            retryable_exceptions: Tuple of exceptions to retry on

        Returns:
            Operation result

        Raises:
            Last exception if all attempts fail

        Example:
            response = ErrorHandler.retry_with_backoff(
                lambda: requests.get(url),
                max_attempts=3,
                retryable_exceptions=(requests.Timeout, requests.ConnectionError)
            )
        """
        delay = initial_delay
        last_exception = None

        for attempt in range(max_attempts):
            try:
                return operation()
            except retryable_exceptions as e:
                last_exception = e

                if attempt == max_attempts - 1:
                    # Last attempt failed
                    raise

                logger.warning(
                    f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay *= backoff_factor

        # Should not reach here, but for type safety
        raise last_exception  # type: ignore

    @staticmethod
    def ignore_errors(
        operation: Callable[[], T],
        error_msg: str = "Operation failed",
        log_level: str = "warning"
    ) -> Optional[T]:
        """
        Execute operation, ignoring all errors.

        Returns None on error. Use with caution!

        Example:
            # Non-critical cache write
            ErrorHandler.ignore_errors(
                lambda: cache.set(key, value),
                error_msg="Cache write failed (non-critical)"
            )
        """
        try:
            return operation()
        except Exception as e:
            getattr(logger, log_level)(f"{error_msg}: {e}")
            return None


def with_retry(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0
):
    """
    Decorator for automatic retry with exponential backoff.

    Example:
        @with_retry(max_attempts=3)
        def fetch_data(url: str) -> dict:
            return requests.get(url).json()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return ErrorHandler.retry_with_backoff(
                lambda: func(*args, **kwargs),
                max_attempts=max_attempts,
                backoff_factor=backoff_factor,
                initial_delay=initial_delay
            )
        return wrapper
    return decorator
