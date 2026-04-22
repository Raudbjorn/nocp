"""Enhanced exception classes with rich context"""

from typing import Any
from datetime import datetime


class ProxyAgentError(Exception):
    """Base exception with enhanced context and metadata"""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        retry_after: float | None = None,
        recoverable: bool = False,
        user_message: str | None = None,
    ):
        """
        Initialize exception with context.

        Args:
            message: Technical error message for logs
            details: Additional context (dict for structured logging)
            retry_after: Seconds to wait before retrying (if applicable)
            recoverable: Whether error is recoverable with retry
            user_message: User-friendly error message
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.retry_after = retry_after
        self.recoverable = recoverable
        self.user_message = user_message or message
        self.timestamp = datetime.now()

    def __str__(self) -> str:
        parts = [self.message]

        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"({details_str})")

        if self.retry_after:
            parts.append(f"[retry after {self.retry_after}s]")

        if self.recoverable:
            parts.append("[recoverable]")

        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "retry_after": self.retry_after,
            "recoverable": self.recoverable,
            "user_message": self.user_message,
            "timestamp": self.timestamp.isoformat(),
        }


class LLMError(ProxyAgentError):
    """LLM API errors with retry information"""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        status_code: int | None = None,
        retry_after: float | None = None,
    ):
        # Rate limits (429) are usually recoverable
        recoverable = status_code == 429

        # Generate user-friendly message
        if status_code == 429:
            user_message = "API rate limit exceeded. Please try again in a moment."
        elif status_code == 401:
            user_message = "API authentication failed. Please check your API key."
        elif status_code == 503:
            user_message = "Service temporarily unavailable. Please try again."
        else:
            user_message = "An error occurred while calling the LLM API."

        super().__init__(
            message=message,
            details=details or {},
            retry_after=retry_after,
            recoverable=recoverable,
            user_message=user_message,
        )
        self.status_code = status_code


class ToolExecutionError(ProxyAgentError):
    """Tool execution errors"""

    def __init__(
        self,
        message: str,
        tool_id: str,
        details: dict[str, Any] | None = None,
        retry_count: int = 0,
    ):
        details = details or {}
        details.update({"tool_id": tool_id, "retry_count": retry_count})

        super().__init__(
            message=message,
            details=details,
            recoverable=retry_count < 3,  # Recoverable if retries remain
            user_message=f"Tool '{tool_id}' execution failed.",
        )
        self.tool_id = tool_id
        self.retry_count = retry_count


class CompressionError(ProxyAgentError):
    """Context compression errors"""

    def __init__(self, message: str, strategy: str, details: dict[str, Any] | None = None):
        details = details or {}
        details["strategy"] = strategy

        super().__init__(
            message=message,
            details=details,
            recoverable=True,  # Can fall back to uncompressed
            user_message="Context compression failed. Using uncompressed context.",
        )
        self.strategy = strategy


class SerializationError(ProxyAgentError):
    """Output serialization errors"""

    def __init__(self, message: str, format: str, details: dict[str, Any] | None = None):
        details = details or {}
        details["format"] = format

        super().__init__(
            message=message,
            details=details,
            recoverable=True,  # Can fall back to JSON
            user_message=f"Failed to serialize to {format}. Using fallback format.",
        )
        self.format = format


class ConfigurationError(ProxyAgentError):
    """Configuration validation errors"""

    def __init__(self, message: str, field: str | None = None, value: Any = None):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = value

        super().__init__(
            message=message,
            details=details,
            recoverable=False,  # Config errors require fix
            user_message=f"Configuration error: {message}",
        )
        self.field = field
        self.value = value


class ValidationError(ProxyAgentError):
    """Raised when Pydantic validation fails."""

    pass
