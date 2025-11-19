"""Result type for explicit error handling (Rust-style)"""
from dataclasses import dataclass, field
from typing import Generic, TypeVar, Optional, List, Any, Callable

T = TypeVar('T')
E = TypeVar('E')

@dataclass
class Result(Generic[T]):
    """
    Result wrapper implementing "error as value" pattern.

    Inspired by Rust's Result<T, E> and functional programming.
    Eliminates exceptions for flow control.

    Examples:
        # Create successful result
        result = Result.ok({"user": "Alice"})

        # Create failed result
        result = Result.err("Database connection failed")

        # Check success
        if result.success:
            print(result.data)
        else:
            print(result.error)

        # Unwrap (raises if failed)
        data = result.unwrap()

        # Unwrap with default
        data = result.unwrap_or(default_value)

        # Chain operations
        result.map(process_data).map(transform_data)
    """

    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate invariants after initialization"""
        if self.success and self.data is None:
            raise ValueError("Successful result must have data")
        if not self.success and self.error is None:
            raise ValueError("Failed result must have error")

    def add_warning(self, warning: str) -> 'Result[T]':
        """
        Add a warning to the result.

        Args:
            warning: Warning message to add

        Returns:
            Self for chaining
        """
        self.warnings.append(warning)
        return self

    def unwrap(self) -> T:
        """
        Unwrap the result, raising exception if failed.

        Use only when you're certain the operation succeeded.
        Prefer unwrap_or() or explicit success checks.

        Returns:
            The wrapped data

        Raises:
            ValueError: If result is not successful
        """
        if not self.success:
            raise ValueError(f"Unwrap called on failed result: {self.error}")
        return self.data

    def unwrap_or(self, default: T) -> T:
        """
        Unwrap the result or return default value.

        Args:
            default: Value to return if result is failed

        Returns:
            The wrapped data if successful, otherwise default
        """
        return self.data if self.success else default

    def unwrap_or_else(self, func: Callable[[], T]) -> T:
        """
        Unwrap or compute default via function.

        Args:
            func: Function to call if result is failed

        Returns:
            The wrapped data if successful, otherwise func()
        """
        return self.data if self.success else func()

    def map(self, func: Callable[[T], Any]) -> 'Result[Any]':
        """
        Apply function to data if successful.

        Returns new Result with transformed data or original error.

        Args:
            func: Function to apply to the data

        Returns:
            New Result with transformed data or original error
        """
        if self.success and self.data is not None:
            try:
                new_data = func(self.data)
                return Result(
                    success=True,
                    data=new_data,
                    warnings=self.warnings.copy()
                )
            except Exception as e:
                return Result(
                    success=False,
                    error=f"Map failed: {str(e)}",
                    warnings=self.warnings.copy()
                )

        return Result(
            success=False,
            error=self.error,
            warnings=self.warnings.copy()
        )

    def and_then(self, func: Callable[[T], 'Result[Any]']) -> 'Result[Any]':
        """
        Chain Result-returning functions (flatMap/bind).

        If successful, applies func to data and returns its Result.
        If failed, returns original error.

        Args:
            func: Function that takes data and returns a Result

        Returns:
            Result returned by func, or original error
        """
        if self.success and self.data is not None:
            try:
                result = func(self.data)
                # Merge warnings
                if isinstance(result, Result):
                    result.warnings = self.warnings + result.warnings
                return result
            except Exception as e:
                return Result.err(f"and_then failed: {str(e)}")

        return Result(success=False, error=self.error, warnings=self.warnings.copy())

    def or_else(self, func: Callable[[str], 'Result[T]']) -> 'Result[T]':
        """
        Provide fallback if failed.

        If failed, calls func with error and returns its Result.
        If successful, returns original result.

        Args:
            func: Function that takes error and returns a Result

        Returns:
            Original result if successful, otherwise func(error)
        """
        if not self.success:
            try:
                return func(self.error)
            except Exception as e:
                return Result.err(f"or_else failed: {str(e)}")

        return self

    def is_ok(self) -> bool:
        """Check if result is successful"""
        return self.success

    def is_err(self) -> bool:
        """Check if result is failed"""
        return not self.success

    def expect(self, msg: str) -> T:
        """
        Unwrap with custom error message.

        Args:
            msg: Error message if result is failed

        Returns:
            The wrapped data

        Raises:
            ValueError: If result is not successful, with custom message
        """
        if not self.success:
            raise ValueError(f"{msg}: {self.error}")
        return self.data

    @classmethod
    def ok(cls, data: T) -> 'Result[T]':
        """
        Create successful result.

        Args:
            data: The data to wrap

        Returns:
            Successful Result containing data
        """
        return cls(success=True, data=data)

    @classmethod
    def err(cls, error: str) -> 'Result[T]':
        """
        Create failed result.

        Args:
            error: Error message

        Returns:
            Failed Result containing error
        """
        return cls(success=False, error=error)

    @classmethod
    def from_optional(cls, value: Optional[T], error_msg: str = "Value is None") -> 'Result[T]':
        """
        Create Result from Optional value.

        Args:
            value: Optional value to wrap
            error_msg: Error message if value is None

        Returns:
            Result.ok(value) if value is not None, otherwise Result.err(error_msg)
        """
        if value is not None:
            return cls.ok(value)
        return cls.err(error_msg)

    @classmethod
    def from_exception(cls, func: Callable[[], T], error_prefix: str = "Operation failed") -> 'Result[T]':
        """
        Create Result by catching exceptions from a function.

        Args:
            func: Function to execute
            error_prefix: Prefix for error message

        Returns:
            Result.ok(func()) if successful, otherwise Result.err(error)
        """
        try:
            data = func()
            return cls.ok(data)
        except Exception as e:
            return cls.err(f"{error_prefix}: {str(e)}")

    def __repr__(self) -> str:
        if self.success:
            warnings_str = f", warnings={self.warnings}" if self.warnings else ""
            return f"Result.ok({self.data!r}{warnings_str})"
        else:
            warnings_str = f", warnings={self.warnings}" if self.warnings else ""
            return f"Result.err({self.error!r}{warnings_str})"

    def __bool__(self) -> bool:
        """Allow using Result in boolean context (checks success)"""
        return self.success
