"""
Act Module: Tool Executor

Manages tool registration and execution with retry logic and timeout handling.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from ..exceptions import ToolExecutionError
from ..models.contracts import ToolRequest, ToolResult, ToolType
from .cache import CacheBackend

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    Manages tool registration and execution with retry logic and caching.

    Example:
        executor = ToolExecutor()

        @executor.register_tool("fetch_data")
        def fetch_data(param1: str) -> dict:
            return {"result": param1}

        request = ToolRequest(
            tool_id="fetch_data",
            tool_type=ToolType.PYTHON_FUNCTION,
            function_name="fetch_data",
            parameters={"param1": "test"}
        )
        result = executor.execute(request)
    """

    def __init__(self, cache: Optional[CacheBackend] = None):
        """
        Initialize ToolExecutor with optional caching.

        Args:
            cache: Optional CacheBackend instance for caching tool results
        """
        self._registry: Dict[str, Callable] = {}
        self._async_registry: Dict[str, Callable] = {}
        self._cache = cache

    def register_tool(
        self,
        tool_id: str,
        tool_type: ToolType = ToolType.PYTHON_FUNCTION
    ) -> Callable:
        """
        Decorator to register a synchronous tool.

        Args:
            tool_id: Unique identifier for the tool
            tool_type: Category of tool (default: PYTHON_FUNCTION)

        Returns:
            Decorator function

        Example:
            @executor.register_tool("my_tool")
            def my_tool(param: str) -> str:
                return param.upper()
        """
        def decorator(func: Callable) -> Callable:
            self._registry[tool_id] = func
            return func
        return decorator

    def register_async_tool(self, tool_id: str) -> Callable:
        """Decorator to register an async tool."""
        def decorator(func: Callable) -> Callable:
            self._async_registry[tool_id] = func
            return func
        return decorator

    def execute(self, request: ToolRequest, use_cache: bool = True) -> ToolResult:
        """
        Execute a registered tool with retry logic and caching.

        Implementation steps:
        1. Check cache if enabled
        2. Validate tool exists in registry
        3. Execute with timeout
        4. Retry on failure if configured
        5. Estimate token count of result
        6. Cache result if enabled
        7. Return ToolResult

        Args:
            request: ToolRequest with execution parameters
            use_cache: Whether to use cache for this request (default: True)

        Returns:
            ToolResult with execution outcome and metadata

        Raises:
            ToolExecutionError: If execution fails after all retries
            TimeoutError: If execution exceeds timeout
        """
        # Check cache first
        if use_cache and self._cache is not None:
            cached_result = self._cache.get_by_request(request)
            if cached_result is not None:
                logger.debug(f"Cache hit for tool '{request.tool_id}'")
                return cached_result

        if request.tool_id not in self._registry:
            raise ToolExecutionError(
                f"Tool '{request.tool_id}' not found in registry",
                details={"tool_id": request.tool_id}
            )

        retry_config = request.retry_config
        max_attempts = retry_config.max_attempts if retry_config else 1
        last_error = None
        is_timeout = False

        for attempt in range(max_attempts):
            try:
                start = time.perf_counter()

                # Execute tool with timeout
                func = self._registry[request.tool_id]
                result = self._execute_with_timeout(
                    func,
                    request.parameters,
                    request.timeout_seconds
                )

                execution_time = (time.perf_counter() - start) * 1000

                # Estimate tokens
                token_estimate = self._estimate_tokens(result)

                tool_result = ToolResult(
                    tool_id=request.tool_id,
                    success=True,
                    data=result,
                    error=None,
                    execution_time_ms=execution_time,
                    timestamp=datetime.now(),
                    token_estimate=token_estimate,
                    retry_count=attempt
                )

                # Cache successful result
                if use_cache and self._cache is not None:
                    self._cache.set_by_request(request, tool_result)

                return tool_result

            except TimeoutError as e:
                last_error = f"Tool execution exceeded {request.timeout_seconds}s timeout"
                is_timeout = True
                if attempt < max_attempts - 1:
                    # Exponential backoff
                    if retry_config:
                        delay = (retry_config.initial_delay_ms / 1000) * \
                                (retry_config.backoff_multiplier ** attempt)
                        time.sleep(delay)
                continue

            except Exception as e:
                last_error = str(e)
                is_timeout = False
                if attempt < max_attempts - 1:
                    # Exponential backoff
                    if retry_config:
                        delay = (retry_config.initial_delay_ms / 1000) * \
                                (retry_config.backoff_multiplier ** attempt)
                        time.sleep(delay)
                continue

        # All retries failed - raise appropriate exception
        if is_timeout:
            raise TimeoutError(last_error)
        else:
            raise ToolExecutionError(
                f"Tool execution failed after {max_attempts} attempts",
                details={
                    "tool_id": request.tool_id,
                    "last_error": last_error,
                    "attempts": max_attempts
                }
            )

    async def execute_async(self, request: ToolRequest, use_cache: bool = True) -> ToolResult:
        """
        Async version for concurrent execution with caching.

        Args:
            request: ToolRequest with execution parameters
            use_cache: Whether to use cache for this request (default: True)

        Returns:
            ToolResult with execution outcome and metadata

        Raises:
            ToolExecutionError: If execution fails after all retries
            TimeoutError: If execution exceeds timeout
        """
        # Check cache first
        if use_cache and self._cache is not None:
            # For async execution, we need to use the cache's async methods if available
            if hasattr(self._cache, 'get_by_request_async'):
                cached_result = await self._cache.get_by_request_async(request)
            else:
                # Fallback to sync method wrapped in async
                cached_result = self._cache.get_by_request(request)

            if cached_result is not None:
                logger.debug(f"Cache hit for async tool '{request.tool_id}'")
                return cached_result

        if request.tool_id not in self._async_registry:
            raise ToolExecutionError(
                f"Async tool '{request.tool_id}' not found in registry",
                details={"tool_id": request.tool_id}
            )

        retry_config = request.retry_config
        max_attempts = retry_config.max_attempts if retry_config else 1
        last_error = None
        is_timeout = False

        for attempt in range(max_attempts):
            try:
                start = time.perf_counter()

                # Execute async tool with timeout
                func = self._async_registry[request.tool_id]
                result = await asyncio.wait_for(
                    func(**request.parameters),
                    timeout=request.timeout_seconds
                )

                execution_time = (time.perf_counter() - start) * 1000
                token_estimate = self._estimate_tokens(result)

                tool_result = ToolResult(
                    tool_id=request.tool_id,
                    success=True,
                    data=result,
                    error=None,
                    execution_time_ms=execution_time,
                    timestamp=datetime.now(),
                    token_estimate=token_estimate,
                    retry_count=attempt
                )

                # Cache successful result
                if use_cache and self._cache is not None:
                    if hasattr(self._cache, 'set_by_request_async'):
                        await self._cache.set_by_request_async(request, tool_result)
                    else:
                        # Fallback to sync method
                        self._cache.set_by_request(request, tool_result)

                return tool_result

            except asyncio.TimeoutError:
                last_error = f"Tool execution exceeded {request.timeout_seconds}s timeout"
                is_timeout = True
                if attempt < max_attempts - 1 and retry_config:
                    delay = (retry_config.initial_delay_ms / 1000) * \
                            (retry_config.backoff_multiplier ** attempt)
                    await asyncio.sleep(delay)
                continue

            except Exception as e:
                last_error = str(e)
                is_timeout = False
                if attempt < max_attempts - 1 and retry_config:
                    delay = (retry_config.initial_delay_ms / 1000) * \
                            (retry_config.backoff_multiplier ** attempt)
                    await asyncio.sleep(delay)
                continue

        # All retries failed - raise appropriate exception
        if is_timeout:
            raise TimeoutError(last_error)
        else:
            raise ToolExecutionError(
                f"Tool execution failed after {max_attempts} attempts",
                details={
                    "tool_id": request.tool_id,
                    "last_error": last_error,
                    "attempts": max_attempts
                }
            )

    def _execute_with_timeout(
        self,
        func: Callable,
        params: Dict[str, Any],
        timeout: int
    ) -> Any:
        """
        Execute function with timeout using concurrent.futures for cross-platform support.

        Uses ThreadPoolExecutor to run the function in a separate thread with a timeout,
        which works on all platforms (Unix, Windows, macOS).

        Args:
            func: Function to execute
            params: Parameters to pass to the function
            timeout: Timeout in seconds

        Returns:
            Result from function execution

        Raises:
            TimeoutError: If execution exceeds timeout
        """
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, **params)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeoutError:
                raise TimeoutError(f"Tool execution exceeded {timeout}s timeout")

    def _estimate_tokens(self, data: Any) -> int:
        """
        Rough token estimate using character count.
        Rule of thumb: 1 token ≈ 4 characters for English text.

        Args:
            data: Data to estimate tokens for

        Returns:
            Estimated token count
        """
        if isinstance(data, str):
            return len(data) // 4
        elif isinstance(data, (dict, list)):
            text = json.dumps(data)
            return len(text) // 4
        else:
            return len(str(data)) // 4

    def list_tools(self) -> list[str]:
        """Get list of registered tool IDs."""
        return list(self._registry.keys())

    def validate_tool(self, tool_id: str) -> bool:
        """Check if tool is registered and available."""
        return tool_id in self._registry or tool_id in self._async_registry
