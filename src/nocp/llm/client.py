"""
LLM Client Wrapper - Unified interface to LiteLLM.

Provides a consistent API for calling different LLM providers through LiteLLM,
with support for structured output, error handling, and retries.
"""

import time
from typing import Any, Dict, List, Optional, Type, Union
import json

from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from ..exceptions import LLMError
from ..models.contracts import LLMRequest, LLMResponse, MessageRole


class LLMClient:
    """
    Unified LLM client using LiteLLM for multi-provider support.

    Example:
        client = LLMClient(default_model="gemini/gemini-2.0-flash-exp")

        response = client.complete(
            messages=[{"role": "user", "content": "Hello"}],
            response_schema=OutputSchema,
            max_tokens=1000
        )
    """

    def __init__(
        self,
        default_model: str = "gemini/gemini-2.0-flash-exp",
        api_key: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        max_retries: int = 3,
        timeout: int = 60,
    ):
        """
        Initialize LLM client.

        Args:
            default_model: Default model to use (LiteLLM format: "provider/model")
            api_key: Optional API key (can also use env vars)
            fallback_models: List of fallback models if primary fails
            max_retries: Maximum retry attempts on transient failures
            timeout: Request timeout in seconds
        """
        self.default_model = default_model
        self.fallback_models = fallback_models or []
        self.max_retries = max_retries
        self.timeout = timeout

        # Import litellm
        try:
            import litellm
            self.litellm = litellm

            # Configure litellm
            if api_key:
                # Set API key for the provider
                provider = default_model.split("/")[0]
                if provider == "gemini":
                    litellm.api_key = api_key
                elif provider == "openai":
                    litellm.openai_key = api_key
                # Add more providers as needed

            # Enable caching for token counting
            litellm.cache = None  # Disable by default, can be enabled

        except ImportError as e:
            raise ImportError(
                "litellm is required for LLMClient. "
                "Install with: pip install litellm"
            ) from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    )
    def complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        response_schema: Optional[Type[BaseModel]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        """
        Call LLM with messages and optional structured output.

        Args:
            messages: List of messages in chat format
            model: Optional model override
            response_schema: Optional Pydantic model for structured output
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            **kwargs: Additional parameters for litellm.completion()

        Returns:
            LLMResponse with content and metadata

        Raises:
            LLMError: If all retry attempts fail
        """
        model = model or self.default_model
        start_time = time.perf_counter()

        try:
            # Build completion kwargs
            completion_kwargs = {
                "model": model,
                "messages": messages,
                "timeout": self.timeout,
                "temperature": temperature,
            }

            if max_tokens:
                completion_kwargs["max_tokens"] = max_tokens

            # Handle structured output if schema provided
            if response_schema:
                # Extract provider from model string (format: "provider/model")
                provider = model.split("/")[0] if "/" in model else "unknown"

                # LiteLLM supports response_format for some providers
                # For Gemini, we'll use function calling to enforce schema
                if provider == "gemini":
                    # Use function calling for structured output
                    schema_json = response_schema.model_json_schema()
                    completion_kwargs["tools"] = [
                        {
                            "type": "function",
                            "function": {
                                "name": "format_response",
                                "description": "Format the response according to schema",
                                "parameters": schema_json,
                            }
                        }
                    ]
                    completion_kwargs["tool_choice"] = {"type": "function", "function": {"name": "format_response"}}
                else:
                    # For OpenAI and compatible models
                    completion_kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": response_schema.__name__,
                            "schema": response_schema.model_json_schema(),
                        }
                    }

            # Merge additional kwargs
            completion_kwargs.update(kwargs)

            # Call LiteLLM
            response = self.litellm.completion(**completion_kwargs)

            # Extract content
            provider = model.split("/")[0] if "/" in model else "unknown"
            if response_schema and provider == "gemini":
                # Extract from function call
                if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                    tool_call = response.choices[0].message.tool_calls[0]
                    content = tool_call.function.arguments
                    # Parse and validate against schema
                    parsed_content = response_schema.model_validate_json(content)
                else:
                    # Fallback: try to parse message content as JSON
                    content = response.choices[0].message.content
                    try:
                        parsed_content = response_schema.model_validate_json(content)
                    except Exception:
                        # Last resort: return raw content
                        parsed_content = content
            else:
                content = response.choices[0].message.content
                if response_schema:
                    try:
                        parsed_content = response_schema.model_validate_json(content)
                    except Exception as e:
                        raise LLMError(
                            f"Failed to parse response into {response_schema.__name__}: {e}",
                            details={"raw_content": content}
                        ) from e
                else:
                    parsed_content = content

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Extract token usage
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0

            return LLMResponse(
                content=parsed_content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                finish_reason=response.choices[0].finish_reason,
                raw_response=response,
            )

        except Exception as e:
            # Try fallback models if available
            if self.fallback_models and model == self.default_model:
                for fallback_model in self.fallback_models:
                    try:
                        # Retry with fallback
                        return self.complete(
                            messages=messages,
                            model=fallback_model,
                            response_schema=response_schema,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            **kwargs,
                        )
                    except Exception:
                        continue

            # All attempts failed
            raise LLMError(
                f"LLM completion failed: {str(e)}",
                details={
                    "model": model,
                    "error_type": type(e).__name__,
                    "fallbacks_tried": self.fallback_models,
                }
            ) from e

    def complete_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        """
        Call LLM with tool/function calling support.

        Args:
            messages: List of messages
            tools: List of tool definitions (OpenAI format)
            model: Optional model override
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            LLMResponse with tool calls if any
        """
        model = model or self.default_model
        start_time = time.perf_counter()

        try:
            response = self.litellm.completion(
                model=model,
                messages=messages,
                tools=tools,
                timeout=self.timeout,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Extract tool calls if present
            message = response.choices[0].message
            tool_calls = []
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                    }
                    for tc in message.tool_calls
                ]

            return LLMResponse(
                content=message.content or "",
                model=model,
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
                latency_ms=latency_ms,
                finish_reason=response.choices[0].finish_reason,
                tool_calls=tool_calls,
                raw_response=response,
            )

        except Exception as e:
            raise LLMError(
                f"LLM tool completion failed: {str(e)}",
                details={"model": model, "error_type": type(e).__name__}
            ) from e

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Count tokens for a given text using LiteLLM's token counter.

        Args:
            text: Text to count tokens for
            model: Model to use for tokenization

        Returns:
            Token count
        """
        model = model or self.default_model
        try:
            return self.litellm.token_counter(model=model, text=text)
        except Exception:
            # Fallback to rough estimate (1 token ≈ 4 chars)
            return len(text) // 4

    def get_model_info(self, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a model (context window, pricing, etc.).

        Args:
            model: Model to get info for

        Returns:
            Model information dictionary
        """
        model = model or self.default_model
        try:
            # LiteLLM has model cost tracking
            return self.litellm.get_model_info(model)
        except Exception:
            return {"model": model, "info": "unavailable"}
