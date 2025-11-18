"""
Token counting utilities using Gemini's CountTokens API.

This module provides a centralized interface for token counting,
which is fundamental to the dynamic compression policy.
"""

import json
from typing import Any, Dict, List, Optional
import google.generativeai as genai

from ..core.config import get_config


class TokenCounter:
    """
    Token counter using Gemini's CountTokens API.

    This is a critical component for the dynamic compression policy,
    enabling the agent to measure token usage before submission.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize token counter.

        Args:
            model_name: Gemini model name (defaults to config value)
        """
        config = get_config()
        self.model_name = model_name or config.gemini_model

        # Configure Gemini API
        genai.configure(api_key=config.gemini_api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def count_text(self, text: str) -> int:
        """
        Count tokens in a text string.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        try:
            result = self.model.count_tokens(text)
            return result.total_tokens
        except Exception as e:
            # Fallback to character-based estimation if API fails
            # Rough approximation: ~4 characters per token
            return len(text) // 4

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        """
        Count tokens in a list of messages.

        Args:
            messages: List of message dictionaries. Each message should contain a 'content' key,
                      but alternative keys like 'text', 'message', etc. are also supported.

        Returns:
            Total number of tokens
        """
        def extract_content(msg: Dict[str, Any]) -> str:
            # Try common keys for message content
            for key in ("content", "text", "message", "body"):
                if key in msg and isinstance(msg[key], str):
                    return msg[key]
            # If no valid key found, return empty string
            return ""

        try:
            # Format messages for Gemini, supporting alternative schemas
            contents = [extract_content(msg) for msg in messages]
            combined_text = "\n".join(contents)
            return self.count_text(combined_text)
        except Exception as e:
            return sum(len(extract_content(msg)) // 4 for msg in messages)

    def count_structured(self, data: Dict[str, Any]) -> int:
        """
        Count tokens in structured data (dict/JSON).

        Args:
            data: Structured data to count

        Returns:
            Number of tokens
        """
        try:
            json_str = json.dumps(data)
            return self.count_text(json_str)
        except Exception as e:
            # Fallback estimation
            return len(str(data)) // 4

    def will_fit(
        self,
        current_tokens: int,
        additional_tokens: int,
        max_tokens: Optional[int] = None,
    ) -> bool:
        """
        Check if additional tokens will fit in the context window.

        Args:
            current_tokens: Current token count
            additional_tokens: Tokens to add
            max_tokens: Maximum tokens (defaults to config)

        Returns:
            True if additional tokens will fit
        """
        config = get_config()
        limit = max_tokens or config.max_input_tokens
        return (current_tokens + additional_tokens) <= limit

    def get_budget_remaining(
        self,
        current_tokens: int,
        max_tokens: Optional[int] = None,
    ) -> int:
        """
        Get remaining token budget.

        Args:
            current_tokens: Current token count
            max_tokens: Maximum tokens (defaults to config)

        Returns:
            Remaining tokens available
        """
        config = get_config()
        limit = max_tokens or config.max_input_tokens
        return max(0, limit - current_tokens)
