"""
Configuration management for the NOCP proxy agent.

Loads settings from environment variables and provides a centralized
configuration object for all components.
"""

import os
import logging
from typing import Optional, Dict, List, ClassVar, FrozenSet
from pathlib import Path
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..models.enums import OutputFormat, LogLevel, CompressionStrategy, LLMProvider

logger = logging.getLogger(__name__)


class ProxyConfig(BaseSettings):
    """
    Main configuration class for the High-Efficiency Proxy Agent.

    Loads configuration from environment variables with sensible defaults
    based on the architectural blueprint.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Secret fields that should be excluded from exports by default
    SECRET_FIELDS: ClassVar[FrozenSet[str]] = frozenset({
        'gemini_api_key',
        'openai_api_key',
        'anthropic_api_key',
    })

    # Gemini API Configuration
    gemini_api_key: str = Field(..., description="Google Gemini API key")
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Primary Gemini model to use"
    )

    # Context Window Limits (Gemini 2.5 Flash defaults)
    max_input_tokens: int = Field(
        default=1_048_576,
        description="Maximum input tokens (1M for Gemini 2.5 Flash)"
    )
    max_output_tokens: int = Field(
        default=65_535,
        description="Maximum output tokens for Gemini 2.5 Flash"
    )

    # Dynamic Compression Configuration
    default_compression_threshold: int = Field(
        default=5000,
        description="Default T_comp: activate compression above this token count"
    )
    compression_cost_multiplier: float = Field(
        default=1.5,
        description="Minimum savings multiplier to justify compression overhead"
    )

    # Student Summarizer Configuration
    student_summarizer_model: str = Field(
        default="gemini-1.5-flash-8b",
        description="Lightweight model for knowledge distillation"
    )
    student_summarizer_max_tokens: int = Field(
        default=2000,
        description="Maximum output tokens for student summarizer"
    )

    # Compression Strategy Configuration
    compression_strategies: List[CompressionStrategy] = Field(
        default=[
            CompressionStrategy.SEMANTIC_PRUNING,
            CompressionStrategy.KNOWLEDGE_DISTILLATION,
            CompressionStrategy.HISTORY_COMPACTION,
        ],
        description="List of enabled compression strategies"
    )

    # Legacy boolean flags (deprecated, but kept for backward compatibility)
    enable_semantic_pruning: bool = Field(
        default=True,
        description="[DEPRECATED] Use compression_strategies instead. Enable semantic pruning for RAG/document outputs"
    )
    enable_knowledge_distillation: bool = Field(
        default=True,
        description="[DEPRECATED] Use compression_strategies instead. Enable knowledge distillation via student summarizer"
    )
    enable_history_compaction: bool = Field(
        default=True,
        description="[DEPRECATED] Use compression_strategies instead. Enable conversation history compaction"
    )

    # Output Serialization Configuration
    default_output_format: OutputFormat = Field(
        default=OutputFormat.TOON,
        description="Default output format: toon, compact_json, or json"
    )
    toon_fallback_threshold: float = Field(
        default=0.3,
        description="Tabularity threshold below which to fallback to compact JSON"
    )
    enable_format_negotiation: bool = Field(
        default=True,
        description="Enable automatic format negotiation based on data structure"
    )

    # Monitoring and Logging
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level"
    )
    enable_metrics_logging: bool = Field(
        default=True,
        description="Enable detailed metrics logging"
    )
    metrics_log_file: Path = Field(
        default=Path("./logs/metrics.jsonl"),
        description="Path to metrics log file"
    )
    drift_detection_threshold: float = Field(
        default=-1000.0,
        description="Threshold for drift detection warning (negative delta trend)"
    )

    # Log Rotation Configuration
    log_file: Optional[Path] = Field(
        default=None,
        description="Path to main application log file (set to None to disable file logging)"
    )
    log_max_bytes: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum log file size before rotation"
    )
    log_backup_count: int = Field(
        default=5,
        description="Number of backup log files to keep"
    )

    # Multi-Cloud Configuration (LiteLLM)
    enable_litellm: bool = Field(
        default=True,
        description="Enable LiteLLM for multi-cloud routing"
    )
    litellm_default_model: str = Field(
        default="gemini/gemini-2.0-flash-exp",
        description="Default model in LiteLLM format (provider/model)"
    )
    litellm_fallback_models: Optional[str] = Field(
        default="gemini/gemini-1.5-flash,openai/gpt-4o-mini",
        description="Comma-separated list of fallback models"
    )
    litellm_max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for LiteLLM calls"
    )
    litellm_timeout: int = Field(
        default=60,
        description="Request timeout in seconds for LiteLLM"
    )

    # OpenAI API Configuration (Optional for multi-cloud)
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for multi-cloud routing"
    )

    # Anthropic API Configuration (Optional for multi-cloud)
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key for multi-cloud routing"
    )

    # Tool-specific compression thresholds (runtime registry)
    _compression_thresholds: Dict[str, int] = {}

    @field_validator('default_compression_threshold')
    @classmethod
    def validate_compression_threshold(cls, v: int) -> int:
        """Ensure compression threshold is reasonable"""
        if v < 1000:
            raise ValueError(
                f"default_compression_threshold ({v}) is too low. "
                "Minimum recommended: 1000 tokens"
            )
        if v > 100_000:
            logger.warning(
                f"Very high default_compression_threshold ({v:,}). "
                "Compression may rarely trigger."
            )
        return v

    @field_validator('compression_cost_multiplier')
    @classmethod
    def validate_compression_cost_multiplier(cls, v: float) -> float:
        """Ensure compression cost multiplier is valid"""
        if v < 1.0:
            raise ValueError(
                f"compression_cost_multiplier must be >= 1.0, got {v}. "
                "Values < 1.0 would accept compression even when it increases cost."
            )
        if v > 10.0:
            logger.warning(
                f"Very high compression_cost_multiplier ({v}). "
                "Compression may be rejected even when beneficial."
            )
        return v

    @field_validator('toon_fallback_threshold')
    @classmethod
    def validate_toon_threshold(cls, v: float) -> float:
        """Validate TOON fallback threshold"""
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"toon_fallback_threshold must be 0.0-1.0, got {v}"
            )
        return v

    @field_validator('student_summarizer_max_tokens')
    @classmethod
    def validate_student_max_tokens(cls, v: int) -> int:
        """Ensure student summarizer max tokens is reasonable"""
        if v < 100:
            raise ValueError(
                f"student_summarizer_max_tokens ({v}) is too low. "
                "Minimum recommended: 100 tokens"
            )
        if v > 10_000:
            logger.warning(
                f"Very high student_summarizer_max_tokens ({v:,}). "
                "This may reduce compression effectiveness."
            )
        return v

    @field_validator('max_input_tokens', 'max_output_tokens')
    @classmethod
    def validate_token_limits(cls, v: int) -> int:
        """Ensure token limits are positive and reasonable"""
        if v <= 0:
            raise ValueError(
                f"Token limit must be positive, got {v}"
            )
        if v > 10_000_000:
            logger.warning(
                f"Very high token limit ({v:,}). "
                "Ensure this matches your model's capabilities."
            )
        return v

    @model_validator(mode='after')
    def sync_compression_strategies(self) -> 'ProxyConfig':
        """
        Sync legacy boolean flags with compression_strategies for backward compatibility.

        This validator ensures a single source of truth by checking if legacy boolean
        flags were explicitly set and updating the compression_strategies list accordingly.
        This prevents inconsistent behavior where legacy flags and the new list could
        conflict.

        Returns:
            Updated ProxyConfig instance with synchronized configuration
        """
        strategies = set(self.compression_strategies)

        # Check if legacy flags were explicitly set by the user via environment variables
        if "enable_semantic_pruning" in self.model_fields_set:
            if self.enable_semantic_pruning:
                strategies.add(CompressionStrategy.SEMANTIC_PRUNING)
            else:
                strategies.discard(CompressionStrategy.SEMANTIC_PRUNING)

        if "enable_knowledge_distillation" in self.model_fields_set:
            if self.enable_knowledge_distillation:
                strategies.add(CompressionStrategy.KNOWLEDGE_DISTILLATION)
            else:
                strategies.discard(CompressionStrategy.KNOWLEDGE_DISTILLATION)

        if "enable_history_compaction" in self.model_fields_set:
            if self.enable_history_compaction:
                strategies.add(CompressionStrategy.HISTORY_COMPACTION)
            else:
                strategies.discard(CompressionStrategy.HISTORY_COMPACTION)

        # Update the strategies list (sorted for deterministic behavior)
        self.compression_strategies = sorted(list(strategies), key=lambda s: s.value)
        return self

    @model_validator(mode='after')
    def validate_cross_field_constraints(self) -> 'ProxyConfig':
        """Cross-field validation of configuration constraints"""
        # Check if max_output_tokens exceeds max_input_tokens
        if self.max_output_tokens > self.max_input_tokens:
            logger.warning(
                f"max_output_tokens ({self.max_output_tokens:,}) > "
                f"max_input_tokens ({self.max_input_tokens:,}). "
                "This may cause issues with some models."
            )

        # Check if compression threshold exceeds max input tokens
        if self.default_compression_threshold > self.max_input_tokens:
            raise ValueError(
                f"default_compression_threshold ({self.default_compression_threshold:,}) "
                f"exceeds max_input_tokens ({self.max_input_tokens:,}). "
                "Compression would never trigger."
            )

        # Check if student summarizer output is reasonable compared to compression threshold
        if self.student_summarizer_max_tokens > self.default_compression_threshold:
            logger.warning(
                f"student_summarizer_max_tokens ({self.student_summarizer_max_tokens:,}) > "
                f"default_compression_threshold ({self.default_compression_threshold:,}). "
                "Student summarizer may produce outputs larger than compression trigger."
            )

        return self

    def is_strategy_enabled(self, strategy: CompressionStrategy) -> bool:
        """
        Check if a specific compression strategy is enabled.

        Args:
            strategy: The compression strategy to check

        Returns:
            True if the strategy is in the enabled list
        """
        return strategy in self.compression_strategies

    def register_tool_threshold(self, tool_name: str, threshold: int) -> None:
        """
        Register a tool-specific compression threshold.

        Args:
            tool_name: Name of the tool
            threshold: Compression threshold in tokens
        """
        self._compression_thresholds[tool_name] = threshold

    def get_compression_threshold(self, tool_name: Optional[str] = None) -> int:
        """
        Get compression threshold for a specific tool or default.

        Args:
            tool_name: Optional tool name for custom threshold

        Returns:
            Compression threshold in tokens
        """
        if tool_name and tool_name in self._compression_thresholds:
            return self._compression_thresholds[tool_name]
        return self.default_compression_threshold

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        DEPRECATED: Calculate estimated cost in USD for a request.

        Cost tracking has been removed in favor of token efficiency metrics.
        This method is kept for backward compatibility but always returns 0.0.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            0.0 (cost tracking deprecated)
        """
        import warnings
        warnings.warn(
            "calculate_cost() is deprecated and will be removed in v2.0. "
            "Focus on token reduction metrics instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return 0.0

    def ensure_log_directory(self) -> None:
        """Ensure the log directories exist."""
        for log_path in [self.metrics_log_file, self.log_file]:
            if log_path and log_path.parent:
                log_path.parent.mkdir(parents=True, exist_ok=True)


# Global configuration instance
_config: Optional[ProxyConfig] = None


def get_config() -> ProxyConfig:
    """
    Get the global configuration instance.

    Returns:
        ProxyConfig instance

    Raises:
        ValueError: If configuration hasn't been initialized
    """
    global _config
    if _config is None:
        _config = ProxyConfig()
        _config.ensure_log_directory()
    return _config


def reset_config() -> None:
    """Reset the global configuration (mainly for testing)."""
    global _config
    _config = None
