"""
Model Router - Cost-based routing logic for LLM selection.

Implements intelligent model selection based on:
- Request complexity scoring
- Cost optimization
- Tiered model configuration
"""

from enum import Enum

from pydantic import BaseModel, Field


class ModelTier(str, Enum):
    """Model tier categories based on capability and cost."""

    ULTRA_CHEAP = "ultra_cheap"  # For simple summarization, basic tasks
    CHEAP = "cheap"  # For most compression tasks
    STANDARD = "standard"  # For general agent tasks
    PREMIUM = "premium"  # For complex reasoning
    FLAGSHIP = "flagship"  # For most demanding tasks


class ModelConfig(BaseModel):
    """Configuration for a specific model."""

    name: str = Field(..., description="Model name in LiteLLM format")
    tier: ModelTier
    input_cost_per_million: float = Field(..., description="Cost per 1M input tokens (USD)")
    output_cost_per_million: float = Field(..., description="Cost per 1M output tokens (USD)")
    max_input_tokens: int = Field(..., description="Maximum input context window")
    max_output_tokens: int = Field(..., description="Maximum output tokens")
    supports_tools: bool = Field(default=True, description="Supports function calling")
    supports_structured_output: bool = Field(default=False, description="Native structured output")


class RequestComplexity(str, Enum):
    """Complexity levels for routing decisions."""

    TRIVIAL = "trivial"  # Simple compression, summarization
    SIMPLE = "simple"  # Basic tool usage, Q&A
    MODERATE = "moderate"  # Multi-step reasoning
    COMPLEX = "complex"  # Complex tool orchestration
    EXPERT = "expert"  # Advanced reasoning, planning


class ModelRouter:
    """
    Routes requests to appropriate models based on complexity and cost.

    Example:
        router = ModelRouter()
        router.register_model(ModelConfig(
            name="gemini/gemini-2.0-flash-exp",
            tier=ModelTier.CHEAP,
            input_cost_per_million=0.075,
            output_cost_per_million=0.30,
            max_input_tokens=1_048_576,
            max_output_tokens=8_192,
        ))

        model = router.select_model(complexity=RequestComplexity.SIMPLE)
    """

    def __init__(self):
        self.models: dict[str, ModelConfig] = {}
        self._tier_mapping: dict[ModelTier, list[str]] = {tier: [] for tier in ModelTier}
        self._initialize_defaults()

    def _initialize_defaults(self):
        """Initialize with common model configurations."""
        default_models = [
            # Gemini models
            ModelConfig(
                name="gemini/gemini-2.0-flash-exp",
                tier=ModelTier.CHEAP,
                input_cost_per_million=0.0,  # Free during preview
                output_cost_per_million=0.0,
                max_input_tokens=1_048_576,
                max_output_tokens=8_192,
                supports_tools=True,
                supports_structured_output=False,
            ),
            ModelConfig(
                name="gemini/gemini-1.5-flash",
                tier=ModelTier.CHEAP,
                input_cost_per_million=0.075,
                output_cost_per_million=0.30,
                max_input_tokens=1_048_576,
                max_output_tokens=8_192,
                supports_tools=True,
                supports_structured_output=False,
            ),
            ModelConfig(
                name="gemini/gemini-1.5-flash-8b",
                tier=ModelTier.ULTRA_CHEAP,
                input_cost_per_million=0.0375,
                output_cost_per_million=0.15,
                max_input_tokens=1_048_576,
                max_output_tokens=8_192,
                supports_tools=True,
                supports_structured_output=False,
            ),
            ModelConfig(
                name="gemini/gemini-1.5-pro",
                tier=ModelTier.PREMIUM,
                input_cost_per_million=1.25,
                output_cost_per_million=5.00,
                max_input_tokens=2_097_152,
                max_output_tokens=8_192,
                supports_tools=True,
                supports_structured_output=False,
            ),
            # OpenAI models
            ModelConfig(
                name="openai/gpt-4o-mini",
                tier=ModelTier.CHEAP,
                input_cost_per_million=0.15,
                output_cost_per_million=0.60,
                max_input_tokens=128_000,
                max_output_tokens=16_384,
                supports_tools=True,
                supports_structured_output=True,
            ),
            ModelConfig(
                name="openai/gpt-4o",
                tier=ModelTier.STANDARD,
                input_cost_per_million=2.50,
                output_cost_per_million=10.00,
                max_input_tokens=128_000,
                max_output_tokens=16_384,
                supports_tools=True,
                supports_structured_output=True,
            ),
            # Claude models
            ModelConfig(
                name="anthropic/claude-3-haiku-20240307",
                tier=ModelTier.CHEAP,
                input_cost_per_million=0.25,
                output_cost_per_million=1.25,
                max_input_tokens=200_000,
                max_output_tokens=4_096,
                supports_tools=True,
                supports_structured_output=False,
            ),
            ModelConfig(
                name="anthropic/claude-3-5-sonnet-20241022",
                tier=ModelTier.PREMIUM,
                input_cost_per_million=3.00,
                output_cost_per_million=15.00,
                max_input_tokens=200_000,
                max_output_tokens=8_192,
                supports_tools=True,
                supports_structured_output=False,
            ),
        ]

        for model in default_models:
            self.register_model(model)

    def register_model(self, config: ModelConfig):
        """
        Register a model configuration.

        Args:
            config: ModelConfig to register
        """
        self.models[config.name] = config
        self._tier_mapping[config.tier].append(config.name)

    def select_model(
        self,
        complexity: RequestComplexity = RequestComplexity.SIMPLE,
        requires_tools: bool = False,
        requires_structured_output: bool = False,
        max_input_tokens: int | None = None,
        prefer_cheap: bool = True,
    ) -> str:
        """
        Select the most appropriate model based on requirements.

        Args:
            complexity: Request complexity level
            requires_tools: Whether tool calling is required
            requires_structured_output: Whether native structured output is needed
            max_input_tokens: Minimum required input token limit
            prefer_cheap: Prefer cheaper models when multiple options available

        Returns:
            Model name (LiteLLM format)
        """
        # Map complexity to tier
        tier = self._complexity_to_tier(complexity)

        # Find suitable models
        candidates = []
        for model_name in self._tier_mapping[tier]:
            model = self.models[model_name]

            # Check requirements
            if requires_tools and not model.supports_tools:
                continue
            if requires_structured_output and not model.supports_structured_output:
                continue
            if max_input_tokens and model.max_input_tokens < max_input_tokens:
                continue

            candidates.append(model)

        # If no candidates in preferred tier, try adjacent tiers
        if not candidates:
            candidates = self._search_adjacent_tiers(
                tier,
                requires_tools,
                requires_structured_output,
                max_input_tokens,
            )

        # If still no candidates, use fallback
        if not candidates:
            return self._get_fallback_model()

        # Sort by cost (input + output average)
        if prefer_cheap:
            candidates.sort(key=lambda m: m.input_cost_per_million + m.output_cost_per_million)
        else:
            # Prefer more capable models
            candidates.sort(key=lambda m: -(m.input_cost_per_million + m.output_cost_per_million))

        return candidates[0].name

    def _complexity_to_tier(self, complexity: RequestComplexity) -> ModelTier:
        """Map complexity to model tier."""
        mapping = {
            RequestComplexity.TRIVIAL: ModelTier.ULTRA_CHEAP,
            RequestComplexity.SIMPLE: ModelTier.CHEAP,
            RequestComplexity.MODERATE: ModelTier.STANDARD,
            RequestComplexity.COMPLEX: ModelTier.PREMIUM,
            RequestComplexity.EXPERT: ModelTier.FLAGSHIP,
        }
        return mapping.get(complexity, ModelTier.STANDARD)

    def _search_adjacent_tiers(
        self,
        tier: ModelTier,
        requires_tools: bool,
        requires_structured_output: bool,
        max_input_tokens: int | None,
    ) -> list[ModelConfig]:
        """Search adjacent tiers for suitable models."""
        tier_order = [
            ModelTier.ULTRA_CHEAP,
            ModelTier.CHEAP,
            ModelTier.STANDARD,
            ModelTier.PREMIUM,
            ModelTier.FLAGSHIP,
        ]

        current_idx = tier_order.index(tier)
        candidates = []

        # Search up and down
        for offset in [1, -1, 2, -2]:
            idx = current_idx + offset
            if 0 <= idx < len(tier_order):
                search_tier = tier_order[idx]
                for model_name in self._tier_mapping[search_tier]:
                    model = self.models[model_name]

                    if requires_tools and not model.supports_tools:
                        continue
                    if requires_structured_output and not model.supports_structured_output:
                        continue
                    if max_input_tokens and model.max_input_tokens < max_input_tokens:
                        continue

                    candidates.append(model)

                if candidates:
                    return candidates

        return candidates

    def _get_fallback_model(self) -> str:
        """Get fallback model when no suitable model found."""
        # Default to gemini-2.0-flash-exp (free during preview)
        if "gemini/gemini-2.0-flash-exp" in self.models:
            return "gemini/gemini-2.0-flash-exp"
        # Otherwise use first available model
        return list(self.models.keys())[0] if self.models else "gemini/gemini-1.5-flash"

    def score_complexity(
        self,
        num_tools: int = 0,
        message_history_length: int = 0,
        has_multi_step_reasoning: bool = False,
        requires_planning: bool = False,
    ) -> RequestComplexity:
        """
        Score request complexity based on heuristics.

        Args:
            num_tools: Number of tools required
            message_history_length: Length of conversation history
            has_multi_step_reasoning: Whether multi-step reasoning needed
            requires_planning: Whether planning/decomposition needed

        Returns:
            RequestComplexity score
        """
        score = 0

        # Tool usage complexity
        if num_tools == 0:
            score += 0
        elif num_tools == 1:
            score += 1
        elif num_tools <= 3:
            score += 2
        else:
            score += 3

        # Conversation complexity
        if message_history_length > 10:
            score += 1
        if message_history_length > 20:
            score += 1

        # Reasoning requirements
        if has_multi_step_reasoning:
            score += 2
        if requires_planning:
            score += 2

        # Map score to complexity
        if score == 0:
            return RequestComplexity.TRIVIAL
        elif score <= 2:
            return RequestComplexity.SIMPLE
        elif score <= 4:
            return RequestComplexity.MODERATE
        elif score <= 6:
            return RequestComplexity.COMPLEX
        else:
            return RequestComplexity.EXPERT

    def get_model_info(self, model_name: str) -> ModelConfig | None:
        """Get configuration for a specific model."""
        return self.models.get(model_name)

    def calculate_cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Calculate cost for a request.

        Args:
            model_name: Model to use
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        model = self.models.get(model_name)
        if not model:
            return 0.0

        input_cost = (input_tokens / 1_000_000) * model.input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * model.output_cost_per_million
        return input_cost + output_cost
