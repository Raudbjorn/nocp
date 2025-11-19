"""
Unit tests for LLM Client and Model Router.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pydantic import BaseModel

from nocp.llm.client import LLMClient
from nocp.llm.router import (
    ModelRouter,
    ModelConfig,
    ModelTier,
    RequestComplexity,
)
from nocp.models.contracts import LLMResponse


class SampleSchema(BaseModel):
    """Sample Pydantic schema for testing."""
    name: str
    age: int


class TestLLMClient:
    """Tests for LLMClient."""

    def test_client_initialization(self):
        """Test that client initializes correctly."""
        with patch('nocp.llm.client.litellm') as mock_litellm:
            client = LLMClient(
                default_model="gemini/gemini-2.0-flash-exp",
                max_retries=3,
                timeout=60,
            )
            assert client.default_model == "gemini/gemini-2.0-flash-exp"
            assert client.max_retries == 3
            assert client.timeout == 60

    def test_client_missing_litellm(self):
        """Test that client raises error if litellm not available."""
        with patch('nocp.llm.client.litellm', side_effect=ImportError):
            with pytest.raises(ImportError) as exc_info:
                # This import will fail
                from nocp.llm.client import LLMClient
                LLMClient()
            # The error should mention litellm and installation instructions
            error_message = str(exc_info.value).lower()
            assert "litellm" in error_message or "install" in error_message

    @patch('nocp.llm.client.litellm')
    def test_complete_basic(self, mock_litellm):
        """Test basic completion without structured output."""
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello, world!"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        mock_litellm.completion.return_value = mock_response

        client = LLMClient(default_model="gemini/gemini-2.0-flash-exp")
        client.litellm = mock_litellm

        response = client.complete(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
        )

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello, world!"
        assert response.input_tokens == 10
        assert response.output_tokens == 5
        assert response.finish_reason == "stop"

    @patch('nocp.llm.client.litellm')
    def test_complete_with_tools(self, mock_litellm):
        """Test completion with tool calling."""
        # Setup mock response with tool calls
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"location": "NYC"}'

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = ""
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_response.choices[0].finish_reason = "tool_calls"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 10

        mock_litellm.completion.return_value = mock_response

        client = LLMClient(default_model="openai/gpt-4o")
        client.litellm = mock_litellm

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                }
            }
        ]

        response = client.complete_with_tools(
            messages=[{"role": "user", "content": "What's the weather in NYC?"}],
            tools=tools,
        )

        assert isinstance(response, LLMResponse)
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "get_weather"
        assert response.tool_calls[0]["arguments"]["location"] == "NYC"

    @patch('nocp.llm.client.litellm')
    def test_count_tokens(self, mock_litellm):
        """Test token counting."""
        mock_litellm.token_counter.return_value = 25

        client = LLMClient()
        client.litellm = mock_litellm

        count = client.count_tokens("This is a test message")
        assert count == 25

    @patch('nocp.llm.client.litellm')
    def test_count_tokens_fallback(self, mock_litellm):
        """Test token counting with fallback."""
        mock_litellm.token_counter.side_effect = Exception("Token counter failed")

        client = LLMClient()
        client.litellm = mock_litellm

        # Should fall back to character-based estimate
        count = client.count_tokens("This is a test")  # 14 chars -> ~3 tokens
        assert count > 0

    @patch('nocp.llm.client.litellm')
    def test_fallback_model_on_failure(self, mock_litellm):
        """Test that client falls back to alternative models on failure."""
        # Setup: First call fails, second succeeds
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Fallback response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        # First call fails, second succeeds
        mock_litellm.completion.side_effect = [
            Exception("Primary model failed"),
            mock_response
        ]

        client = LLMClient(
            default_model="primary/model",
            fallback_models=["fallback/model"]
        )
        client.litellm = mock_litellm

        response = client.complete(
            messages=[{"role": "user", "content": "Test"}]
        )

        # Should succeed with fallback model
        assert isinstance(response, LLMResponse)
        assert response.content == "Fallback response"
        # Should have tried twice (primary + fallback)
        assert mock_litellm.completion.call_count == 2


class TestModelRouter:
    """Tests for ModelRouter."""

    def test_router_initialization(self):
        """Test that router initializes with default models."""
        router = ModelRouter()
        assert len(router.models) > 0
        assert "gemini/gemini-2.0-flash-exp" in router.models

    def test_register_model(self):
        """Test registering a custom model."""
        router = ModelRouter()
        config = ModelConfig(
            name="test/test-model",
            tier=ModelTier.STANDARD,
            input_cost_per_million=1.0,
            output_cost_per_million=2.0,
            max_input_tokens=100_000,
            max_output_tokens=4_096,
        )
        router.register_model(config)
        assert "test/test-model" in router.models
        assert router.models["test/test-model"].tier == ModelTier.STANDARD

    def test_select_model_by_complexity(self):
        """Test model selection based on complexity."""
        router = ModelRouter()

        # Trivial tasks should get ultra cheap models
        model = router.select_model(complexity=RequestComplexity.TRIVIAL)
        config = router.get_model_info(model)
        assert config.tier in [ModelTier.ULTRA_CHEAP, ModelTier.CHEAP]

        # Expert tasks should get premium/flagship models
        model = router.select_model(complexity=RequestComplexity.EXPERT)
        config = router.get_model_info(model)
        assert config.tier in [ModelTier.PREMIUM, ModelTier.FLAGSHIP]

    def test_select_model_with_requirements(self):
        """Test model selection with specific requirements."""
        router = ModelRouter()

        # Require structured output
        model = router.select_model(
            complexity=RequestComplexity.SIMPLE,
            requires_structured_output=True,
        )
        config = router.get_model_info(model)
        assert config.supports_structured_output is True

        # Require large context window
        model = router.select_model(
            complexity=RequestComplexity.MODERATE,
            max_input_tokens=500_000,
        )
        config = router.get_model_info(model)
        assert config.max_input_tokens >= 500_000

    def test_score_complexity(self):
        """Test complexity scoring heuristic."""
        router = ModelRouter()

        # No tools, no history -> trivial
        score = router.score_complexity(num_tools=0, message_history_length=0)
        assert score == RequestComplexity.TRIVIAL

        # Few tools -> simple
        score = router.score_complexity(num_tools=1, message_history_length=0)
        assert score in [RequestComplexity.SIMPLE, RequestComplexity.MODERATE]

        # Planning required -> expert
        score = router.score_complexity(
            num_tools=3,
            message_history_length=15,
            requires_planning=True,
        )
        assert score in [RequestComplexity.COMPLEX, RequestComplexity.EXPERT]

    def test_calculate_cost(self):
        """Test cost calculation."""
        router = ModelRouter()

        # Get a known model
        model = "gemini/gemini-1.5-flash"
        cost = router.calculate_cost(
            model_name=model,
            input_tokens=1_000_000,
            output_tokens=500_000,
        )

        # Should be non-zero for known model
        assert cost > 0

        # Cost should scale with tokens
        cost2 = router.calculate_cost(
            model_name=model,
            input_tokens=2_000_000,
            output_tokens=1_000_000,
        )
        assert cost2 > cost

    def test_prefer_cheap_models(self):
        """Test that prefer_cheap selects cheaper models."""
        router = ModelRouter()

        cheap_model = router.select_model(
            complexity=RequestComplexity.SIMPLE,
            prefer_cheap=True,
        )
        expensive_model = router.select_model(
            complexity=RequestComplexity.SIMPLE,
            prefer_cheap=False,
        )

        cheap_config = router.get_model_info(cheap_model)
        expensive_config = router.get_model_info(expensive_model)

        # The expensive model should cost more (or equal if only one option)
        cheap_cost = cheap_config.input_cost_per_million + cheap_config.output_cost_per_million
        expensive_cost = expensive_config.input_cost_per_million + expensive_config.output_cost_per_million
        assert expensive_cost >= cheap_cost

    def test_get_fallback_model(self):
        """Test that _get_fallback_model returns fallback when no models match."""
        # Create router with no models
        router = ModelRouter()
        router.models = {}
        router._tier_mapping = {tier: [] for tier in ModelTier}

        # Should return hardcoded fallback
        fallback = router._get_fallback_model()
        assert fallback == "gemini/gemini-1.5-flash"

        # Add a different model
        router.register_model(ModelConfig(
            name="custom/test-model",
            tier=ModelTier.CHEAP,
            input_cost_per_million=0.1,
            output_cost_per_million=0.2,
            max_input_tokens=100_000,
            max_output_tokens=4_096,
        ))

        # Should return first available model (not the preferred gemini-2.0)
        fallback = router._get_fallback_model()
        assert fallback == "custom/test-model"

        # Add preferred model
        router.register_model(ModelConfig(
            name="gemini/gemini-2.0-flash-exp",
            tier=ModelTier.CHEAP,
            input_cost_per_million=0.0,
            output_cost_per_million=0.0,
            max_input_tokens=1_000_000,
            max_output_tokens=8_192,
        ))

        # Should return preferred fallback
        fallback = router._get_fallback_model()
        assert fallback == "gemini/gemini-2.0-flash-exp"

    def test_search_adjacent_tiers(self):
        """Test that _search_adjacent_tiers finds models in nearby tiers."""
        router = ModelRouter()

        # Clear all models and add specific test models
        router.models = {}
        router._tier_mapping = {tier: [] for tier in ModelTier}

        # Add a model only in PREMIUM tier
        premium_model = ModelConfig(
            name="test/premium-model",
            tier=ModelTier.PREMIUM,
            input_cost_per_million=5.0,
            output_cost_per_million=10.0,
            max_input_tokens=200_000,
            max_output_tokens=8_192,
            supports_tools=True,
            supports_structured_output=True,
        )
        router.register_model(premium_model)

        # Search from STANDARD tier (should find PREMIUM at offset +1)
        candidates = router._search_adjacent_tiers(
            tier=ModelTier.STANDARD,
            requires_tools=False,
            requires_structured_output=False,
            max_input_tokens=None,
        )

        assert len(candidates) > 0
        assert candidates[0].name == "test/premium-model"

        # Search with structured output requirement
        candidates = router._search_adjacent_tiers(
            tier=ModelTier.STANDARD,
            requires_tools=False,
            requires_structured_output=True,
            max_input_tokens=None,
        )

        assert len(candidates) > 0
        assert candidates[0].supports_structured_output is True

        # Search with requirement that model doesn't meet
        candidates = router._search_adjacent_tiers(
            tier=ModelTier.STANDARD,
            requires_tools=False,
            requires_structured_output=False,
            max_input_tokens=500_000,  # Exceeds model's 200k limit
        )

        # Should not find the premium model
        assert len(candidates) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
