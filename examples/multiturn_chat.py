"""
Advanced Example: Multi-Turn Chat with NOCP

This example demonstrates how to build a production-ready chatbot with:
- History management and compaction
- Tool calling during conversation
- Context optimization for long conversations
- State management across turns
"""

import asyncio
import sys
from datetime import datetime
from typing import List, Optional

sys.path.insert(0, '/home/user/nocp/src')

from pydantic import BaseModel, Field

from nocp.core.act import ToolExecutor
from nocp.core.assess import ContextManager
from nocp.core.cache import LRUCache
from nocp.models.contracts import (
    ToolRequest,
    ToolType,
    ContextData,
    ChatMessage,
    CompressionMethod
)


# ============================================================================
# Data Models
# ============================================================================

class ConversationTurn(BaseModel):
    """A single turn in the conversation."""
    user_message: str
    assistant_message: str
    tool_calls: List[dict] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    tokens_used: int = 0


class ChatSession(BaseModel):
    """A chat session with history."""
    session_id: str
    turns: List[ConversationTurn] = Field(default_factory=list)
    total_tokens: int = 0
    created_at: datetime = Field(default_factory=datetime.now)


# ============================================================================
# Chat System
# ============================================================================

class ChatBot:
    """Production-ready chatbot with NOCP optimizations."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.session = ChatSession(session_id=session_id)

        # Setup components
        self.cache = LRUCache(max_size=500, default_ttl=1800)
        self.executor = ToolExecutor(cache=self.cache)

        # Context manager for history compaction
        self.context_manager = ContextManager(
            compression_threshold=5000,  # Compress history after 5k tokens
            target_compression_ratio=0.30,  # Aggressive compression
            enable_litellm=False
        )

        # Register tools
        self._register_tools()

    def _register_tools(self):
        """Register tools available to the chatbot."""

        @self.executor.register_async_tool("search_web")
        async def search_web(query: str) -> dict:
            """Search the web for information."""
            # Simulate web search
            await asyncio.sleep(0.1)
            return {
                "query": query,
                "results": [
                    {"title": f"Result about {query}", "snippet": f"Information about {query}..."},
                ]
            }

        @self.executor.register_async_tool("get_weather")
        async def get_weather(location: str) -> dict:
            """Get current weather for a location."""
            # Simulate weather API call
            await asyncio.sleep(0.05)
            return {
                "location": location,
                "temperature": 72,
                "condition": "Sunny",
                "humidity": 45
            }

        @self.executor.register_async_tool("calculate")
        async def calculate(expression: str) -> dict:
            """Perform a calculation."""
            try:
                # In production, use safer evaluation
                result = eval(expression, {"__builtins__": {}}, {})
                return {"expression": expression, "result": result}
            except Exception as e:
                return {"expression": expression, "error": str(e)}

        @self.executor.register_async_tool("set_reminder")
        async def set_reminder(task: str, time: str) -> dict:
            """Set a reminder."""
            return {
                "task": task,
                "time": time,
                "status": "Reminder set successfully"
            }

    def _detect_intent(self, message: str) -> Optional[dict]:
        """
        Detect user intent and determine which tool to call.

        In production, this would use an NLU model or LLM for intent detection.
        """
        message_lower = message.lower()

        # Simple keyword-based intent detection
        if any(word in message_lower for word in ['weather', 'temperature', 'forecast']):
            # Extract location (simplified)
            words = message.split()
            for i, word in enumerate(words):
                if word.lower() in ['in', 'at', 'for']:
                    if i + 1 < len(words):
                        location = words[i + 1].strip('.,?!')
                        return {
                            "tool_id": "get_weather",
                            "parameters": {"location": location}
                        }
            return {"tool_id": "get_weather", "parameters": {"location": "San Francisco"}}

        elif any(word in message_lower for word in ['search', 'find', 'look up', 'google']):
            # Extract search query
            for trigger in ['search for', 'find', 'look up', 'google']:
                if trigger in message_lower:
                    query = message_lower.split(trigger, 1)[1].strip('.,?!')
                    return {
                        "tool_id": "search_web",
                        "parameters": {"query": query}
                    }

        elif any(word in message_lower for word in ['calculate', 'compute', '+', '-', '*', '/']):
            # Extract mathematical expression
            for trigger in ['calculate', 'compute', 'what is', 'what\'s']:
                if trigger in message_lower:
                    expression = message_lower.split(trigger, 1)[-1].strip('.,?!')
                    return {
                        "tool_id": "calculate",
                        "parameters": {"expression": expression}
                    }

        elif any(word in message_lower for word in ['remind', 'reminder', 'schedule']):
            # Extract reminder details (simplified)
            return {
                "tool_id": "set_reminder",
                "parameters": {
                    "task": message,
                    "time": "later"  # Simplified
                }
            }

        return None

    async def _execute_tool(self, tool_info: dict) -> dict:
        """Execute a tool based on detected intent."""
        request = ToolRequest(
            tool_id=tool_info["tool_id"],
            tool_type=ToolType.API_CALL,
            function_name=tool_info["tool_id"],
            parameters=tool_info["parameters"]
        )

        result = await self.executor.execute_async(request)

        return {
            "tool": tool_info["tool_id"],
            "parameters": tool_info["parameters"],
            "result": result.data,
            "execution_time_ms": result.execution_time_ms
        }

    def _build_message_history(self) -> List[ChatMessage]:
        """Build message history from conversation turns."""
        messages = []

        for turn in self.session.turns:
            messages.append(ChatMessage(
                role="user",
                content=turn.user_message,
                timestamp=turn.timestamp,
                tokens=len(turn.user_message) // 4
            ))

            messages.append(ChatMessage(
                role="assistant",
                content=turn.assistant_message,
                timestamp=turn.timestamp,
                tokens=len(turn.assistant_message) // 4
            ))

        return messages

    async def chat(self, user_message: str) -> str:
        """
        Process a chat message with tool calling and history management.

        Steps:
        1. Detect if tools need to be called
        2. Execute tools if needed
        3. Optimize conversation history
        4. Generate response
        5. Update session
        """
        print(f"\n{'='*60}")
        print(f"User: {user_message}")
        print('-'*60)

        tool_calls = []

        # Step 1: Detect intent and execute tools
        intent = self._detect_intent(user_message)
        if intent:
            print(f"[Detected intent: {intent['tool_id']}]")
            tool_result = await self._execute_tool(intent)
            tool_calls.append(tool_result)
            print(f"[Tool executed: {tool_result['tool']} in {tool_result['execution_time_ms']:.2f}ms]")

        # Step 2: Optimize conversation history if needed
        messages = self._build_message_history()

        if len(messages) > 0:
            context = ContextData(
                tool_results=[],
                message_history=messages,
                max_tokens=100_000
            )

            optimized = self.context_manager.optimize(context)

            if optimized.method_used != CompressionMethod.NONE:
                print(f"[History compressed: {optimized.original_tokens} → {optimized.optimized_tokens} tokens]")
                print(f"[Method: {optimized.method_used.value}]")

        # Step 3: Generate response
        # In production, this would call an actual LLM
        if tool_calls:
            tool_result = tool_calls[0]['result']
            if 'weather' in tool_calls[0]['tool']:
                response = f"The weather in {tool_result['location']} is {tool_result['condition']} with a temperature of {tool_result['temperature']}°F."
            elif 'search' in tool_calls[0]['tool']:
                response = f"I found some information about {tool_result['query']}: {tool_result['results'][0]['snippet']}"
            elif 'calculate' in tool_calls[0]['tool']:
                response = f"The result of {tool_result['expression']} is {tool_result.get('result', 'error')}"
            elif 'remind' in tool_calls[0]['tool']:
                response = f"I've set a reminder for: {tool_result['task']}"
            else:
                response = f"I processed your request using {tool_calls[0]['tool']}"
        else:
            # Simple conversational response
            response = f"I understand you said: '{user_message}'. How can I help you further?"

        # Step 4: Update session
        turn = ConversationTurn(
            user_message=user_message,
            assistant_message=response,
            tool_calls=tool_calls,
            timestamp=datetime.now(),
            tokens_used=(len(user_message) + len(response)) // 4
        )

        self.session.turns.append(turn)
        self.session.total_tokens += turn.tokens_used

        print(f"Assistant: {response}")
        print('-'*60)
        print(f"[Turn {len(self.session.turns)} | Total tokens: {self.session.total_tokens}]")
        print('='*60)

        return response

    def get_session_summary(self) -> dict:
        """Get summary of the current session."""
        cache_stats = self.cache.stats()

        return {
            "session_id": self.session_id,
            "turn_count": len(self.session.turns),
            "total_tokens": self.session.total_tokens,
            "tool_calls": sum(len(turn.tool_calls) for turn in self.session.turns),
            "cache_hit_rate": cache_stats['hit_rate'],
            "duration": (datetime.now() - self.session.created_at).total_seconds()
        }


# ============================================================================
# Example Usage
# ============================================================================

async def main():
    """Demonstrate multi-turn chat with various intents."""
    print("="*60)
    print("Multi-Turn Chat Example with NOCP")
    print("="*60)

    # Create chatbot
    bot = ChatBot(session_id="demo_session_001")

    # Example conversation
    conversation = [
        "Hello! What can you help me with?",
        "What's the weather like in New York?",
        "Can you search for information about Python programming?",
        "Calculate 25 * 4 + 10",
        "Set a reminder to call mom tomorrow",
        "Thanks for your help!",
        "What was the weather you told me about earlier?",  # Test memory
        "Can you calculate 100 / 5?",
        "What other things can you do?",
        "That's all, thank you!",
    ]

    # Process conversation
    for i, message in enumerate(conversation, 1):
        print(f"\n\n=== Turn {i}/{len(conversation)} ===")
        await bot.chat(message)

        # Small delay for readability
        await asyncio.sleep(0.3)

    # Print session summary
    print("\n\n" + "="*60)
    print("Session Summary")
    print("="*60)

    summary = bot.get_session_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\n" + "="*60)
    print("Multi-Turn Chat Example Complete!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
