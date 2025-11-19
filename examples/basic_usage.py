"""
Basic usage example for nocp.

Demonstrates the complete Act -> Assess -> Articulate pipeline.
"""

from nocp.core import ContextManager, OutputSerializer, ToolExecutor
from nocp.models.contracts import (
    ContextData,
    SerializationRequest,
    ToolRequest,
    ToolType,
)
from pydantic import BaseModel


# Define response schema
class User(BaseModel):
    id: str
    name: str
    email: str


class UserList(BaseModel):
    users: list[User]
    total: int


def main():
    print("🚀 nocp - High-Efficiency LLM Proxy Agent Demo\n")

    # ========================================================================
    # Step 1: ACT - Execute Tools
    # ========================================================================
    print("📋 Step 1: ACT (Tool Execution)")
    print("-" * 50)

    executor = ToolExecutor()

    @executor.register_tool("fetch_users")
    def fetch_users(count: int) -> list:
        """Simulate fetching users from a database."""
        return [
            {"id": str(i), "name": f"User{i}", "email": f"user{i}@example.com"}
            for i in range(count)
        ]

    # Execute tool
    request = ToolRequest(
        tool_id="fetch_users",
        tool_type=ToolType.PYTHON_FUNCTION,
        function_name="fetch_users",
        parameters={"count": 100},
    )

    tool_result = executor.execute(request)

    print(f"✓ Executed tool: {tool_result.tool_id}")
    print(f"  Success: {tool_result.success}")
    print(f"  Execution time: {tool_result.execution_time_ms:.2f}ms")
    print(f"  Token estimate: {tool_result.token_estimate} tokens")
    print(f"  Data size: {len(tool_result.data)} users")

    # ========================================================================
    # Step 2: ASSESS - Compress Context
    # ========================================================================
    print("\n🔍 Step 2: ASSESS (Context Compression)")
    print("-" * 50)

    manager = ContextManager(compression_threshold=1000, enable_litellm=False)  # Disabled for demo

    context = ContextData(
        tool_results=[tool_result],
        transient_context={"query": "Fetch all active users"},
        max_tokens=50_000,
    )

    optimized = manager.optimize(context)

    print(f"✓ Compression method: {optimized.method_used.value}")
    print(f"  Original tokens: {optimized.original_tokens}")
    print(f"  Optimized tokens: {optimized.optimized_tokens}")
    print(f"  Compression ratio: {optimized.compression_ratio:.2%}")
    print(f"  Token savings: {optimized.original_tokens - optimized.optimized_tokens}")
    print(f"  Cost savings: ${optimized.estimated_cost_savings:.6f}")
    print(f"  Compression time: {optimized.compression_time_ms:.2f}ms")

    # ========================================================================
    # Step 3: ARTICULATE - Optimize Output Serialization
    # ========================================================================
    print("\n📝 Step 3: ARTICULATE (Output Serialization)")
    print("-" * 50)

    # Create a structured response
    response_data = UserList(
        users=[User(**user) for user in tool_result.data[:10]],  # First 10 users
        total=len(tool_result.data),
    )

    serializer = OutputSerializer()
    serialization_request = SerializationRequest(
        data=response_data, include_length_markers=True, validate_output=True
    )

    serialized = serializer.serialize(serialization_request)

    print(f"✓ Serialization format: {serialized.format_used.value}")
    print(f"  Schema complexity: {serialized.schema_complexity}")
    print(f"  Original tokens: {serialized.original_tokens}")
    print(f"  Optimized tokens: {serialized.optimized_tokens}")
    print(f"  Savings ratio: {serialized.savings_ratio:.2%}")
    print(f"  Token savings: {serialized.original_tokens - serialized.optimized_tokens}")
    print(f"  Serialization time: {serialized.serialization_time_ms:.2f}ms")
    print(f"  Valid: {serialized.is_valid}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)

    total_input_savings = optimized.original_tokens - optimized.optimized_tokens
    total_output_savings = serialized.original_tokens - serialized.optimized_tokens
    total_cost_savings = optimized.estimated_cost_savings

    print(
        f"Input Token Reduction: {total_input_savings} tokens ({optimized.compression_ratio:.1%})"
    )
    print(f"Output Token Reduction: {total_output_savings} tokens ({serialized.savings_ratio:.1%})")
    print(f"Total Cost Savings: ${total_cost_savings:.6f}")
    print(
        f"Total Latency Overhead: {optimized.compression_time_ms + serialized.serialization_time_ms:.2f}ms"
    )

    print("\n✅ Pipeline completed successfully!")

    # Optional: Show serialized output sample
    print("\n📄 Serialized Output Sample:")
    print("-" * 50)
    print(serialized.serialized_text[:500])
    if len(serialized.serialized_text) > 500:
        print("...")


if __name__ == "__main__":
    main()
