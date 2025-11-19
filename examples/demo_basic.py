"""
Basic demonstration of the High-Efficiency Proxy Agent.

This example shows:
1. Agent initialization
2. Tool registration
3. Processing a request with tool usage
4. Token optimization in action
5. Metrics analysis
"""

import os

from nocp import AgentRequest, HighEfficiencyProxyAgent
from nocp.tools import (
    create_analyze_data_tool,
    create_fetch_document_tool,
    create_search_database_tool,
)
from nocp.utils.logging import setup_logging


def main():
    """Run the basic demo."""
    print("=" * 80)
    print("NOCP - High-Efficiency LLM Proxy Agent Demo")
    print("=" * 80)
    print()

    # Setup logging
    setup_logging()

    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY environment variable not set")
        print("Please set your Gemini API key:")
        print("  export GEMINI_API_KEY='your-api-key-here'")
        print()
        print("Or create a .env file in the project root:")
        print("  GEMINI_API_KEY=your-api-key-here")
        return

    print("Initializing High-Efficiency Proxy Agent...")
    print()

    # Initialize agent
    agent = HighEfficiencyProxyAgent()

    print("Registering tools...")
    print()

    # Register example tools
    search_def, search_func = create_search_database_tool()
    agent.register_tool(search_def, search_func)

    analyze_def, analyze_func = create_analyze_data_tool()
    agent.register_tool(analyze_def, analyze_func)

    fetch_def, fetch_func = create_fetch_document_tool()
    agent.register_tool(fetch_def, fetch_func)

    tool_definitions = agent.tool_executor.get_tool_definitions()
    print(f"✓ Registered {len(tool_definitions)} tools:")
    for tool_def in tool_definitions:
        print(f"  - {tool_def.name}")
    print()

    # Create a request
    request = AgentRequest(
        query="Search for wireless headphones and provide a summary",
        session_id="demo-session-001",
        available_tools=["search_database"],
    )

    print("-" * 80)
    print("Processing Request")
    print("-" * 80)
    print(f"Query: {request.query}")
    print(f"Session: {request.session_id}")
    print()

    print("Executing agent pipeline...")
    print("  1. Route request and prepare context")
    print("  2. Execute tools (Act)")
    print("  3. Compress context (Assess)")
    print("  4. Generate LLM response")
    print("  5. Serialize output (Articulate)")
    print()

    try:
        # Process the request
        response, metrics = agent.process_request(request)

        print("-" * 80)
        print("Response")
        print("-" * 80)
        print(response)
        print()

        print("-" * 80)
        print("Performance Metrics")
        print("-" * 80)
        print(f"Transaction ID: {metrics.transaction_id}")
        print()

        print("Input Token Optimization:")
        print(f"  Raw Input Tokens:        {metrics.raw_input_tokens:,}")
        print(f"  Compressed Input Tokens: {metrics.compressed_input_tokens:,}")
        print(f"  Compression Ratio:       {metrics.input_compression_ratio:.2%}")
        print(
            f"  Token Savings:           {metrics.raw_input_tokens - metrics.compressed_input_tokens:,}"
        )
        print()

        print("Output Token Optimization:")
        print(f"  Raw Output Tokens:       {metrics.raw_output_tokens:,}")
        print(f"  Output Format:           {metrics.final_output_format}")
        print(f"  Token Savings:           {metrics.output_token_savings:,}")
        print()

        print("Latency Breakdown:")
        print(f"  Total Latency:           {metrics.total_latency_ms:.2f} ms")
        print(f"  LLM Inference:           {metrics.llm_inference_latency_ms:.2f} ms")
        print(f"  Compression:             {metrics.compression_latency_ms:.2f} ms")
        print()

        print("Cost Analysis:")
        print(f"  Estimated Cost:          ${metrics.estimated_cost_usd:.6f}")
        if metrics.estimated_savings_usd:
            print(f"  Estimated Savings:       ${metrics.estimated_savings_usd:.6f}")
            total_cost = metrics.estimated_cost_usd + metrics.estimated_savings_usd
            if total_cost > 0:
                savings_pct = (metrics.estimated_savings_usd / total_cost) * 100
                print(f"  Savings Percentage:      {savings_pct:.1f}%")
        print()

        if metrics.compression_operations:
            print("Compression Operations:")
            for i, comp in enumerate(metrics.compression_operations, 1):
                print(f"  Operation {i}:")
                print(f"    Method:         {comp.compression_method}")
                print(f"    Original:       {comp.original_tokens:,} tokens")
                print(f"    Compressed:     {comp.compressed_tokens:,} tokens")
                print(f"    Cost:           {comp.compression_cost_tokens:,} tokens")
                print(f"    Net Savings:    {comp.net_savings:,} tokens")
                print(f"    Ratio:          {comp.compression_ratio:.2%}")
            print()

        if metrics.tools_used:
            print(f"Tools Used: {', '.join(metrics.tools_used)}")
            print()

        print("-" * 80)
        print("Summary")
        print("-" * 80)
        print("The High-Efficiency Proxy Agent successfully:")
        print("✓ Executed tool calls with validated parameters")
        print("✓ Applied dynamic compression based on T_comp threshold")
        print("✓ Verified Cost-of-Compression Calculus")
        print("✓ Serialized output to token-efficient format")
        print("✓ Tracked comprehensive performance metrics")
        print()
        print("For production deployment, enable drift detection monitoring")
        print("to maintain optimal compression ratios over time.")
        print()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
