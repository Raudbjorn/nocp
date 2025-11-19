#!/usr/bin/env python3
"""
Benchmarking framework for NOCP.

Compares optimized vs baseline pipeline performance across different scenarios:
- RAG retrieval with large document sets
- API calls with verbose responses
- Database queries with structured results
- Multi-turn conversations

Generates comprehensive reports with metrics, charts, and analysis.
"""

import sys
import time
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from dataclasses import dataclass, asdict
import statistics

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nocp.models.schemas import TransactionLog
from nocp.models.context import TransientContext, PersistentContext, ConversationMessage
from nocp.modules.context_manager import ContextManager
from nocp.modules.output_serializer import OutputSerializer
from nocp.core.config import get_config
from nocp.utils.logging import get_logger
from nocp.utils.token_counter import TokenCounter


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    benchmark_name: str
    scenario: str
    dataset_size: Literal["small", "medium", "large"]

    # Input metrics
    raw_input_tokens: int
    optimized_input_tokens: int
    input_compression_ratio: float
    input_reduction_pct: float

    # Output metrics
    raw_output_tokens: int
    optimized_output_tokens: int
    output_compression_ratio: float
    output_reduction_pct: float

    # Performance metrics
    baseline_latency_ms: float
    optimized_latency_ms: float
    latency_overhead_ratio: float

    # Cost metrics (estimated)
    baseline_cost_usd: float
    optimized_cost_usd: float
    cost_savings_usd: float
    cost_reduction_pct: float

    # Success criteria
    meets_input_reduction_target: bool  # >50%
    meets_output_reduction_target: bool  # >30%
    meets_latency_target: bool  # <2x baseline
    meets_cost_reduction_target: bool  # >40%

    timestamp: str
    metadata: Dict[str, Any]


class BenchmarkRunner:
    """
    Runs benchmarks comparing baseline vs optimized pipelines.
    """

    def __init__(self, output_dir: str = "./benchmarks/results"):
        """
        Initialize benchmark runner.

        Args:
            output_dir: Directory for benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(__name__)
        self.token_counter = TokenCounter()
        self.context_manager = ContextManager()
        self.output_serializer = OutputSerializer()

        # Pricing (approximate, Gemini 2.0 Flash)
        self.input_cost_per_1m = 0.30  # $0.30 per 1M input tokens
        self.output_cost_per_1m = 1.20  # $1.20 per 1M output tokens

    def run_baseline_pipeline(
        self,
        input_data: str,
        output_data: Any,
    ) -> Dict[str, Any]:
        """
        Run baseline pipeline (no optimization).

        Args:
            input_data: Raw input data
            output_data: Raw output data

        Returns:
            Dictionary with metrics
        """
        start_time = time.perf_counter()

        # Count tokens
        input_tokens = self.token_counter.count_text(input_data)
        output_tokens = self.token_counter.count_text(json.dumps(output_data, default=str))

        # Simulate LLM processing (without actual API call)
        time.sleep(0.1)  # Simulate network/processing time

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Calculate cost
        cost_usd = (
            (input_tokens / 1_000_000) * self.input_cost_per_1m +
            (output_tokens / 1_000_000) * self.output_cost_per_1m
        )

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
        }

    def run_optimized_pipeline(
        self,
        input_data: str,
        output_data: Any,
        tool_name: str = "test_tool",
    ) -> Dict[str, Any]:
        """
        Run optimized pipeline (with compression and serialization).

        Args:
            input_data: Raw input data
            output_data: Raw output data
            tool_name: Tool name for compression strategy selection

        Returns:
            Dictionary with metrics
        """
        from nocp.models.schemas import ToolExecutionResult, SerializationRequest

        start_time = time.perf_counter()

        # Count raw tokens
        raw_input_tokens = self.token_counter.count_text(input_data)
        raw_output_tokens = self.token_counter.count_text(json.dumps(output_data, default=str))

        # Apply input compression
        tool_result = ToolExecutionResult(
            tool_name=tool_name,
            raw_output=input_data,
            raw_token_count=raw_input_tokens,
            execution_time_ms=0,
        )

        compressed_input, compression_result = self.context_manager.manage_tool_output(tool_result)
        optimized_input_tokens = (
            compression_result.compressed_tokens if compression_result
            else raw_input_tokens
        )

        # Apply output serialization (mock response)
        # In real scenario, this would be the structured LLM response
        # For benchmarking, we use the output_data directly
        if isinstance(output_data, dict):
            from pydantic import BaseModel, create_model
            from typing import Any

            # Create dynamic model
            # Using Any for field types to handle complex nested structures
            DynamicModel = create_model(
                'DynamicResponse',
                __base__=BaseModel,
                **{k: (Any, v) for k, v in output_data.items()}
            )
            serialization_request = SerializationRequest(data=DynamicModel(**output_data))
            serialized = self.output_serializer.serialize(serialization_request)
            optimized_output_tokens = serialized.optimized_tokens
        else:
            optimized_output_tokens = raw_output_tokens

        # Simulate LLM processing
        time.sleep(0.1)

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Calculate cost (using optimized tokens)
        cost_usd = (
            (optimized_input_tokens / 1_000_000) * self.input_cost_per_1m +
            (optimized_output_tokens / 1_000_000) * self.output_cost_per_1m
        )

        return {
            "raw_input_tokens": raw_input_tokens,
            "optimized_input_tokens": optimized_input_tokens,
            "raw_output_tokens": raw_output_tokens,
            "optimized_output_tokens": optimized_output_tokens,
            "total_tokens": optimized_input_tokens + optimized_output_tokens,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            "compression_result": compression_result,
        }

    def run_benchmark(
        self,
        benchmark_name: str,
        scenario: str,
        input_data: str,
        output_data: Any,
        dataset_size: Literal["small", "medium", "large"],
        tool_name: str = "test_tool",
    ) -> BenchmarkResult:
        """
        Run a single benchmark comparing baseline vs optimized.

        Args:
            benchmark_name: Name of benchmark
            scenario: Scenario description
            input_data: Input data
            output_data: Output data
            dataset_size: Size category
            tool_name: Tool name

        Returns:
            BenchmarkResult
        """
        self.logger.info(
            "running_benchmark",
            name=benchmark_name,
            scenario=scenario,
            dataset_size=dataset_size,
        )

        # Run baseline
        baseline = self.run_baseline_pipeline(input_data, output_data)

        # Run optimized
        optimized = self.run_optimized_pipeline(input_data, output_data, tool_name)

        # Calculate metrics
        input_reduction_pct = (
            (baseline["input_tokens"] - optimized["optimized_input_tokens"])
            / baseline["input_tokens"] * 100
            if baseline["input_tokens"] > 0 else 0
        )

        output_reduction_pct = (
            (baseline["output_tokens"] - optimized["optimized_output_tokens"])
            / baseline["output_tokens"] * 100
            if baseline["output_tokens"] > 0 else 0
        )

        cost_reduction_pct = (
            (baseline["cost_usd"] - optimized["cost_usd"])
            / baseline["cost_usd"] * 100
            if baseline["cost_usd"] > 0 else 0
        )

        latency_overhead_ratio = (
            optimized["latency_ms"] / baseline["latency_ms"]
            if baseline["latency_ms"] > 0 else float('inf')
        )

        # Check success criteria
        result = BenchmarkResult(
            benchmark_name=benchmark_name,
            scenario=scenario,
            dataset_size=dataset_size,
            raw_input_tokens=baseline["input_tokens"],
            optimized_input_tokens=optimized["optimized_input_tokens"],
            input_compression_ratio=(
                optimized["optimized_input_tokens"] / baseline["input_tokens"]
                if baseline["input_tokens"] > 0 else 0.0
            ),
            input_reduction_pct=input_reduction_pct,
            raw_output_tokens=baseline["output_tokens"],
            optimized_output_tokens=optimized["optimized_output_tokens"],
            output_compression_ratio=(
                optimized["optimized_output_tokens"] / baseline["output_tokens"]
                if baseline["output_tokens"] > 0 else 0.0
            ),
            output_reduction_pct=output_reduction_pct,
            baseline_latency_ms=baseline["latency_ms"],
            optimized_latency_ms=optimized["latency_ms"],
            latency_overhead_ratio=latency_overhead_ratio,
            baseline_cost_usd=baseline["cost_usd"],
            optimized_cost_usd=optimized["cost_usd"],
            cost_savings_usd=baseline["cost_usd"] - optimized["cost_usd"],
            cost_reduction_pct=cost_reduction_pct,
            meets_input_reduction_target=input_reduction_pct >= 50,
            meets_output_reduction_target=output_reduction_pct >= 30,
            meets_latency_target=latency_overhead_ratio < 2.0,
            meets_cost_reduction_target=cost_reduction_pct >= 40,
            timestamp=datetime.utcnow().isoformat(),
            metadata={
                "compression_method": optimized.get("compression_result", {}).compression_method if optimized.get("compression_result") else "none",
            },
        )

        self.logger.info(
            "benchmark_completed",
            name=benchmark_name,
            input_reduction=f"{input_reduction_pct:.1f}%",
            output_reduction=f"{output_reduction_pct:.1f}%",
            cost_reduction=f"{cost_reduction_pct:.1f}%",
        )

        return result

    def save_result(self, result: BenchmarkResult) -> str:
        """
        Save benchmark result to file.

        Args:
            result: Benchmark result

        Returns:
            Path to saved file
        """
        filename = f"{result.benchmark_name}_{result.dataset_size}_{int(time.time())}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(asdict(result), f, indent=2)

        self.logger.info("benchmark_result_saved", filepath=str(filepath))

        return str(filepath)


class BenchmarkSuite:
    """
    Complete benchmark suite with multiple scenarios.
    """

    def __init__(self, output_dir: str = "./benchmarks/results"):
        """
        Initialize benchmark suite.

        Args:
            output_dir: Directory for results
        """
        self.runner = BenchmarkRunner(output_dir)
        self.output_dir = Path(output_dir)
        self.results: List[BenchmarkResult] = []

    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """
        Run all benchmark scenarios.

        Returns:
            List of all benchmark results
        """
        from .test_datasets import (
            generate_rag_dataset,
            generate_api_dataset,
            generate_database_dataset,
        )

        self.results = []

        # RAG benchmarks
        for size in ["small", "medium", "large"]:
            dataset = generate_rag_dataset(size)
            result = self.runner.run_benchmark(
                benchmark_name="rag_retrieval",
                scenario=f"RAG document retrieval ({size} dataset)",
                input_data=dataset["input"],
                output_data=dataset["output"],
                dataset_size=size,
                tool_name="rag_search",
            )
            self.results.append(result)
            self.runner.save_result(result)

        # API call benchmarks
        for size in ["small", "medium", "large"]:
            dataset = generate_api_dataset(size)
            result = self.runner.run_benchmark(
                benchmark_name="api_call",
                scenario=f"API call with verbose response ({size} dataset)",
                input_data=dataset["input"],
                output_data=dataset["output"],
                dataset_size=size,
                tool_name="api_call",
            )
            self.results.append(result)
            self.runner.save_result(result)

        # Database query benchmarks
        for size in ["small", "medium", "large"]:
            dataset = generate_database_dataset(size)
            result = self.runner.run_benchmark(
                benchmark_name="database_query",
                scenario=f"Database query with structured results ({size} dataset)",
                input_data=dataset["input"],
                output_data=dataset["output"],
                dataset_size=size,
                tool_name="database_query",
            )
            self.results.append(result)
            self.runner.save_result(result)

        return self.results

    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate summary report across all benchmarks.

        Returns:
            Summary report dictionary
        """
        if not self.results:
            return {"error": "No results available"}

        # Aggregate metrics
        input_reductions = [r.input_reduction_pct for r in self.results]
        output_reductions = [r.output_reduction_pct for r in self.results]
        cost_reductions = [r.cost_reduction_pct for r in self.results]
        latency_ratios = [r.latency_overhead_ratio for r in self.results]

        # Success criteria checks
        input_target_met = sum(1 for r in self.results if r.meets_input_reduction_target)
        output_target_met = sum(1 for r in self.results if r.meets_output_reduction_target)
        latency_target_met = sum(1 for r in self.results if r.meets_latency_target)
        cost_target_met = sum(1 for r in self.results if r.meets_cost_reduction_target)

        total = len(self.results)

        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_benchmarks": total,
            "metrics": {
                "input_token_reduction": {
                    "mean": statistics.mean(input_reductions),
                    "median": statistics.median(input_reductions),
                    "min": min(input_reductions),
                    "max": max(input_reductions),
                },
                "output_token_reduction": {
                    "mean": statistics.mean(output_reductions),
                    "median": statistics.median(output_reductions),
                    "min": min(output_reductions),
                    "max": max(output_reductions),
                },
                "cost_reduction": {
                    "mean": statistics.mean(cost_reductions),
                    "median": statistics.median(cost_reductions),
                    "min": min(cost_reductions),
                    "max": max(cost_reductions),
                },
                "latency_overhead_ratio": {
                    "mean": statistics.mean(latency_ratios),
                    "median": statistics.median(latency_ratios),
                    "min": min(latency_ratios),
                    "max": max(latency_ratios),
                },
            },
            "success_criteria": {
                "input_reduction_>50%": {
                    "met": input_target_met,
                    "total": total,
                    "percentage": (input_target_met / total * 100),
                    "status": "PASS" if input_target_met == total else "PARTIAL",
                },
                "output_reduction_>30%": {
                    "met": output_target_met,
                    "total": total,
                    "percentage": (output_target_met / total * 100),
                    "status": "PASS" if output_target_met == total else "PARTIAL",
                },
                "latency_<2x": {
                    "met": latency_target_met,
                    "total": total,
                    "percentage": (latency_target_met / total * 100),
                    "status": "PASS" if latency_target_met == total else "PARTIAL",
                },
                "cost_reduction_>40%": {
                    "met": cost_target_met,
                    "total": total,
                    "percentage": (cost_target_met / total * 100),
                    "status": "PASS" if cost_target_met == total else "PARTIAL",
                },
            },
            "results_by_scenario": {},
        }

        # Group results by benchmark name
        by_scenario = {}
        for result in self.results:
            if result.benchmark_name not in by_scenario:
                by_scenario[result.benchmark_name] = []
            by_scenario[result.benchmark_name].append(result)

        for scenario_name, scenario_results in by_scenario.items():
            summary["results_by_scenario"][scenario_name] = {
                "count": len(scenario_results),
                "avg_input_reduction": statistics.mean(r.input_reduction_pct for r in scenario_results),
                "avg_output_reduction": statistics.mean(r.output_reduction_pct for r in scenario_results),
                "avg_cost_reduction": statistics.mean(r.cost_reduction_pct for r in scenario_results),
            }

        return summary


def main():
    """Run benchmark suite and generate reports."""
    print("=" * 80)
    print("NOCP Benchmark Suite")
    print("=" * 80)
    print()

    # Run benchmarks
    suite = BenchmarkSuite()

    print("Running all benchmarks...")
    results = suite.run_all_benchmarks()

    print(f"\nCompleted {len(results)} benchmarks")
    print()

    # Generate summary
    summary = suite.generate_summary_report()

    # Save summary
    summary_file = suite.output_dir / "summary_report.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("Summary Report")
    print("-" * 80)
    print(f"Total benchmarks: {summary['total_benchmarks']}")
    print()

    print("Average Reductions:")
    print(f"  Input tokens:  {summary['metrics']['input_token_reduction']['mean']:.1f}%")
    print(f"  Output tokens: {summary['metrics']['output_token_reduction']['mean']:.1f}%")
    print(f"  Cost:          {summary['metrics']['cost_reduction']['mean']:.1f}%")
    print(f"  Latency ratio: {summary['metrics']['latency_overhead_ratio']['mean']:.2f}x")
    print()

    print("Success Criteria:")
    for criterion, data in summary['success_criteria'].items():
        status_symbol = "✓" if data['status'] == "PASS" else "✗"
        print(f"  {status_symbol} {criterion}: {data['met']}/{data['total']} ({data['percentage']:.0f}%) - {data['status']}")
    print()

    print(f"Summary saved to: {summary_file}")
    print(f"Individual results saved to: {suite.output_dir}")
    print()


if __name__ == "__main__":
    main()
