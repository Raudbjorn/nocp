"""
Performance report generator for benchmark results.

Generates:
- Markdown reports with tables
- ASCII charts for visualization
- Summary statistics
- Success criteria validation
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from dataclasses import asdict

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def generate_ascii_bar_chart(
    data: Dict[str, float],
    width: int = 50,
    title: str = "Chart",
) -> str:
    """
    Generate ASCII bar chart.

    Args:
        data: Dictionary of label: value pairs
        width: Maximum bar width
        title: Chart title

    Returns:
        ASCII bar chart string
    """
    if not data:
        return "No data available"

    max_value = max(data.values())
    if max_value == 0:
        max_value = 1

    lines = [title, "=" * (width + 30), ""]

    for label, value in data.items():
        bar_length = int((value / max_value) * width)
        bar = "█" * bar_length
        lines.append(f"{label:20s} | {bar} {value:.1f}")

    lines.append("")
    return "\n".join(lines)


def generate_markdown_table(
    headers: List[str],
    rows: List[List[Any]],
) -> str:
    """
    Generate markdown table.

    Args:
        headers: Table headers
        rows: Table rows

    Returns:
        Markdown table string
    """
    # Create header
    table = "| " + " | ".join(str(h) for h in headers) + " |\n"

    # Create separator
    table += "| " + " | ".join("---" for _ in headers) + " |\n"

    # Create rows
    for row in rows:
        table += "| " + " | ".join(str(cell) for cell in row) + " |\n"

    return table


class ReportGenerator:
    """
    Generates comprehensive performance reports from benchmark results.
    """

    def __init__(self, results_dir: str = "./benchmarks/results"):
        """
        Initialize report generator.

        Args:
            results_dir: Directory containing benchmark results
        """
        self.results_dir = Path(results_dir)

    def load_results(self) -> List[Dict[str, Any]]:
        """
        Load all benchmark results from directory.

        Returns:
            List of result dictionaries
        """
        results = []

        for result_file in self.results_dir.glob("*.json"):
            if result_file.name == "summary_report.json":
                continue

            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                    results.append(result)
            except (json.JSONDecodeError, IOError, OSError) as e:
                logger.warning(f"Failed to load or parse {result_file}: {e}")

        return results

    def load_summary(self) -> Dict[str, Any]:
        """
        Load summary report.

        Returns:
            Summary report dictionary
        """
        summary_file = self.results_dir / "summary_report.json"

        if not summary_file.exists():
            return {}

        with open(summary_file, 'r') as f:
            return json.load(f)

    def generate_overview_section(self, summary: Dict[str, Any]) -> str:
        """
        Generate overview section of report.

        Args:
            summary: Summary report data

        Returns:
            Markdown overview section
        """
        if not summary:
            return "## Overview\n\nNo summary data available.\n\n"

        metrics = summary.get("metrics", {})

        md = "## Overview\n\n"
        md += f"**Report Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
        md += f"**Total Benchmarks:** {summary.get('total_benchmarks', 0)}\n\n"

        md += "### Average Performance Metrics\n\n"

        # Create metrics table
        headers = ["Metric", "Mean", "Median", "Min", "Max"]
        rows = []

        if "input_token_reduction" in metrics:
            m = metrics["input_token_reduction"]
            rows.append([
                "Input Token Reduction (%)",
                f"{m['mean']:.1f}",
                f"{m['median']:.1f}",
                f"{m['min']:.1f}",
                f"{m['max']:.1f}",
            ])

        if "output_token_reduction" in metrics:
            m = metrics["output_token_reduction"]
            rows.append([
                "Output Token Reduction (%)",
                f"{m['mean']:.1f}",
                f"{m['median']:.1f}",
                f"{m['min']:.1f}",
                f"{m['max']:.1f}",
            ])

        if "cost_reduction" in metrics:
            m = metrics["cost_reduction"]
            rows.append([
                "Cost Reduction (%)",
                f"{m['mean']:.1f}",
                f"{m['median']:.1f}",
                f"{m['min']:.1f}",
                f"{m['max']:.1f}",
            ])

        if "latency_overhead_ratio" in metrics:
            m = metrics["latency_overhead_ratio"]
            rows.append([
                "Latency Overhead Ratio",
                f"{m['mean']:.2f}x",
                f"{m['median']:.2f}x",
                f"{m['min']:.2f}x",
                f"{m['max']:.2f}x",
            ])

        md += generate_markdown_table(headers, rows)
        md += "\n"

        return md

    def generate_success_criteria_section(self, summary: Dict[str, Any]) -> str:
        """
        Generate success criteria section.

        Args:
            summary: Summary report data

        Returns:
            Markdown success criteria section
        """
        if not summary or "success_criteria" not in summary:
            return "## Success Criteria\n\nNo criteria data available.\n\n"

        criteria = summary["success_criteria"]

        md = "## Success Criteria\n\n"
        md += "The roadmap specifies the following success criteria:\n\n"

        # Create criteria table
        headers = ["Criterion", "Target", "Met", "Total", "Percentage", "Status"]
        rows = []

        criteria_targets = {
            "input_reduction_>50%": "Input token reduction > 50%",
            "output_reduction_>30%": "Output token reduction > 30%",
            "latency_<2x": "End-to-end latency < 2x baseline",
            "cost_reduction_>40%": "Cost reduction > 40%",
        }

        for key, target in criteria_targets.items():
            if key in criteria:
                c = criteria[key]
                status_emoji = "✅" if c["status"] == "PASS" else "⚠️"
                rows.append([
                    target,
                    key.split("_")[1] if "_" in key else "",
                    c["met"],
                    c["total"],
                    f"{c['percentage']:.0f}%",
                    f"{status_emoji} {c['status']}",
                ])

        md += generate_markdown_table(headers, rows)
        md += "\n"

        # Overall assessment
        all_passed = all(
            criteria.get(k, {}).get("status") == "PASS"
            for k in criteria_targets.keys()
        )

        if all_passed:
            md += "### ✅ Overall Assessment: **PASS**\n\n"
            md += "All success criteria have been met!\n\n"
        else:
            md += "### ⚠️ Overall Assessment: **PARTIAL**\n\n"
            md += "Some success criteria have not been fully met. See details above.\n\n"

        return md

    def generate_scenario_breakdown_section(
        self,
        summary: Dict[str, Any],
        results: List[Dict[str, Any]],
    ) -> str:
        """
        Generate scenario breakdown section.

        Args:
            summary: Summary report data
            results: Individual benchmark results

        Returns:
            Markdown scenario breakdown section
        """
        md = "## Results by Scenario\n\n"

        scenarios_summary = summary.get("results_by_scenario", {})

        for scenario_name, scenario_data in scenarios_summary.items():
            md += f"### {scenario_name.replace('_', ' ').title()}\n\n"

            md += f"**Number of benchmarks:** {scenario_data['count']}\n\n"

            # Get detailed results for this scenario
            scenario_results = [
                r for r in results
                if r.get("benchmark_name") == scenario_name
            ]

            if scenario_results:
                # Create detailed table
                headers = ["Size", "Input Reduction", "Output Reduction", "Cost Reduction", "Latency Ratio"]
                rows = []

                for result in sorted(scenario_results, key=lambda r: r.get("dataset_size", "")):
                    size_emoji = {
                        "small": "🔹",
                        "medium": "🔸",
                        "large": "🔶",
                    }.get(result.get("dataset_size", ""), "")

                    rows.append([
                        f"{size_emoji} {result.get('dataset_size', 'N/A').title()}",
                        f"{result.get('input_reduction_pct', 0):.1f}%",
                        f"{result.get('output_reduction_pct', 0):.1f}%",
                        f"{result.get('cost_reduction_pct', 0):.1f}%",
                        f"{result.get('latency_overhead_ratio', 0):.2f}x",
                    ])

                md += generate_markdown_table(headers, rows)
                md += "\n"

            # Add bar chart for this scenario
            if scenario_results:
                chart_data = {
                    result.get("dataset_size", "unknown"): result.get("input_reduction_pct", 0)
                    for result in scenario_results
                }
                md += "```\n"
                md += generate_ascii_bar_chart(
                    chart_data,
                    title=f"{scenario_name} - Input Token Reduction by Size"
                )
                md += "```\n\n"

        return md

    def generate_detailed_results_section(self, results: List[Dict[str, Any]]) -> str:
        """
        Generate detailed results section.

        Args:
            results: Individual benchmark results

        Returns:
            Markdown detailed results section
        """
        md = "## Detailed Results\n\n"

        # Create comprehensive table
        headers = [
            "Scenario",
            "Size",
            "Raw Input",
            "Optimized Input",
            "Input Reduction",
            "Raw Output",
            "Optimized Output",
            "Output Reduction",
            "Cost Savings",
            "Latency Ratio",
        ]

        rows = []

        for result in results:
            rows.append([
                result.get("benchmark_name", "N/A"),
                result.get("dataset_size", "N/A"),
                f"{result.get('raw_input_tokens', 0):,}",
                f"{result.get('optimized_input_tokens', 0):,}",
                f"{result.get('input_reduction_pct', 0):.1f}%",
                f"{result.get('raw_output_tokens', 0):,}",
                f"{result.get('optimized_output_tokens', 0):,}",
                f"{result.get('output_reduction_pct', 0):.1f}%",
                f"${result.get('cost_savings_usd', 0):.4f}",
                f"{result.get('latency_overhead_ratio', 0):.2f}x",
            ])

        md += generate_markdown_table(headers, rows)
        md += "\n"

        return md

    def generate_kpi_tracking_section(self, summary: Dict[str, Any]) -> str:
        """
        Generate KPI tracking section with charts.

        Args:
            summary: Summary report data

        Returns:
            Markdown KPI tracking section
        """
        md = "## Key Performance Indicators (KPIs)\n\n"

        metrics = summary.get("metrics", {})

        # Token reduction KPI
        if "input_token_reduction" in metrics and "output_token_reduction" in metrics:
            md += "### Token Reduction\n\n"

            chart_data = {
                "Input Tokens": metrics["input_token_reduction"]["mean"],
                "Output Tokens": metrics["output_token_reduction"]["mean"],
            }

            md += "```\n"
            md += generate_ascii_bar_chart(chart_data, title="Average Token Reduction (%)")
            md += "```\n\n"

        # Cost savings KPI
        if "cost_reduction" in metrics:
            md += "### Cost Savings\n\n"
            md += f"**Average Cost Reduction:** {metrics['cost_reduction']['mean']:.1f}%\n\n"

            md += "```\n"
            chart_data = {
                "Cost Reduction": metrics['cost_reduction']['mean'],
                "Target (40%)": 40.0,
            }
            md += generate_ascii_bar_chart(chart_data, title="Cost Reduction vs Target")
            md += "```\n\n"

        # Latency KPI
        if "latency_overhead_ratio" in metrics:
            md += "### Latency Overhead\n\n"
            md += f"**Average Latency Ratio:** {metrics['latency_overhead_ratio']['mean']:.2f}x\n\n"
            md += f"**Target:** < 2.0x baseline\n\n"

            if metrics['latency_overhead_ratio']['mean'] < 2.0:
                md += "✅ **Status:** Within target\n\n"
            else:
                md += "⚠️ **Status:** Exceeds target\n\n"

        return md

    def generate_full_report(self, output_file: str = "performance_report.md") -> str:
        """
        Generate complete performance report.

        Args:
            output_file: Output file path

        Returns:
            Path to generated report
        """
        # Load data
        results = self.load_results()
        summary = self.load_summary()

        # Generate report
        md = "# NOCP Performance Benchmark Report\n\n"

        md += self.generate_overview_section(summary)
        md += self.generate_success_criteria_section(summary)
        md += self.generate_kpi_tracking_section(summary)
        md += self.generate_scenario_breakdown_section(summary, results)
        md += self.generate_detailed_results_section(results)

        # Add footer
        md += "---\n\n"
        md += f"*Report generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}*\n"

        # Save report
        output_path = self.results_dir / output_file

        with open(output_path, 'w') as f:
            f.write(md)

        logger.info(f"Performance report generated: {output_path}")

        return str(output_path)


def main():
    """Generate performance report from benchmark results."""
    generator = ReportGenerator()
    report_path = generator.generate_full_report()
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
