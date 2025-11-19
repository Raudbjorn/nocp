"""Enhanced logging with Rich library"""

from typing import TYPE_CHECKING, Any, Optional

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.theme import Theme
from rich.traceback import install as install_rich_traceback
from rich.tree import Tree

if TYPE_CHECKING:
    from ..core.config import ProxyConfig
    from ..models.schemas import ContextMetrics

# Custom NOCP theme
NOCP_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "metric": "magenta",
        "token": "blue",
        "cost": "green",
        "savings": "bright_green",
        "latency": "cyan",
        "tool": "blue",
        "compression": "yellow",
    }
)


class NOCPConsole:
    """Singleton console with NOCP branding and theme"""

    _instance: Optional["NOCPConsole"] = None

    def __new__(cls) -> "NOCPConsole":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.console = Console(theme=NOCP_THEME)
            self.initialized = True

    def print_banner(self):
        """Print NOCP startup banner"""
        self.console.print(
            Panel.fit(
                "[bold cyan]NOCP[/bold cyan] - High-Efficiency LLM Proxy Agent\n"
                "[dim]Token Optimization • Cost Reduction • Smart Compression[/dim]",
                border_style="cyan",
            )
        )

    def print_config_summary(self, config: "ProxyConfig"):
        """Print configuration summary table"""
        table = Table(title="Configuration", show_header=False, border_style="cyan")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="yellow")

        # Core settings
        model_display = (
            config.litellm_default_model if config.enable_litellm else config.gemini_model
        )
        table.add_row("Model", model_display)
        table.add_row("Max Input Tokens", f"{config.max_input_tokens:,}")
        table.add_row("Max Output Tokens", f"{config.max_output_tokens:,}")
        table.add_row("Compression Threshold", f"{config.default_compression_threshold:,}")

        # Compression strategies
        strategies = []
        if config.enable_semantic_pruning:
            strategies.append("✓ Semantic Pruning")
        if config.enable_knowledge_distillation:
            strategies.append("✓ Knowledge Distillation")
        if config.enable_history_compaction:
            strategies.append("✓ History Compaction")

        if strategies:
            table.add_row("Strategies", ", ".join(strategies))

        # Output format
        table.add_row("Output Format", config.default_output_format.upper())

        # LiteLLM info
        if config.enable_litellm:
            table.add_row("Multi-Cloud", "Enabled (LiteLLM)")
            if config.litellm_fallback_models:
                fallbacks = config.litellm_fallback_models.split(",")
                table.add_row("Fallback Models", f"{len(fallbacks)} configured")

        self.console.print(table)

    def print_metrics(self, metrics: "ContextMetrics"):
        """Print transaction metrics in beautiful table"""
        table = Table(title="Transaction Metrics", show_header=True, border_style="cyan")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="yellow", justify="right")
        table.add_column("Details", style="dim")

        # Transaction ID
        table.add_row("Transaction ID", metrics.transaction_id[:8] + "...", "")

        # Token metrics
        input_savings = metrics.raw_input_tokens - metrics.compressed_input_tokens
        has_raw_input_tokens = metrics.raw_input_tokens > 0
        input_reduction_pct = (
            (input_savings / metrics.raw_input_tokens * 100) if has_raw_input_tokens else 0
        )
        table.add_row(
            "Input Tokens",
            f"{metrics.compressed_input_tokens:,}",
            f"[dim]saved {input_savings:,} ({input_reduction_pct:.1f}%)[/dim]",
        )

        # Output tokens
        output_savings = metrics.output_token_savings
        table.add_row(
            "Output Tokens",
            f"{metrics.raw_output_tokens:,}",
            f"[dim]saved {output_savings:,} via {metrics.final_output_format.upper()}[/dim]",
        )

        # Total savings
        total_savings = input_savings + output_savings
        table.add_row(
            "[bold]Total Token Savings[/bold]", f"[bold green]{total_savings:,}[/bold green]", ""
        )

        # Latency breakdown
        table.add_row(
            "Latency",
            f"{metrics.total_latency_ms:.0f}ms",
            f"[dim]compression: {metrics.compression_latency_ms:.0f}ms, "
            f"LLM: {metrics.llm_inference_latency_ms:.0f}ms[/dim]",
        )

        # Format and compression
        table.add_row(
            "Output Format",
            metrics.final_output_format.upper(),
            f"[dim]{len(metrics.tools_used)} tools used[/dim]",
        )

        if metrics.compression_operations:
            ops = ", ".join(
                [
                    f"{op.compression_method} ({op.compression_ratio:.0%})"
                    for op in metrics.compression_operations
                ]
            )
            table.add_row(
                "Compression", f"{len(metrics.compression_operations)} ops", f"[dim]{ops}[/dim]"
            )

        self.console.print(table)

    def print_operation_tree(self, operations: dict[str, Any]):
        """Print operation hierarchy as tree"""
        tree = Tree("[bold cyan]Request Pipeline[/bold cyan]")

        # Act phase
        act_branch = tree.add("[blue]⚡ Act[/blue] - Tool Execution")
        for tool in operations.get("tools", []):
            duration = tool.get("duration_ms", 0)
            tool_name = tool.get("name", "Unknown Tool")
            act_branch.add(f"✓ {tool_name} ({duration:.0f}ms)")

        # Assess phase
        assess_branch = tree.add("[yellow]🔍 Assess[/yellow] - Context Optimization")
        for compression in operations.get("compressions", []):
            ratio = compression.get("ratio", 1.0)
            method = compression.get("method", "unknown")
            reduction_pct = (1 - ratio) * 100
            assess_branch.add(f"✓ {method} " f"({reduction_pct:.0%} reduction)")

        # LLM phase
        llm_branch = tree.add("[magenta]🤖 LLM[/magenta] - Inference")
        llm_branch.add(f"Model: {operations.get('model', 'N/A')}")
        llm_branch.add(f"Tokens: {operations.get('tokens', 0):,}")

        # Articulate phase
        articulate_branch = tree.add("[green]📝 Articulate[/green] - Serialization")
        articulate_branch.add(f"Format: {operations.get('format', 'N/A')}")

        self.console.print(tree)

    def create_progress(self, description: str = "Processing") -> Progress:
        """Create progress bar for long operations"""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )

    def print_success(self, message: str):
        """Print success message"""
        self.console.print(f"[success]✓[/success] {message}")

    def print_error(self, message: str):
        """Print error message"""
        self.console.print(f"[error]✗[/error] {message}")

    def print_warning(self, message: str):
        """Print warning message"""
        self.console.print(f"[warning]⚠[/warning] {message}")

    def print_info(self, message: str):
        """Print info message"""
        self.console.print(f"[info]ℹ[/info] {message}")


# Global console instance
console = NOCPConsole()


def setup_rich_logging() -> None:
    """
    Setup Rich traceback formatting for better error messages.

    This installs Rich's enhanced traceback handler globally, which provides:
    - Syntax-highlighted code in tracebacks
    - Local variable inspection
    - Better formatting and readability

    Note: This function installs rich tracebacks globally. Call it once at
    application startup if you want enhanced error messages.

    Note: structlog configuration is handled separately in utils/logging.py.
    This function only handles rich traceback installation, not log formatting.
    """
    # Install rich traceback handler for better error messages
    install_rich_traceback(
        show_locals=True,
        width=120,
        extra_lines=3,
        theme="monokai",
        word_wrap=False,
        suppress=[structlog],
    )


def format_metrics_summary(metrics: "ContextMetrics") -> str:
    """
    Format metrics as a compact string for inline logging.

    Args:
        metrics: ContextMetrics to format

    Returns:
        Formatted metrics string
    """
    input_savings = metrics.raw_input_tokens - metrics.compressed_input_tokens
    total_savings = input_savings + metrics.output_token_savings

    return (
        f"[savings]↓ {total_savings:,} tokens[/savings] | "
        f"[latency]{metrics.total_latency_ms:.0f}ms[/latency] | "
        f"[compression]{len(metrics.compression_operations)} compressions[/compression]"
    )
