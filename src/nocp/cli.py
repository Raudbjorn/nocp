"""
Command-line interface for nocp.

Provides commands for setup, running scripts, testing, and benchmarking.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from . import __version__
from .bootstrap import get_uv_command

app = typer.Typer(
    name="nocp",
    help="High-Efficiency LLM Proxy Agent with token optimization",
    add_completion=False,
)

console = Console()


@app.command()
def version():
    """Show version information."""
    console.print(f"[bold cyan]nocp[/bold cyan] version {__version__}")
    console.print("High-Efficiency LLM Proxy Agent")


@app.command()
def setup(
    dev: bool = typer.Option(False, "--dev", help="Install development dependencies")
):
    """
    Initialize the project and install dependencies.

    This command uses uv to create a virtual environment and install all dependencies.
    """
    console.print("[bold]🚀 Setting up nocp...[/bold]")

    uv_cmd = get_uv_command()

    try:
        # Sync dependencies
        if dev:
            console.print("📦 Installing project with dev dependencies...")
            result = subprocess.run(
                [*uv_cmd, "sync", "--all-extras"],
                check=True,
                capture_output=False
            )
        else:
            console.print("📦 Installing project dependencies...")
            result = subprocess.run(
                [*uv_cmd, "sync"],
                check=True,
                capture_output=False
            )

        console.print("[bold green]✅ Setup complete![/bold green]")
        console.print("\n[dim]You can now run:[/dim]")
        console.print("  [cyan]./nocp --help[/cyan]         - Show available commands")
        console.print("  [cyan]./nocp run <script>[/cyan]   - Run a Python script")
        console.print("  [cyan]./nocp test[/cyan]           - Run tests")

    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]❌ Setup failed:[/bold red] {e}")
        sys.exit(1)


@app.command()
def run(
    script: Path = typer.Argument(..., help="Python script to run"),
    model: Optional[str] = typer.Option(None, "--model", help="Override default LLM model"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
):
    """
    Run a Python script with nocp environment.

    This ensures all dependencies are available and environment is configured.
    """
    if not script.exists():
        console.print(f"[bold red]❌ Script not found:[/bold red] {script}")
        sys.exit(1)

    uv_cmd = get_uv_command()

    # Build environment variables
    env = {}
    if model:
        env["NOCP_DEFAULT_MODEL"] = model
    if debug:
        env["NOCP_LOG_LEVEL"] = "DEBUG"

    try:
        # Run script via uv
        result = subprocess.run(
            [*uv_cmd, "run", "python", str(script)],
            env={**subprocess.os.environ, **env},
            check=True
        )
        sys.exit(result.returncode)

    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]❌ Script failed:[/bold red] {e}")
        sys.exit(1)


@app.command()
def test(
    path: Optional[str] = typer.Argument(None, help="Specific test file or directory"),
    cov: bool = typer.Option(False, "--cov", help="Generate coverage report"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
):
    """
    Run the test suite.

    Examples:
        ./nocp test                    - Run all tests
        ./nocp test tests/core/        - Run tests in specific directory
        ./nocp test --cov              - Run with coverage report
    """
    console.print("[bold]🧪 Running tests...[/bold]")

    uv_cmd = get_uv_command()

    # Build pytest command
    pytest_args = [*uv_cmd, "run", "pytest"]

    if path:
        pytest_args.append(path)

    if verbose:
        pytest_args.append("-v")

    if cov:
        pytest_args.extend(["--cov=src/nocp", "--cov-report=term", "--cov-report=html"])

    try:
        result = subprocess.run(pytest_args, check=True)
        console.print("[bold green]✅ Tests passed![/bold green]")

        if cov:
            console.print("\n[dim]Coverage report generated at:[/dim] htmlcov/index.html")

        sys.exit(result.returncode)

    except subprocess.CalledProcessError:
        console.print("[bold red]❌ Tests failed![/bold red]")
        sys.exit(1)


@app.command()
def benchmark(
    component: Optional[str] = typer.Option(
        None,
        "--component",
        help="Benchmark specific component: act, assess, articulate, or full"
    ),
    iterations: int = typer.Option(100, "--iterations", "-n", help="Number of iterations"),
):
    """
    Run performance benchmarks.

    This compares the optimized pipeline against baseline to measure token savings,
    latency overhead, and cost reduction.
    """
    console.print("[bold]📊 Running benchmarks...[/bold]")

    # TODO: Implement benchmarking
    console.print("[yellow]⚠️  Benchmarking not yet implemented[/yellow]")
    console.print("[dim]Will be available in Phase 3[/dim]")


@app.command()
def shell():
    """
    Start an interactive Python shell with nocp environment.

    This provides a REPL with all nocp modules pre-imported.
    """
    console.print("[bold]🐚 Starting interactive shell...[/bold]")

    uv_cmd = get_uv_command()

    try:
        subprocess.run(
            [*uv_cmd, "run", "python", "-i", "-c", "from nocp import *"],
            check=True
        )
    except subprocess.CalledProcessError:
        sys.exit(1)


@app.command()
def health():
    """
    Check system health and configuration.

    Verifies:
    - uv installation
    - Dependencies
    - LLM connectivity (if API keys set)
    """
    console.print("[bold]🏥 Health Check[/bold]\n")

    from .bootstrap import UVBootstrap

    bootstrap = UVBootstrap()

    # Check uv
    if bootstrap.is_installed():
        console.print(f"[green]✓[/green] uv installation: OK ({bootstrap.uv_bin})")
    else:
        console.print("[red]✗[/red] uv installation: NOT FOUND")
        return

    # Check dependencies
    try:
        import pydantic
        import litellm
        import rich
        import typer
        console.print("[green]✓[/green] Dependencies: OK")
    except ImportError as e:
        console.print(f"[red]✗[/red] Dependencies: MISSING ({e.name})")
        console.print("[dim]Run: ./nocp setup[/dim]")
        return

    # Check API keys (optional)
    import os
    if os.getenv("OPENAI_API_KEY"):
        console.print("[green]✓[/green] OpenAI API key: SET")
    if os.getenv("ANTHROPIC_API_KEY"):
        console.print("[green]✓[/green] Anthropic API key: SET")
    if os.getenv("GOOGLE_API_KEY"):
        console.print("[green]✓[/green] Google API key: SET")

    console.print("\n[bold green]System is healthy![/bold green]")


@app.command()
def info():
    """
    Display project information and configuration.
    """
    panel = Panel(
        f"""[bold cyan]nocp[/bold cyan] - High-Efficiency LLM Proxy Agent

[bold]Version:[/bold] {__version__}
[bold]Purpose:[/bold] Token optimization middleware for LLM integration

[bold]Components:[/bold]
  • Act (Tool Executor)      - Execute external tools
  • Assess (Context Manager) - Compress input context (50-70% reduction)
  • Articulate (Serializer)  - Optimize output with TOON (30-60% reduction)

[bold]Documentation:[/bold] docs/
[bold]Examples:[/bold] examples/

[dim]For help: ./nocp --help[/dim]
        """,
        title="🚀 Project Info",
        border_style="cyan",
    )
    console.print(panel)


if __name__ == "__main__":
    app()
