"""
Structured logging and metrics tracking for the NOCP proxy agent.

Implements comprehensive monitoring as outlined in the architectural blueprint,
including the Context Watchdog for drift detection.
"""

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import structlog
from rich.console import Console

from ..core.config import get_config
from ..models.enums import LogLevel
from ..models.schemas import ContextMetrics


def setup_file_logging(
    log_file: Path, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5  # 10MB
) -> RotatingFileHandler:
    """
    Configure rotating file handler for logs.

    Args:
        log_file: Path to the log file
        max_bytes: Maximum log file size before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)

    Returns:
        Configured RotatingFileHandler instance

    Note:
        Directory creation is handled by get_config().ensure_log_directory()
    """
    # Create rotating file handler
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )

    # Set formatter for the file handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)

    return file_handler


def setup_logging() -> None:
    """
    Configure structured logging with appropriate processors.

    Sets up structlog with timestamping, log level filtering, JSON formatting
    for production environments, and optional file logging with rotation.
    """
    config = get_config()

    # Shared processors for pre-rendering
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    # Configure standard library logging for file output
    if config.log_file is not None:
        # Set up rotating file handler
        file_handler = setup_file_logging(
            log_file=config.log_file,
            max_bytes=config.log_max_bytes,
            backup_count=config.log_backup_count,
        )

        # Override file handler formatter to use structlog's JSONRenderer
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(),
                ],
            )
        )

        # Configure console handler with ConsoleRenderer
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.dev.ConsoleRenderer(),
                ],
            )
        )

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        root_logger.setLevel(getattr(logging, config.log_level.value, logging.INFO))

        # Use stdlib logger factory and no final renderer (handlers do the rendering)
        logger_factory = structlog.stdlib.LoggerFactory()
        processors = shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ]
    else:
        # Use print logger factory for console-only mode with final renderer
        logger_factory = structlog.PrintLoggerFactory()  # type: ignore[assignment]
        processors = shared_processors + [
            (
                structlog.processors.JSONRenderer()
                if config.log_level == LogLevel.DEBUG
                else structlog.dev.ConsoleRenderer()
            ),
        ]

    # Configure structlog with appropriate processors
    structlog.configure(
        processors=processors,  # type: ignore[arg-type]
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(structlog.stdlib, config.log_level.value, structlog.INFO)
        ),
        context_class=dict,
        logger_factory=logger_factory,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically module name)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


class MetricsLogger:
    """
    Dedicated metrics logger for tracking token efficiency and cost.

    Implements the Context Watchdog pattern for drift detection by monitoring
    the efficiency delta across transactions.
    """

    def __init__(self, log_file: Path | None = None):
        """
        Initialize metrics logger.

        Args:
            log_file: Path to metrics log file (JSONL format)
        """
        config = get_config()
        self.log_file = log_file or config.metrics_log_file
        self.enabled = config.enable_metrics_logging
        self.logger = get_logger("metrics")

        # Set up rotating file handler for metrics using shared setup function
        if self.enabled:
            # Create file handler using shared setup function
            self.file_handler = setup_file_logging(
                log_file=self.log_file,
                max_bytes=config.log_max_bytes,
                backup_count=config.log_backup_count,
            )
            # Override formatter to preserve JSONL format (output only the message)
            self.file_handler.setFormatter(logging.Formatter("%(message)s"))

            # Create dedicated logger for writing to metrics file
            self.file_logger = logging.getLogger("metrics.file")
            self.file_logger.setLevel(logging.INFO)
            self.file_logger.addHandler(self.file_handler)
            self.file_logger.propagate = False  # Prevent logs from going to parent loggers

    def log_transaction(self, metrics: ContextMetrics) -> None:
        """
        Log a complete transaction with all metrics.

        Args:
            metrics: ContextMetrics object containing transaction data
        """
        if not self.enabled:
            return

        # Convert to dict for JSON serialization
        metrics_dict = metrics.model_dump(mode="json")

        # Calculate efficiency delta (drift detection metric)
        efficiency_delta = metrics.raw_input_tokens - metrics.compressed_input_tokens

        # Add derived metrics
        metrics_dict["efficiency_delta"] = efficiency_delta
        metrics_dict["compression_justified"] = all(
            comp.net_savings > 0 for comp in metrics.compression_operations
        )

        # Write to JSONL file using dedicated logger
        log_line = json.dumps(metrics_dict)
        self.file_logger.info(log_line)

        # Log summary to console
        self.logger.info(
            "transaction_completed",
            transaction_id=metrics.transaction_id,
            input_tokens=metrics.compressed_input_tokens,
            output_tokens=metrics.raw_output_tokens,
            output_format=metrics.final_output_format,
            total_latency_ms=metrics.total_latency_ms,
            estimated_cost_usd=metrics.estimated_cost_usd,  # type: ignore[attr-defined]
            efficiency_delta=efficiency_delta,
        )

    def log_compression_event(
        self,
        transaction_id: str,
        tool_name: str,
        original_tokens: int,
        compressed_tokens: int,
        method: str,
        compression_cost: int,
    ) -> None:
        """
        Log individual compression events for analysis.

        Args:
            transaction_id: Transaction identifier
            tool_name: Name of tool that produced the output
            original_tokens: Original token count
            compressed_tokens: Compressed token count
            method: Compression method used
            compression_cost: Tokens consumed by compression
        """
        self.logger.info(
            "compression_applied",
            transaction_id=transaction_id,
            tool_name=tool_name,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 0,
            method=method,
            compression_cost=compression_cost,
            net_savings=original_tokens - compressed_tokens - compression_cost,
        )

    def check_drift(self, window_size: int = 100) -> dict[str, float]:
        """
        Analyze recent transactions for context drift.

        Implements the Context Watchdog by monitoring efficiency delta trends.

        Args:
            window_size: Number of recent transactions to analyze

        Returns:
            Dictionary containing drift metrics
        """
        if not self.enabled or not self.log_file.exists():
            return {}

        # Read recent transactions
        transactions = []
        with self.log_file.open("r") as f:
            lines = f.readlines()
            recent_lines = lines[-window_size:] if len(lines) > window_size else lines

            for line in recent_lines:
                try:
                    transactions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not transactions:
            return {}

        # Calculate drift metrics
        efficiency_deltas = [t.get("efficiency_delta", 0) for t in transactions]
        compression_ratios = [t.get("input_compression_ratio", 1.0) for t in transactions]

        avg_efficiency_delta = sum(efficiency_deltas) / len(efficiency_deltas)
        avg_compression_ratio = sum(compression_ratios) / len(compression_ratios)

        # Calculate trend (simple linear regression on last 20 vs previous)
        if len(efficiency_deltas) >= 40:
            recent_avg = sum(efficiency_deltas[-20:]) / 20
            previous_avg = sum(efficiency_deltas[-40:-20]) / 20
            delta_trend = recent_avg - previous_avg
        else:
            delta_trend = 0

        # Get drift threshold from config
        from ..core.config import get_config

        config = get_config()

        drift_metrics = {
            "avg_efficiency_delta": avg_efficiency_delta,
            "avg_compression_ratio": avg_compression_ratio,
            "delta_trend": delta_trend,
            "transactions_analyzed": len(transactions),
            "drift_detected": delta_trend < config.drift_detection_threshold,
        }

        if drift_metrics["drift_detected"]:
            self.logger.warning(
                "context_drift_detected",
                **drift_metrics,
            )

        return drift_metrics


# Global metrics logger instance
_metrics_logger: MetricsLogger | None = None


def get_metrics_logger() -> MetricsLogger:
    """Get the global metrics logger instance."""
    global _metrics_logger
    if _metrics_logger is None:
        _metrics_logger = MetricsLogger()
    return _metrics_logger


def log_metrics(metrics: ContextMetrics) -> None:
    """
    Convenience function to log metrics.

    Args:
        metrics: ContextMetrics to log
    """
    get_metrics_logger().log_transaction(metrics)


class ComponentLogger:
    """Base class for component-specific structured logging"""

    def __init__(self, component_name: str):
        self.logger = structlog.get_logger(component_name)
        self.component = component_name
        self.console = Console()

    def log_operation_start(self, operation: str, details: dict | None = None):
        """Log operation start with emoji"""
        self.console.print(f"[cyan]▶[/cyan] [{self.component}] Starting: {operation}")
        self.logger.info(f"{operation}_started", component=self.component, **(details or {}))

    def log_operation_complete(
        self, operation: str, duration_ms: float | None = None, details: dict | None = None
    ):
        """Log operation completion"""
        msg = f"[green]✅[/green] [{self.component}] Completed: {operation}"
        if duration_ms is not None:
            msg += f" ({duration_ms:.0f}ms)"

        self.console.print(msg)

        log_data = details.copy() if details else {}
        log_data["component"] = self.component
        if duration_ms is not None:
            log_data["duration_ms"] = duration_ms

        self.logger.info(f"{operation}_completed", **log_data)

    def log_operation_error(self, operation: str, error: Exception, details: dict | None = None):
        """Log operation error"""
        from rich.traceback import Traceback

        self.console.print(f"[red]❌[/red] [{self.component}] Failed: {operation}")
        trace = Traceback.from_exception(
            type(error),
            error,
            error.__traceback__,
        )
        self.console.print(trace)

        log_data = details.copy() if details else {}
        log_data.update(
            {
                "component": self.component,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )

        self.logger.error(f"{operation}_failed", **log_data, exc_info=error)

    def log_metric(self, metric_name: str, value: Any, unit: str = ""):
        """Log a metric"""
        self.logger.info(
            "metric", component=self.component, metric=metric_name, value=value, unit=unit
        )


# Create component-specific loggers
act_logger = ComponentLogger("act")
assess_logger = ComponentLogger("assess")
articulate_logger = ComponentLogger("articulate")
agent_logger = ComponentLogger("agent")
