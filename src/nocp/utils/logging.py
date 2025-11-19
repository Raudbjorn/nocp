"""
Structured logging and metrics tracking for the NOCP proxy agent.

Implements comprehensive monitoring as outlined in the architectural blueprint,
including the Context Watchdog for drift detection.
"""

import json
from pathlib import Path

import structlog

from ..core.config import get_config
from ..models.schemas import ContextMetrics


def setup_logging() -> None:
    """
    Configure structured logging with appropriate processors.

    Sets up structlog with timestamping, log level filtering, and JSON formatting
    for production environments.
    """
    config = get_config()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            (
                structlog.processors.JSONRenderer()
                if config.log_level == "DEBUG"
                else structlog.dev.ConsoleRenderer()
            ),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(structlog.stdlib, config.log_level.upper(), structlog.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
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

        # Ensure log directory exists
        if self.enabled:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

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

        # Write to JSONL file
        with self.log_file.open("a") as f:
            f.write(json.dumps(metrics_dict) + "\n")

        # Log summary to console
        self.logger.info(
            "transaction_completed",
            transaction_id=metrics.transaction_id,
            input_tokens=metrics.compressed_input_tokens,
            output_tokens=metrics.raw_output_tokens,
            output_format=metrics.final_output_format,
            total_latency_ms=metrics.total_latency_ms,
            estimated_cost_usd=metrics.estimated_cost_usd,
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
