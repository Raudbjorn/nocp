"""
Observability infrastructure for NOCP.

Provides:
- Structured logging with TransactionLog schema
- Metrics collection and aggregation
- Drift detection and alerting
- Performance monitoring
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from collections import deque
import statistics

from ..models.schemas import TransactionLog, ContextMetrics
from ..utils.logging import get_logger


class TransactionLogger:
    """
    Logs all transactions with comprehensive metrics.

    Handles:
    - JSON-structured logging to file
    - Transaction log retention
    - Query/analysis of historical logs
    """

    def __init__(self, log_file: str = "./logs/transactions.jsonl"):
        """
        Initialize the transaction logger.

        Args:
            log_file: Path to JSONL log file
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)

    def log_transaction(self, transaction: TransactionLog) -> None:
        """
        Log a transaction to the JSONL file.

        Args:
            transaction: Transaction to log
        """
        try:
            # Serialize to JSON
            transaction_dict = transaction.model_dump(mode='json')

            # Append to JSONL file
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(transaction_dict, default=str) + '\n')

            self.logger.info(
                "transaction_logged",
                transaction_id=transaction.transaction_id,
                efficiency_delta=transaction.efficiency_delta,
                total_savings=transaction.total_token_savings,
            )

        except (IOError, OSError, TypeError) as e:
            self.logger.error(
                "failed_to_log_transaction",
                transaction_id=transaction.transaction_id,
                error=str(e),
            )

    def load_recent_transactions(
        self,
        limit: int = 100,
        since: Optional[datetime] = None,
    ) -> List[TransactionLog]:
        """
        Load recent transactions from log file.

        Args:
            limit: Maximum number of transactions to load
            since: Optional datetime to filter transactions after

        Returns:
            List of TransactionLog instances
        """
        if not self.log_file.exists():
            return []

        transactions = []

        try:
            with open(self.log_file, 'r') as f:
                # Read from end of file (most recent)
                lines = deque(f, maxlen=limit * 2)  # Read more in case of filtering

            for line in reversed(list(lines)):
                if len(transactions) >= limit:
                    break

                try:
                    transaction_dict = json.loads(line.strip())

                    # Apply time filter if specified
                    if since:
                        tx_time = datetime.fromisoformat(transaction_dict['timestamp'])
                        if tx_time < since:
                            continue

                    transaction = TransactionLog(**transaction_dict)
                    transactions.append(transaction)

                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    self.logger.warning("failed_to_parse_transaction_log", error=str(e))
                    continue

            return transactions

        except (IOError, OSError, json.JSONDecodeError, ValueError) as e:
            self.logger.error("failed_to_load_transactions", error=str(e))
            return []

    def get_summary_stats(
        self,
        window_minutes: int = 60,
    ) -> Dict[str, Any]:
        """
        Get summary statistics for recent transactions.

        Args:
            window_minutes: Time window in minutes

        Returns:
            Dictionary of summary statistics
        """
        since = datetime.utcnow() - timedelta(minutes=window_minutes)
        transactions = self.load_recent_transactions(limit=1000, since=since)

        if not transactions:
            return {
                "transaction_count": 0,
                "window_minutes": window_minutes,
            }

        # Calculate statistics
        efficiency_deltas = [tx.efficiency_delta for tx in transactions]
        total_savings = [tx.total_token_savings for tx in transactions]
        input_ratios = [tx.input_compression_ratio for tx in transactions]
        output_ratios = [tx.output_compression_ratio for tx in transactions]
        latencies = [tx.total_latency_ms for tx in transactions]

        return {
            "transaction_count": len(transactions),
            "window_minutes": window_minutes,
            "efficiency_delta": {
                "mean": statistics.mean(efficiency_deltas) if efficiency_deltas else 0,
                "median": statistics.median(efficiency_deltas) if efficiency_deltas else 0,
                "min": min(efficiency_deltas) if efficiency_deltas else 0,
                "max": max(efficiency_deltas) if efficiency_deltas else 0,
            },
            "total_token_savings": {
                "mean": statistics.mean(total_savings) if total_savings else 0,
                "total": sum(total_savings),
            },
            "input_compression_ratio": {
                "mean": statistics.mean(input_ratios) if input_ratios else 1.0,
                "median": statistics.median(input_ratios) if input_ratios else 1.0,
            },
            "output_compression_ratio": {
                "mean": statistics.mean(output_ratios) if output_ratios else 1.0,
                "median": statistics.median(output_ratios) if output_ratios else 1.0,
            },
            "latency_ms": {
                "mean": statistics.mean(latencies) if latencies else 0,
                "p50": statistics.median(latencies) if latencies else 0,
                "p95": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else max(latencies, default=0),
                "p99": statistics.quantiles(latencies, n=100)[98] if len(latencies) > 100 else max(latencies, default=0),
            },
        }


class MetricsCollector:
    """
    Collects and aggregates performance metrics.

    Tracks:
    - Compression ratios
    - Token savings
    - Latency
    - Cost savings
    - Success rates
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize metrics collector.

        Args:
            window_size: Number of recent transactions to track in memory
        """
        self.window_size = window_size
        self.recent_transactions: deque[TransactionLog] = deque(maxlen=window_size)
        self.logger = get_logger(__name__)

        # Cumulative counters
        self.total_transactions = 0
        self.total_input_tokens_saved = 0
        self.total_output_tokens_saved = 0
        self.total_cost_saved_usd = 0.0

    def record_transaction(self, transaction: TransactionLog) -> None:
        """
        Record a transaction for metrics collection.

        Args:
            transaction: Transaction to record
        """
        self.recent_transactions.append(transaction)
        self.total_transactions += 1
        self.total_input_tokens_saved += transaction.efficiency_delta
        self.total_output_tokens_saved += (
            transaction.raw_output_tokens - transaction.optimized_output_tokens
        )

        if transaction.cost_savings_usd:
            self.total_cost_saved_usd += transaction.cost_savings_usd

    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics from recent transactions.

        Returns:
            Dictionary of current metrics
        """
        if not self.recent_transactions:
            return {
                "status": "no_data",
                "recent_transaction_count": 0,
            }

        transactions = list(self.recent_transactions)

        # Calculate averages
        avg_input_compression = statistics.mean(
            tx.input_compression_ratio for tx in transactions
        )
        avg_output_compression = statistics.mean(
            tx.output_compression_ratio for tx in transactions
        )
        avg_efficiency_delta = statistics.mean(
            tx.efficiency_delta for tx in transactions
        )
        avg_latency = statistics.mean(
            tx.total_latency_ms for tx in transactions
        )

        # Success rate
        successful = sum(1 for tx in transactions if tx.compression_success)
        success_rate = successful / len(transactions)

        return {
            "recent_transaction_count": len(transactions),
            "window_size": self.window_size,
            "averages": {
                "input_compression_ratio": avg_input_compression,
                "output_compression_ratio": avg_output_compression,
                "efficiency_delta": avg_efficiency_delta,
                "total_latency_ms": avg_latency,
            },
            "success_rate": success_rate,
            "cumulative": {
                "total_transactions": self.total_transactions,
                "total_input_tokens_saved": self.total_input_tokens_saved,
                "total_output_tokens_saved": self.total_output_tokens_saved,
                "total_tokens_saved": self.total_input_tokens_saved + self.total_output_tokens_saved,
                "total_cost_saved_usd": self.total_cost_saved_usd,
            },
        }

    def export_metrics(self, output_file: str) -> None:
        """
        Export current metrics to JSON file.

        Args:
            output_file: Path to output file
        """
        try:
            metrics = self.get_current_metrics()
            metrics["export_timestamp"] = datetime.utcnow().isoformat()

            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            self.logger.info("metrics_exported", output_file=output_file)

        except (IOError, OSError, TypeError) as e:
            self.logger.error("failed_to_export_metrics", error=str(e))


class DriftDetector:
    """
    Detects performance drift and degradation.

    Monitors:
    - Efficiency delta trends
    - Compression ratio degradation
    - Latency increases
    - Error rate spikes
    """

    def __init__(
        self,
        window_size: int = 100,
        alert_threshold: float = -1000.0,
        comparison_window: int = 50,
    ):
        """
        Initialize drift detector.

        Args:
            window_size: Rolling window size for analysis
            alert_threshold: Efficiency delta threshold for alerts (negative = degradation)
            comparison_window: Window size for trend comparison
        """
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.comparison_window = comparison_window
        self.recent_transactions: deque[TransactionLog] = deque(maxlen=window_size)
        self.logger = get_logger(__name__)

        # Alert state
        self.alerts: List[Dict[str, Any]] = []

    def analyze_transaction(self, transaction: TransactionLog) -> Optional[Dict[str, Any]]:
        """
        Analyze a transaction for drift detection.

        Args:
            transaction: Transaction to analyze

        Returns:
            Alert dictionary if drift detected, None otherwise
        """
        self.recent_transactions.append(transaction)

        # Need enough data for comparison
        if len(self.recent_transactions) < self.comparison_window * 2:
            return None

        # Split into recent and previous windows
        all_transactions = list(self.recent_transactions)
        recent_window = all_transactions[-self.comparison_window:]
        previous_window = all_transactions[-2*self.comparison_window:-self.comparison_window]

        # Calculate average efficiency delta for each window
        recent_avg_delta = statistics.mean(tx.efficiency_delta for tx in recent_window)
        previous_avg_delta = statistics.mean(tx.efficiency_delta for tx in previous_window)

        # Calculate trend (delta change)
        delta_trend = recent_avg_delta - previous_avg_delta

        # Check for degradation
        if delta_trend < self.alert_threshold:
            alert = {
                "timestamp": datetime.utcnow().isoformat(),
                "alert_type": "efficiency_degradation",
                "severity": "warning" if delta_trend > self.alert_threshold * 1.5 else "critical",
                "delta_trend": delta_trend,
                "recent_avg_delta": recent_avg_delta,
                "previous_avg_delta": previous_avg_delta,
                "threshold": self.alert_threshold,
                "message": f"Efficiency degradation detected: {delta_trend:.0f} token delta trend",
            }

            self.alerts.append(alert)

            self.logger.warning(
                "drift_detected",
                **alert
            )

            return alert

        # Check for latency increase
        recent_avg_latency = statistics.mean(tx.total_latency_ms for tx in recent_window)
        previous_avg_latency = statistics.mean(tx.total_latency_ms for tx in previous_window)

        latency_increase_pct = (
            (recent_avg_latency - previous_avg_latency) / previous_avg_latency * 100
            if previous_avg_latency > 0 else 0
        )

        if latency_increase_pct > 50:  # 50% increase
            alert = {
                "timestamp": datetime.utcnow().isoformat(),
                "alert_type": "latency_increase",
                "severity": "warning",
                "latency_increase_pct": latency_increase_pct,
                "recent_avg_latency_ms": recent_avg_latency,
                "previous_avg_latency_ms": previous_avg_latency,
                "message": f"Latency increased by {latency_increase_pct:.1f}%",
            }

            self.alerts.append(alert)

            self.logger.warning(
                "latency_increase_detected",
                **alert
            )

            return alert

        return None

    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent alerts.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of recent alerts
        """
        return self.alerts[-limit:]

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self.alerts.clear()


class ObservabilityHub:
    """
    Central hub for all observability features.

    Integrates:
    - Transaction logging
    - Metrics collection
    - Drift detection
    - Reporting
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        metrics_window: int = 100,
        drift_threshold: float = -1000.0,
    ):
        """
        Initialize observability hub.

        Args:
            log_dir: Directory for log files
            metrics_window: Window size for metrics collection
            drift_threshold: Threshold for drift detection
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.transaction_logger = TransactionLogger(
            log_file=str(self.log_dir / "transactions.jsonl")
        )
        self.metrics_collector = MetricsCollector(window_size=metrics_window)
        self.drift_detector = DriftDetector(
            window_size=metrics_window,
            alert_threshold=drift_threshold,
        )

        self.logger = get_logger(__name__)

    def log_transaction(self, transaction: TransactionLog) -> None:
        """
        Log a transaction to all observability systems.

        Args:
            transaction: Transaction to log
        """
        # Log to file
        self.transaction_logger.log_transaction(transaction)

        # Record in metrics collector
        self.metrics_collector.record_transaction(transaction)

        # Analyze for drift
        alert = self.drift_detector.analyze_transaction(transaction)

        if alert:
            self.logger.warning("observability_alert", **alert)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get comprehensive dashboard data.

        Returns:
            Dictionary with all observability metrics
        """
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "current_metrics": self.metrics_collector.get_current_metrics(),
            "recent_summary": self.transaction_logger.get_summary_stats(window_minutes=60),
            "recent_alerts": self.drift_detector.get_recent_alerts(limit=5),
        }

    def export_report(self, output_file: Optional[str] = None) -> str:
        """
        Export comprehensive observability report.

        Args:
            output_file: Optional output file path

        Returns:
            Path to generated report
        """
        if output_file is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_file = str(self.log_dir / f"observability_report_{timestamp}.json")

        dashboard_data = self.get_dashboard_data()

        with open(output_file, 'w') as f:
            json.dump(dashboard_data, f, indent=2)

        self.logger.info("observability_report_exported", output_file=output_file)

        return output_file


# Global instance
_observability_hub: Optional[ObservabilityHub] = None


def get_observability_hub(
    log_dir: Optional[str] = None,
    metrics_window: Optional[int] = None,
    drift_threshold: Optional[float] = None,
) -> ObservabilityHub:
    """
    Get or create the global observability hub instance.

    Args:
        log_dir: Log directory (only used on first call)
        metrics_window: Metrics window size (only used on first call)
        drift_threshold: Drift detection threshold (only used on first call)

    Returns:
        ObservabilityHub instance
    """
    global _observability_hub

    if _observability_hub is None:
        _observability_hub = ObservabilityHub(
            log_dir=log_dir or "./logs",
            metrics_window=metrics_window or 100,
            drift_threshold=drift_threshold or -1000.0,
        )

    return _observability_hub
