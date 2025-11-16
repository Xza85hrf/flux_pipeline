"""Performance metrics collection and reporting.

This module provides tools for tracking and analyzing performance metrics
during image generation, including timing, memory usage, and GPU utilization.

Examples:
    Basic usage with context manager:
        >>> from utils.metrics import PerformanceMetrics
        >>> metrics = PerformanceMetrics()
        >>> with metrics.measure("image_generation"):
        ...     image = generate_image(prompt)
        >>> report = metrics.report()
        >>> print(f"Average time: {report['avg_generation_time']:.2f}s")

    Manual tracking:
        >>> metrics = PerformanceMetrics()
        >>> metrics.start_operation("model_load")
        >>> load_model()
        >>> metrics.end_operation("model_load")
        >>> print(metrics.get_operation_stats("model_load"))

    Export metrics:
        >>> metrics.export_to_json("metrics.json")
        >>> metrics.export_to_csv("metrics.csv")
"""

import time
import psutil
from contextlib import contextmanager
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import json
import csv

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PerformanceMetrics:
    """Collect and report performance metrics for FluxPipeline operations.

    Tracks timing, memory usage, and GPU utilization for various operations.
    Provides statistical analysis and export capabilities.

    Attributes:
        metrics: Dictionary storing all collected metrics
        active_operations: Dictionary of currently running operations
    """

    def __init__(self):
        """Initialize performance metrics collector."""
        self.metrics: Dict[str, List[Dict[str, Any]]] = {
            "generation": [],
            "model_load": [],
            "preprocessing": [],
            "postprocessing": [],
            "custom": [],
        }
        self.active_operations: Dict[str, Dict[str, Any]] = {}

    def _get_memory_info(self) -> Dict[str, float]:
        """Get current memory information.

        Returns:
            Dict with CPU and optionally GPU memory info in GB
        """
        info = {
            "cpu_memory_used_gb": psutil.Process().memory_info().rss / 1024**3,
            "cpu_memory_percent": psutil.Process().memory_percent(),
        }

        if TORCH_AVAILABLE and torch.cuda.is_available():
            info["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            info["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
            info["gpu_memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3

        return info

    def _get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization percentage.

        Returns:
            GPU utilization as percentage, or None if not available
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return None

        try:
            # This requires nvidia-ml-py3
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            pynvml.nvmlShutdown()
            return util.gpu
        except (ImportError, Exception):
            # If pynvml not available or fails, return None
            return None

    @contextmanager
    def measure(self, operation: str, category: str = "custom"):
        """Context manager to measure operation performance.

        Args:
            operation: Name of the operation being measured
            category: Category to store metrics in (default: "custom")

        Yields:
            None

        Example:
            >>> with metrics.measure("image_generation", "generation"):
            ...     generate_image()
        """
        start_time = time.time()
        start_memory = self._get_memory_info()

        try:
            yield
        finally:
            elapsed = time.time() - start_time
            end_memory = self._get_memory_info()

            metric = {
                "operation": operation,
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": elapsed,
                "cpu_memory_delta_gb": end_memory["cpu_memory_used_gb"] - start_memory["cpu_memory_used_gb"],
                "cpu_memory_end_gb": end_memory["cpu_memory_used_gb"],
            }

            if "gpu_memory_allocated_gb" in start_memory:
                metric["gpu_memory_delta_gb"] = (
                    end_memory["gpu_memory_allocated_gb"] - start_memory["gpu_memory_allocated_gb"]
                )
                metric["gpu_memory_end_gb"] = end_memory["gpu_memory_allocated_gb"]

            gpu_util = self._get_gpu_utilization()
            if gpu_util is not None:
                metric["gpu_utilization_percent"] = gpu_util

            # Store in appropriate category
            if category in self.metrics:
                self.metrics[category].append(metric)
            else:
                self.metrics[category] = [metric]

    def start_operation(self, operation: str):
        """Start tracking an operation manually.

        Args:
            operation: Name of the operation
        """
        self.active_operations[operation] = {
            "start_time": time.time(),
            "start_memory": self._get_memory_info(),
        }

    def end_operation(self, operation: str, category: str = "custom") -> Dict[str, Any]:
        """End tracking an operation and record metrics.

        Args:
            operation: Name of the operation
            category: Category to store metrics in

        Returns:
            Dict containing the recorded metrics

        Raises:
            KeyError: If operation was not started
        """
        if operation not in self.active_operations:
            raise KeyError(f"Operation '{operation}' was not started")

        op_data = self.active_operations.pop(operation)
        elapsed = time.time() - op_data["start_time"]
        end_memory = self._get_memory_info()

        metric = {
            "operation": operation,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": elapsed,
            "cpu_memory_delta_gb": end_memory["cpu_memory_used_gb"] - op_data["start_memory"]["cpu_memory_used_gb"],
            "cpu_memory_end_gb": end_memory["cpu_memory_used_gb"],
        }

        if "gpu_memory_allocated_gb" in op_data["start_memory"]:
            metric["gpu_memory_delta_gb"] = (
                end_memory["gpu_memory_allocated_gb"] - op_data["start_memory"]["gpu_memory_allocated_gb"]
            )
            metric["gpu_memory_end_gb"] = end_memory["gpu_memory_allocated_gb"]

        if category in self.metrics:
            self.metrics[category].append(metric)
        else:
            self.metrics[category] = [metric]

        return metric

    def get_operation_stats(self, operation: Optional[str] = None, category: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for a specific operation or category.

        Args:
            operation: Specific operation name (optional)
            category: Category to analyze (optional, defaults to all)

        Returns:
            Dict with statistical analysis (count, avg, min, max, total)
        """
        # Collect relevant metrics
        relevant_metrics = []

        if category:
            relevant_metrics = self.metrics.get(category, [])
        else:
            for cat_metrics in self.metrics.values():
                relevant_metrics.extend(cat_metrics)

        # Filter by operation if specified
        if operation:
            relevant_metrics = [m for m in relevant_metrics if m["operation"] == operation]

        if not relevant_metrics:
            return {"count": 0, "message": "No metrics found"}

        # Calculate statistics
        durations = [m["duration_seconds"] for m in relevant_metrics]
        cpu_deltas = [m.get("cpu_memory_delta_gb", 0) for m in relevant_metrics]

        stats = {
            "count": len(relevant_metrics),
            "total_time_seconds": sum(durations),
            "avg_time_seconds": sum(durations) / len(durations),
            "min_time_seconds": min(durations),
            "max_time_seconds": max(durations),
            "avg_cpu_memory_delta_gb": sum(cpu_deltas) / len(cpu_deltas),
        }

        # Add GPU stats if available
        gpu_deltas = [m.get("gpu_memory_delta_gb", 0) for m in relevant_metrics if "gpu_memory_delta_gb" in m]
        if gpu_deltas:
            stats["avg_gpu_memory_delta_gb"] = sum(gpu_deltas) / len(gpu_deltas)

        return stats

    def report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report.

        Returns:
            Dict with overall statistics and category breakdowns
        """
        total_operations = sum(len(v) for v in self.metrics.values())

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_operations": total_operations,
            "categories": {},
        }

        for category, metrics_list in self.metrics.items():
            if metrics_list:
                report["categories"][category] = self.get_operation_stats(category=category)

        return report

    def clear(self):
        """Clear all collected metrics."""
        self.metrics = {
            "generation": [],
            "model_load": [],
            "preprocessing": [],
            "postprocessing": [],
            "custom": [],
        }
        self.active_operations.clear()

    def export_to_json(self, filepath: Path) -> None:
        """Export metrics to JSON file.

        Args:
            filepath: Path to save JSON file
        """
        filepath = Path(filepath)
        data = {
            "export_time": datetime.now().isoformat(),
            "metrics": self.metrics,
            "summary": self.report(),
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def export_to_csv(self, filepath: Path, category: Optional[str] = None) -> None:
        """Export metrics to CSV file.

        Args:
            filepath: Path to save CSV file
            category: Specific category to export (optional, defaults to all)
        """
        filepath = Path(filepath)

        # Collect metrics to export
        if category:
            metrics_to_export = self.metrics.get(category, [])
        else:
            metrics_to_export = []
            for cat_metrics in self.metrics.values():
                metrics_to_export.extend(cat_metrics)

        if not metrics_to_export:
            return

        # Get all possible field names
        fieldnames = set()
        for metric in metrics_to_export:
            fieldnames.update(metric.keys())
        fieldnames = sorted(fieldnames)

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_to_export)


# Singleton instance for global metrics
_global_metrics = PerformanceMetrics()


def get_global_metrics() -> PerformanceMetrics:
    """Get the global metrics instance.

    Returns:
        Global PerformanceMetrics instance
    """
    return _global_metrics


def log_metric(operation: str, duration: float, **kwargs):
    """Log a performance metric to the global metrics instance.

    Args:
        operation: Name of the operation
        duration: Duration in seconds
        **kwargs: Additional metric data
    """
    metric = {
        "operation": operation,
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": duration,
        **kwargs
    }
    _global_metrics.metrics["custom"].append(metric)


if __name__ == "__main__":
    # Test performance metrics
    print("Testing PerformanceMetrics...")

    metrics = PerformanceMetrics()

    # Test context manager
    print("\n1. Testing context manager:")
    with metrics.measure("test_operation", "custom"):
        time.sleep(0.1)
        # Simulate some work
        _ = [i**2 for i in range(1000000)]

    stats = metrics.get_operation_stats("test_operation")
    print(f"   Duration: {stats['avg_time_seconds']:.3f}s")
    print(f"   Memory: {stats['avg_cpu_memory_delta_gb']:.4f}GB")

    # Test manual tracking
    print("\n2. Testing manual tracking:")
    metrics.start_operation("manual_test")
    time.sleep(0.05)
    result = metrics.end_operation("manual_test")
    print(f"   Duration: {result['duration_seconds']:.3f}s")

    # Test report
    print("\n3. Testing report:")
    report = metrics.report()
    print(f"   Total operations: {report['total_operations']}")

    print("\n✅ All tests passed!")
