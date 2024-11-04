"""Advanced logging utilities for the FluxPipeline project.

This module provides comprehensive logging capabilities including:
- Session-based logging management
- Performance monitoring and metrics
- Progress tracking with ETA calculation
- GPU metrics logging
- Exception handling and logging
- Operation timing and performance analysis

The module supports both file and console logging with:
- Colored output for different log levels
- Performance metrics tracking
- Progress visualization
- GPU resource monitoring
- Session management

Example:
    Basic usage:
    ```python
    from utils.logging_utils import setup_session_logging, performance_logger
    from pathlib import Path
    
    # Setup session logging
    log_manager = setup_session_logging(Path("logs"))
    
    # Use performance logging decorator
    @performance_logger("image_processing")
    def process_image(image):
        # Process image...
        pass
    
    # Use context manager for operation logging
    with log_manager.log_operation("batch_processing"):
        process_image(image)
    ```

Note:
    This module automatically handles log file creation and rotation,
    making it suitable for both development and production use.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from functools import wraps
from contextlib import contextmanager

from config.logging_config import logger


class LogManager:
    """Advanced logging management system with performance tracking.

    This class provides comprehensive logging capabilities including:
    - Session-based logging
    - Performance metrics tracking
    - Operation timing
    - Log file management

    Attributes:
        base_log_dir (Path): Base directory for log files
        session_start (datetime): Session start timestamp
        performance_logs (List[Dict[str, Any]]): Performance metrics history

    Example:
        ```python
        log_manager = LogManager(Path("logs"))

        # Log an operation with timing
        with log_manager.log_operation("data_processing"):
            process_data()

        # Setup error handling
        log_manager.setup_error_logging()
        ```
    """

    def __init__(self, base_log_dir: Path):
        """Initialize the log manager.

        Args:
            base_log_dir (Path): Base directory for storing log files
        """
        self.base_log_dir = base_log_dir
        self.base_log_dir.mkdir(parents=True, exist_ok=True)
        self.session_start = datetime.now()
        self.performance_logs: List[Dict[str, Any]] = []

    def get_session_log_path(self) -> Path:
        """Generate unique log file path for current session.

        Returns:
            Path: Path to session-specific log file

        Example:
            ```python
            log_path = log_manager.get_session_log_path()
            print(f"Logging to: {log_path}")
            ```
        """
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        return self.base_log_dir / f"session_{timestamp}.log"

    def setup_error_logging(self):
        """Configure system-wide exception handling and logging.

        This method sets up a custom exception handler that:
        1. Preserves keyboard interrupt behavior
        2. Logs uncaught exceptions
        3. Maintains stack traces

        Example:
            ```python
            log_manager = LogManager(Path("logs"))
            log_manager.setup_error_logging()
            # All uncaught exceptions will now be logged
            ```
        """

        def exception_handler(exc_type, exc_value, exc_traceback):
            # Preserve keyboard interrupt behavior
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return

            # Log uncaught exception with full stack trace
            logger.critical(
                "Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback)
            )

        sys.excepthook = exception_handler

    def log_performance(self, operation: str, duration: float, details: Dict[str, Any]):
        """Log performance metrics for an operation.

        Args:
            operation (str): Name of the operation
            duration (float): Duration in seconds
            details (Dict[str, Any]): Additional performance details

        Example:
            ```python
            log_manager.log_performance(
                "image_processing",
                1.23,
                {"batch_size": 32, "resolution": "1024x1024"}
            )
            ```
        """
        # Create performance log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "duration": duration,
            **details,
        }
        self.performance_logs.append(log_entry)

        # Save to performance log file
        perf_log_path = self.base_log_dir / "performance.log"
        with open(perf_log_path, "a") as f:
            f.write(f"{log_entry}\n")

    @contextmanager
    def log_operation(self, operation_name: str, **kwargs):
        """Context manager for logging operations with timing.

        Args:
            operation_name (str): Name of the operation to log
            **kwargs: Additional details to log with the operation

        Example:
            ```python
            with log_manager.log_operation("data_processing",
                                         batch_size=32,
                                         dataset="train"):
                process_data()
            ```
        """
        start_time = time.time()
        try:
            logger.info(f"Starting {operation_name}")
            yield
        except Exception as e:
            logger.error(f"Error in {operation_name}: {str(e)}")
            raise
        finally:
            duration = time.time() - start_time
            self.log_performance(operation_name, duration, kwargs)
            logger.info(f"Completed {operation_name} in {duration:.2f}s")


def performance_logger(operation_name: Optional[str] = None):
    """Decorator for logging function performance metrics.

    Args:
        operation_name (Optional[str], optional): Custom name for the operation.
            Defaults to function name if None.

    Returns:
        Callable: Decorated function with performance logging

    Example:
        ```python
        @performance_logger("image_processing")
        def process_image(image):
            # Process image...
            pass

        # Or use function name
        @performance_logger()
        def process_data():
            # Process data...
            pass
        ```
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            op_name = operation_name or func.__name__

            try:
                logger.debug(f"Starting {op_name}")
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.debug(f"Completed {op_name} in {duration:.2f}s")
                return result
            except Exception as e:
                logger.error(f"Error in {op_name}: {str(e)}")
                raise

        return wrapper

    return decorator


class ProgressLogger:
    """Progress tracking with ETA calculation.

    This class provides:
    - Progress percentage tracking
    - ETA calculation
    - Rate calculation
    - Periodic progress updates

    Attributes:
        total (int): Total number of steps
        current (int): Current step
        operation (str): Operation name
        log_interval (int): Seconds between progress updates

    Example:
        ```python
        progress = ProgressLogger(total=100, operation="Processing")
        for item in items:
            process_item(item)
            progress.update()
        ```
    """

    def __init__(self, total: int, operation: str, log_interval: int = 1):
        """Initialize progress logger.

        Args:
            total (int): Total number of steps
            operation (str): Name of the operation
            log_interval (int, optional): Seconds between updates. Defaults to 1.
        """
        self.total = total
        self.current = 0
        self.operation = operation
        self.start_time = time.time()
        self.log_interval = log_interval
        self.last_log_time = self.start_time

    def update(self, amount: int = 1):
        """Update progress counter and log if interval elapsed.

        Args:
            amount (int, optional): Steps to increment. Defaults to 1.
        """
        self.current += amount
        current_time = time.time()

        # Log progress if interval elapsed
        if current_time - self.last_log_time >= self.log_interval:
            self._log_progress()
            self.last_log_time = current_time

    def _log_progress(self):
        """Log current progress with ETA calculation."""
        if self.current == 0:
            return

        # Calculate metrics
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed
        eta = (self.total - self.current) / rate if rate > 0 else 0

        # Calculate progress percentage
        progress = (self.current / self.total) * 100
        logger.info(
            f"{self.operation}: {progress:.1f}% complete "
            f"({self.current}/{self.total}) "
            f"[ETA: {eta:.1f}s]"
        )


def log_gpu_metrics():
    """Log GPU usage metrics if available.

    This function logs:
    - Memory allocation
    - Memory reservation
    - GPU temperature (if available)

    Example:
        ```python
        # Log current GPU metrics
        log_gpu_metrics()
        ```
    """
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                # Get GPU properties
                props = torch.cuda.get_device_properties(i)
                memory = torch.cuda.memory_stats(i)

                # Log GPU information
                logger.info(f"GPU {i} ({props.name}):")
                logger.info(
                    f"  Memory Allocated: {memory['allocated_bytes.all.current'] / 1e9:.2f}GB"
                )
                logger.info(
                    f"  Memory Reserved: {memory['reserved_bytes.all.current'] / 1e9:.2f}GB"
                )

                # Log temperature if available
                if hasattr(torch.cuda, "get_device_temperature"):
                    temp = torch.cuda.get_device_temperature(i)
                    logger.info(f"  Temperature: {temp}°C")
    except Exception as e:
        logger.warning(f"Error logging GPU metrics: {str(e)}")


def setup_session_logging(base_dir: Optional[Path] = None) -> LogManager:
    """Setup complete logging configuration for a new session.

    Args:
        base_dir (Optional[Path], optional): Base directory for logs.
            Defaults to ./logs.

    Returns:
        LogManager: Configured log manager instance

    Example:
        ```python
        # Setup with default location
        log_manager = setup_session_logging()

        # Setup with custom location
        log_manager = setup_session_logging(Path("/var/log/myapp"))
        ```
    """
    if base_dir is None:
        base_dir = Path.cwd() / "logs"

    # Initialize log manager
    log_manager = LogManager(base_dir)
    log_manager.setup_error_logging()

    # Setup session log file
    session_log_path = log_manager.get_session_log_path()
    file_handler = logging.FileHandler(session_log_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    # Log session start
    logger.info(f"Session started: {log_manager.session_start}")
    logger.info(f"Log file: {session_log_path}")

    return log_manager


@performance_logger("image_processing")
def process_image(*args, **kwargs):
    """Example function demonstrating performance logging decorator.

    This is a placeholder function showing decorator usage.
    """
    pass


def log_generation_stats(stats: Dict[str, Any]):
    """Log image generation statistics.

    Args:
        stats (Dict[str, Any]): Generation statistics to log

    Example:
        ```python
        stats = {
            "generation_time": 1.23,
            "batch_size": 32,
            "success_rate": 0.95
        }
        log_generation_stats(stats)
        ```
    """
    logger.info("Generation Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
