"""Logging configuration module for the FluxPipeline project.

This module provides a comprehensive logging setup with colored output and file logging capabilities.
Key features include:
- Custom ColorFormatter for visually distinct log levels and message types
- Configurable file and console logging handlers
- Special highlighting for GPU, Memory, Error, and Warning messages
- Thread-safe logging implementation
- Global logger instance for project-wide use

Example:
    Basic usage with default settings:
    ```python
    from config.logging_config import logger
    
    logger.info("Starting pipeline")
    logger.warning("Resource usage high")
    logger.error("Process failed")
    ```

    Custom logging setup:
    ```python
    from config.logging_config import setup_logging
    from pathlib import Path
    
    custom_logger = setup_logging(Path("custom_log.log"))
    custom_logger.info("Custom logging initialized")
    ```
"""

import logging
from pathlib import Path
from typing import Optional


class ColorFormatter(logging.Formatter):
    """Custom formatter that adds ANSI color coding to log messages.
    
    This formatter enhances log readability by:
    - Adding distinct colors for different log levels
    - Highlighting specific keywords (GPU, Memory, Error, Warning)
    - Colorizing timestamps
    - Supporting bold and underline text formatting
    
    Attributes:
        COLORS (dict): Mapping of text styles to ANSI color codes:
            - DEBUG: Cyan
            - INFO: Green
            - WARNING: Yellow
            - ERROR: Red
            - CRITICAL: Magenta
            - Additional styles for timestamps and special keywords
    
    Example:
        ```python
        handler = logging.StreamHandler()
        handler.setFormatter(ColorFormatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        ```
    """

    # ANSI color code mapping for various log elements
    COLORS = {
        "DEBUG": "\033[0;36m",     # Cyan for debug messages
        "INFO": "\033[0;32m",      # Green for info messages
        "WARNING": "\033[0;33m",   # Yellow for warnings
        "ERROR": "\033[0;31m",     # Red for errors
        "CRITICAL": "\033[0;35m",  # Magenta for critical errors
        "RESET": "\033[0m",        # Reset all formatting
        "BLUE": "\033[0;34m",      # Blue for timestamps and GPU messages
        "MAGENTA": "\033[0;35m",   # Magenta for memory-related messages
        "BOLD": "\033[1m",         # Bold text formatting
        "UNDERLINE": "\033[4m",    # Underlined text formatting
    }

    def formatTime(self, record, datefmt=None):
        """Format the log timestamp with blue coloring.

        Args:
            record (logging.LogRecord): The log record containing timestamp information.
            datefmt (str, optional): Custom date format string. Defaults to None.

        Returns:
            str: Formatted timestamp string with blue coloring.

        Example:
            The timestamp will appear as: [blue]2023-01-01 12:00:00[/blue]
        """
        # Get the basic formatted time using the parent class
        asctime = super().formatTime(record, datefmt)
        # Add blue coloring to the timestamp
        return f"{self.COLORS['BLUE']}{asctime}{self.COLORS['RESET']}"

    def format(self, record):
        """Format the log record with appropriate colors and styling.

        Applies color formatting based on:
        - Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        - Special keywords in the message (GPU, Memory, Error, Warning)

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The fully formatted log message with color coding.

        Example:
            Input message: "GPU utilization at 80%"
            Output: [blue]GPU utilization at 80%[/blue]
        """
        # Add color to the log level name with fixed width padding
        level_color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        record.levelname = f"{level_color}{record.levelname:8}{self.COLORS['RESET']}"

        # Convert message to string and apply keyword-based coloring
        msg = str(record.msg)
        if "GPU" in msg:
            # Highlight GPU-related messages in blue
            record.msg = f"{self.COLORS['BLUE']}{msg}{self.COLORS['RESET']}"
        elif "Memory" in msg:
            # Highlight memory-related messages in magenta
            record.msg = f"{self.COLORS['MAGENTA']}{msg}{self.COLORS['RESET']}"
        elif "Error" in msg or "Failed" in msg:
            # Highlight error messages in red
            record.msg = f"{self.COLORS['ERROR']}{msg}{self.COLORS['RESET']}"
        elif "Warning" in msg:
            # Highlight warning messages in yellow
            record.msg = f"{self.COLORS['WARNING']}{msg}{self.COLORS['RESET']}"
        else:
            # Use level-specific color for other messages
            record.msg = f"{level_color}{msg}{self.COLORS['RESET']}"

        return super().format(record)


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """Configure and return a logger with console and optional file output.

    Sets up a logger with the following features:
    - Console output with color formatting
    - Optional file output with standard formatting
    - INFO level logging by default
    - Thread-safe implementation

    Args:
        log_file (Path, optional): Path to the log file. If None, only console
            logging is enabled. Defaults to None.

    Returns:
        logging.Logger: Configured logger instance ready for use.

    Example:
        ```python
        # Setup with both console and file logging
        logger = setup_logging(Path("app.log"))
        logger.info("Application started")
        
        # Setup with console logging only
        console_logger = setup_logging()
        console_logger.warning("Console only warning")
        ```
    """
    # Create or get logger instance
    logger = logging.getLogger("FluxPipeline")
    logger.setLevel(logging.INFO)
    # Clear any existing handlers to avoid duplication
    logger.handlers.clear()

    # Setup console handler with color formatting
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        ColorFormatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(console_handler)

    # Setup file handler if log file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    return logger


# Global logger instance for project-wide use
# Uses default log file location: flux_pipeline.log
logger = setup_logging(Path("flux_pipeline.log"))
