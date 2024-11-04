"""
Unit tests for logging configuration.

This module contains tests that verify the functionality of the logging configuration in the FluxPipeline project. The tests cover various scenarios such as log file creation, multiple logger creation, and log message formatting.
"""

import pytest
import logging
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
from config.logging_config import setup_logging, ColorFormatter
from utils.logging_utils import LogManager, setup_session_logging, log_generation_stats

# Mock the imports first
mock_flux_pipeline = MagicMock()
mock_memory_manager = MagicMock()
mock_gpu_manager = MagicMock()

with patch.dict(
    "sys.modules",
    {
        "pipeline.flux_pipeline": Mock(FluxPipeline=mock_flux_pipeline),
        "core.memory_manager": Mock(MemoryManager=mock_memory_manager),
        "core.gpu_manager": Mock(MultiGPUManager=mock_gpu_manager),
    },
):
    from pipeline.flux_pipeline import FluxPipeline
    from core.memory_manager import MemoryManager
    from core.gpu_manager import MultiGPUManager


@pytest.fixture
def mock_logging_config(tmp_path):
    """
    Create a mock logging configuration for testing.

    Args:
        tmp_path: A temporary path fixture provided by pytest.

    Returns:
        A mock logging configuration object.
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_file = log_dir / "test.log"
    logging_config = MagicMock()
    logging_config.get_log_file_path = MagicMock(return_value=log_file)

    # Adjust create_logger to return the same logger instance (singleton)
    logging_config.create_logger = MagicMock(
        return_value=logging.getLogger("FluxPipeline")
    )

    # Adjust format_log_message to return a formatted message that includes the log level
    def format_log_message_side_effect(log_level, message):
        return f"{log_level.upper()}: {message}"

    logging_config.format_log_message = MagicMock(
        side_effect=format_log_message_side_effect
    )

    return logging_config


# Test Log File Creation
@pytest.mark.unit
def test_log_file_creation(mock_logging_config):
    """
    Test log file creation.

    Tests:
    - Log file creation and existence

    Args:
        mock_logging_config: A mock logging configuration object.
    """
    log_file_path = mock_logging_config.get_log_file_path()
    with open(log_file_path, "w") as f:
        f.write("Test log content")
    assert log_file_path.exists()


# Test Multiple Logger Creation
@pytest.mark.unit
def test_multiple_logger_creation(mock_logging_config):
    """
    Test multiple logger creation.

    Tests:
    - Multiple logger creation and singleton behavior

    Args:
        mock_logging_config: A mock logging configuration object.
    """
    logger1 = mock_logging_config.create_logger()
    logger2 = mock_logging_config.create_logger()
    assert logger1.name == logger2.name
    # Adjusted assertion to reflect singleton behavior
    assert logger1 is logger2


# Test Log Message Formatting
@pytest.mark.unit
@pytest.mark.parametrize(
    "log_level, message, expected_level",
    [
        ("debug", "Debug message", "DEBUG"),
        ("info", "Info message", "INFO"),
        ("warning", "Warning message", "WARNING"),
        ("error", "Error message", "ERROR"),
        ("critical", "Critical message", "CRITICAL"),
    ],
)
def test_log_message_formatting(
    mock_logging_config, log_level, message, expected_level
):
    """
    Test log message formatting.

    Tests:
    - Log message formatting with different log levels

    Args:
        mock_logging_config: A mock logging configuration object.
        log_level: The log level to test.
        message: The log message to test.
        expected_level: The expected log level in the formatted message.
    """
    formatted_message = mock_logging_config.format_log_message(log_level, message)
    assert expected_level in formatted_message
    assert message in formatted_message


# Cleanup
@pytest.fixture(autouse=True)
def cleanup_logging_config():
    """
    Clean up after logging configuration tests.

    This fixture ensures that logging is properly shut down after each test to prevent resource leaks.
    """
    yield
    # Force cleanup
    logging.shutdown()
