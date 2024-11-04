"""Unit tests for logging utilities."""

import pytest
import os
import time
from pathlib import Path
from unittest.mock import patch, Mock
from datetime import datetime
from utils.logging_utils import LogManager


def test_log_manager_init(tmp_path):
    """Test LogManager initialization and configuration."""
    manager = LogManager(base_log_dir=tmp_path)
    assert manager.base_log_dir == tmp_path
    assert manager.base_log_dir.exists()


def test_multiple_session_handling(tmp_path):
    """Test handling of multiple logging sessions."""
    # Create a unique subdirectory for this test to avoid interference
    test_dir = tmp_path / "test_multiple_sessions"
    test_dir.mkdir()

    # Create two timestamps for testing
    timestamp1 = datetime(2023, 1, 1, 12, 0, 0)
    timestamp2 = datetime(2023, 1, 1, 12, 0, 1)

    # Mock both datetime.now() and the session_start attribute
    with patch("datetime.datetime") as mock_dt:
        # First manager
        mock_dt.now.return_value = timestamp1
        manager1 = LogManager(base_log_dir=test_dir)
        # Explicitly set the session_start to ensure it uses our mocked time
        manager1.session_start = timestamp1
        path1 = manager1.get_session_log_path()

        # Second manager
        mock_dt.now.return_value = timestamp2
        manager2 = LogManager(base_log_dir=test_dir)
        # Explicitly set the session_start to ensure it uses our mocked time
        manager2.session_start = timestamp2
        path2 = manager2.get_session_log_path()

        # Create the log files
        path1.parent.mkdir(parents=True, exist_ok=True)
        path2.parent.mkdir(parents=True, exist_ok=True)
        path1.touch()
        path2.touch()

        # Get the expected filenames
        expected_file1 = f"session_{timestamp1.strftime('%Y%m%d_%H%M%S')}.log"
        expected_file2 = f"session_{timestamp2.strftime('%Y%m%d_%H%M%S')}.log"

        # Verify paths are different and files exist
        assert path1.exists(), "First log file should exist"
        assert path2.exists(), "Second log file should exist"
        assert (
            path1.name == expected_file1
        ), f"Expected {expected_file1}, got {path1.name}"
        assert (
            path2.name == expected_file2
        ), f"Expected {expected_file2}, got {path2.name}"
        assert path1 != path2, "Log paths should be different"
        assert str(path1) != str(path2), "Log paths as strings should be different"


@pytest.mark.parametrize(
    "scenario",
    [
        "invalid_directory",
        "concurrent_file_access",
    ],
)
def test_logging_edge_cases(scenario, tmp_path):
    """Test logging edge cases."""
    if scenario == "invalid_directory":
        with patch("pathlib.Path.mkdir", side_effect=PermissionError):
            with pytest.raises(Exception):
                LogManager(base_log_dir=tmp_path)

    elif scenario == "concurrent_file_access":
        # Create a unique subdirectory for this test
        test_dir = tmp_path / "test_concurrent"
        test_dir.mkdir()

        timestamp = datetime(2023, 1, 1, 12, 0, 0)

        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = timestamp

            # Create first manager and its log file
            manager1 = LogManager(base_log_dir=test_dir)
            manager1.session_start = timestamp
            path1 = manager1.get_session_log_path()
            path1.parent.mkdir(parents=True, exist_ok=True)
            path1.touch()

            # Create second manager with same timestamp
            manager2 = LogManager(base_log_dir=test_dir)
            manager2.session_start = timestamp
            path2 = manager2.get_session_log_path()

            # Attempt to create the file, which should raise FileExistsError
            with pytest.raises(FileExistsError):
                path2.touch(exist_ok=False)


# Add more tests for other LogManager functionality
def test_performance_logging(tmp_path):
    """Test performance logging functionality."""
    manager = LogManager(base_log_dir=tmp_path)

    # Test logging performance metrics
    operation = "test_operation"
    duration = 1.5
    details = {"param1": "value1", "param2": 42}

    manager.log_performance(operation, duration, details)

    # Verify performance log file exists and contains the entry
    perf_log_path = tmp_path / "performance.log"
    assert perf_log_path.exists()

    # Read and verify log content
    content = perf_log_path.read_text()
    assert operation in content
    assert str(duration) in content
    assert all(str(value) in content for value in details.values())


def test_log_operation_context(tmp_path):
    """Test log_operation context manager."""
    manager = LogManager(base_log_dir=tmp_path)

    # Test successful operation
    with manager.log_operation("test_operation", param="test"):
        # Simulate some work
        time.sleep(0.1)

    # Verify performance log was created
    assert len(manager.performance_logs) == 1
    assert manager.performance_logs[0]["operation"] == "test_operation"
    assert manager.performance_logs[0]["param"] == "test"

    # Test operation with exception
    with pytest.raises(ValueError):
        with manager.log_operation("failed_operation"):
            raise ValueError("Test error")

    # Verify error case was logged
    assert len(manager.performance_logs) == 2
    assert manager.performance_logs[1]["operation"] == "failed_operation"


def test_log_operation_timing_accuracy(tmp_path):
    """Test accuracy of operation timing measurements."""
    manager = LogManager(base_log_dir=tmp_path)

    sleep_duration = 0.1
    tolerance = 0.05  # 50ms tolerance

    with manager.log_operation("timing_test"):
        time.sleep(sleep_duration)

    # Verify timing accuracy
    logged_duration = manager.performance_logs[-1]["duration"]
    assert (
        abs(logged_duration - sleep_duration) < tolerance
    ), f"Expected duration ~{sleep_duration}s, got {logged_duration}s"


def test_concurrent_operations(tmp_path):
    """Test handling of nested/concurrent operations."""
    manager = LogManager(base_log_dir=tmp_path)

    with manager.log_operation("outer"):
        with manager.log_operation("inner1"):
            time.sleep(0.1)
        with manager.log_operation("inner2"):
            time.sleep(0.1)

    # Verify all operations were logged
    operations = [log["operation"] for log in manager.performance_logs]
    assert "outer" in operations
    assert "inner1" in operations
    assert "inner2" in operations

    # Verify timing relationships
    outer_time = next(
        log["duration"]
        for log in manager.performance_logs
        if log["operation"] == "outer"
    )
    inner_times = sum(
        log["duration"]
        for log in manager.performance_logs
        if log["operation"].startswith("inner")
    )

    # Outer operation should take at least as long as sum of inner operations
    assert outer_time >= inner_times


def test_log_operation_memory_leak(tmp_path):
    """Test for memory leaks in long-running logging."""
    manager = LogManager(base_log_dir=tmp_path)

    import psutil

    process = psutil.Process()
    initial_memory = process.memory_info().rss

    # Run many operations
    for _ in range(1000):
        with manager.log_operation("memory_test"):
            pass

    final_memory = process.memory_info().rss
    memory_growth = final_memory - initial_memory

    # Allow for some memory overhead, but catch significant leaks
    assert memory_growth < 10 * 1024 * 1024  # 10MB limit
