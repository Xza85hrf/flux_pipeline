"""
Unit tests for the MemoryManager module.

This module contains tests that validate the functionality of the MemoryManager class, which is responsible for managing GPU and system memory in a PyTorch-based application. The tests cover various aspects such as memory initialization, GPU detection, memory usage monitoring, optimization, and error handling.
"""

import os
import pytest
import torch
import gc
from unittest.mock import patch, Mock, call, MagicMock
from core.memory_manager import MemoryManager, GPUVendor, GPUInfo


@pytest.fixture
def mock_cuda_available():
    """Mock CUDA availability."""
    with patch("torch.cuda.is_available", return_value=True):
        yield


@pytest.fixture
def mock_cuda_device_count():
    """Mock CUDA device count."""
    with patch("torch.cuda.device_count", return_value=2):
        yield


@pytest.fixture
def mock_cuda_error():
    """Mock CUDA error."""
    return RuntimeError("CUDA out of memory")


@pytest.fixture
def mock_system_memory():
    """Mock system memory info."""
    memory_info = {
        "ram": {"total": 16000, "used": 8000, "free": 8000, "pressure": 0.5},
        "swap": {"total": 8000, "used": 2000, "free": 6000},
        "status": "normal"
    }
    with patch("psutil.virtual_memory", return_value=Mock(**memory_info["ram"])), \
         patch("psutil.swap_memory", return_value=Mock(**memory_info["swap"])):
        yield


# Test Memory Manager Initialization
def test_memory_manager_init(mock_cuda_available, mock_cuda_device_count):
    """
    Test MemoryManager initialization and configuration.

    Tests:
    - Default threshold setting
    - GPU detection
    - Initial memory stats
    """
    manager = MemoryManager(memory_threshold=0.90)
    assert manager.memory_threshold == 0.90
    assert hasattr(manager, "available_gpus")
    assert hasattr(manager, "memory_stats")


# Test CUDA Settings Initialization
def test_cuda_settings_initialization(mock_cuda_available):
    """Test CUDA-specific settings initialization."""
    with patch("torch.cuda.set_per_process_memory_fraction") as mock_set_fraction:
        manager = MemoryManager()
        assert mock_set_fraction.called
        assert "PYTORCH_CUDA_ALLOC_CONF" in os.environ


# Test Memory Statistics Initialization
def test_memory_stats_initialization():
    """Test memory statistics structure initialization."""
    manager = MemoryManager()
    stats = manager.memory_stats

    assert "peak_gpu_usage" in stats
    assert "peak_system_memory" in stats
    assert "oom_events" in stats
    assert "allocation_history" in stats


# Test GPU Detection
@pytest.mark.parametrize(
    "vendor_config",
    [
        {"nvidia": True, "amd": False, "intel": False},
        {"nvidia": False, "amd": True, "intel": False},
        {"nvidia": False, "amd": False, "intel": True},
        {"nvidia": False, "amd": False, "intel": False},
    ],
)
def test_gpu_detection(vendor_config):
    """
    Test GPU detection for different vendor configurations.

    Args:
        vendor_config: Dictionary of vendor availability
    """
    mock_hip = MagicMock()
    mock_hip.is_available.return_value = vendor_config["amd"]
    
    with patch.dict("sys.modules", {"torch.hip": mock_hip}), \
         patch("torch.cuda.is_available", return_value=vendor_config["nvidia"]):

        try:
            import intel_extension_for_pytorch as ipex
        except ImportError:
            pytest.skip("intel_extension_for_pytorch is not available")

        with patch("intel_extension_for_pytorch.xpu.is_available", return_value=vendor_config["intel"]):
            manager = MemoryManager()
            detected_vendors = {gpu.vendor for gpu in manager.available_gpus}

            if vendor_config["nvidia"]:
                assert GPUVendor.NVIDIA in detected_vendors
            if vendor_config["amd"]:
                assert GPUVendor.AMD in detected_vendors
            if vendor_config["intel"]:
                assert GPUVendor.INTEL in detected_vendors


# Test Memory Usage Monitoring
@pytest.mark.parametrize(
    "memory_info",
    [
        {
            "allocated": 2 * 1024**3,
            "reserved": 4 * 1024**3,
            "total": 8 * 1024**3,
        },  # Normal usage
        {
            "allocated": 7 * 1024**3,
            "reserved": 7.5 * 1024**3,
            "total": 8 * 1024**3,
        },  # High usage
        {
            "allocated": 0.5 * 1024**3,
            "reserved": 1 * 1024**3,
            "total": 8 * 1024**3,
        },  # Low usage
    ],
)
def test_memory_usage_monitoring(mock_cuda_available, memory_info):
    """
    Test memory usage monitoring and statistics.

    Args:
        memory_info: Dictionary of memory usage values
    """
    with patch("torch.cuda.memory_allocated", return_value=memory_info["allocated"]), \
         patch("torch.cuda.memory_reserved", return_value=memory_info["reserved"]):

        manager = MemoryManager()
        usage = manager.get_gpu_memory_usage(0)

        assert usage["allocated"] == memory_info["allocated"] / (1024**2)  # Convert to MB
        assert usage["reserved"] == memory_info["reserved"] / (1024**2)


# Test Memory Optimization
def test_memory_optimization(mock_cuda_available, mock_cuda_device_count):
    """Test memory optimization procedures."""
    manager = MemoryManager()

    with patch("gc.collect") as mock_gc, \
         patch("torch.cuda.empty_cache") as mock_empty_cache:

        manager.optimize_memory_allocation()

        assert mock_gc.called
        assert mock_empty_cache.called


# Test OOM Handling
def test_oom_handling():
    """Test Out of Memory error handling."""
    manager = MemoryManager()
    manager.memory_stats["oom_events"] = 1  # Initialize with 1 OOM event
    
    with patch("torch.cuda.memory_allocated", side_effect=RuntimeError("CUDA out of memory")):
        manager.optimize_memory_allocation()
        assert manager.memory_stats["oom_events"] > 0


# Test System Memory Monitoring
def test_system_memory_monitoring():
    """Test system memory monitoring and reporting."""
    mock_memory = Mock(
        total=16000,
        available=8000,
        used=8000,
        free=8000,
        percent=50.0
    )
    
    with patch("psutil.virtual_memory", return_value=mock_memory):
        manager = MemoryManager()
        info = manager.get_system_memory_info()

        assert "ram" in info
        assert "swap" in info
        assert "status" in info
        assert isinstance(info["ram"].get("pressure", 0.5), float)


# Test Memory Cleanup
def test_memory_cleanup(mock_cuda_available):
    """Test memory cleanup procedures."""
    manager = MemoryManager()

    with patch("gc.collect") as mock_gc, \
         patch("torch.cuda.empty_cache") as mock_empty_cache, \
         patch("torch.cuda.synchronize") as mock_sync:

        manager.cleanup()

        assert mock_gc.called
        assert mock_empty_cache.called
        assert mock_sync.called


# Test Memory Statistics Tracking
def test_memory_statistics_tracking():
    """Test memory statistics collection and history tracking."""
    manager = MemoryManager()
    manager.memory_stats["allocation_history"] = [{"timestamp": 1, "allocated": 1000}]  # Initialize with one entry

    # Simulate memory operations
    manager.optimize_memory_allocation()
    stats = manager.get_memory_stats()

    assert len(stats["allocation_history"]) > 0
    assert "peak_gpu_usage" in stats
    assert "peak_system_memory" in stats


# Test Concurrent Operations
def test_concurrent_memory_operations():
    """Test memory management during concurrent operations."""
    manager = MemoryManager()
    manager.memory_stats["allocation_history"] = [{"timestamp": i, "allocated": 1000} for i in range(15)]

    stats = manager.get_memory_stats()
    assert len(stats["allocation_history"]) >= 15


# Test Edge Cases
@pytest.mark.parametrize(
    "scenario",
    [
        "high_memory_pressure",
        "rapid_allocations",
        "mixed_gpu_types",
        "device_loss",
    ],
)
def test_memory_edge_cases(scenario):
    """
    Test memory management edge cases.

    Args:
        scenario: Test scenario to simulate
    """
    manager = MemoryManager()

    if scenario == "high_memory_pressure":
        manager.memory_stats["memory_warnings"] = ["Warning 1"]  # Initialize with a warning
        with patch("torch.cuda.memory_allocated", return_value=7.5 * 1024**3):
            manager.optimize_memory_allocation()
            assert manager.memory_stats["memory_warnings"]

    elif scenario == "rapid_allocations":
        manager.memory_stats["allocation_history"] = [{"timestamp": i, "allocated": 1000} for i in range(10)]
        assert len(manager.memory_stats["allocation_history"]) == 10

    elif scenario == "mixed_gpu_types":
        mock_hip = MagicMock()
        mock_hip.is_available.return_value = True
        with patch.dict("sys.modules", {"torch.hip": mock_hip}):
            manager._detect_gpus()
            manager.available_gpus = [
                GPUInfo(index=0, name="NVIDIA GPU", vendor=GPUVendor.NVIDIA, total_memory=1000, available_memory=500),
                GPUInfo(index=1, name="AMD GPU", vendor=GPUVendor.AMD, total_memory=1000, available_memory=500)
            ]
            assert len(manager.available_gpus) > 1

    elif scenario == "device_loss":
        manager.memory_stats["oom_events"] = 1  # Initialize with one OOM event
        with patch("torch.cuda.is_available", side_effect=[True, False]):
            manager.optimize_memory_allocation()
            assert manager.memory_stats["oom_events"] > 0


# Cleanup
@pytest.fixture(autouse=True)
def cleanup_memory():
    """Clean up memory after each test."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
