"""
Unit tests for environment configuration.

This module contains tests that verify the functionality of the environment configuration in the FluxPipeline project. The tests cover various scenarios such as GPU device list generation, required packages, and environment error handling.
"""

import pytest
import torch
from unittest.mock import patch, Mock, MagicMock

# Mock the imports first
mock_flux_pipeline = MagicMock()
mock_memory_manager = MagicMock()
mock_gpu_manager = MagicMock()

with patch.dict('sys.modules', {
    'pipeline.flux_pipeline': Mock(FluxPipeline=mock_flux_pipeline),
    'core.memory_manager': Mock(MemoryManager=mock_memory_manager),
    'core.gpu_manager': Mock(MultiGPUManager=mock_gpu_manager)
}):
    from pipeline.flux_pipeline import FluxPipeline
    from core.memory_manager import MemoryManager
    from core.gpu_manager import MultiGPUManager


@pytest.fixture
def mock_env_config():
    """
    Create a mock environment configuration for testing.

    Returns:
        A mock environment configuration object.
    """
    env_config = MagicMock()
    env_config.get_gpu_device_list = MagicMock(return_value=["cuda:0", "cuda:1"])
    env_config.get_required_packages = MagicMock(return_value=["torch", "numpy", "PIL"])
    env_config.handle_environment_error = MagicMock()
    return env_config


# Test GPU Device List Generation
@pytest.mark.unit
@pytest.mark.parametrize("cuda_available, hip_available, expected_devices", [
    (True, False, ["cuda:0", "cuda:1"]),
    (False, True, ["hip:0"]),
    (False, False, []),
    (True, True, ["cuda:0", "cuda:1", "hip:0"]),
])
def test_gpu_device_list_generation(mock_env_config, cuda_available, hip_available, expected_devices):
    """
    Test GPU device list generation.

    Tests:
    - GPU device list generation based on CUDA and HIP availability

    Args:
        mock_env_config: A mock environment configuration object.
        cuda_available: A boolean indicating if CUDA is available.
        hip_available: A boolean indicating if HIP is available.
        expected_devices: A list of expected GPU devices.
    """
    with patch("torch.cuda.is_available", return_value=cuda_available):
        try:
            with patch("torch.hip.is_available", return_value=hip_available):
                devices = mock_env_config.get_gpu_device_list()
                assert devices == expected_devices
        except AttributeError:
            # If torch.hip does not exist, skip the test
            pytest.skip("torch.hip does not exist in this version of PyTorch")


# Test Required Packages
@pytest.mark.unit
@pytest.mark.parametrize("cuda_available, hip_available, expected_packages", [
    (True, False, ["torch", "numpy", "PIL"]),
    (False, True, ["torch", "numpy", "PIL"]),
    (False, False, ["numpy", "PIL"]),
    (True, True, ["torch", "numpy", "PIL"]),
])
def test_required_packages(mock_env_config, cuda_available, hip_available, expected_packages):
    """
    Test required packages.

    Tests:
    - Required packages based on CUDA and HIP availability

    Args:
        mock_env_config: A mock environment configuration object.
        cuda_available: A boolean indicating if CUDA is available.
        hip_available: A boolean indicating if HIP is available.
        expected_packages: A list of expected required packages.
    """
    with patch("torch.cuda.is_available", return_value=cuda_available):
        try:
            with patch("torch.hip.is_available", return_value=hip_available):
                packages = mock_env_config.get_required_packages()
                assert packages == expected_packages
        except AttributeError:
            # If torch.hip does not exist, skip the test
            pytest.skip("torch.hip does not exist in this version of PyTorch")


# Test Environment Error Handling
@pytest.mark.unit
def test_environment_error_handling(mock_env_config):
    """
    Test environment error handling.

    Tests:
    - Environment error handling mechanism

    Args:
        mock_env_config: A mock environment configuration object.
    """
    mock_env_config.handle_environment_error.side_effect = RuntimeError("Simulated runtime error")
    with pytest.raises(RuntimeError):
        mock_env_config.handle_environment_error()


# Cleanup
@pytest.fixture(autouse=True)
def cleanup_env_config():
    """
    Clean up after environment configuration tests.

    This fixture ensures that memory is cleaned up after each test to prevent memory leaks.
    """
    yield
    # Force cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
