"""Global test configuration and fixtures for the Flux Pipeline project.

This module provides shared fixtures and configuration for all test modules,
including mock GPU environments, test data generators, and common test utilities.
"""

import sys
import os
import pytest
import torch
import PIL.Image
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Generator
import asyncio

# Add the main package directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def pytest_configure(config):
    """Configure pytest with custom settings."""
    try:
        import torch.cuda
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            torch.cuda.init()
        else:
            print("WARNING: CUDA is not available. Tests requiring GPU will be skipped.")
    except Exception as e:
        print(f"WARNING: Error initializing CUDA: {e}")
        print("Tests requiring GPU will be skipped.")

def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if CUDA is not available."""
    try:
        import torch.cuda
        if not torch.cuda.is_available():
            skip_gpu = pytest.mark.skip(reason="CUDA is not available")
            for item in items:
                if "gpu" in item.keywords:
                    item.add_marker(skip_gpu)
    except Exception as e:
        print(f"WARNING: Error checking CUDA availability: {e}")
        skip_gpu = pytest.mark.skip(reason="Error checking CUDA availability")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

# Mock Image Fixtures
@pytest.fixture
def mock_pil_image():
    """Create a properly configured PIL Image mock."""
    image_mock = MagicMock(spec=PIL.Image.Image)
    image_mock._mock_spec = PIL.Image.Image
    return image_mock

@pytest.fixture
def mock_image_generator(mock_pil_image):
    """Mock image generator that returns PIL Image mocks."""
    def generate_image(*args, **kwargs):
        return mock_pil_image
    return generate_image

# Mock GPU Environment Fixtures
@pytest.fixture
def mock_cuda_available():
    """Mock CUDA availability for testing."""
    with patch("torch.cuda.is_available", return_value=True) as mock:
        yield mock

@pytest.fixture
def mock_hip():
    """Mock AMD HIP support."""
    mock_hip = MagicMock()
    mock_hip.is_available.return_value = True
    mock_hip.device_count.return_value = 2
    with patch.object(torch, 'hip', mock_hip):
        yield mock_hip

@pytest.fixture
def mock_cuda_device_count():
    """Mock CUDA device count for testing."""
    with patch("torch.cuda.device_count", return_value=2) as mock:
        yield mock

@pytest.fixture
def mock_gpu_properties():
    """Mock GPU device properties for testing."""
    properties = Mock()
    properties.name = "NVIDIA GeForce RTX 3090 Ti"
    properties.total_memory = 24 * 1024 * 1024 * 1024  # 24GB
    properties.major = 8
    properties.minor = 6
    with patch("torch.cuda.get_device_properties", return_value=properties) as mock:
        yield mock

# Test Data Fixtures
@pytest.fixture
def test_prompt() -> str:
    """Provide a test prompt for image generation."""
    return "A test prompt for image generation with specific details and style"

@pytest.fixture
def test_negative_prompt() -> str:
    """Provide a test negative prompt."""
    return "low quality, blurry, distorted"

@pytest.fixture
def test_workspace(tmp_path) -> Path:
    """Create a temporary workspace for testing."""
    workspace = tmp_path / "test_workspace"
    workspace.mkdir(exist_ok=True)
    (workspace / "outputs").mkdir(exist_ok=True)
    (workspace / "cache").mkdir(exist_ok=True)
    (workspace / "logs").mkdir(exist_ok=True)
    return workspace

# Mock Model Fixtures
@pytest.fixture
def mock_flux_model(mock_pil_image):
    """Mock Flux model for testing."""
    model = Mock()
    images = [mock_pil_image]
    model.generate.return_value = Mock(images=images)
    with patch("diffusers.FluxPipeline.from_pretrained", return_value=model) as mock:
        yield mock

# Memory Management Fixtures
@pytest.fixture
def mock_memory_stats() -> Dict[str, Any]:
    """Provide mock memory statistics."""
    return {
        "allocated": 2 * 1024 * 1024 * 1024,  # 2GB
        "reserved": 4 * 1024 * 1024 * 1024,  # 4GB
        "free": 4 * 1024 * 1024 * 1024,  # 4GB
        "pressure": 50.0  # Add pressure metric
    }

@pytest.fixture
def mock_system_memory():
    """Mock system memory information."""
    memory = Mock()
    memory.total = 16 * 1024 * 1024 * 1024  # 16GB
    memory.available = 8 * 1024 * 1024 * 1024  # 8GB
    memory.percent = 50.0
    with patch("psutil.virtual_memory", return_value=memory) as mock:
        yield mock

# Error Simulation Fixtures
@pytest.fixture
def mock_cuda_error():
    """Simulate CUDA out of memory error."""
    def raise_oom(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")
    return raise_oom

@pytest.fixture
def mock_runtime_error():
    """Simulate runtime error."""
    def raise_runtime(*args, **kwargs):
        raise RuntimeError("Simulated runtime error")
    return raise_runtime

# Performance Test Utilities
@pytest.fixture
def performance_tracker():
    """Track performance metrics during tests."""
    class PerfTracker:
        def __init__(self):
            self.start_memory = 0
            self.peak_memory = 0
            self.execution_time = 0.0

        def start(self):
            try:
                import torch.cuda
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    self.start_memory = torch.cuda.memory_allocated()
            except Exception as e:
                print(f"Warning: Could not start performance tracking: {e}")

        def stop(self):
            try:
                import torch.cuda
                if torch.cuda.is_available():
                    self.peak_memory = torch.cuda.max_memory_allocated()
                    current_memory = torch.cuda.memory_allocated()
                    torch.cuda.empty_cache()
                    return {
                        "start_memory": self.start_memory,
                        "peak_memory": self.peak_memory,
                        "end_memory": current_memory,
                    }
            except Exception as e:
                print(f"Warning: Could not stop performance tracking: {e}")
            return {}

    return PerfTracker()

# Test Data Generators
@pytest.fixture
def generate_test_prompts() -> Generator[str, None, None]:
    """Generate test prompts."""
    prompts = [
        "A simple test prompt",
        "A complex prompt with multiple elements and style descriptions",
        "An extremely detailed prompt with specific requirements",
        "A prompt with technical specifications and parameters",
    ]
    for prompt in prompts:
        yield prompt

@pytest.fixture
def generate_test_seeds() -> Generator[int, None, None]:
    """Generate test seeds."""
    seeds = [42, 123, 7777, 999999]
    for seed in seeds:
        yield seed

# Environment Setup Utilities
@pytest.fixture
def setup_test_env(test_workspace):
    """Setup test environment with required directories and files."""
    def _setup(additional_dirs: list = None):
        if additional_dirs:
            for dir_name in additional_dirs:
                (test_workspace / dir_name).mkdir(exist_ok=True)
        return test_workspace
    return _setup

# Pipeline Import Mock
@pytest.fixture(autouse=True)
def mock_pipeline_import():
    """Mock pipeline import to prevent import errors."""
    with patch.dict('sys.modules', {
        'diffusers': Mock(),
        'diffusers.pipelines': Mock(),
        'diffusers.pipelines.flux': Mock(),
        'diffusers.pipelines.flux.pipeline_flux': Mock(),
        'transformers': Mock(),
        'transformers.models': Mock(),
        'transformers.models.clip': Mock(),
        'transformers.models.clip.modeling_clip': Mock(),
    }):
        yield

# Cleanup Utilities
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    try:
        import torch.cuda
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        print(f"Warning: Could not cleanup CUDA resources: {e}")
    import gc
    gc.collect()

# Test Environment Setup
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment."""
    try:
        import torch.cuda
        if torch.cuda.is_available():
            torch.cuda.init()
            torch.set_default_device('cuda')
    except Exception as e:
        print(f"Warning: Could not setup CUDA environment: {e}")
    yield
    try:
        import torch.cuda
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        print(f"Warning: Could not cleanup CUDA environment: {e}")

# File System Fixtures
@pytest.fixture
def setup_log_directories(tmp_path):
    """Setup log directories for testing."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir

# Token Group Fixtures
@pytest.fixture
def token_groups():
    """Provide mock token groups for testing."""
    return {
        "quality": ["high quality", "detailed", "professional"],
        "style": ["portrait", "minimalist", "realistic"],
        "technical": ["8k resolution", "clean lines", "lighting"]
    }
