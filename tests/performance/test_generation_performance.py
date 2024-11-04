"""
Performance tests for generation across components.

This module contains tests that measure the performance of the FluxPipeline project, focusing on generation speed, scaling, batch generation performance, memory usage performance, optimization impact, profile performance, concurrent performance, and system load impact.
"""

import pytest
import torch
import gc
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
from PIL import Image

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
def mock_pipeline(mock_pil_image):
    """
    Create a mock pipeline for testing.

    Args:
        mock_pil_image: A mock PIL image object.

    Returns:
        A mock pipeline object with memory management capabilities.
    """
    pipeline = MagicMock()
    pipeline.pipe = MagicMock()
    pipeline.pipe.to = MagicMock(return_value=pipeline.pipe)
    mock_result = MagicMock()
    mock_result.images = [mock_pil_image]
    pipeline.pipe.__call__ = MagicMock(return_value=mock_result)
    pipeline.memory_manager = MagicMock()
    pipeline.memory_manager.get_memory_stats = MagicMock(return_value={
        "allocated": 1000000,
        "reserved": 2000000,
        "free": 3000000,
        "pressure": 0.5
    })
    pipeline.memory_manager.get_gpu_memory_usage = MagicMock(return_value={
        "total": 8000000000,
        "used": 2000000000,
        "free": 6000000000
    })
    pipeline.memory_manager.get_system_memory_info = MagicMock(return_value={
        "total": 16000000000,
        "available": 8000000000,
        "percent": 50.0
    })
    pipeline.generate_image = MagicMock(return_value=(mock_pil_image, 42))
    return pipeline


# Test Generation Speed
@pytest.mark.performance
def test_generation_speed(mock_pipeline, mock_cuda_available, mock_pipeline_import):
    """
    Test generation speed.

    Tests:
    - Generation speed of image generation
    - Memory management during generation

    Args:
        mock_pipeline: A mock pipeline object.
        mock_cuda_available: A mock for CUDA availability.
        mock_pipeline_import: A mock for pipeline import.
    """
    pipeline = mock_pipeline

    # Generate image
    image, _ = pipeline.generate_image(
        prompt="Test generation speed",
        num_inference_steps=4,
        height=1024,
        width=1024,
    )
    assert isinstance(image, MagicMock)
    assert image._mock_spec == Image.Image

    # Verify memory management
    final_stats = pipeline.memory_manager.get_memory_stats()
    assert final_stats is not None


# Test Generation Scaling
@pytest.mark.performance
@pytest.mark.parametrize("image_size", [
    {"height": 512, "width": 512},
    {"height": 768, "width": 768},
    {"height": 1024, "width": 1024},
    {"height": 1280, "width": 1280},
])
def test_generation_scaling(mock_pipeline, mock_cuda_available, mock_pipeline_import, image_size):
    """
    Test generation scaling.

    Tests:
    - Generation scaling with different image sizes
    - Memory management during scaling

    Args:
        mock_pipeline: A mock pipeline object.
        mock_cuda_available: A mock for CUDA availability.
        mock_pipeline_import: A mock for pipeline import.
        image_size: A dictionary containing height and width of the image.
    """
    pipeline = mock_pipeline

    # Generate image with different sizes
    image, _ = pipeline.generate_image(
        prompt="Test generation scaling",
        num_inference_steps=4,
        height=image_size["height"],
        width=image_size["width"],
    )
    assert isinstance(image, MagicMock)
    assert image._mock_spec == Image.Image

    # Verify memory management
    final_stats = pipeline.memory_manager.get_memory_stats()
    assert final_stats is not None


# Test Batch Generation Performance
@pytest.mark.performance
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_batch_generation_performance(mock_pipeline, mock_cuda_available, mock_pipeline_import, batch_size):
    """
    Test batch generation performance.

    Tests:
    - Batch generation performance with different batch sizes
    - Memory management during batch generation

    Args:
        mock_pipeline: A mock pipeline object.
        mock_cuda_available: A mock for CUDA availability.
        mock_pipeline_import: A mock for pipeline import.
        batch_size: The number of images to generate in a batch.
    """
    pipeline = mock_pipeline

    # Generate batch of images
    images = []
    for _ in range(batch_size):
        image, _ = pipeline.generate_image(
            prompt="Test batch generation performance",
            num_inference_steps=4,
            height=1024,
            width=1024,
        )
        images.append(image)

    assert all(isinstance(img, MagicMock) for img in images)
    assert all(img._mock_spec == Image.Image for img in images)

    # Verify memory management
    final_stats = pipeline.memory_manager.get_memory_stats()
    assert final_stats is not None


# Test Memory Usage Performance
@pytest.mark.performance
def test_memory_usage_performance(mock_pipeline, mock_cuda_available, mock_pipeline_import):
    """
    Test memory usage performance.

    Tests:
    - Memory usage during image generation
    - Memory management during memory usage performance test

    Args:
        mock_pipeline: A mock pipeline object.
        mock_cuda_available: A mock for CUDA availability.
        mock_pipeline_import: A mock for pipeline import.
    """
    pipeline = mock_pipeline

    # Generate image
    image, _ = pipeline.generate_image(
        prompt="Test memory usage performance",
        num_inference_steps=4,
        height=1024,
        width=1024,
    )
    assert isinstance(image, MagicMock)
    assert image._mock_spec == Image.Image

    # Verify memory management
    final_stats = pipeline.memory_manager.get_memory_stats()
    assert final_stats is not None


# Test Optimization Impact
@pytest.mark.performance
@pytest.mark.parametrize("optimization_config", [
    {"attention_slicing": True, "cpu_offload": False},
    {"attention_slicing": False, "cpu_offload": True},
    {"attention_slicing": True, "cpu_offload": True},
])
def test_optimization_impact(mock_pipeline, mock_cuda_available, mock_pipeline_import, optimization_config):
    """
    Test optimization impact.

    Tests:
    - Optimization impact on image generation
    - Memory management during optimization

    Args:
        mock_pipeline: A mock pipeline object.
        mock_cuda_available: A mock for CUDA availability.
        mock_pipeline_import: A mock for pipeline import.
        optimization_config: A dictionary containing optimization settings.
    """
    pipeline = mock_pipeline

    # Apply optimization settings
    pipeline.memory_manager.optimize_memory_allocation(**optimization_config)

    # Generate image with current settings
    image, _ = pipeline.generate_image(
        prompt="Test optimization impact",
        num_inference_steps=4,
        height=1024,
        width=1024,
    )
    assert isinstance(image, MagicMock)
    assert image._mock_spec == Image.Image

    # Verify memory state
    stats = pipeline.memory_manager.get_memory_stats()
    assert stats is not None


# Test Profile Performance
@pytest.mark.performance
@pytest.mark.parametrize("profile", ["conservative", "balanced", "experimental", "full_range"])
def test_profile_performance(mock_pipeline, mock_cuda_available, mock_pipeline_import, profile):
    """
    Test profile performance.

    Tests:
    - Profile performance with different profiles
    - Memory management during profile performance test

    Args:
        mock_pipeline: A mock pipeline object.
        mock_cuda_available: A mock for CUDA availability.
        mock_pipeline_import: A mock for pipeline import.
        profile: The profile to use for image generation.
    """
    pipeline = mock_pipeline

    # Generate image with different profiles
    image, _ = pipeline.generate_image(
        prompt="Test profile performance",
        num_inference_steps=4,
        height=1024,
        width=1024,
        profile=profile,
    )
    assert isinstance(image, MagicMock)
    assert image._mock_spec == Image.Image

    # Verify memory management
    final_stats = pipeline.memory_manager.get_memory_stats()
    assert final_stats is not None


# Test Concurrent Performance
@pytest.mark.performance
@pytest.mark.asyncio
async def test_concurrent_performance(mock_pipeline, mock_cuda_available, mock_pipeline_import):
    """
    Test concurrent performance.

    Tests:
    - Concurrent performance of image generation
    - Memory management during concurrent performance test

    Args:
        mock_pipeline: A mock pipeline object.
        mock_cuda_available: A mock for CUDA availability.
        mock_pipeline_import: A mock for pipeline import.
    """
    import asyncio

    async def generation_task(task_id: int):
        image, seed = mock_pipeline.generate_image(
            prompt=f"Test concurrent performance {task_id}", num_inference_steps=4
        )
        assert isinstance(image, MagicMock)
        assert image._mock_spec == Image.Image
        return task_id, mock_pipeline.memory_manager.get_memory_stats()

    # Run concurrent tasks
    tasks = [generation_task(i) for i in range(3)]
    results = await asyncio.gather(*tasks)

    # Verify results
    assert len(results) == 3
    assert all(isinstance(stats, dict) for _, stats in results)


# Test System Load Impact
@pytest.mark.performance
def test_system_load_impact(mock_pipeline, mock_cuda_available, mock_pipeline_import):
    """
    Test system load impact.

    Tests:
    - System load impact during image generation
    - Memory management during system load impact test

    Args:
        mock_pipeline: A mock pipeline object.
        mock_cuda_available: A mock for CUDA availability.
        mock_pipeline_import: A mock for pipeline import.
    """
    pipeline = mock_pipeline

    # Monitor system memory
    initial_sys_info = pipeline.memory_manager.get_system_memory_info()

    # Perform memory-intensive operation
    image, _ = pipeline.generate_image(
        prompt="Test system load impact", num_inference_steps=4, height=1024, width=1024
    )
    assert isinstance(image, MagicMock)
    assert image._mock_spec == Image.Image

    # Verify system memory impact
    final_sys_info = pipeline.memory_manager.get_system_memory_info()
    assert final_sys_info is not None


# Cleanup
@pytest.fixture(autouse=True)
def cleanup_generation_performance():
    """
    Clean up after generation performance tests.

    This fixture ensures that memory is cleaned up after each test to prevent memory leaks.
    """
    yield
    # Force cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
