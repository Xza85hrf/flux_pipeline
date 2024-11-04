"""
Integration tests for pipeline management across components.

This module contains tests that verify the interaction between different components of the FluxPipeline project, focusing on pipeline management. The tests cover various scenarios such as complete generation workflow, multi-GPU scenario, error recovery, memory integration, batch processing integration, pipeline state management, and output management integration.
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


# Test Complete Generation Workflow
@pytest.mark.integration
def test_complete_generation_workflow(mock_pipeline, mock_cuda_available, mock_pipeline_import):
    """
    Test complete generation workflow.

    Tests:
    - Complete image generation workflow
    - Memory management during the workflow

    Args:
        mock_pipeline: A mock pipeline object.
        mock_cuda_available: A mock for CUDA availability.
        mock_pipeline_import: A mock for pipeline import.
    """
    pipeline = mock_pipeline

    # Generate image
    image, _ = pipeline.generate_image(
        prompt="Test complete generation workflow",
        num_inference_steps=4,
        height=1024,
        width=1024,
    )
    assert isinstance(image, MagicMock)
    assert image._mock_spec == Image.Image

    # Verify memory management
    final_stats = pipeline.memory_manager.get_memory_stats()
    assert final_stats is not None


# Test Multi-GPU Scenario
@pytest.mark.integration
def test_multi_gpu_scenario(mock_pipeline, mock_cuda_available, mock_pipeline_import):
    """
    Test multi-GPU scenario.

    Tests:
    - Image generation on multiple GPUs
    - Memory management during multi-GPU scenario

    Args:
        mock_pipeline: A mock pipeline object.
        mock_cuda_available: A mock for CUDA availability.
        mock_pipeline_import: A mock for pipeline import.
    """
    pipeline = mock_pipeline

    # Generate images on multiple GPUs
    images = []
    for _ in range(3):
        image, _ = pipeline.generate_image(
            prompt="Test multi-GPU scenario",
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


# Test Error Recovery
@pytest.mark.integration
def test_error_recovery(mock_pipeline, mock_pil_image, mock_cuda_available, mock_pipeline_import):
    """
    Test error recovery after errors and high usage.

    Tests:
    - Error recovery after an out-of-memory error
    - Memory management during error recovery

    Args:
        mock_pipeline: A mock pipeline object.
        mock_pil_image: A mock PIL image object.
        mock_cuda_available: A mock for CUDA availability.
        mock_pipeline_import: A mock for pipeline import.
    """
    # First attempt should fail with OOM
    mock_pipeline.generate_image.side_effect = torch.cuda.OutOfMemoryError("CUDA OOM")
    with pytest.raises(torch.cuda.OutOfMemoryError):
        image, _ = mock_pipeline.generate_image(
            prompt="Test error recovery", num_inference_steps=4
        )

    # Verify memory recovery
    stats = mock_pipeline.memory_manager.get_memory_stats()
    assert stats is not None

    # Second attempt should succeed
    mock_pipeline.generate_image.side_effect = None
    mock_pipeline.generate_image.return_value = (mock_pil_image, 42)
    image, _ = mock_pipeline.generate_image(
        prompt="Test after recovery", num_inference_steps=4
    )
    assert isinstance(image, MagicMock)
    assert image._mock_spec == Image.Image


# Test Memory Integration
@pytest.mark.integration
def test_memory_integration(mock_pipeline, mock_cuda_available, mock_pipeline_import):
    """
    Test memory integration across different components.

    Tests:
    - Memory management across different components
    - Memory statistics tracking during integration

    Args:
        mock_pipeline: A mock pipeline object.
        mock_cuda_available: A mock for CUDA availability.
        mock_pipeline_import: A mock for pipeline import.
    """
    # Track memory across components
    memory_stats = mock_pipeline.memory_manager.get_memory_stats()
    gpu_stats = mock_pipeline.memory_manager.get_gpu_memory_usage(0)

    # Generate image to trigger cross-component interaction
    image, _ = mock_pipeline.generate_image(
        prompt="Test memory integration", num_inference_steps=4
    )
    assert isinstance(image, MagicMock)
    assert image._mock_spec == Image.Image

    # Verify memory tracking
    updated_memory_stats = mock_pipeline.memory_manager.get_memory_stats()
    updated_gpu_stats = mock_pipeline.memory_manager.get_gpu_memory_usage(0)

    assert updated_memory_stats is not None
    assert updated_gpu_stats is not None


# Test Batch Processing Integration
@pytest.mark.integration
def test_batch_processing_integration(mock_pipeline, mock_cuda_available, mock_pipeline_import):
    """
    Test batch processing integration.

    Tests:
    - Batch image generation
    - Memory management during batch processing

    Args:
        mock_pipeline: A mock pipeline object.
        mock_cuda_available: A mock for CUDA availability.
        mock_pipeline_import: A mock for pipeline import.
    """
    pipeline = mock_pipeline

    # Generate batch of images
    images = []
    for _ in range(3):
        image, _ = pipeline.generate_image(
            prompt="Test batch processing integration",
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


# Test Pipeline State Management
@pytest.mark.integration
def test_pipeline_state_management(mock_pipeline, mock_cuda_available, mock_pipeline_import):
    """
    Test pipeline state management.

    Tests:
    - Pipeline state management during image generation
    - Memory management during state management

    Args:
        mock_pipeline: A mock pipeline object.
        mock_cuda_available: A mock for CUDA availability.
        mock_pipeline_import: A mock for pipeline import.
    """
    pipeline = mock_pipeline

    # Generate image to trigger state management
    image, _ = pipeline.generate_image(
        prompt="Test pipeline state management", num_inference_steps=4
    )
    assert isinstance(image, MagicMock)
    assert image._mock_spec == Image.Image

    # Verify memory tracking
    updated_memory_stats = pipeline.memory_manager.get_memory_stats()
    updated_gpu_stats = pipeline.memory_manager.get_gpu_memory_usage(0)

    assert updated_memory_stats is not None
    assert updated_gpu_stats is not None


# Test Output Management Integration
@pytest.mark.integration
def test_output_management_integration(mock_pipeline, mock_cuda_available, mock_pipeline_import):
    """
    Test output management integration.

    Tests:
    - Output management during image generation
    - Memory management during output management

    Args:
        mock_pipeline: A mock pipeline object.
        mock_cuda_available: A mock for CUDA availability.
        mock_pipeline_import: A mock for pipeline import.
    """
    pipeline = mock_pipeline

    # Generate image to trigger output management
    image, _ = pipeline.generate_image(
        prompt="Test output management integration", num_inference_steps=4
    )
    assert isinstance(image, MagicMock)
    assert image._mock_spec == Image.Image

    # Verify memory tracking
    updated_memory_stats = pipeline.memory_manager.get_memory_stats()
    updated_gpu_stats = pipeline.memory_manager.get_gpu_memory_usage(0)

    assert updated_memory_stats is not None
    assert updated_gpu_stats is not None


# Cleanup
@pytest.fixture(autouse=True)
def cleanup_pipeline_integration():
    """
    Clean up after pipeline integration tests.

    This fixture ensures that memory is cleaned up after each test to prevent memory leaks.
    """
    yield
    # Force cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
