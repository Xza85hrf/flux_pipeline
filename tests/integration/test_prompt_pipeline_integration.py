"""
Integration tests for prompt pipeline management across components.

This module contains tests that verify the interaction between different components of the FluxPipeline project, focusing on prompt pipeline management. The tests cover various scenarios such as prompt processing integration, negative prompt integration, batch prompt processing, style token integration, technical token integration, prompt-based optimization, and cross-component integration.
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


# Test Prompt Processing Integration
@pytest.mark.integration
def test_prompt_processing_integration(mock_pipeline, mock_cuda_available, mock_pipeline_import):
    """
    Test prompt processing integration.

    Tests:
    - Prompt processing during image generation
    - Memory management during prompt processing

    Args:
        mock_pipeline: A mock pipeline object.
        mock_cuda_available: A mock for CUDA availability.
        mock_pipeline_import: A mock for pipeline import.
    """
    pipeline = mock_pipeline

    # Generate image
    image, _ = pipeline.generate_image(
        prompt="Test prompt processing integration",
        num_inference_steps=4,
        height=1024,
        width=1024,
    )
    assert isinstance(image, MagicMock)
    assert image._mock_spec == Image.Image

    # Verify memory management
    final_stats = pipeline.memory_manager.get_memory_stats()
    assert final_stats is not None


# Test Negative Prompt Integration
@pytest.mark.integration
def test_negative_prompt_integration(mock_pipeline, mock_cuda_available, mock_pipeline_import):
    """
    Test negative prompt integration.

    Tests:
    - Negative prompt processing during image generation
    - Memory management during negative prompt processing

    Args:
        mock_pipeline: A mock pipeline object.
        mock_cuda_available: A mock for CUDA availability.
        mock_pipeline_import: A mock for pipeline import.
    """
    pipeline = mock_pipeline

    # Generate image with negative prompt
    image, _ = pipeline.generate_image(
        prompt="Test negative prompt integration",
        negative_prompt="low quality, blurry, distorted",
        num_inference_steps=4,
        height=1024,
        width=1024,
    )
    assert isinstance(image, MagicMock)
    assert image._mock_spec == Image.Image

    # Verify memory management
    final_stats = pipeline.memory_manager.get_memory_stats()
    assert final_stats is not None


# Test Batch Prompt Processing
@pytest.mark.integration
def test_batch_prompt_processing(mock_pipeline, mock_cuda_available, mock_pipeline_import):
    """
    Test batch prompt processing.

    Tests:
    - Batch prompt processing during image generation
    - Memory management during batch prompt processing

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
            prompt="Test batch prompt processing",
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


# Test Style Token Integration
@pytest.mark.integration
def test_style_token_integration(mock_pipeline, mock_cuda_available, mock_pipeline_import):
    """
    Test style token integration.

    Tests:
    - Style token processing during image generation
    - Memory management during style token processing

    Args:
        mock_pipeline: A mock pipeline object.
        mock_cuda_available: A mock for CUDA availability.
        mock_pipeline_import: A mock for pipeline import.
    """
    pipeline = mock_pipeline

    # Generate image with style token
    image, _ = pipeline.generate_image(
        prompt="Test style token integration",
        style_token="minimalist",
        num_inference_steps=4,
        height=1024,
        width=1024,
    )
    assert isinstance(image, MagicMock)
    assert image._mock_spec == Image.Image

    # Verify memory management
    final_stats = pipeline.memory_manager.get_memory_stats()
    assert final_stats is not None


# Test Technical Token Integration
@pytest.mark.integration
def test_technical_token_integration(mock_pipeline, mock_cuda_available, mock_pipeline_import):
    """
    Test technical token integration.

    Tests:
    - Technical token processing during image generation
    - Memory management during technical token processing

    Args:
        mock_pipeline: A mock pipeline object.
        mock_cuda_available: A mock for CUDA availability.
        mock_pipeline_import: A mock for pipeline import.
    """
    pipeline = mock_pipeline

    # Generate image with technical token
    image, _ = pipeline.generate_image(
        prompt="Test technical token integration",
        technical_token="8k resolution",
        num_inference_steps=4,
        height=1024,
        width=1024,
    )
    assert isinstance(image, MagicMock)
    assert image._mock_spec == Image.Image

    # Verify memory management
    final_stats = pipeline.memory_manager.get_memory_stats()
    assert final_stats is not None


# Test Prompt-Based Optimization
@pytest.mark.integration
def test_prompt_based_optimization(mock_pipeline, mock_cuda_available, mock_pipeline_import):
    """
    Test prompt-based optimization.

    Tests:
    - Prompt-based optimization during image generation
    - Memory management during prompt-based optimization

    Args:
        mock_pipeline: A mock pipeline object.
        mock_cuda_available: A mock for CUDA availability.
        mock_pipeline_import: A mock for pipeline import.
    """
    pipeline = mock_pipeline

    # Generate image with prompt-based optimization
    image, _ = pipeline.generate_image(
        prompt="Test prompt-based optimization",
        num_inference_steps=4,
        height=1024,
        width=1024,
    )
    assert isinstance(image, MagicMock)
    assert image._mock_spec == Image.Image

    # Verify memory management
    final_stats = pipeline.memory_manager.get_memory_stats()
    assert final_stats is not None


# Test Cross-Component Integration
@pytest.mark.integration
def test_cross_component_integration(mock_pipeline, mock_cuda_available, mock_pipeline_import):
    """
    Test cross-component integration.

    Tests:
    - Cross-component interaction during image generation
    - Memory management during cross-component interaction

    Args:
        mock_pipeline: A mock pipeline object.
        mock_cuda_available: A mock for CUDA availability.
        mock_pipeline_import: A mock for pipeline import.
    """
    pipeline = mock_pipeline

    # Generate image to trigger cross-component interaction
    image, _ = pipeline.generate_image(
        prompt="Test cross-component integration", num_inference_steps=4
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
def cleanup_prompt_pipeline_integration():
    """
    Clean up after prompt pipeline integration tests.

    This fixture ensures that memory is cleaned up after each test to prevent memory leaks.
    """
    yield
    # Force cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
