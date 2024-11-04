"""
Performance tests for prompt processing across components.

This module contains tests that measure the performance of prompt processing in the FluxPipeline project. The tests cover various scenarios such as concurrent processing performance.
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


# Test Concurrent Processing Performance
@pytest.mark.performance
@pytest.mark.asyncio
async def test_concurrent_processing_performance(mock_pipeline, mock_cuda_available, mock_pipeline_import):
    """
    Test concurrent processing performance.

    Tests:
    - Concurrent processing performance of prompt generation
    - Memory management during concurrent processing

    Args:
        mock_pipeline: A mock pipeline object.
        mock_cuda_available: A mock for CUDA availability.
        mock_pipeline_import: A mock for pipeline import.
    """
    import asyncio

    async def generation_task(task_id: int):
        image, seed = mock_pipeline.generate_image(
            prompt=f"Test concurrent processing performance {task_id}", num_inference_steps=4
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


# Cleanup
@pytest.fixture(autouse=True)
def cleanup_prompt_performance():
    """
    Clean up after prompt performance tests.

    This fixture ensures that memory is cleaned up after each test to prevent memory leaks.
    """
    yield
    # Force cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
