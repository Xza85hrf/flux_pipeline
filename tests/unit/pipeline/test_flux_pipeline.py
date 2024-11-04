"""Unit tests for Flux Pipeline module."""

import pytest
import torch
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
from PIL import Image
from pipeline.flux_pipeline import FluxPipeline
from core.seed_manager import SeedProfile


@pytest.fixture
def mock_cuda_available():
    """Mock CUDA availability."""
    with patch("torch.cuda.is_available", return_value=True) as mock:
        yield mock


@pytest.fixture
def mock_flux_model():
    """Mock Flux model with basic functionality."""
    mock = Mock()
    mock.from_pretrained.return_value = Mock(spec=["enable_attention_slicing"])
    return mock


@pytest.mark.parametrize(
    "cuda_available,expected_device",
    [
        (True, "cuda"),
        (False, "cpu"),
    ],
)
def test_model_loading(
    mock_cuda_available, mock_flux_model, cuda_available, expected_device
):
    """Test model loading with different device configurations."""
    with patch("torch.cuda.is_available", return_value=cuda_available):
        # Create a mock pipeline class
        mock_pipeline_class = Mock()
        mock_pipeline_instance = Mock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

        # Patch the safe_import_flux to return our mock
        with patch(
            "pipeline.flux_pipeline.safe_import_flux",
            return_value=(mock_pipeline_class, None),
        ):
            pipeline = FluxPipeline()
            success = pipeline.load_model()

            assert success
            mock_pipeline_class.from_pretrained.assert_called_once()

            # Verify correct device configuration
            if cuda_available:
                mock_pipeline_class.from_pretrained.assert_called_with(
                    pipeline.model_id,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    use_fast_tokenizer=pipeline.use_fast_tokenizer,
                    device_map="balanced",
                )
            else:
                mock_pipeline_class.from_pretrained.assert_called_with(
                    pipeline.model_id,
                    torch_dtype=torch.float32,
                    use_safetensors=True,
                    use_fast_tokenizer=pipeline.use_fast_tokenizer,
                )


@pytest.mark.parametrize(
    "error_scenario",
    [
        ("cuda_error", torch.cuda.OutOfMemoryError("CUDA out of memory")),
        ("runtime_error", RuntimeError("Model error")),
        ("value_error", ValueError("Invalid parameter")),
    ],
)
def test_error_handling(error_scenario):
    """Test error handling during generation."""
    scenario_name, error = error_scenario
    pipeline = FluxPipeline()

    # Mock the load_model to return True and set a mock pipe
    with patch.object(pipeline, "load_model", return_value=True):
        # Set up the pipeline mock
        pipeline.pipe = Mock()
        pipeline.pipe.return_value = Mock(images=[None])

        # Mock the memory manager device
        pipeline.memory_manager.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set up error simulation
        if isinstance(error, RuntimeError):
            # For RuntimeError, simulate OOM behavior with retry
            pipeline.pipe.side_effect = [
                torch.cuda.OutOfMemoryError(
                    "CUDA out of memory"
                ),  # First attempt fails
                Mock(images=[None]),  # Second attempt succeeds but returns no image
            ]
        else:
            # For other errors, simulate single failure
            pipeline.pipe.side_effect = error

        # Mock seed generation
        test_seed = 42
        with patch.object(
            pipeline.seed_manager, "generate_seed", return_value=test_seed
        ), patch.object(
            pipeline.seed_manager, "_validate_seed", return_value=test_seed
        ):

            if isinstance(error, RuntimeError):
                # Test OOM retry behavior
                with patch.object(pipeline, "_handle_oom_error"):
                    image, seed = pipeline.generate_image(prompt="Test error handling")
                    assert image is None
                    assert seed == test_seed
            elif isinstance(error, torch.cuda.OutOfMemoryError):
                # Test CUDA OOM handling
                with patch.object(pipeline, "_handle_oom_error"):
                    image, seed = pipeline.generate_image(prompt="Test error handling")
                    assert image is None
                    assert seed == test_seed
            else:
                # For other errors (like ValueError), expect (None, None)
                image, seed = pipeline.generate_image(prompt="Test error handling")
                assert image is None
                assert (
                    seed is None
                )  # Changed: ValueError should return None for both image and seed


@pytest.mark.parametrize(
    "scenario",
    [
        "empty_prompt",
        "very_long_prompt",
        "special_characters",
        "extreme_dimensions",
    ],
)
def test_edge_cases(scenario):
    """Test pipeline edge cases."""
    pipeline = FluxPipeline()

    # Mock the load_model to return True
    with patch.object(pipeline, "load_model", return_value=True):
        # Create a mock image
        mock_image = MagicMock(spec=Image.Image)
        pipeline.pipe = Mock()
        pipeline.pipe.return_value = Mock(images=[mock_image])

        # Mock prompt processing
        with patch.object(pipeline.prompt_manager, "process_prompt") as mock_process:
            if scenario == "empty_prompt":
                mock_process.return_value = ""
                image, seed = pipeline.generate_image(prompt="")
                assert image is None

            elif scenario == "very_long_prompt":
                mock_process.return_value = (
                    "processed " * 10
                )  # Simulated processed prompt
                long_prompt = "test " * 100
                image, seed = pipeline.generate_image(prompt=long_prompt)
                assert isinstance(image, mock_image.__class__)

            elif scenario == "special_characters":
                mock_process.return_value = "processed special chars"
                image, seed = pipeline.generate_image(prompt="Test!@#$%^&*()")
                assert isinstance(image, mock_image.__class__)

            elif scenario == "extreme_dimensions":
                pipeline.pipe.side_effect = RuntimeError("Dimensions too large")
                image, seed = pipeline.generate_image(
                    prompt="Test dimensions", height=2048, width=2048
                )
                assert image is None
                assert seed is not None  # Seed should still be returned


@pytest.mark.parametrize(
    "profile",
    [SeedProfile.CONSERVATIVE, SeedProfile.BALANCED, SeedProfile.EXPERIMENTAL],
)
def test_seed_profile_usage(profile):
    """Test generation with different seed profiles."""
    pipeline = FluxPipeline()

    with patch.object(pipeline, "load_model", return_value=True):
        mock_image = MagicMock(spec=Image.Image)
        pipeline.pipe = Mock(return_value=Mock(images=[mock_image]))

        image, seed = pipeline.generate_image(
            prompt="Test with seed profile", seed_profile=profile
        )

        assert isinstance(image, mock_image.__class__)
        assert seed is not None
        # Verify seed is within profile range
        seed_range = pipeline.seed_manager.profiles[profile]
        assert seed_range.start <= seed <= seed_range.end
