"""FluxPipeline - AI Image Generation Framework using FLUX.1-schnell.

This package provides a comprehensive framework for AI-powered image generation
with support for multiple GPU vendors, advanced memory management, and both
GUI and CLI interfaces.

Key Features:
    - Multi-vendor GPU support (NVIDIA CUDA, AMD ROCm, Intel OneAPI)
    - Advanced memory management and optimization
    - Web-based GUI interface (Gradio)
    - Command-line interface
    - Batch processing and GIF generation
    - Reproducible generation with seed management

Example:
    Basic usage::

        from pipeline.flux_pipeline import FluxPipeline

        # Initialize pipeline
        pipeline = FluxPipeline()
        pipeline.load_model()

        # Generate image
        image, seed = pipeline.generate_image(
            prompt="A beautiful sunset over mountains",
            num_inference_steps=4
        )

License:
    CC BY-NC 4.0 - Not for commercial use

See Also:
    - README.md: Full documentation
    - CONTRIBUTING.md: Contribution guidelines
    - SECURITY.md: Security policy
"""

from ._version import __version__, __version_info__, get_version

__all__ = [
    "__version__",
    "__version_info__",
    "get_version",
]

# Package metadata
__author__ = "Xza85hrf"
__license__ = "CC BY-NC 4.0"
__url__ = "https://github.com/Xza85hrf/flux_pipeline"
