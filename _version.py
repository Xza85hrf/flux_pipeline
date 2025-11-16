"""Version information for FluxPipeline.

This module contains version metadata for the FluxPipeline package.
Version numbers follow Semantic Versioning (https://semver.org/).
"""

__version__ = "0.1.0"
__version_info__ = (0, 1, 0)

# Release information
__author__ = "Xza85hrf"
__license__ = "CC BY-NC 4.0"
__url__ = "https://github.com/Xza85hrf/flux_pipeline"

# Version history
VERSION_HISTORY = {
    "0.1.0": {
        "date": "2024-11-16",
        "changes": [
            "Initial release with FLUX.1-schnell support",
            "Multi-GPU support (NVIDIA, AMD, Intel)",
            "Gradio web interface",
            "CLI and interactive modes",
            "Comprehensive testing suite",
            "Professional project infrastructure",
        ]
    }
}

def get_version() -> str:
    """Get the current version string.

    Returns:
        str: Version string in format 'MAJOR.MINOR.PATCH'

    Example:
        >>> from _version import get_version
        >>> print(get_version())
        0.1.0
    """
    return __version__


def get_version_info() -> tuple:
    """Get the current version as a tuple.

    Returns:
        tuple: Version tuple (major, minor, patch)

    Example:
        >>> from _version import get_version_info
        >>> major, minor, patch = get_version_info()
        >>> print(f"Major: {major}, Minor: {minor}, Patch: {patch}")
        Major: 0, Minor: 1, Patch: 0
    """
    return __version_info__
