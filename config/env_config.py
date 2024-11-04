"""Environment configuration and hardware management module.

This module handles environment setup and hardware-specific configurations for the Flux AI pipeline.
It provides functionality to:
- Detect and configure available GPU devices across vendors (NVIDIA, AMD, Intel)
- Set optimal environment variables for deep learning workloads
- Manage hardware-specific package requirements
- Define default configurations for models and generation settings

Example:
    Basic usage to setup environment:
    ```python
    from config.env_config import setup_environment
    
    # Setup environment with default settings
    required_packages = setup_environment()
    
    # Force environment variable updates
    required_packages = setup_environment(force=True)
    ```

Note:
    This module automatically handles multi-vendor GPU support and will configure
    the environment appropriately based on available hardware.
"""

import os
import torch
from typing import Dict, List


def _get_gpu_device_list() -> str:
    """Get a comma-separated list of available GPU devices from all vendors.

    Checks for available GPUs across multiple vendors:
    - NVIDIA GPUs via PyTorch CUDA
    - AMD GPUs via ROCm/HIP
    - Intel GPUs via Intel Extension for PyTorch

    Returns:
        str: Comma-separated list of GPU device indices (e.g., "0,1,2"),
             or empty string if no GPUs are available.

    Example:
        ```python
        devices = _get_gpu_device_list()
        print(f"Available GPU devices: {devices}")  # e.g., "0,1" for 2 GPUs
        ```
    """
    devices = []

    # Check NVIDIA GPUs through CUDA
    if torch.cuda.is_available():
        devices.extend(range(torch.cuda.device_count()))

    # Check AMD GPUs through ROCm/HIP support
    try:
        import torch_hip

        if torch_hip.is_available():
            devices.extend(range(torch_hip.device_count()))
    except ImportError:
        pass

    # Check Intel GPUs through Intel Extension for PyTorch
    try:
        import intel_extension_for_pytorch as ipex

        if ipex.xpu.is_available():
            devices.extend(range(ipex.xpu.device_count()))
    except ImportError:
        pass

    return ",".join(map(str, devices)) if devices else ""


# Environment variables configuration dictionary
ENVIRONMENT_VARS: Dict[str, str] = {
    # PyTorch and CUDA configurations
    "CUDA_VISIBLE_DEVICES": _get_gpu_device_list(),
    "PYTORCH_CUDA_ARCH_LIST": "8.6",
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    # CUDA specific settings
    "CUDA_LAUNCH_BLOCKING": "1",
    "KMP_DUPLICATE_LIB_OK": "True",
    # Hugging Face configurations
    "TRANSFORMERS_OFFLINE": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "DIFFUSERS_VERBOSITY": "error",
    "TOKENIZERS_PARALLELISM": "true",
    # TensorFlow and warning suppressions
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "PYTHONWARNINGS": "ignore",
    # AMD GPU support
    "HSA_OVERRIDE_GFX_VERSION": "10.3.0",
    "HIP_VISIBLE_DEVICES": _get_gpu_device_list(),
    # Intel GPU support
    "INTEL_EXTENSION_FOR_PYTORCH": "1",
    "SYCL_DEVICE_FILTER": "gpu",
}


def get_required_packages() -> List[str]:
    """Determine required Python packages based on available hardware.

    Checks available hardware and returns a list of necessary packages for
    optimal performance. This includes base PyTorch and hardware-specific
    packages for NVIDIA, AMD, or Intel GPUs.

    Returns:
        List[str]: List of required package names with version specifications.

    Example:
        ```python
        packages = get_required_packages()
        print("Required packages:", packages)
        # Output: ['torch>=2.0.0', 'torch-cuda', ...]
        ```
    """
    packages = ["torch>=2.0.0"]

    # Add NVIDIA support if CUDA is available
    if torch.cuda.is_available():
        packages.append("torch-cuda")

    # Add AMD support if ROCm/HIP is available
    try:
        import torch_hip

        packages.append("torch-hip")
    except ImportError:
        pass

    # Add Intel support if available
    try:
        import intel_extension_for_pytorch

        packages.append("intel-extension-for-pytorch")
    except ImportError:
        pass

    return packages


def setup_environment(force: bool = False) -> List[str]:
    """Configure environment variables for optimal performance.

    Sets up environment variables for deep learning workloads and returns
    a list of required packages based on available hardware.

    Args:
        force (bool, optional): If True, overwrites existing environment variables.
            If False, only sets variables that aren't already defined.
            Defaults to False.

    Returns:
        List[str]: List of required package names for the current hardware setup.

    Example:
        ```python
        # Setup environment and get required packages
        packages = setup_environment(force=True)
        print(f"Environment configured. Required packages: {packages}")
        ```
    """
    # Get hardware-specific package requirements
    required_packages = get_required_packages()

    # Configure environment variables based on force parameter
    for key, value in ENVIRONMENT_VARS.items():
        if force or key not in os.environ:
            os.environ[key] = value

    return required_packages


def ensure_environment() -> None:
    """Ensure all required environment variables are properly set.

    This is a convenience function that calls setup_environment with default
    settings to ensure the environment is properly configured without
    overwriting existing variables.

    Example:
        ```python
        # Ensure environment is properly configured
        ensure_environment()
        ```
    """
    setup_environment(force=False)


# Default model configuration settings
DEFAULT_MODEL_CONFIG: Dict[str, any] = {
    "model_id": "<model here>",  # Default model identifier
    "memory_threshold": 0.90,  # Maximum memory utilization threshold (90%)
    "max_retries": 3,  # Maximum retry attempts for model operations
    "enable_xformers": False,  # Memory-efficient attention disabled by default
    "use_fast_tokenizer": True,  # Use fast tokenizer implementation
    "max_prompt_tokens": 77,  # Maximum tokens for input prompts
}

# Default generation parameters
GENERATION_DEFAULTS: Dict[str, any] = {
    "default_steps": 4,  # Number of generation steps
    "guidance_scale": 0.0,  # Guidance scale for generation
    "height": 1024,  # Default output image height
    "width": 1024,  # Default output image width
}
