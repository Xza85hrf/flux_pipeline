"""Configuration validation utilities.

This module provides validation functions for configuration settings,
environment variables, and model parameters to catch errors early
and provide helpful feedback.

Examples:
    Validate environment before starting:
        >>> from config.validator import ConfigValidator
        >>> validator = ConfigValidator()
        >>> warnings = validator.validate_env()
        >>> if warnings:
        ...     for warning in warnings:
        ...         print(f"Warning: {warning}")

    Validate model configuration:
        >>> config = {"memory_threshold": 0.9, "num_inference_steps": 4}
        >>> errors = validator.validate_model_config(config)
        >>> if errors:
        ...     raise ValueError(f"Invalid configuration: {errors}")
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.logging_config import logger


class ConfigValidator:
    """Configuration validator for FluxPipeline settings."""

    @staticmethod
    def validate_env() -> List[str]:
        """Validate environment variables.

        Checks for:
        - Valid CUDA_VISIBLE_DEVICES format
        - Workspace directory existence
        - HF_TOKEN presence (warning if missing)
        - Valid log level
        - Memory threshold range

        Returns:
            List of warning messages (empty if all valid)
        """
        warnings = []

        # Check GPU visibility
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            devices = os.environ["CUDA_VISIBLE_DEVICES"]
            # Should be comma-separated integers or empty string
            if devices and not all(d.strip().isdigit() for d in devices.split(",")):
                warnings.append(
                    f"Invalid CUDA_VISIBLE_DEVICES format: '{devices}'. "
                    "Expected comma-separated integers (e.g., '0,1,2')"
                )

        # Check workspace directory
        workspace = os.getenv("WORKSPACE_DIR", "./workspace")
        workspace_path = Path(workspace)
        if not workspace_path.exists():
            warnings.append(
                f"Workspace directory does not exist: {workspace}. "
                "It will be created on first run."
            )

        # Check HuggingFace token (warning, not error)
        if "HF_TOKEN" not in os.environ and "HUGGING_FACE_HUB_TOKEN" not in os.environ:
            warnings.append(
                "No HuggingFace token found (HF_TOKEN or HUGGING_FACE_HUB_TOKEN). "
                "Some models may not be accessible. "
                "Set HF_TOKEN in .env if needed."
            )

        # Check log level
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if log_level not in valid_levels:
            warnings.append(
                f"Invalid LOG_LEVEL: '{log_level}'. "
                f"Valid options: {', '.join(valid_levels)}"
            )

        # Check memory threshold
        memory_threshold = os.getenv("MEMORY_THRESHOLD")
        if memory_threshold:
            try:
                threshold_val = float(memory_threshold)
                if not 0 < threshold_val <= 1:
                    warnings.append(
                        f"MEMORY_THRESHOLD must be between 0 and 1, got {threshold_val}"
                    )
            except ValueError:
                warnings.append(
                    f"MEMORY_THRESHOLD must be a float, got '{memory_threshold}'"
                )

        return warnings

    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> List[str]:
        """Validate model configuration parameters.

        Args:
            config: Dictionary containing model configuration

        Returns:
            List of error messages (empty if all valid)
        """
        errors = []

        # Validate memory threshold
        if "memory_threshold" in config:
            threshold = config["memory_threshold"]
            if not isinstance(threshold, (int, float)):
                errors.append(
                    f"memory_threshold must be numeric, got {type(threshold).__name__}"
                )
            elif not 0 < threshold <= 1:
                errors.append(
                    f"memory_threshold must be between 0 and 1, got {threshold}"
                )

        # Validate num_inference_steps
        if "num_inference_steps" in config:
            steps = config["num_inference_steps"]
            if not isinstance(steps, int):
                errors.append(
                    f"num_inference_steps must be an integer, got {type(steps).__name__}"
                )
            elif steps < 1:
                errors.append(
                    f"num_inference_steps must be at least 1, got {steps}"
                )
            elif steps > 100:
                errors.append(
                    f"num_inference_steps seems too high ({steps}). "
                    "FLUX.1-schnell typically uses 1-4 steps."
                )

        # Validate guidance_scale
        if "guidance_scale" in config:
            scale = config["guidance_scale"]
            if not isinstance(scale, (int, float)):
                errors.append(
                    f"guidance_scale must be numeric, got {type(scale).__name__}"
                )
            elif scale < 0:
                errors.append(
                    f"guidance_scale must be non-negative, got {scale}"
                )
            elif scale > 20:
                errors.append(
                    f"guidance_scale seems too high ({scale}). "
                    "Typical range is 0-10. FLUX.1-schnell uses 0.0."
                )

        # Validate image dimensions
        for dim in ["height", "width"]:
            if dim in config:
                value = config[dim]
                if not isinstance(value, int):
                    errors.append(
                        f"{dim} must be an integer, got {type(value).__name__}"
                    )
                elif value < 256:
                    errors.append(
                        f"{dim} must be at least 256, got {value}"
                    )
                elif value > 2048:
                    errors.append(
                        f"{dim} seems too large ({value}). "
                        "Values over 2048 may cause OOM errors."
                    )
                elif value % 8 != 0:
                    errors.append(
                        f"{dim} must be divisible by 8, got {value}"
                    )

        # Validate model name
        if "model_name" in config:
            model = config["model_name"]
            if not isinstance(model, str):
                errors.append(
                    f"model_name must be a string, got {type(model).__name__}"
                )
            elif not model:
                errors.append("model_name cannot be empty")

        # Validate torch dtype
        if "torch_dtype" in config:
            dtype = config["torch_dtype"]
            valid_dtypes = ["float32", "float16", "bfloat16"]
            if isinstance(dtype, str) and dtype not in valid_dtypes:
                errors.append(
                    f"torch_dtype must be one of {valid_dtypes}, got '{dtype}'"
                )

        return errors

    @staticmethod
    def validate_generation_params(
        prompt: str,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: Optional[int] = None
    ) -> List[str]:
        """Validate parameters for image generation.

        Args:
            prompt: Text prompt for generation
            height: Image height in pixels
            width: Image width in pixels
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            seed: Random seed (optional)

        Returns:
            List of error messages (empty if all valid)
        """
        errors = []

        # Validate prompt
        if not prompt or not isinstance(prompt, str):
            errors.append("Prompt must be a non-empty string")
        elif len(prompt) > 10000:
            errors.append(f"Prompt too long ({len(prompt)} chars). Max 10000 chars.")

        # Validate dimensions
        if height < 256 or height > 2048:
            errors.append(f"Height must be between 256 and 2048, got {height}")
        if width < 256 or width > 2048:
            errors.append(f"Width must be between 256 and 2048, got {width}")
        if height % 8 != 0:
            errors.append(f"Height must be divisible by 8, got {height}")
        if width % 8 != 0:
            errors.append(f"Width must be divisible by 8, got {width}")

        # Validate inference steps
        if num_inference_steps < 1 or num_inference_steps > 100:
            errors.append(
                f"num_inference_steps must be between 1 and 100, got {num_inference_steps}"
            )

        # Validate guidance scale
        if guidance_scale < 0 or guidance_scale > 20:
            errors.append(
                f"guidance_scale must be between 0 and 20, got {guidance_scale}"
            )

        # Validate seed
        if seed is not None:
            if not isinstance(seed, int):
                errors.append(f"seed must be an integer, got {type(seed).__name__}")
            elif seed < 0:
                errors.append(f"seed must be non-negative, got {seed}")

        return errors

    @staticmethod
    def validate_paths(config: Dict[str, Any]) -> List[str]:
        """Validate file and directory paths in configuration.

        Args:
            config: Dictionary containing path configurations

        Returns:
            List of error messages (empty if all valid)
        """
        errors = []

        # Check workspace path
        if "workspace" in config:
            workspace = Path(config["workspace"])
            if workspace.exists() and not workspace.is_dir():
                errors.append(
                    f"Workspace path exists but is not a directory: {workspace}"
                )

        # Check model cache directory
        if "model_cache_dir" in config:
            cache_dir = Path(config["model_cache_dir"])
            if cache_dir.exists() and not cache_dir.is_dir():
                errors.append(
                    f"Model cache path exists but is not a directory: {cache_dir}"
                )

        # Check log file path
        if "log_file" in config:
            log_file = Path(config["log_file"])
            if log_file.exists() and not log_file.is_file():
                errors.append(
                    f"Log path exists but is not a file: {log_file}"
                )
            # Check if parent directory exists
            if not log_file.parent.exists():
                errors.append(
                    f"Log file parent directory does not exist: {log_file.parent}"
                )

        return errors

    @classmethod
    def validate_all(
        cls,
        model_config: Optional[Dict[str, Any]] = None,
        check_env: bool = True,
        check_paths: bool = True
    ) -> tuple[List[str], List[str]]:
        """Run all validations and return warnings and errors.

        Args:
            model_config: Optional model configuration to validate
            check_env: Whether to check environment variables
            check_paths: Whether to validate paths

        Returns:
            Tuple of (warnings, errors)
        """
        warnings = []
        errors = []

        # Validate environment
        if check_env:
            env_warnings = cls.validate_env()
            warnings.extend(env_warnings)

        # Validate model config
        if model_config:
            config_errors = cls.validate_model_config(model_config)
            errors.extend(config_errors)

            # Validate paths in config
            if check_paths:
                path_errors = cls.validate_paths(model_config)
                errors.extend(path_errors)

        return warnings, errors


def validate_and_log(
    model_config: Optional[Dict[str, Any]] = None,
    check_env: bool = True,
    check_paths: bool = True,
    raise_on_error: bool = False
) -> bool:
    """Validate configuration and log results.

    Convenience function that validates and logs warnings/errors.

    Args:
        model_config: Optional model configuration to validate
        check_env: Whether to check environment variables
        check_paths: Whether to validate paths
        raise_on_error: Whether to raise exception on validation errors

    Returns:
        True if validation passed (no errors), False otherwise

    Raises:
        ValueError: If raise_on_error is True and errors are found
    """
    validator = ConfigValidator()
    warnings, errors = validator.validate_all(model_config, check_env, check_paths)

    # Log warnings
    for warning in warnings:
        logger.warning(f"Configuration warning: {warning}")

    # Log errors
    for error in errors:
        logger.error(f"Configuration error: {error}")

    if errors:
        if raise_on_error:
            raise ValueError(f"Configuration validation failed: {errors}")
        return False

    if not warnings:
        logger.info("✅ Configuration validation passed")
    else:
        logger.info(f"Configuration validation passed with {len(warnings)} warnings")

    return True


if __name__ == "__main__":
    # Test validation
    print("=" * 60)
    print("Testing Configuration Validation")
    print("=" * 60)

    validator = ConfigValidator()

    print("\n1. Testing environment validation:")
    env_warnings = validator.validate_env()
    if env_warnings:
        for warning in env_warnings:
            print(f"  ⚠️  {warning}")
    else:
        print("  ✅ Environment validation passed")

    print("\n2. Testing model config validation:")
    test_config = {
        "memory_threshold": 0.9,
        "num_inference_steps": 4,
        "guidance_scale": 0.0,
        "height": 1024,
        "width": 1024,
    }
    config_errors = validator.validate_model_config(test_config)
    if config_errors:
        for error in config_errors:
            print(f"  ❌ {error}")
    else:
        print("  ✅ Model config validation passed")

    print("\n3. Testing invalid config:")
    invalid_config = {
        "memory_threshold": 1.5,  # Invalid: > 1
        "num_inference_steps": -1,  # Invalid: < 1
        "height": 1023,  # Invalid: not divisible by 8
    }
    invalid_errors = validator.validate_model_config(invalid_config)
    print(f"  Found {len(invalid_errors)} errors (expected):")
    for error in invalid_errors:
        print(f"    - {error}")

    print("\n4. Testing generation params:")
    gen_errors = validator.validate_generation_params(
        prompt="A beautiful landscape",
        height=1024,
        width=1024,
        num_inference_steps=4,
        guidance_scale=0.0,
        seed=42
    )
    if gen_errors:
        for error in gen_errors:
            print(f"  ❌ {error}")
    else:
        print("  ✅ Generation params validation passed")

    print("\n" + "=" * 60)
