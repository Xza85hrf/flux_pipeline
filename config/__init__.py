"""Configuration modules for FluxPipeline.

This package contains configuration and setup utilities:
    - Environment configuration and hardware detection
    - Logging setup with colored output
    - Default model configurations
    - Configuration validation
"""

from .env_config import (
    setup_environment,
    ensure_environment,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_GENERATION_CONFIG,
)
from .logging_config import logger, setup_logging, ColorFormatter
from .validator import ConfigValidator, validate_and_log

__all__ = [
    # Environment Configuration
    "setup_environment",
    "ensure_environment",
    "DEFAULT_MODEL_CONFIG",
    "DEFAULT_GENERATION_CONFIG",
    # Logging
    "logger",
    "setup_logging",
    "ColorFormatter",
    # Validation
    "ConfigValidator",
    "validate_and_log",
]
