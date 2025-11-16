"""Environment-specific configuration loader.

This module loads configuration from YAML files based on the environment.
Environment is determined by FLUXPIPELINE_ENV environment variable.

Examples:
    Load configuration for current environment:
        >>> from config.env_loader import load_env_config
        >>> config = load_env_config()
        >>> print(config['environment'])

    Load specific environment:
        >>> config = load_env_config('production')
        >>> print(config['logging']['level'])

    Use configuration in code:
        >>> config = load_env_config()
        >>> if config['debug']['save_intermediate_outputs']:
        ...     save_intermediates()
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from config.logging_config import logger


class EnvironmentConfig:
    """Environment configuration manager.

    Handles loading and accessing environment-specific configuration
    from YAML files.

    Attributes:
        env_name: Current environment name
        config: Loaded configuration dictionary
        config_dir: Directory containing environment configs
    """

    def __init__(self, env_name: Optional[str] = None):
        """Initialize environment configuration.

        Args:
            env_name: Environment name (development, production, testing).
                     If None, uses FLUXPIPELINE_ENV environment variable.
                     Defaults to 'development' if not set.
        """
        self.env_name = env_name or os.getenv("FLUXPIPELINE_ENV", "development")
        self.config_dir = Path(__file__).parent / "environments"
        self.config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        config_file = self.config_dir / f"{self.env_name}.yaml"

        if not config_file.exists():
            logger.warning(
                f"Configuration file not found: {config_file}. "
                f"Using default configuration."
            )
            self.config = self._get_default_config()
            return

        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_file}: {e}")
            self.config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration when file is not found.

        Returns:
            Default configuration dictionary
        """
        return {
            "environment": "development",
            "logging": {
                "level": "INFO",
                "console": True,
            },
            "gpu": {
                "vendor": "cuda",
                "memory_threshold": 0.9,
            },
            "model": {
                "name": "black-forest-labs/FLUX.1-schnell",
                "torch_dtype": "float16",
            },
            "generation": {
                "default_height": 1024,
                "default_width": 1024,
                "default_steps": 4,
                "default_guidance_scale": 0.0,
            },
            "workspace": {
                "base_dir": "./workspace",
            },
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Supports nested keys with dot notation.

        Args:
            key: Configuration key (e.g., "logging.level")
            default: Default value if key not found

        Returns:
            Configuration value or default

        Examples:
            >>> config = EnvironmentConfig()
            >>> level = config.get("logging.level")
            >>> height = config.get("generation.default_height", 512)
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value if value is not None else default

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section.

        Args:
            section: Section name (e.g., "logging", "gpu")

        Returns:
            Section configuration dictionary

        Examples:
            >>> config = EnvironmentConfig()
            >>> logging_config = config.get_section("logging")
            >>> print(logging_config['level'])
        """
        return self.config.get(section, {})

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.

        Args:
            updates: Dictionary of configuration updates

        Examples:
            >>> config = EnvironmentConfig()
            >>> config.update({"logging": {"level": "DEBUG"}})
        """
        self._deep_update(self.config, updates)
        logger.debug(f"Updated configuration with: {updates}")

    def _deep_update(self, base: Dict, updates: Dict) -> None:
        """Recursively update nested dictionaries.

        Args:
            base: Base dictionary to update
            updates: Updates to apply
        """
        for key, value in updates.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary.

        Returns:
            Complete configuration dictionary
        """
        return self.config.copy()

    def __repr__(self) -> str:
        """String representation of config."""
        return f"EnvironmentConfig(env={self.env_name})"


# Global instance
_env_config: Optional[EnvironmentConfig] = None


def load_env_config(env_name: Optional[str] = None) -> Dict[str, Any]:
    """Load environment configuration.

    This is a convenience function that returns the configuration dictionary.

    Args:
        env_name: Environment name (optional)

    Returns:
        Configuration dictionary

    Examples:
        >>> config = load_env_config()
        >>> print(config['environment'])

        >>> prod_config = load_env_config('production')
        >>> print(prod_config['logging']['level'])
    """
    global _env_config

    if _env_config is None or (env_name and env_name != _env_config.env_name):
        _env_config = EnvironmentConfig(env_name)

    return _env_config.to_dict()


def get_env_config() -> EnvironmentConfig:
    """Get global environment configuration instance.

    Returns:
        EnvironmentConfig instance

    Examples:
        >>> config = get_env_config()
        >>> level = config.get("logging.level")
    """
    global _env_config

    if _env_config is None:
        _env_config = EnvironmentConfig()

    return _env_config


if __name__ == "__main__":
    # Test environment configuration
    print("Testing Environment Configuration\n")

    # Test loading different environments
    for env in ["development", "production", "testing"]:
        print(f"Loading {env} environment...")
        config = load_env_config(env)
        print(f"  Environment: {config.get('environment')}")
        print(f"  Log Level: {config.get('logging', {}).get('level')}")
        print(f"  Default Size: {config.get('generation', {}).get('default_height')}x{config.get('generation', {}).get('default_width')}")
        print()

    # Test dot notation
    config_obj = get_env_config()
    print("Testing dot notation:")
    print(f"  logging.level = {config_obj.get('logging.level')}")
    print(f"  generation.default_height = {config_obj.get('generation.default_height')}")
    print(f"  gpu.vendor = {config_obj.get('gpu.vendor')}")
