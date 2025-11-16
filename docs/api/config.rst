Configuration Module
====================

The config module handles environment setup, logging, and configuration validation.

Environment Configuration
--------------------------

.. automodule:: config.env_config
   :members:
   :undoc-members:
   :show-inheritance:

Logging Configuration
---------------------

.. automodule:: config.logging_config
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Validation
-------------------------

.. automodule:: config.validator
   :members:
   :undoc-members:
   :show-inheritance:

ConfigValidator
~~~~~~~~~~~~~~~

.. autoclass:: config.validator.ConfigValidator
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Environment Setup
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from config import setup_environment, logger

   # Setup environment (detects GPU, configures settings)
   setup_environment()

   # Use logger
   logger.info("Environment configured successfully")

Configuration Validation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from config import ConfigValidator, validate_and_log

   # Validate environment
   validator = ConfigValidator()
   warnings = validator.validate_env()

   if warnings:
       for warning in warnings:
           print(f"⚠️  {warning}")

   # Validate model config
   config = {
       "memory_threshold": 0.9,
       "num_inference_steps": 4,
       "height": 1024,
       "width": 1024
   }

   errors = validator.validate_model_config(config)
   if errors:
       print("Configuration errors:", errors)

   # Quick validation with logging
   validate_and_log(config, raise_on_error=True)

Custom Logging
~~~~~~~~~~~~~~

.. code-block:: python

   from config import logger, setup_logging
   from pathlib import Path

   # Custom log file
   setup_logging(Path("my_custom.log"))

   # Use logger
   logger.debug("Debug message")
   logger.info("Info message")
   logger.warning("Warning message")
   logger.error("Error message")
